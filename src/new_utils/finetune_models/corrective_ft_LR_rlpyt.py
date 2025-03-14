import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
import pdb
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil

from collections import deque
from functools import partial
from new_utils.model_utils import weight_init
from new_utils.new_agent.encoder import cnn_mlp
from new_utils.tensor_utils import torchify_buffer, get_leading_dims, numpify_buffer
from new_utils.draw_utils import ax_plot_img, ax_plot_bar, ax_plot_heatmap

from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.buffer import buffer_from_example

SamplesToCFDataBuffer = namedarraytuple("SamplesToCFDataBuffer",
                            ["observation", "done", "action", "GT_reward",
                            "oracle_act", "oracle_act_prob", "oracle_q"])
# QueriesToRewardBuffer = namedarraytuple("QueriesToRewardBuffer",
#     ["observation", "action", "oracle_act"])

class CorrectiveRlpytFinetuneLRModel:
    def __init__(self,
                 B,
                 obs_shape,
                 action_dim,
                 OptimCls,
                 optim_kwargs,
                 gamma,
                 lr=3e-4,
                 query_batch=128,  # old name in PEBBLE is mb_size
                 train_batch_size=128,
                 size_segment=25,  # timesteps per segment
                 cf_per_seg=2,  # number of corrective feedback per segment
                 max_size=100000,  # max timesteps of trajectories for query sampling
                 label_capacity=3000,  # "labeled" query buffer capacity, each query corresponds to a segment (old name in PEBBLE is capacity default 5e5)
                 device='cuda',
                 env_name=None,
                 loss_name='Exp',
                 clip_grad_norm=None,
                 exp_loss_beta=1.,
                 loss_margine=0.1,  # margine used to train reward model
                 margine_decay=None,
                 loss_square=False,
                 loss_square_coef=1.,
                 oracle_type='oe',  # {'oe': 'oracle_entropy', 'oq': 'oracle_q_value'}
                 softmax_tau=1.0,
                 ckpt_path=None,
                 distributional=True,
                 log_q_path=None,
                 total_itr=None,
                 ignore_small_qdiff=None,
                 LR=None,
                 ):
        self.device = device

        self.obs_shape = obs_shape  # (frame_stack, H, W)
        self.action_dim = action_dim  # number of available discrete actions
        self.max_entropy = -self.action_dim * ((1.0/self.action_dim) * np.log((1.0/self.action_dim)))
        print(f"[FT Model] max_entropy {self.max_entropy} for action_dim {self.action_dim}.")
        # self.model_input_shape = self.obs_shape[:]  # (frame_stack, H, W)
        # self.model_input_shape[0] += self.action_dim  # input shape for reward models: [frame_stack+action_dim, H, W]
        self.B = int(B)
        assert self.B == 1

        self.OptimCls = OptimCls
        self.optim_kwargs = optim_kwargs
        self.lr = lr

        self.gamma = gamma

        self.opt = None
        assert max_size % self.B == 0
        self.max_size_T = int(max_size // self.B)
        self.max_size = max_size
        self.loss_square = loss_square
        self.loss_square_coef = loss_square_coef
        self.size_segment = size_segment

        self.cf_per_seg = cf_per_seg
        self.clip_grad_norm = clip_grad_norm

        self.label_capacity = int(label_capacity)  # "labeled" query buffer capacity

        self.query_batch = query_batch  # reward batch size
        self.origin_query_batch = query_batch  # reward batch size may change according to the current training step
        self.train_batch_size = train_batch_size
        
        self.env_name = env_name

        if loss_name == 'Exp':
            self.finetune_loss = self.finetune_exp_loss
            self.exp_loss_beta = exp_loss_beta
            if margine_decay.flag:
                raise NotImplementedError
        elif loss_name == 'MargineLossDQfD':
            self.finetune_loss = self.finetune_margine_loss_DQfD
            self.init_loss_margine = loss_margine
            self.loss_margine = loss_margine
            if self.loss_square:
                raise NotImplementedError
        elif loss_name == 'MargineLossMin0Fix':
            self.finetune_loss = self.finetune_margine_loss_min0_fix
            self.init_loss_margine = loss_margine
            self.loss_margine = loss_margine
            if margine_decay.flag:
                raise NotImplementedError
            if self.loss_square:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        self.loss_name = loss_name
        self.margine_decay = margine_decay
        
        self.oracle_type = oracle_type
        assert self.oracle_type in ['oe','oq', 'oqLRd']
        self.softmax_tau = softmax_tau
        self.ckpt_path = ckpt_path

        self.distributional = distributional
        self.log_q_path = log_q_path
        if self.log_q_path:
            os.makedirs(self.log_q_path, exist_ok=True)
        self.total_itr = total_itr
        self.ignore_small_qdiff = ignore_small_qdiff
        self.LR = LR
        if self.LR.flag:
            self.label_capacity += 10  # to avoid label buffer capacity & index issue in self.put_queries_with_LR
            if self.LR.buffer_type == 1:  # kept all LR relation
                self.LR_buffer_capacity = self.label_capacity
            elif self.LR.buffer_type == 0:  # only kept LR relation w.r.t segments generated from the current policy
                self.LR_buffer_capacity = self.origin_query_batch
            else:
                raise NotImplementedError
            
            # assert self.LR.schedule in ['b', 'a', 'm']  # [b, a, m] -> [before_SL, after_SL, mixed_only]
            assert self.LR.RR_type in [0, 1]  # when calculate the return of the right segment: # 0: start from Q(s_t,a_t); 1: start from Q(s_{t+1},a_{t+1})
            assert self.LR.scale in [0, 1, 2]  # 0: longest subseg related to s_t with reward normalization; 1: longest same-length-subseg related to s_t; 2: every subseg related to s_t;
        self.gamma_exp = torch.tensor([self.gamma**k for k in range(self.size_segment)])
        self.discount_norm = torch.zeros_like(self.gamma_exp)
        self.discount_norm[0] = 1.
        for k in range(1, self.size_segment):
            self.discount_norm[k] = self.discount_norm[k - 1] + self.gamma_exp[k]
        self.np_gamma_exp = self.gamma_exp.detach().cpu().numpy()
        self.np_discount_norm = self.discount_norm.detach().cpu().numpy()
        self.gamma_exp = self.gamma_exp.float().to(self.device)
        self.discount_norm = self.discount_norm.float().to(self.device)

    def config_agent(self, agent):
        self.agent = agent
        if ("eps" not in self.optim_kwargs) or\
           (self.optim_kwargs["eps"] is None):  # Assume optim.Adam
            self.optim_kwargs["eps"] = 0.01 / self.train_batch_size
        # Because agent.search only related to agent.model and do
        #   not need agent.target_model, so only need to optimize this
        self.opt = eval(self.OptimCls)(self.agent.model.parameters(),
                        lr=self.lr, **self.optim_kwargs)

        if self.distributional:
            self.distri_z = torch.linspace(
                self.agent.V_min, self.agent.V_max, self.agent.n_atoms).\
                to(self.device)

    def initialize_buffer(self, agent, oracle, env,
                          check_label_path=None, segment_log_path=None):
        sample_examples = dict()
        label_examples = dict()
        # From get_example_outputs(agent, env, examples)
        env.reset()
        a = env.action_space.sample()
        o, r, terminated, truncated, env_info = env.step(a)
        r = np.asarray(r, dtype="float32")  # Must match torch float dtype here.
        # agent.reset()
        agent_inputs = torchify_buffer(o)
        a, agent_info = agent.step(agent_inputs)
        oracle_a, oracle_info = oracle.step(agent_inputs)
        assert agent_info.value.shape == oracle_info.value.shape

        # NOTE: !!!! the order of keys in sample_examples should be consistent with samples_to_reward_buffer() 
        sample_examples["observation"] = o[:]
        sample_examples["done"] = terminated
        sample_examples["action"] = a
        sample_examples["GT_reward"] = r
        sample_examples["oracle_act"] = oracle_a
        sample_examples["oracle_act_prob"] = F.softmax(oracle_info.value, dim=-1)  # (action_dim,)
        sample_examples["oracle_q"] = oracle_info.value  # (action_dim,)

        field_names = [f for f in sample_examples.keys() if f != "observation"]
        global RewardBufferSamples
        RewardBufferSamples = namedarraytuple("RewardBufferSamples", field_names)
        
        reward_buffer_example = RewardBufferSamples(*(v for \
                                    k, v in sample_examples.items() \
                                    if k != "observation"))
        self.input_buffer = buffer_from_example(reward_buffer_example,
                                                (self.max_size_T, self.B))
        # self.input_buffer.action&oracle_act.shape: (self.max_size_T, self.B), int64
        # self.input_buffer.oracle_act_prob.shape: (self.max_size_T, self.B, act_dim), float32
        self.n_frames = n_frames = get_leading_dims(o[:], n_dim=1)[0]
        print(f"[FT Buffer-Inputs] Frame-based buffer using {n_frames}-frame sequences.")
        self.input_frames = buffer_from_example(o[0],  # avoid saving duplicated frames
                                (self.max_size_T + n_frames - 1, self.B))
        # self.input_frames.shape: (self.max_size_T+n_frames-1, self.B, H, W), uint8
        self.input_new_frames = self.input_frames[n_frames - 1:]  # [self.max_size_T,B,H,W]
        
        self.input_t = 0
        self._input_buffer_full = False
        
        label_examples["observation"] = o[:]
        label_examples["action"] = a
        label_examples["oracle_act"] = oracle_a
        if self.loss_name in ['MargineLossDQfD', 'MargineLossMin0Fix']:
            label_examples["margine"] = self.init_loss_margine
        if self.LR.flag:
            label_examples["with_LR"] = False
            label_examples["LR_index"] = 0
        
        global RewardLabelBufferSamples
        RewardLabelBufferSamples = namedarraytuple("RewardLabelBufferSamples",
                                                   label_examples.keys())
        reward_label_buffer_example = RewardLabelBufferSamples(*(v for \
                                            k, v in label_examples.items()))
        
        self.label_buffer = buffer_from_example(reward_label_buffer_example,
                                                (self.label_capacity,))
        
        if self.LR.flag:  # LR relation should only works on trajectories sampled from the current policy
            assert self.cf_per_seg == 1
            # assert (self.ignore_small_qdiff.flag and self.ignore_small_qdiff.thres>=0.)  # this is not necessary, but I think's it's the preferred one
            LR_example = dict()
            LR_example["observation"] = np.repeat(np.expand_dims(o[:],axis=0),
                                            repeats=self.size_segment, axis=0)  # shape: (size_seg, *obs_shape)
            LR_example["action"] = np.repeat([a], repeats=self.size_segment, axis=0)  # shape: (size_seg,)
            LR_example["oracle_action"] = np.repeat([a], repeats=self.size_segment, axis=0)  # shape: (size_seg,)
            LR_example["CF_index"] = 0
            global LRBufferSamples
            LRBufferSamples = namedarraytuple("LRBufferSamples",
                                              LR_example.keys())
            LR_buffer_example = LRBufferSamples(*(v for \
                                                k, v in LR_example.items()))
            self.LR_buffer = buffer_from_example(LR_buffer_example,
                                                 (self.LR_buffer_capacity,))
            self.LR_buffer_t = 0

        if self.log_q_path:
            qlog_examples = dict()
            qlog_examples["observation"] = copy.deepcopy(o[:])
            qlog_examples["oracle_act"] = oracle_a
            qlog_examples["oracle_q"] = copy.deepcopy(oracle_info.value)  # oracle_q: (action_dim,)
            qlog_examples["agent_act"] = a
            qlog_examples["agent_q"] = np.repeat(copy.deepcopy(agent_info.value).reshape(1, -1),
                                                  repeats=self.total_itr+1,  # extra one for the initial ckpt
                                                  axis=0)  # agent_q: (total_itr+1, action_dim)
            qlog_examples["margine"] = self.init_loss_margine

            # note that return_0 and return_1 are considered for same length
            qlog_examples["Q_diff_aE"] = 0.0  # Q_E(s_t,a_E) - Q_E(s_t,a_t)
            qlog_examples["num_ne_act"] = 0  # number of non-expert actions in this segment
            if self.cf_per_seg == 1:
                qlog_examples["return_l"] = [0.0] * (self.size_segment // 2)  # discounted return - left, before CF
                qlog_examples["return_r"] = [0.0] * (self.size_segment // 2) # discounted return - right, after CF
                self.oracle_return_l_large = [0] * (self.size_segment // 2) 
                self.oracle_return_cnt = [0] * (self.size_segment // 2) 
            global QLogBufferSamples
            QLogBufferSamples = namedarraytuple("QLogBufferSamples",
                                                qlog_examples.keys())
            qlog_buffer_example = QLogBufferSamples(*(v for \
                                                k, v in qlog_examples.items()))
            
            self.qlog_buffer = buffer_from_example(qlog_buffer_example,
                                                   (self.label_capacity,))  # shape: (capacity, total_itr+1, action_dim)
            self.qlog_t = 0

        # self.label_buffer.observation.shape: (self.label_capacity, C, H, W), uint8
        # self.label_buffer.action&oracle_act.shape: (self.label_capacity,), int64
        self.label_t = 0
        self._label_buffer_full = False
        self.total_label_seg = 0  # the number of segments that have been labeled (queried)
        self.check_label_path = check_label_path
        if self.check_label_path:
            os.makedirs(self.check_label_path, exist_ok=True)
        self.segment_log_path = segment_log_path
        os.makedirs(self.segment_log_path, exist_ok=True)
        self.env_action_meaning = env.unwrapped.get_action_meanings()
    
    def change_batch(self, new_frac):
        self.query_batch = max(int(self.origin_query_batch * new_frac), 1)
    
    def set_batch(self, new_batch):
        self.query_batch = int(new_batch)

    def update_margine(self, itr):
        total_itr = self.total_itr - 1 # -1 since itr counts from 0 (i.e. itr \in [0, 1, ..., total_itr - 1])
        if self.margine_decay.type == 'cosine':
            frac = (itr * 1.0) / (total_itr * 1.0)
            self.loss_margine = self.margine_decay.min_loss_margine + \
                0.5 * (1.0 + np.cos(frac * np.pi)) * (self.init_loss_margine - self.margine_decay.min_loss_margine)
        elif self.margine_decay.type == 'linear':
            frac = 1.0 - (itr * 1.0) / (total_itr * 1.0)
            self.loss_margine = self.margine_decay.min_loss_margine + \
                frac * (self.init_loss_margine - self.margine_decay.min_loss_margine)
        else:
            raise NotImplementedError
        return self.loss_margine

    def add_data(self, samples):
        # NOTE: extra penalty for episode end
        # NOTE: concatenate all episodes into a single one
        # TODO: Then what if one segment contains two episodes?
        # explanation from RLHF: 
        #    In Atari games, we do not send life loss or episode end signals to the agent 
        #    (we do continue to actually reset the environment),
        #    effectively converting the environment into a single continuous episode.
        #    When providing synthetic oracle feedback we replace episode ends with a penalty
        #    in all games except Pong; the agent must learn this penalty.
        
        # add one batch & one step data
        # For img states, we save obs and act separately to save memory
        # (T, B) for done, action, oracle_act; (T, B, action_dim) for oracle_act_prob, (T, B, 4, H, W) for observation
        t, fm1 = self.input_t, self.n_frames - 1
        reward_buffer_samples = RewardBufferSamples(*(v \
                                for k, v in samples.items() \
                                if k != "observation"))
        T, B = get_leading_dims(reward_buffer_samples, n_dim=2)  # samples.env.reward.shape[:2]
        assert B == self.B
        if t + T > self.max_size_T:  # Wrap.
            idxs = np.arange(t, t + T) % self.max_size_T
        else:
            idxs = slice(t, t + T)
        self.input_buffer[idxs] = reward_buffer_samples
        if not self._input_buffer_full and t + T >= self.max_size_T:
            self._input_buffer_full = True  # Only changes on first around.
        self.input_t = (t + T) % self.max_size_T

        assert samples.observation.ndim == 5 and\
               samples.observation.shape[2] == fm1 + 1 # (T, B, frame_stack, H, W)
        # self.samples_new_frames[idxs] = samples.observation[:, :, -1]
        # self.input_frames.shape: (size+fm1, B, H, W)
        self.input_new_frames[idxs] = samples.observation[:, :, -1]
        if t == 0:  # Starting: write early frames
            for f in range(fm1):
                self.input_frames[f] = samples.observation[0, :, f]
        elif self.input_t <= t and fm1 > 0:  # Wrapped: copy any duplicate frames. In the case that T==self.max_size_T, self.input_t will == t
            self.input_frames[:fm1] = self.input_frames[-fm1:]
        # return T, idxs
    
    def save(self, model_dir, title):
        for member in range(self.de):
            torch.save(
                self.ensemble[member].state_dict(),
                '%s/reward_model_%s_%s.pt' % (model_dir, title, member)
            )
    
    def load(self, model_dir, title):
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load('%s/reward_model_%s_%s.pt' % (model_dir, title, member))
            )

    def load_init(self, model_dir, title):
        self.init_model_dict = []
        for member in range(self.de):
            self.init_model_dict.append(
                torch.load('%s/reward_model_%s_%s.pt' % (model_dir, title, member))
            )
    
    @property
    def len_inputs_T(self):
        return self.input_t if not self._input_buffer_full else self.max_size_T
    
    @property
    def len_label(self):
        return self.label_t if not self._label_buffer_full else self.label_capacity
    
    def extract_observation(self, T_idxs, B_idxs):
        """Assembles multi-frame observations from frame-wise buffer.  Frames
        are ordered OLDEST to NEWEST along C dim: [B,C,H,W].  Where
        ``done=True`` is found, the history is not full due to recent
        environment reset, so these frames are zero-ed.
        """
        # Begin/end frames duplicated in samples_frames so no wrapping here.
        # return np.stack([self.samples_frames[t:t + self.n_frames, b]
        #     for t, b in zip(T_idxs, B_idxs)], axis=0)  # [B,C,H,W]
        observation = np.stack([self.input_frames[t:t + self.n_frames, b]
            for t, b in zip(T_idxs, B_idxs)], axis=0)  # [B,C,H,W]
        
        # Populate empty (zero) frames after environment done.
        for f in range(1, self.n_frames):
            # e.g. if done 1 step prior, all but newest frame go blank.
            b_blanks = np.where(self.input_buffer.done[T_idxs - f, B_idxs])[0]
            observation[b_blanks, :self.n_frames - f] = 0
        return observation

    def get_queries(self, query_batch, use_seg=False):
        # check shape. TODO: consider if done=True should be considered;
        # get train traj
        max_index_T = self.len_inputs_T - self.size_segment
        # Batch = query_batch
        if use_seg:
            assert query_batch is None
            assert self.B == 1
            batch_index = np.arange(0, max_index_T, step=self.size_segment).reshape(-1, 1)  # (query_batch, 1)
            query_batch = batch_index.shape[0]
        else:
            assert (max_index_T * self.B) > query_batch
            batch_index = np.random.choice(max_index_T*self.B, size=query_batch, replace=True).reshape(-1, 1)  # (query_batch, 1)
        
        batch_index_T, batch_index_B = np.divmod(batch_index, self.B)  # (x // y, x % y), batch_index_B & batch_index_T.shape: (query_batch, 1)
        take_index_T = (batch_index_T + np.arange(0, self.size_segment)).reshape(-1)  # shape: (query_batch * size_segment, )
        take_index_B = batch_index_B.repeat(self.size_segment, axis=-1).reshape(-1)  # shape: (query_batch * size_segment, )

        # obs_t = self.input_frames[take_index_B, take_index_T, ...]  # (query_batch * size_segment, *obs_shape)
        obs_t = self.extract_observation(take_index_T, take_index_B)  # (query_batch * size_segment, *obs_shape)
        assert (obs_t.ndim == 4) and (obs_t.shape[0] == query_batch * self.size_segment)
        obs_t = obs_t.reshape(query_batch, self.size_segment, *self.obs_shape)  # (query_batch, size_segment, *obs_shape)

        # NOTE: for ndarray, array can be indexed with other array
        act_t = self.input_buffer.action[take_index_T, take_index_B]  # (query_batch * size_segment, )
        act_t = act_t.reshape(query_batch, self.size_segment)  # (query_batch, size_segment)

        oracle_act_t = self.input_buffer.oracle_act[take_index_T, take_index_B]  # (query_batch * size_segment, )
        oracle_act_t = oracle_act_t.reshape(query_batch, self.size_segment)  # (query_batch, size_segment)

        oracle_act_prob_t = self.input_buffer.oracle_act_prob[take_index_T, take_index_B]  # (query_batch * size_segment, action_dim)
        oracle_act_prob_t = oracle_act_prob_t.reshape(query_batch, self.size_segment, self.action_dim)  # (query_batch, size_segment, action_dim)

        oracle_q_t = self.input_buffer.oracle_q[take_index_T, take_index_B]  # (query_batch * size_segment, action_dim)
        oracle_q_t = oracle_q_t.reshape(query_batch, self.size_segment, self.action_dim)  # (query_batch, size_segment, action_dim)

        GT_reward_t = self.input_buffer.GT_reward[take_index_T, take_index_B]
        GT_reward_t = GT_reward_t.reshape(query_batch, self.size_segment)  # (query_batch, size_segment)

        return obs_t, act_t, oracle_act_t, oracle_act_prob_t, oracle_q_t, GT_reward_t
    
    def put_queries(self, obs_t, act_t, oracle_act_t, early_advertising=False):
        # obs_t.shape: (query_batch * cf_per_seg * neighbor_size, *obs_shape) or (len_label, *obs_shape)
        # act_t & oracle_act_t.shape: (query_batch * cf_per_seg * neighbor_size,) or (len_label,)
        total_sample = obs_t.shape[0]
        next_index = self.label_t + total_sample  # new index in the query buffer after adding new queries
        self.total_label_seg += self.query_batch if (not early_advertising) else 0
        # NOTE: np.copyto(dest, src) is deepcopy for scalars
        if next_index >= self.label_capacity:
            self._label_buffer_full = True
            maximum_index = self.label_capacity - self.label_t
            np.copyto(self.label_buffer.observation[self.label_t:self.label_capacity], obs_t[:maximum_index])
            np.copyto(self.label_buffer.action[self.label_t:self.label_capacity], act_t[:maximum_index])
            np.copyto(self.label_buffer.oracle_act[self.label_t:self.label_capacity], oracle_act_t[:maximum_index])
            self.label_buffer.margine[self.label_t:self.label_capacity] = self.loss_margine

            remain = total_sample - (maximum_index)
            if remain > 0:  # if next_index exceed capacity, extra new queries will be added to the beginning of query buffer
                np.copyto(self.label_buffer.observation[0:remain], obs_t[maximum_index:])
                np.copyto(self.label_buffer.action[0:remain], act_t[maximum_index:])
                np.copyto(self.label_buffer.oracle_act[0:remain], oracle_act_t[maximum_index:])
                self.label_buffer.margine[0:remain] = self.loss_margine

            self.label_t = remain
        else:
            np.copyto(self.label_buffer.observation[self.label_t:next_index], obs_t)
            np.copyto(self.label_buffer.action[self.label_t:next_index], act_t)
            np.copyto(self.label_buffer.oracle_act[self.label_t:next_index], oracle_act_t)
            self.label_buffer.margine[self.label_t:next_index] = self.loss_margine
            self.label_t = next_index

    def put_queries_with_LR(self, obs_t, act_t, oracle_act_t,
                            obs_t_seg, act_t_seg, oracle_act_t_seg, kept_index_timestep_seg,
                            early_advertising=False,):
        # obs_t.shape: (query_batch * cf_per_seg * neighbor_size, *obs_shape) or (len_label, *obs_shape)
        # act_t & oracle_act_t.shape: (query_batch * cf_per_seg * neighbor_size,) or (len_label,)
        total_sample = obs_t.shape[0]
        assert total_sample == obs_t_seg.shape[0] == kept_index_timestep_seg.shape[0]
        # obs_t_seg.shape: (total_sample, size_seg, *obs_shape)
        # act_t_seg.shape = oracle_act_t_seg.shape: (total_sample, size_seg)
        # kept_index_timestep_seg.shape: (total_sample,)
        next_index = self.label_t + total_sample  # new index in the query buffer after adding new queries
        self.total_label_seg += self.query_batch if (not early_advertising) else 0
        
        if self.LR.buffer_type == 0: # only kept LR relation w.r.t segments generated from the current policy
            self.LR_buffer_t = 0
        next_LR_index = self.LR_buffer_t + total_sample  # new index in the query buffer after adding new queries
        assert next_LR_index <= self.LR_buffer_capacity

        # NOTE: np.copyto(dest, src) is deepcopy for scalars
        if next_index >= self.label_capacity:
            assert not self.LR.flag
            self._label_buffer_full = True
            maximum_index = self.label_capacity - self.label_t
            np.copyto(self.label_buffer.observation[self.label_t:self.label_capacity], obs_t[:maximum_index])
            np.copyto(self.label_buffer.action[self.label_t:self.label_capacity], act_t[:maximum_index])
            np.copyto(self.label_buffer.oracle_act[self.label_t:self.label_capacity], oracle_act_t[:maximum_index])
            self.label_buffer.margine[self.label_t:self.label_capacity] = self.loss_margine

            remain = total_sample - (maximum_index)
            if remain > 0:  # if next_index exceed capacity, extra new queries will be added to the beginning of query buffer
                np.copyto(self.label_buffer.observation[0:remain], obs_t[maximum_index:])
                np.copyto(self.label_buffer.action[0:remain], act_t[maximum_index:])
                np.copyto(self.label_buffer.oracle_act[0:remain], oracle_act_t[maximum_index:])
                self.label_buffer.margine[0:remain] = self.loss_margine

            self.label_t = remain
        else:
            np.copyto(self.label_buffer.observation[self.label_t:next_index], obs_t)
            np.copyto(self.label_buffer.action[self.label_t:next_index], act_t)
            np.copyto(self.label_buffer.oracle_act[self.label_t:next_index], oracle_act_t)
            self.label_buffer.margine[self.label_t:next_index] = self.loss_margine

            if self.LR.buffer_type == 0:  # only generated from the current policy
                # reset label_buffer.with_LR and label_buffer.LR_index
                self.label_buffer.with_LR[:] = False  # with_LR.shape: (label_capacity,)
                self.label_buffer.LR_index[:] = 0  # LR_index.shape: (label_capacity,)
            if self.LR.RR_type == 0:
                with_LR_flag = np.where(
                    (kept_index_timestep_seg>0) & (kept_index_timestep_seg<self.size_segment-1),
                    True, False)
            elif self.LR.RR_type == 1:
                with_LR_flag = np.where(
                    (kept_index_timestep_seg>0) & (kept_index_timestep_seg<self.size_segment-2),
                    True, False)
            else:
                raise NotImplementedError
            np.copyto(self.label_buffer.with_LR[self.label_t:next_index], with_LR_flag)
            np.copyto(self.label_buffer.LR_index[self.label_t:next_index], 
                        np.arange(self.LR_buffer_t, next_LR_index))
            self.label_t = next_index
            
            np.copyto(self.LR_buffer.observation[self.LR_buffer_t: next_LR_index], obs_t_seg)
            np.copyto(self.LR_buffer.action[self.LR_buffer_t: next_LR_index], act_t_seg)
            np.copyto(self.LR_buffer.oracle_action[self.LR_buffer_t: next_LR_index], oracle_act_t_seg)
            np.copyto(self.LR_buffer.CF_index[self.LR_buffer_t: next_LR_index], kept_index_timestep_seg)
            self.LR_buffer_t = next_LR_index

    # def put_LR_segments(self, obs_t_seg, act_t_seg, oracle_act_t_seg, kept_index_timestep_seg):
    #     total_sample = obs_t_seg.shape[0]
    #     if self.LR.buffer_type == 0: # only kept LR relation w.r.t segments generated from the current policy
    #         self.LR_buffer_t = 0
    #     next_index = self.LR_buffer_t + total_sample  # new index in the query buffer after adding new queries
    #     assert next_index <= self.LR_buffer_capacity
    #     pdb.set_trace()  # check LR_buffer can be updated correctly
    #     np.copyto(self.LR_buffer.observation[self.LR_buffer_t:next_index], obs_t_seg)
    #     np.copyto(self.LR_buffer.action[self.LR_buffer_t:next_index], act_t_seg)
    #     np.copyto(self.LR_buffer.oracle_action[self.LR_buffer_t:next_index], oracle_act_t_seg)
    #     np.copyto(self.LR_buffer.CF_index[self.LR_buffer_t:next_index], kept_index_timestep_seg)
    #     self.LR_buffer_t = next_index

    def put_qlog(self, itr, obs_t, act_t, oracle_act_t, oracle_q_t):
        if obs_t is not None:
            total_sample = obs_t.shape[0]
            next_index = self.qlog_t + total_sample  # new index in the query buffer after adding new queries
            if next_index > self.label_capacity:
                raise NotImplementedError  # this case need more careful consideration (e.g. maintain max_len and selt.qlog_t). For now we suppose keeping all labels
            else:
                np.copyto(self.qlog_buffer.observation[self.qlog_t:next_index], obs_t)
                np.copyto(self.qlog_buffer.agent_act[self.qlog_t:next_index], act_t)
                np.copyto(self.qlog_buffer.oracle_act[self.qlog_t:next_index], oracle_act_t)
                np.copyto(self.qlog_buffer.oracle_q[self.qlog_t:next_index], oracle_q_t)
                self.qlog_buffer.margine[self.qlog_t:next_index] = self.loss_margine
                self.qlog_t = next_index

        self.agent.eval_mode(itr=1, eps=0.0)  # in fact, here the eps is useless since we just want to check its Q-value
        num_epochs = int(np.ceil(self.qlog_t/self.train_batch_size))  # NOTE: use 'cile', so all labeled data will be used to train reward predictor!
        for epoch in range(num_epochs): 
            last_index = (epoch+1)*self.train_batch_size
            if last_index > self.qlog_t:
                last_index = self.qlog_t
            idxs = np.arange(epoch*self.train_batch_size, last_index)
            obs_batch = self.qlog_buffer.observation[idxs]
            obs_batch = torch.from_numpy(obs_batch).float().to(self.device)
            _, agent_info = self.agent.step(obs_batch)  # agent_info.value.shape: (B, action_dim)
            # np.copyto(self.qlog_buffer.agent_q[idxs, itr], agent_info.value.numpy())  # slice not work for np.copyto, since index slices will return a new object instead of the original memory
            self.qlog_buffer.agent_q[idxs, itr] = agent_info.value.numpy()
        
        if obs_t is None:  # after finishing the last iteration
            save_path = os.path.join(self.log_q_path, 'qlog_buffer')
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, 'oracle_q.npy'), self.qlog_buffer.oracle_q[:self.qlog_t])
            np.save(os.path.join(save_path, 'oracle_act.npy'), self.qlog_buffer.oracle_act[:self.qlog_t])
            np.save(os.path.join(save_path, 'margine.npy'), self.qlog_buffer.margine[:self.qlog_t])
            np.save(os.path.join(save_path, 'agent_q.npy'), self.qlog_buffer.agent_q[:self.qlog_t])
            np.save(os.path.join(save_path, 'Q_diff_aE.npy'), self.qlog_buffer.Q_diff_aE[:self.qlog_t])
            np.save(os.path.join(save_path, 'num_ne_act.npy'), self.qlog_buffer.num_ne_act[:self.qlog_t])
            if self.cf_per_seg == 1:
                np.save(os.path.join(save_path, 'oracle_return_cnt.npy'), self.oracle_return_cnt)
                np.save(os.path.join(save_path, 'oracle_return_l_large.npy'), self.oracle_return_l_large)
                oracle_return_l_large_ratio = [(x*1.0)/(y*1.0) for x, y \
                                               in zip(self.oracle_return_l_large, self.oracle_return_cnt)]
                np.save(os.path.join(save_path, 'oracle_return_l_large_ratio.npy'), oracle_return_l_large_ratio)
                print(f"[Q-LOG] oracle_return_cnt: {self.oracle_return_cnt}")
                print(f"[Q-LOG] oracle_return_l_large: {self.oracle_return_l_large}")
                print(f"[Q-LOG] oracle_return_l_large_ratio: {oracle_return_l_large_ratio}")

    def plot_q(self, cnt_samples, action_names):
        itv = self.qlog_t // cnt_samples  # interval to sample data
        for cnt in range(cnt_samples):
            idx = cnt * itv
            oracle_a = self.qlog_buffer.oracle_act[idx]
            oracle_q = self.qlog_buffer.oracle_q[idx]
            agent_a = self.qlog_buffer.agent_act[idx]
            agent_q = self.qlog_buffer.agent_q[idx]
            margine = self.qlog_buffer.margine[idx]

            width = 3  # the width of the bars
            x = np.arange(self.action_dim) * (width * (self.total_itr + 2) + width) # the label locations
            # multiplier = 0
            fig, ax = plt.subplots(layout='constrained',
                                   figsize=(self.action_dim * (self.total_itr + 2) * 0.1, 7))

            bar_label = ['E'] + [_ for _ in range(self.total_itr+1)]

            for id_bar, itr_name in enumerate(bar_label):
                offset = width * id_bar
                if itr_name == 'E':
                    heights = oracle_q  # (action_dim,)
                else:
                    heights = agent_q[itr_name]  # (action_dim,)
                rects = ax.bar(x + offset, heights, width, label=itr_name)
                ax.bar_label(rects, padding=1.5, rotation=90, fontsize=6.5)
            
            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel('Q')
            ax.set_title(f'Q-values in different finetuning iterations, margine={margine}')
            ax.set_xticks(x + width*((self.total_itr+3)//2), action_names, rotation=40)

            # ax.hlines(y=[oracle_q[oracle_a], oracle_q[agent_a],
            #              agent_q[-1][oracle_a], agent_q[-1][agent_a]],
            #              xmin=x[0], xmax=x[-1]+offset,
            #              labels=[r'$Q_E(a_E)$', r'$Q_E(a_t)$', r'$Q_N(a_E)$', r'$Q_N(a_t)$'],
            #              colors=['tomato', 'tomato', 'blue', 'blue'],
            #              linestyles=['solid', 'dashed', 'solid', 'dashed'],
            #              linewidth=0.5)
            # since hlines do not support a list of labels, I have to split them into single lines
            ax.hlines(y=oracle_q[oracle_a], xmin=x[0], xmax=x[-1]+offset,
                      label=r'$Q_E(a_E)$', colors='tomato', linestyles='solid',
                      linewidth=0.5)
            ax.hlines(y=oracle_q[agent_a], xmin=x[0], xmax=x[-1]+offset,
                      label=r'$Q_E(a_t)$', colors='tomato', linestyles='dashed',
                      linewidth=0.5)
            ax.hlines(y=agent_q[-1][oracle_a], xmin=x[0], xmax=x[-1]+offset,
                      label=r'$Q_N(a_E)$', colors='blue', linestyles='solid',
                      linewidth=0.5)
            ax.hlines(y=agent_q[-1][agent_a], xmin=x[0], xmax=x[-1]+offset,
                      label=r'$Q_N(a_t)$', colors='blue', linestyles='dashed',
                      linewidth=0.5)
            ax.hlines(y=np.max(agent_q[-1]), xmin=x[0], xmax=x[-1]+offset,
                      label=r'$max(Q_N(a))$', colors='green', linestyles='solid',
                      linewidth=0.5)
            ax.hlines(y=np.min(agent_q[-1]), xmin=x[0], xmax=x[-1]+offset,
                      label=r'$min(Q_N(a))$', colors='green', linestyles='dashed',
                      linewidth=0.5)
            
            ax.legend(bbox_to_anchor=(0, -0.5), loc='upper left', prop={'size':8}, framealpha=0.2, ncols=5)
            fig.tight_layout()
            fig.savefig(fname=os.path.join(self.log_q_path,
                                              f'{cnt}.png'),
                            bbox_inches='tight', pad_inches=0)

    def get_label(self, itr, obs_t, act_t, oracle_act_t, oracle_act_prob_t,
                  oracle_q_t, GT_reward_t, salient_oracle=False):
        # obs_t.shape: (query_batch, size_segment, *obs_shape)
        # act_t & oracle_act_t.shape: (query_batch, size_segment)
        # oracle_act_prob_t & oracle_q_t.shape: (query_batch, size_segment, action_ndim)
        # GT_reward_t.shape: (query_batch, size_seg)
        # NOTE: !!!! oracle_act_prob_t's minimal value should be larger than EPS!!! otherwise np.log will has errors!!!!
        segment_log_itr_path = os.path.join(self.segment_log_path, str(itr))
        os.makedirs(segment_log_itr_path, exist_ok=True)
        if self.oracle_type == 'oe':
            raise NotImplementedError  # haven't check related contents
            assert np.min(oracle_act_prob_t) > 1e-11  # because use log to calculate entropy
            oracle_act_entropy = (-oracle_act_prob_t * np.log(oracle_act_prob_t))\
                                    .sum(axis=-1, keepdims=False)  # shape(query_batch, size_segment)
            # target_oracle_index = np.argsort(oracle_act_entropy, axis=-1)[:, -self.cf_per_seg:]  # ascending order, shape (query_batch, cf_per_seg)
            # NOTE: (This argument may not true!) smaller entropy -> more confident about one state
            target_oracle_index = np.argsort(oracle_act_entropy,
                                            axis=-1)[:, :self.cf_per_seg]  # ascending order, shape (query_batch, cf_per_seg)
        elif self.oracle_type == 'oq':
            index_0 = np.arange(act_t.shape[0])
            index_1 = np.arange(act_t.shape[1])
            index_seg, index_batch = np.meshgrid(index_1, index_0)
            oracle_act_oracle_q = oracle_q_t[index_batch, index_seg, oracle_act_t]  # oracle's q for its own action, shape (query_batch, size_seg)
            act_oracle_q = oracle_q_t[index_batch, index_seg, act_t]  # oracle's q for current selected action, shape (query_batch, size_seg)
            
            q_diff = oracle_act_oracle_q - act_oracle_q  # q_diff are positive values, shape (query_batch, size_seg)
            np.save(file=os.path.join(segment_log_itr_path, 'q_diff.npy'), arr=q_diff)
            
            target_oracle_index = np.argsort(q_diff, axis=-1)[:, -self.cf_per_seg:]  # ascending order, shape (query_batch, cf_per_seg)
            q_diff_selected = np.take_along_axis(q_diff,
                                                target_oracle_index,
                                                axis=1)  # (query_batch, cf_per_seg)
            if salient_oracle:
                q_diff_selected_sum = np.sum(q_diff_selected, axis=-1, keepdims=False)  # (self.query_batch*sampler_multiplier)
                q_diff_seg_selected = np.argsort(q_diff_selected_sum, axis=-1)[-self.query_batch:]  # (self.query_batch)
                q_diff_selected = np.take_along_axis(q_diff_selected,
                                                    q_diff_seg_selected[...,None],
                                                    axis=0)  # (self.query_batch, cf_per_seg)
                target_oracle_index = np.take_along_axis(target_oracle_index,
                                                    q_diff_seg_selected[...,None],
                                                    axis=0) # (self.query_batch, cf_per_seg)
                q_diff = np.take_along_axis(q_diff, q_diff_seg_selected[..., None], axis=0)

                obs_t = np.take_along_axis(obs_t, q_diff_seg_selected[..., None, None, None, None], axis=0)
                act_t = np.take_along_axis(act_t, q_diff_seg_selected[...,None], axis=0)
                oracle_act_t = np.take_along_axis(oracle_act_t, q_diff_seg_selected[...,None], axis=0)
                oracle_act_prob_t = np.take_along_axis(oracle_act_prob_t, q_diff_seg_selected[...,None,None], axis=0)
                oracle_q_t = np.take_along_axis(oracle_q_t, q_diff_seg_selected[...,None,None], axis=0)
                GT_reward_t = np.take_along_axis(GT_reward_t, q_diff_seg_selected[...,None], axis=0)
                oracle_act_oracle_q = np.take_along_axis(oracle_act_oracle_q,
                                            q_diff_seg_selected[...,None], axis=0)
                act_oracle_q = np.take_along_axis(act_oracle_q,
                                            q_diff_seg_selected[...,None], axis=0)
            assert q_diff_selected.shape == (self.query_batch, self.cf_per_seg)
        elif self.oracle_type == 'oqLRd':  # after the same procedure as 'oq', we select states which satisfy the LR relation (discounted)
            if salient_oracle:
                raise NotImplementedError
            index_0 = np.arange(act_t.shape[0])
            index_1 = np.arange(act_t.shape[1])
            index_seg, index_batch = np.meshgrid(index_1, index_0)
            oracle_act_oracle_q = oracle_q_t[index_batch, index_seg, oracle_act_t]  # oracle's q for its own action, shape (query_batch, size_seg)
            act_oracle_q = oracle_q_t[index_batch, index_seg, act_t]  # oracle's q for current selected action, shape (query_batch, size_seg)
            q_diff = oracle_act_oracle_q - act_oracle_q  # q_diff are positive values, shape (query_batch, size_seg)
            # target_oracle_index = np.argsort(q_diff, axis=-1)[:, -self.cf_per_seg:]  # ascending order, shape (query_batch, cf_per_seg)
            sort_q_diff_index = np.argsort(q_diff, axis=-1)  # ascending order, shape (query_batch, cf_per_seg)
            
            # GT_return_left_t = copy.deepcopy(GT_reward_t)  # (query_batch, size_seg)
            # GT_return_right_t = copy.deepcopy(GT_reward_t)  # query_batch, size_seg
            GT_return_left_t = np.zeros_like(GT_reward_t)  # (query_batch, size_seg)
            GT_return_right_t = np.zeros_like(GT_reward_t)  # query_batch, size_seg
            for id_t in range(0, self.size_segment):
                GT_return_left_t[:, id_t] = (np.multiply(GT_reward_t[:, :id_t+1], self.np_gamma_exp[:id_t+1][None, ...])\
                                                .sum(axis=-1, keepdims=False))/(self.np_discount_norm[id_t])
                GT_return_right_t[:, id_t] = (np.multiply(GT_reward_t[:, id_t:], self.np_gamma_exp[:self.size_segment-id_t][None, ...])\
                                                .sum(axis=-1, keepdims=False))/(self.np_discount_norm[self.size_segment-id_t-1])
            # for id_t in range(1, self.size_segment):
            #     GT_return_left_t[:, id_t] = GT_return_left_t[:, id_t - 1] + GT_return_left_t[:, id_t]
            #     GT_return_right_t[:, self.size_segment - id_t - 1] = GT_return_right_t[:, self.size_segment - id_t] \
            #                                                         + GT_return_right_t[:, self.size_segment - id_t - 1]
            if self.LR.flag:
                if self.LR.RR_type==0:
                    LR = GT_return_left_t[:, :-1] - GT_return_right_t[:, 1:]  # (query_batch, size_seg-1)
                    np.save(file=os.path.join(segment_log_itr_path, 'q_diff.npy'), arr=q_diff[:, 1:])
                    np.save(file=os.path.join(segment_log_itr_path, 'LR.npy'), arr=LR)
                    LR_completeT = np.concatenate((np.inf*np.ones((LR.shape[0], 1)), LR), axis=1)  # To avoid that the first action is selected as the corrective action
                elif self.LR.RR_type==1:
                    LR = GT_return_left_t[:, :-2] - GT_return_right_t[:, 2:]  # (query_batch, size_seg-1)
                    np.save(file=os.path.join(segment_log_itr_path, 'q_diff.npy'), arr=q_diff[:, 1:-2])
                    np.save(file=os.path.join(segment_log_itr_path, 'LR.npy'), arr=LR)
                    LR_completeT = np.concatenate((np.inf*np.ones((LR.shape[0], 1)),
                                                   LR,
                                                   np.inf*np.ones((LR.shape[0], 1))),
                                                  axis=1)  # To avoid that the first action is selected as the corrective action
                else:
                    raise NotImplementedError
            else:
                LR = GT_return_left_t[:, :-1] - GT_return_right_t[:, 1:]  # (query_batch, size_seg-1)
                np.save(file=os.path.join(segment_log_itr_path, 'q_diff.npy'), arr=q_diff[:, 1:])
                np.save(file=os.path.join(segment_log_itr_path, 'LR.npy'), arr=LR)
                LR_completeT = np.concatenate((np.inf*np.ones((LR.shape[0], 1)), LR), axis=1)  # To avoid that the first action is selected as the corrective action
            
            # np.take_along_axis(arr=q_diff, indices=sort_q_diff_index, axis=-1) obtain the sorted q_diff arr, left->right small->large
            LR_sort_as_qdiff = np.take_along_axis(arr=LR_completeT, indices=sort_q_diff_index, axis=-1)  # (query_batch, size_seg)  from left to right, the corresponding q_diff from small to large
            assert self.cf_per_seg == 1  # because I use argmax to select the first
            # target_oracle_index.shape(query_batch, size_seg=1)
            target_oracle_index_arg = self.size_segment - 1 \
                                - np.argmax(LR_sort_as_qdiff[:,::-1] > 0, axis=-1, keepdims=True)  # argmax will stop at the first True; if there's no True value in its input array, np.argmax will happily return 0 (But since we )
            target_oracle_index = np.take_along_axis(arr=sort_q_diff_index, indices=target_oracle_index_arg, axis=-1) 
            # target_oracle_index = np.argsort(q_diff, axis=-1)[:, -self.cf_per_seg:]  # ascending order, shape (query_batch, cf_per_seg)
            # if self.cf_per_seg == 1:
            #     pdb.set_trace()  # check target_oracle_index
            #     target_oracle_index = np.argmax(LR_sort_as_qdiff > 0, axis=-1)  # argmax will stop at the first True; if there's no True value in its input array, np.argmax will happily return 0
            # else:
            #     pdb.set_trace()  # check target_oracle_index
            #     sort_LR_index = np.argsort(LR_sort_as_qdiff, axis=-1)[:, -self.cf_per_seg]
            #     target_oracle_index = np.take_along_axis(arr=sort_q_diff_index,
            #                                              indices=sort_LR_index, axis=-1)  # argmax will stop at the first True; if there's no True value in its input array, np.argmax will happily return 0
            LR_selected = np.take_along_axis(LR_completeT, target_oracle_index, axis=1)
            q_diff_selected = np.take_along_axis(q_diff, target_oracle_index, axis=1)  # TODO: only a small part of samples satisfy the condition...
            np.save(file=os.path.join(segment_log_itr_path, 'q_diff_selected.npy'), arr=q_diff_selected)
            np.save(file=os.path.join(segment_log_itr_path, 'LR_selected.npy'), arr=LR_selected)
            assert LR_selected.shape == q_diff_selected.shape == (self.query_batch, self.cf_per_seg)
        else:
            # TODO: maybe also consider r_hat, or even training agent's q_value?
            raise NotImplementedError
        
        # target_oracle_index.shape: (query_batch, cf_per_seg)
        if self.check_label_path:
            # check selected (s,a) pair, check if low entropy = confidence about action selection
            query_cnt = obs_t.shape[0]
            for id_seg in range(query_cnt):
                widths = [10, 10, 10, 10, 10, 1, 1, 1]  # 4 frame stack + 1 bar plot + 3 entropy color
                heights = [5 for _ in range(self.size_segment)]
                fig = plt.figure(constrained_layout=True,
                                 figsize=(len(widths)*5, len(heights)*5))
                spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights),)
                                        # width_ratios=widths,
                                        # height_ratios=heights)
                # fig, axes = plt.subplots(ncols=len(widths), nrows=len(heights),
                #                          figsize=(len(widths)*5, len(heights)*5))  # figsize: (weight, height)
                for id_in_seg in range(self.size_segment):
                    for id_fs in range(self.n_frames):  # default is 4 framestack
                        ax_img = fig.add_subplot(spec[id_in_seg, id_fs])
                        # ax_img = axes[id_in_seg, 0]
                        ax_plot_img(ori_img=obs_t[id_seg][id_in_seg][id_fs],
                                    ax=ax_img,
                                    title=f'{id_in_seg}',
                                    vmin=0, vmax=255)
                    
                    ax_prob_bar = fig.add_subplot(spec[id_in_seg, self.n_frames + 1])
                    if self.oracle_type == 'oe':
                        bar_height = oracle_act_prob_t[id_seg][id_in_seg]
                        bars = ax_plot_bar(ax=ax_prob_bar,
                                        xlabel=self.env_action_meaning,
                                        height=bar_height)
                    elif self.oracle_type in ['oq', 'oqLRd']:
                        bar_height = oracle_q_t[id_seg][id_in_seg]  # can have negative values
                        value_min = bar_height.min()
                        bar_height_min0 = bar_height - value_min
                        bars = ax_plot_bar(ax=ax_prob_bar,
                                        xlabel=self.env_action_meaning,
                                        height=bar_height_min0,
                                        yvals=bar_height,
                                        top=bar_height_min0.max()+0.,
                                        )
                    else:
                        raise NotImplementedError
                    # mark oracle act at this step
                    oracle_act_this = oracle_act_t[id_seg][id_in_seg]
                    ax_prob_bar.axvline(x=bars[oracle_act_this].get_x(), color='r')
                    # ax_prob_bar.text(x=bars[oracle_act_this].get_x()+.2,
                    #                  y=bars[oracle_act_this].get_y()+.1,
                    #                  s=self.env_action_meaning[oracle_act_this],
                    #                  color='r')
                    
                    # mark training act at this step
                    act_this = act_t[id_seg][id_in_seg]
                    ax_prob_bar.axvline(x=bars[act_this].get_x()+.2, color='b')
                    # ax_prob_bar.text(x=bars[act_this].get_x()+.4,
                    #                  y=bars[act_this].get_y()+.05,
                    #                  s=self.env_action_meaning[act_this],
                    #                  color='b')
                
                if self.oracle_type == 'oe':
                    ax_entropy = fig.add_subplot(spec[:, -2])
                    # dark: small entropy; light: large entropy
                    ax_plot_heatmap(arr=oracle_act_entropy[id_seg].reshape(-1, 1),
                                    ax=ax_entropy,
                                    highlight_row=target_oracle_index[id_seg].reshape(-1),
                                    title='entropy (darkest <-> smallest)')  # normalized on its own min_max

                    ax_entropy_minmax = fig.add_subplot(spec[:, -1])
                    # minmax scale: [0, N * (-(1/N) * log(1/N))] -> [0, 1]
                    ax_plot_heatmap(arr=oracle_act_entropy[id_seg].reshape(-1, 1),
                                    ax=ax_entropy_minmax,
                                    highlight_row=target_oracle_index[id_seg].reshape(-1),
                                    title=f'entropy (largest={self.max_entropy:.6f})',
                                    vmin=0, vmax=self.max_entropy)
                elif self.oracle_type == 'oq':
                    ax_qE_max = fig.add_subplot(spec[:, -3])
                    pdb.set_trace()  # check oracle_act_oracle_q.shape
                    ax_plot_heatmap(arr=oracle_act_oracle_q[id_seg].reshape(-1, 1),
                                    ax=ax_qE_max,
                                    highlight_row=target_oracle_index[id_seg].reshape(-1),
                                    title=f'oracle_act_oracle_q (lightest <-> largest)')
                    
                    ax_q_diff = fig.add_subplot(spec[:, -2])
                    # dark: small value; light: large value
                    ax_plot_heatmap(arr=q_diff[id_seg].reshape(-1, 1),
                                    ax=ax_q_diff,
                                    highlight_row=target_oracle_index[id_seg].reshape(-1),
                                    title='q_diff (lightest <-> largest)')  # normalized on its own min_max
                    
                    ax_entropy = fig.add_subplot(spec[:, -1])
                    # dark: small entropy; light: large entropy
                    oracle_act_entropy = (-oracle_act_prob_t * np.log(oracle_act_prob_t))\
                                    .sum(axis=-1, keepdims=False)
                    ax_plot_heatmap(arr=oracle_act_entropy[id_seg].reshape(-1, 1),
                                    ax=ax_entropy,
                                    highlight_row=target_oracle_index[id_seg].reshape(-1),
                                    title='entropy (darkest <-> smallest)')  # normalized on its own min_max
                elif self.oracle_type == 'oqLRd':
                    ax_qE_max = fig.add_subplot(spec[:, -3])
                    pdb.set_trace()  # check oracle_act_oracle_q.shape
                    ax_plot_heatmap(arr=oracle_act_oracle_q[id_seg].reshape(-1, 1),
                                    ax=ax_qE_max,
                                    highlight_row=target_oracle_index[id_seg].reshape(-1),
                                    title=f'oracle_act_oracle_q (lightest <-> largest)')
                    
                    ax_q_diff = fig.add_subplot(spec[:, -2])
                    # dark: small value; light: large value
                    ax_plot_heatmap(arr=q_diff[id_seg].reshape(-1, 1),
                                    ax=ax_q_diff,
                                    highlight_row=target_oracle_index[id_seg].reshape(-1),
                                    title='q_diff (lightest <-> largest)')  # normalized on its own min_max
                    
                    ax_LRd = fig.add_subplot(spec[:, -1])
                    # dark: small LR; light: large LR
                    ax_plot_heatmap(arr=LR[id_seg].reshape(-1, 1),
                                    ax=ax_LRd,
                                    highlight_row=target_oracle_index[id_seg].reshape(-1),
                                    title='L-R return -d (darkest <-> smallest)')  # normalized on its own min_max
                else:
                    raise NotImplementedError

                fig.tight_layout()
                fig.savefig(fname=os.path.join(self.check_label_path,
                                              f'{self.total_label_seg + id_seg}.png'),
                            bbox_inches='tight', pad_inches=0)

        if self.log_q_path:
            total_sample = obs_t.shape[0]
            if self.cf_per_seg == 1:
                # target_oracle_index.shape: (query_batch, 1)
                compare_len = np.min(
                    np.concatenate([self.size_segment - 1 - target_oracle_index, target_oracle_index], axis=-1),
                    axis=-1, keepdims=False)  # shape: (quer_batch,) (size_segment-1 because target_oracle_index starts from index 0)
                # oracle_q_t.shape: (query_batch, size_seg, action_dim)
                # oracle_act_t.shape: (query_batch, size_seg)
                for id_seg in range(total_sample):
                    for delt in range(1, compare_len[id_seg] + 1):
                        t_oracle_index = target_oracle_index[id_seg, 0]
                        t_oracle_index_l = t_oracle_index - delt
                        t_oracle_index_r = t_oracle_index + delt
                        return_l = oracle_q_t[id_seg, t_oracle_index_l][oracle_act_t[id_seg, t_oracle_index_l]] \
                                - (self.gamma ** delt) * oracle_q_t[id_seg, t_oracle_index][oracle_act_t[id_seg, t_oracle_index]]
                        return_r =  oracle_q_t[id_seg, t_oracle_index][oracle_act_t[id_seg, t_oracle_index]]\
                                - (self.gamma ** delt) * oracle_q_t[id_seg, t_oracle_index_r][oracle_act_t[id_seg, t_oracle_index_r]]
                        self.qlog_buffer.return_l[self.qlog_t + id_seg, delt - 1] = return_l
                        self.qlog_buffer.return_r[self.qlog_t + id_seg, delt - 1] = return_r
                        self.oracle_return_cnt[delt - 1] += 1
                        self.oracle_return_l_large[delt - 1] += (return_l >= return_r)
            else:
                raise NotImplementedError
            num_ne_act = np.sum(q_diff>0, axis=-1)  # the number of non-expert actions
            np.copyto(self.qlog_buffer.num_ne_act[self.qlog_t: self.qlog_t+total_sample], num_ne_act)
            np.copyto(self.qlog_buffer.Q_diff_aE[self.qlog_t: self.qlog_t+total_sample],
                      q_diff_selected.reshape(self.query_batch * self.cf_per_seg,))
       
        # obs_t.shape: (len_label, *obs_shape)
        if self.oracle_type == 'oqLRd':  # remove labels without LR relation
            if self.ignore_small_qdiff.flag:
                kept_index = np.where((q_diff_selected>self.ignore_small_qdiff.thres) &\
                                      (LR_selected>0))
            else:
                kept_index = np.where(LR_selected>0)
            len_label = kept_index[0].shape[0]
            kept_index_query = kept_index[0]  # shape: (len_label,)
            kept_index_timestep = target_oracle_index[kept_index[0], kept_index[1]]  # shape: (len_label,)
            if self.LR.flag:  # cf_per_seg==1
                obs_t_seg = obs_t[kept_index_query, ...].\
                            reshape(len_label, self.size_segment, *self.obs_shape)
                act_t_seg = act_t[kept_index_query, ...].\
                            reshape(len_label, self.size_segment,)
                oracle_act_t_seg = oracle_act_t[kept_index_query, ...].\
                                reshape(len_label, self.size_segment,)
                kept_index_timestep_seg = kept_index_timestep  # shape: (len_label,)
            obs_t =  obs_t[kept_index_query, kept_index_timestep, ...].\
                        reshape(len_label, *self.obs_shape)
            # act_t.shape: (len_label,)
            act_t = act_t[kept_index_query, kept_index_timestep].\
                        reshape(len_label,)
            # oracle_act_t.shape: (len_label,)
            oracle_act_t = oracle_act_t[kept_index_query, kept_index_timestep].\
                            reshape(len_label,)
            # q_diff_t == q_diff_selected[kept_index]
            q_diff_t = q_diff[kept_index_query, kept_index_timestep].\
                        reshape(len_label,)
            LR_t = LR_completeT[kept_index_query, kept_index_timestep].\
                        reshape(len_label,)
            np.save(file=os.path.join(segment_log_itr_path, 'q_diff_t.npy'), arr=q_diff_t)
            self.q_diff_average = q_diff_t.mean()
            np.save(file=os.path.join(segment_log_itr_path, 'LR_t.npy'), arr=LR_t)
            if self.log_q_path:
                oracle_q_t = oracle_q_t[kept_index_query, kept_index_timestep, ...].\
                            reshape(len_label, self.action_dim)
                # oracle_q_t.shape: (len_label, action_dim)
            else:
                oracle_q_t = None
        elif self.ignore_small_qdiff.flag:
            kept_index = np.where(q_diff_selected>self.ignore_small_qdiff.thres)
            # q_diff_selected.shape = target_oracle_index.shape: (query_batch, cf_per_seg)
            len_label = kept_index[0].shape[0]
            kept_index_query = kept_index[0]  # shape: (len_label,)
            kept_index_timestep = target_oracle_index[kept_index[0], kept_index[1]]  # shape: (len_label,)
            if self.LR.flag:
                obs_t_seg = obs_t[kept_index_query, ...].\
                            reshape(len_label, self.size_segment, *self.obs_shape)
                act_t_seg = act_t[kept_index_query, ...].\
                            reshape(len_label, self.size_segment,)
                oracle_act_t_seg = oracle_act_t[kept_index_query, ...].\
                                reshape(len_label, self.size_segment,)
                kept_index_timestep_seg = kept_index_timestep  # shape: (len_label,)
            obs_t =  obs_t[kept_index_query, kept_index_timestep, ...].\
                        reshape(len_label, *self.obs_shape)
            # act_t.shape: (len_label,)
            act_t = act_t[kept_index_query, kept_index_timestep].\
                        reshape(len_label,)
            # oracle_act_t.shape: (len_label,)
            oracle_act_t = oracle_act_t[kept_index_query, kept_index_timestep].\
                            reshape(len_label,)
            
            # q_diff_t == q_diff_selected[kept_index]
            q_diff_t = q_diff[kept_index_query, kept_index_timestep].\
                        reshape(len_label,)
            np.save(file=os.path.join(segment_log_itr_path, 'q_diff_t.npy'), arr=q_diff_t)
            self.q_diff_average = q_diff_t.mean()

            if self.log_q_path:
                oracle_q_t = oracle_q_t[kept_index_query, kept_index_timestep, ...].\
                            reshape(len_label, self.action_dim)
                # oracle_q_t.shape: (len_label, action_dim)
            else:
                oracle_q_t = None
        else:
            len_label = self.query_batch * self.cf_per_seg
            if self.LR.flag:
                assert self.cf_per_seg == 1
                obs_t_seg = copy.deepcopy(obs_t)
                act_t_seg = copy.deepcopy(act_t)
                oracle_act_t_seg = copy.deepcopy(oracle_act_t)
                kept_index_timestep_seg = target_oracle_index[:,0]  # shape: (len_label,)
            obs_t =  np.take_along_axis(obs_t,
                                        target_oracle_index[..., None, None, None],
                                        axis=1).\
                        reshape(len_label, *self.obs_shape)
            # act_t.shape: (query_batch * cf_per_seg,)
            act_t = np.take_along_axis(act_t,
                                       target_oracle_index,
                                       axis=1).\
                        reshape(len_label,)
            # oracle_act_t.shape: (query_batch * cf_per_seg,)
            oracle_act_t = np.take_along_axis(oracle_act_t,
                                              target_oracle_index,
                                              axis=1).\
                            reshape(len_label,)
            # q_diff_selected.shape: (query_batch, cf_per_seg)
            self.q_diff_average = q_diff_selected.mean()
            if self.log_q_path:
                oracle_q_t = np.take_along_axis(oracle_q_t,
                                                target_oracle_index[..., None],
                                                axis=1).\
                            reshape(len_label, self.action_dim)
            else:
                oracle_q_t = None
        
        if not self.LR.flag:
            return obs_t, act_t, oracle_act_t, oracle_q_t
        else:
            return obs_t, act_t, oracle_act_t, oracle_q_t,\
                   obs_t_seg, act_t_seg, oracle_act_t_seg, kept_index_timestep_seg

    def uniform_sampling(self, itr, salient_oracle=False, sample_multipler=1):
        # get queries
        obs_t, act_t, oracle_act_t, oracle_act_prob_t, oracle_q_t, GT_reward_t =  self.get_queries(
            query_batch=self.query_batch*sample_multipler)
        # obs_t.shape: (query_batch, cf_per_seg, *obs_shape)
        
        if self.LR.flag:
            # get labels
            obs_t, act_t, oracle_act_t, oracle_q_t,\
            obs_t_seg, act_t_seg, oracle_act_t_seg, kept_index_timestep_seg = \
                self.get_label(  # filter queries and 
                    itr, obs_t, act_t, oracle_act_t, oracle_act_prob_t,
                    oracle_q_t, GT_reward_t, salient_oracle=salient_oracle)
            # obs_t.shape: (len_label, *obs_shape)
            self.put_queries_with_LR(obs_t, act_t, oracle_act_t,
                                    obs_t_seg, act_t_seg, oracle_act_t_seg,
                                    kept_index_timestep_seg)
        else:
            # get labels
            obs_t, act_t, oracle_act_t, oracle_q_t = self.get_label(  # filter queries and 
                itr, obs_t, act_t, oracle_act_t, oracle_act_prob_t,
                oracle_q_t, GT_reward_t, salient_oracle=salient_oracle)
            # obs_t.shape: (len_label, *obs_shape)
            self.put_queries(obs_t, act_t, oracle_act_t)

        if self.log_q_path:
            self.put_qlog(itr, obs_t, act_t, oracle_act_t, oracle_q_t)

        return obs_t.shape[0]  # query_batch * cf_per_seg
    
    def top_seg_sampling(self, itr):  # use_seg=True for self.get_queries(); salient_oracle=True for self.get_label()
        assert self.oracle_type == 'oq'
        # get queries
        obs_t, act_t, oracle_act_t, oracle_act_prob_t, oracle_q_t, GT_reward_t =  self.get_queries(
            query_batch=None, use_seg=True)
        # obs_t.shape: (query_batch, cf_per_seg, *obs_shape)
        
        if self.LR.flag:
            # get labels
            obs_t, act_t, oracle_act_t, oracle_q_t,\
            obs_t_seg, act_t_seg, oracle_act_t_seg, kept_index_timestep_seg = \
                self.get_label(  # filter queries and 
                    itr, obs_t, act_t, oracle_act_t, oracle_act_prob_t,
                    oracle_q_t, GT_reward_t, salient_oracle=True)
            # obs_t.shape: (len_label, *obs_shape)
            self.put_queries_with_LR(obs_t, act_t, oracle_act_t,
                                    obs_t_seg, act_t_seg, oracle_act_t_seg,
                                    kept_index_timestep_seg)
        else:
            # get labels
            obs_t, act_t, oracle_act_t, oracle_q_t = self.get_label(  # filter queries and 
                itr, obs_t, act_t, oracle_act_t, oracle_act_prob_t,
                oracle_q_t, GT_reward_t, salient_oracle=True)
            # obs_t.shape: (len_label, *obs_shape)
            self.put_queries(obs_t, act_t, oracle_act_t)

        if self.log_q_path:
            self.put_qlog(itr, obs_t, act_t, oracle_act_t, oracle_q_t)

        return obs_t.shape[0]  # query_batch * cf_per_seg

    def LR_loss(self, LR_index, with_LR, q_s_t, act_t, oracle_act_t):
        available_LR_index = LR_index[with_LR]  # index with boolean type will return the filtered array directly
        len_LR_sample = available_LR_index.shape[0]
        assert q_s_t.shape[0] == with_LR.shape[0]
        LR_q_s_t = q_s_t[with_LR]  # LR_q_s_t.shape: (len_LR_sample, action_dim)
        LR_act_t = act_t[with_LR]  # LR_oracle_act_t.shape:(len_LR_sample, )
        LR_oracle_act_t = oracle_act_t[with_LR]  # LR_oracle_act_t.shape:(len_LR_sample, )
        # obs_t_seg = torch.from_numpy(self.LR_buffer.observation[available_LR_index]).float().to(self.device)  # (len_LR_sample, size_seg, *obs_shape)
        obs_t_seg = self.LR_buffer.observation[available_LR_index]  # (len_LR_sample, size_seg, *obs_shape)
        act_t_seg = torch.from_numpy(self.LR_buffer.action[available_LR_index]).long().to(self.device)  # (len_LR_sample, size_seg)
        oracle_act_t_seg = torch.from_numpy(self.LR_buffer.oracle_action[available_LR_index]).long().to(self.device)  # (len_LR_sample, size_seg)
        target_oracle_index = self.LR_buffer.CF_index[available_LR_index]  # (len_LR_sample,)

        LR_q_s_t_a_E = torch.gather(input=LR_q_s_t, dim=-1, index=LR_oracle_act_t[...,None]).reshape(len_LR_sample)  # LR_q_s_t_a_E.shape: (len_LR_sample,)
        len_left = target_oracle_index  # (len_LR_sample,)
        if self.LR.RR_type == 0:
            len_right = self.size_segment - 1 - target_oracle_index  # (len_LR_sample,)
        elif self.LR.RR_type == 1:
            len_right = self.size_segment - 1 - target_oracle_index - 1  # (len_LR_sample,)
        compare_len = np.min(
                        np.concatenate([len_left[None,...], len_right[None,...]], axis=0),  # (2, len_LR_sample)
                        axis=0, keepdims=False)  # compare_len.shape: (len_LR_sample,)
        if self.LR.scale == 1:  # longest same-length-subseg related to s_t
            raise NotImplementedError
        elif self.LR.scale == 0:  # no matther the length of segment, normalize it wiith 1+\gammq+\gamma^2+...
            LR_total = with_LR.sum()
            len_left = torch.from_numpy(len_left).long().to(self.device)
            len_right = torch.from_numpy(len_right).long().to(self.device)
            obs_t_left = torch.from_numpy(obs_t_seg[:, 0]).float().to(self.device)  # (len_LR_sample, *obs_shape)
            if self.distributional:
                p_s_t_left = self.agent(obs_t_left, train=True)  # q_s_t.shape: (len_LR_sample, action_dim, n_atoms)
                # z = torch.linspace(self.agent.V_min, self.agent.V_max, self.agent.n_atoms)
                q_s_t_left = torch.tensordot(p_s_t_left, self.distri_z, dims=1)  # (len_LR_sample, action_dim)
            else:
                pdb.set_trace()  # check q_s_t.shape
                q_s_t_left = self.agent(obs_t_left, train=True)  # q_s_t.shape: (len_LR_sample, action_dim)
            q_s_t_left_a_t_left = torch.gather(input=q_s_t_left, dim=-1,
                                               index=act_t_seg[:, 0, None]).reshape(len_LR_sample,)  # q_s_t_left_a_t_left.shape: (B, 1)
            left_return = q_s_t_left_a_t_left - \
                          torch.mul(self.gamma_exp[len_left],  # = self.gamma ** len_left.shape
                                    LR_q_s_t_a_E)  # mul for pointwise multiplication
            left_norm = self.discount_norm[len_left-1]
            left_return_norm = left_return / left_norm 
            obs_t_right = torch.from_numpy(obs_t_seg[:, -1]).float().to(self.device)  # (len_LR_sample, *obs_shape)
            # return  q_s_t_left_a_t_left.mean() + left_return_norm.mean()  # this is ok
            if self.distributional:
                p_s_t_right = self.agent(obs_t_right, train=True)  # q_s_t.shape: (len_LR_sample, action_dim, n_atoms)
                # z = torch.linspace(self.agent.V_min, self.agent.V_max, self.agent.n_atoms)
                q_s_t_right = torch.tensordot(p_s_t_right, self.distri_z, dims=1)  # (len_LR_sample, action_dim)
            else:
                pdb.set_trace()  # check q_s_t.shape
                q_s_t_right = self.agent(obs_t_right, train=True)  # q_s_t.shape: (len_LR_sample, action_dim)
            q_s_t_right_a_t_right = torch.gather(input=q_s_t_right, dim=-1,
                                                 index=act_t_seg[:,-1,None]).reshape(len_LR_sample,)  # q_s_t_right_a_t.shape: (len_LR_sample,)
            if self.LR.RR_type == 0:  # for right return, use Q(s_t, a_t)
                LR_q_s_t_a_t = torch.gather(input=LR_q_s_t, dim=-1,
                                            index=LR_act_t[...,None]).reshape(len_LR_sample,)  # LR_q_s_t_a_t.shape: (len_LR_sample,)
                right_return = LR_q_s_t_a_t - \
                               torch.mul(self.gamma_exp[len_right],  #self.gamma ** len_right * q_s_t_right_a_t_right
                                         q_s_t_right_a_t_right)  # right_return.shape: (len_LR_sample,)
            elif self.LR.RR_type == 1:  # for right return, use Q(s_{t+1}, a_{t+1})
                act_t1 = torch.gather(act_t_seg, dim=1, 
                            index=torch.from_numpy(
                                    target_oracle_index[..., None]+1
                                ).long().to(self.device))  # act_t1.shape: (len_LR_sample, 1)
                obs_t1 = np.take_along_axis(obs_t_seg,
                                            target_oracle_index[..., None, None, None, None]+1,
                                            axis=1).\
                                            reshape(len_LR_sample, *self.obs_shape)
                obs_t1 = torch.from_numpy(obs_t1).float().to(self.device)
                if self.distributional:
                    p_s_t1 = self.agent(obs_t1, train=True)  # q_s_t.shape: (len_LR_sample, action_dim, n_atoms)
                    # z = torch.linspace(self.agent.V_min, self.agent.V_max, self.agent.n_atoms)
                    q_s_t1 = torch.tensordot(p_s_t1, self.distri_z, dims=1)  # (len_LR_sample, action_dim)
                else:
                    pdb.set_trace()  # check q_s_t.shape
                    q_s_t1 = self.agent(obs_t1, train=True)  # q_s_t.shape: (B, action_dim)
                LR_q_s_t_a_t1 = torch.gather(input=q_s_t1, dim=-1, index=act_t1[...]).reshape(len_LR_sample,)  # r_hat_s_oa.shape: (B, 1)
                right_return = LR_q_s_t_a_t1 - \
                               torch.mul(self.gamma_exp[len_right],  # self.gamma ** len_right
                                         q_s_t_right_a_t_right)  # pointwise multiply
            right_norm = self.discount_norm[len_right-1]  # right_norm.shape: (len_LR_sample,)
            right_return_norm = right_return / right_norm  # right_return.shape: (len_LR_sample,)
            LR_diff = left_return_norm - right_return_norm + self.LR.loss_slack
            loss = torch.max(-LR_diff, torch.zeros_like(LR_diff)).mean()  # mean() return a scalar
            LR_acc = (LR_diff.detach() >= 0.).sum().item()
            return loss, LR_total, LR_acc
        elif self.LR.scale == 2:
            LR_total = 0
            index_seg = []
            left_seg_index = []
            right_seg_index = []
            len_subseg = []
            for id_seg in range(len_LR_sample):
                for id_t in range(1, compare_len[id_seg] + 1):
                    index_seg.append(id_seg)
                    len_subseg.append(id_t)
                    left_seg_index.append(target_oracle_index[id_seg] - id_t)
                    if self.LR.RR_type == 0:
                        right_seg_index.append(target_oracle_index[id_seg] + id_t)
                    if self.LR.RR_type == 1:
                        right_seg_index.append(target_oracle_index[id_seg] + id_t + 1)
            index_seg = np.array(index_seg).reshape(-1)  # (len_pairs,)
            len_subseg = np.array(len_subseg).reshape(-1)  # (len_pairs,)
            left_seg_index = np.array(left_seg_index).reshape(-1)  # (len_pairs,)
            right_seg_index = np.array(right_seg_index).reshape(-1)  # (len_pairs,)
            len_pairs = index_seg.shape[0]
            left_obs_selected = torch.from_numpy(
                                    obs_t_seg[index_seg, left_seg_index]
                                ).reshape(len_pairs, *self.obs_shape).\
                                float().to(self.device)  # (len_pairs, *obs_shape)
            right_obs_selected = torch.from_numpy(
                                    obs_t_seg[index_seg, right_seg_index]
                                ).reshape(len_pairs, *self.obs_shape).\
                                float().to(self.device)  # (len_pairs, *obs_shape)
            obs_selected_LR = torch.concatenate([left_obs_selected[None, ...],
                                                 right_obs_selected[None, ...]],
                                                dim=0)  # obs_selected_LR.shape: (2, len_pairs, *obs_shape)
            if self.distributional:
                p_s_select = self.agent(obs_selected_LR, train=True)  # p_s_seg.shape: (2, len_pairs, action_dim, n_atoms)
                # z = torch.linspace(self.agent.V_min, self.agent.V_max, self.agent.n_atoms)
                q_s_select = torch.tensordot(p_s_select, self.distri_z, dims=1)  # (2, len_pairs, action_dim)
            else:
                pdb.set_trace()  # check q_s_select.shape
                q_s_select = self.agent(obs_selected_LR)  # q_s_t.shape: (2, len_pairs, action_dim)
            q_s_left_selected = q_s_select[0, ...]  # (len_pairs, action_dim)
            q_s_right_selected = q_s_select[1, ...]  # (len_pairs, action_dim)
            index_seg = torch.from_numpy(index_seg).long().to(self.device)
            len_subseg = torch.from_numpy(len_subseg).long().to(self.device)
            left_seg_index = torch.from_numpy(left_seg_index).long().to(self.device)
            right_seg_index = torch.from_numpy(right_seg_index).long().to(self.device)
            return_norm = self.discount_norm[len_subseg-1]
            act_left_selected = act_t_seg[index_seg, left_seg_index]  # (len_pairs,)
            act_right_selected = act_t_seg[index_seg, right_seg_index]  # (len_pairs,)
            q_s_left_selected_a_selected = torch.gather(input=q_s_left_selected, dim=-1,
                                            index=act_left_selected[...,None]).reshape(len_pairs)
            q_s_rihgt_selected_a_selected = torch.gather(input=q_s_right_selected, dim=-1,
                                            index=act_right_selected[...,None]).reshape(len_pairs)
            
            q_s_t_selected = LR_q_s_t[index_seg]  # (len_pairs, action_dim)
            oracle_act_t_selected = LR_oracle_act_t[index_seg]  # (len_pairs, )
            q_s_t_selected_a_E = torch.gather(input=q_s_t_selected, dim=-1,
                                        index=oracle_act_t_selected[...,None]).reshape(len_pairs)  # (len_pairs,)
            left_returns = q_s_left_selected_a_selected - \
                            torch.mul(self.gamma_exp[len_subseg], q_s_t_selected_a_E)
            left_returns_norm = left_returns / return_norm  # right_return.shape: (len_LR_sample,)
            if self.LR.RR_type == 0:
                act_t_selected = LR_act_t[index_seg]  # (len_pairs, )
                q_s_t_selected_a_t = torch.gather(input=q_s_t_selected, dim=-1,
                                        index=act_t_selected[...,None]).reshape(len_pairs)  # (len_pairs,)
                q_s_tx_selected_a_tx = q_s_t_selected_a_t
            elif self.LR.RR_type == 1:
                act_t1 = torch.gather(act_t_seg, dim=1, 
                                      index=torch.from_numpy(
                                            target_oracle_index[..., None]+1
                                        ).long().to(self.device))  # act_t1.shape: (len_LR_sample, 1)
                obs_t1 = np.take_along_axis(obs_t_seg,
                                            target_oracle_index[..., None, None, None, None]+1,
                                            axis=1).\
                                            reshape(len_LR_sample, *self.obs_shape)
                obs_t1 = torch.from_numpy(obs_t1).float().to(self.device)
                if self.distributional:
                    p_s_t1 = self.agent(obs_t1, train=True)  # q_s_t.shape: (len_LR_sample, action_dim, n_atoms)
                    # z = torch.linspace(self.agent.V_min, self.agent.V_max, self.agent.n_atoms)
                    q_s_t1 = torch.tensordot(p_s_t1, self.distri_z, dims=1)  # (len_LR_sample, action_dim)
                else:
                    pdb.set_trace()  # check q_s_t.shape
                    q_s_t1 = self.agent(obs_t1, train=True)  # q_s_t.shape: (B, action_dim)
                LR_q_s_t1_a_t1 = torch.gather(input=q_s_t1, dim=-1,
                                              index=act_t1[...]).reshape(len_LR_sample,)  # r_hat_s_oa.shape: (B, 1)
                q_s_t1_selected_a_t1 = LR_q_s_t1_a_t1[index_seg]  # (len_pairs,)
                q_s_tx_selected_a_tx = q_s_t1_selected_a_t1
            right_returns = q_s_tx_selected_a_tx - \
                            torch.mul(self.gamma_exp[len_subseg], q_s_rihgt_selected_a_selected)
            right_returns_norm = right_returns / return_norm  # right_return.shape: (len_pairs,)
            LR_diff = left_returns_norm - right_returns_norm + self.LR.loss_slack
            loss = torch.max(-LR_diff, torch.zeros_like(LR_diff)).mean()  # mean() return a scalar
            LR_acc = (LR_diff.detach() >= 0.).sum().item()
            return loss, len_pairs, LR_acc
        else:
            raise NotImplementedError

    def finetune_margine_loss_DQfD(self, q_s, oracle_act, loss_margine):
        # NOTE: output from the reward model is constrained by tanh in (0, 1)
        # q_s.shape: (B, action_ndim), oracle_act.shape: (B)
        assert q_s.ndim == 2
        assert q_s.shape[-1] == self.action_dim
        assert oracle_act.shape == loss_margine.shape
        assert oracle_act.ndim == 1
        q_s_oa = torch.gather(input=q_s, dim=-1, index=oracle_act[...,None])  # r_hat_s_oa.shape: (B, 1)
        # loss_margine = self.loss_margine * torch.ones_like(q_s)  # (B, action_dim)  # old code for constant loss_margine
        loss_margine_arr = loss_margine[..., None].repeat(1, self.action_dim)
        loss_margine_oracle = torch.zeros_like(oracle_act, dtype=loss_margine_arr.dtype)  # (B,)
        loss_margine_arr.scatter_(dim=1, index=oracle_act[..., None],
                              src=loss_margine_oracle[..., None])  # (B, action_dim)
        
        # assert q_s_oa.shape == max_q_val.shape == \
            #    loss_margine.shape == (q_s.shape[0], 1)
        max_q_margine_val, max_q_margine_id = torch.max(q_s + loss_margine_arr, dim=-1, keepdim=True)  # max_q_margine.shape: (B, 1)
        loss = max_q_margine_val - q_s_oa  # loss.shape: (B, 1)
        return loss.mean()  # a scalar
    
    def finetune_margine_loss_min0_fix(self, q_s, oracle_act, loss_margine):
        # NOTE: output from the reward model is constrained by tanh in (0, 1)
        # r_hat_s.shape: (B, action_ndim), oracle_act.shape: (B)
        pdb.set_trace()  # check if the function is correct (shape, function)
        assert q_s.ndim == 2
        assert q_s.shape[-1] == self.action_dim
        assert oracle_act.ndim == 1
        assert oracle_act.shape == loss_margine.shape
        q_s_oa = torch.gather(input=q_s, dim=-1, index=oracle_act[...,None])  # r_hat_s_oa.shape: (B, 1)
        max_q_val, max_q_id = torch.max(q_s, dim=-1, keepdim=True) # (torch.max returns (values, indices))  max action reward
        # max_r_hat_val.shape == max_r_hat_id.sahpe: (B, 1)
        # Assign margine=0 if the max non-oracle reward < oracle reward; else margine=self.loss_margine
        # loss_margine = torch.where(max_q_id == oracle_act[...,None], 0, self.loss_margine)  # old code when loss_margine is a constant
        pdb.set_trace()  # check if loss_margine
        loss_margine_arr = torch.where(max_q_id == oracle_act[...,None], 0, loss_margine)
        assert q_s_oa.shape == max_q_val.shape == \
               loss_margine_arr.shape == (q_s.shape[0], 1)
        loss = max_q_val + loss_margine_arr - q_s_oa  # loss.shape: (B, 1)
        return loss.mean()  # a scalar

    def finetune_exp_loss(self, q_s, oracle_act, loss_margine=None):
        # q_s.shape: (B, action_ndim), oracle_act.shape: (B)
        loss = nn.CrossEntropyLoss()(q_s * self.exp_loss_beta,
                                     oracle_act.reshape(q_s.shape[0]))  # a scalar
        if self.loss_square:
            output_squared = q_s**2  # (B, action_dim)
            loss += self.loss_square_coef * output_squared.mean()  # output_squared.mean() is a scalar
        return loss

    def finetune(self, itr):
        self.agent.train_mode(itr=1)

        losses = []
        losses_margine = []
        losses_LR = []
        grad_norms = []
        acc = 0.
        LR_acc = 0.

        # max_len = self.label_capacity if self._label_buffer_full else self.label_t
        max_len = self.len_label
        total_batch_index = np.random.permutation(max_len)
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))  # NOTE: use 'cile', so all labeled data will be used to train reward predictor!
        # list_debug_loss1, list_debug_loss2 = [], []
        total = 0
        total_LR = 0
        for epoch in range(num_epochs):  # NOTE: larger batch_size should use larger learning_rate, because #epochs will decrease
            self.opt.zero_grad()
            
            if self.agent.model.noisy:  # For noisy net
                self.agent.model.head.reset_noise()
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
            
            idxs = total_batch_index[epoch*self.train_batch_size:last_index]
            # obs_t = torch.from_numpy(self.label_buffer.observation[idxs]).float().to(self.device)
            obs_t = self.label_buffer.observation[idxs]  # obs will be transferred to tensor in r_hat_s_member()
            act_t = torch.from_numpy(self.label_buffer.action[idxs]).long().to(self.device)
            oracle_act_t = torch.from_numpy(self.label_buffer.oracle_act[idxs]).long().to(self.device)  # (B,)
            margine_t = torch.from_numpy(self.label_buffer.margine[idxs]).to(self.device)  # (B,)
            obs_t = torch.from_numpy(obs_t).float().to(self.device)
            if self.distributional:
                p_s = self.agent(obs_t, train=True)  # q_s_t.shape: (B, action_dim, n_atoms)
                # z = torch.linspace(self.agent.V_min, self.agent.V_max, self.agent.n_atoms)
                q_s = torch.tensordot(p_s, self.distri_z, dims=1)  # (B, action_dim)
            else:
                pdb.set_trace()  # check q_s_t.shape
                q_s = self.agent(obs_t, train=True)  # q_s_t.shape: (B, action_dim)
            
            total += oracle_act_t.size(0)
            # compute loss
            loss = self.finetune_loss(q_s=q_s,
                                      oracle_act=oracle_act_t,
                                      loss_margine=margine_t,
                                      )
            if self.LR.flag:
                losses_margine.append(loss.item())
                LR_index = self.label_buffer.LR_index[idxs]
                with_LR = self.label_buffer.with_LR[idxs]
                if with_LR.sum() > 0:
                    loss_LR, LR_total, LR_correct = self.LR_loss(LR_index, with_LR, q_s, act_t, oracle_act_t)
                    losses_LR.append(loss_LR.item())
                    loss += self.LR.weight * loss_LR
                    total_LR += LR_total
                    LR_acc += LR_correct
                    
            losses.append(loss.item())
            loss.backward()

            if self.clip_grad_norm is not None:
                grad_norms.append(torch.nn.utils.clip_grad_norm_(
                        self.agent.model.parameters(), self.clip_grad_norm).item())  # default l2 norm
            # # code to check if there have nan and inf in grad
            # for net in self.ensemble:
            #     for name, param in net.named_parameters():
            #         try:
            #             assert not torch.any(torch.isnan(param.grad))
            #         except:
            #             print(f'[NAN] in {name}')
            #         try:
            #             assert not torch.any(torch.isinf(param.grad))
            #         except:
            #             print(f'[INF] in {name}')
            self.opt.step()
            
            # compute acc
            max_q_a = torch.max(q_s.data, dim=-1)[1]
            # check max_q_a & oracle_act_t shape == (self.train_batch_size (different batch_size for the last batch),)
            assert oracle_act_t.shape == max_q_a.shape == (oracle_act_t.shape[0], )
            correct = (max_q_a == oracle_act_t).sum().item()  # count the number of samples that r_hat assign largest value for oracle_actions
            acc += correct

        losses = np.mean(np.array(losses), axis=-1, keepdims=False)  # shape: (#ensemble,)
        grad_norms = np.mean(np.array(grad_norms), axis=-1, keepdims=False)  # shape: (#ensemble,)
        acc = acc / total
        if not self.LR.flag:
            return acc, losses, grad_norms
        else:
            losses_margine = np.mean(np.array(losses_margine), axis=-1, keepdims=False)  # shape: (#ensemble,)
            if total_LR > 0:
                losses_LR = np.mean(np.array(losses_LR), axis=-1, keepdims=False)  # shape: (#ensemble,)
                LR_acc = LR_acc / total_LR
            else:
                losses_LR = 0
                LR_acc = 0
            return acc, losses, grad_norms, losses_margine, losses_LR, LR_acc

    def samples_to_data_buffer(self, samples):
        # NOTE: data's order must be the same with initialize_buffer()'s sample_examples 
        #       Otherwise a serious bug will occur in add_data: e.g. samples.done be saved in input_buffer.done
        return SamplesToCFDataBuffer(
            observation=samples.env.observation,
            done=samples.env.done,
            action=samples.agent.action,
            GT_reward=samples.env.reward,
            oracle_act=samples.oracle_act,
            oracle_act_prob=samples.oracle_act_prob,
            oracle_q=samples.oracle_q,
        )
