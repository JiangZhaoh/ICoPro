import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pdb
import os
import matplotlib.pyplot as plt
import hydra

from new_utils.new_agent.rainbow_replay_buffer import discount_return_n_step
from new_utils.tensor_utils import torchify_buffer, get_leading_dims, select_at_indexes

from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.buffer import buffer_from_example

SamplesToCFDataBuffer = namedarraytuple("SamplesToCFDataBuffer",
                            ["observation", "done", "action", "GT_reward",
                            "oracle_act", "oracle_act_prob", "oracle_q"])
# QueriesToRewardBuffer = namedarraytuple("QueriesToRewardBuffer",
#     ["observation", "action", "oracle_act"])
EPS = 1e-6  # (NaN-guard)

class CorrectiveRlpytFinetuneRLModel:
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
                 cf_per_seg=1,  # number of corrective feedback per segment
                 query_recent_itr=1,
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
                 RL_loss=None,
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
        assert query_recent_itr >= 1
        self.query_recent_itr = query_recent_itr
        self.T_itr_ls = []
        self.label_itr_ls = []
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
        assert self.oracle_type in ['oq']
        self.softmax_tau = softmax_tau
        self.ckpt_path = ckpt_path

        self.distributional = distributional
        self.log_q_path = log_q_path
        if self.log_q_path:
            os.makedirs(self.log_q_path, exist_ok=True)
        self.total_itr = total_itr
        self.ignore_small_qdiff = ignore_small_qdiff

        self.gamma_exp = torch.tensor([self.gamma**k for k in range(self.size_segment)])
        self.discount_norm = torch.zeros_like(self.gamma_exp)
        self.discount_norm[0] = 1.
        for k in range(1, self.size_segment):
            self.discount_norm[k] = self.discount_norm[k - 1] + self.gamma_exp[k]
        self.np_gamma_exp = self.gamma_exp.detach().cpu().numpy()
        self.np_discount_norm = self.discount_norm.detach().cpu().numpy()
        self.gamma_exp = self.gamma_exp.float().to(self.device)
        self.discount_norm = self.discount_norm.float().to(self.device)
        
        self.RL_loss = RL_loss  # TODO: for RL loss, currently suppose that we use the same data buffer as input_buffer, which only contains recently evaluated trajectories
        if self.RL_loss.flag:
            assert RL_loss.data_recent_itr >= 1
            assert RL_loss.sl_type in ['tgt', 'label', 'AL', 'ALD', 'tlEqu', None]
            assert not self.margine_decay.flag  # if margine_decay, need to consider how to set the margine in RL_finetune

    def config_agent(self, agent):
        self.agent = agent
        if ("eps" not in self.optim_kwargs) or\
           (self.optim_kwargs["eps"] is None):  # Assume optim.Adam
            self.optim_kwargs["eps"] = 0.01 / self.train_batch_size
        # Because agent.search only related to agent.model and do
        #   not need agent.target_model, so only need to optimize this
        self.opt = eval(self.OptimCls)(self.agent.model.parameters(),
                        lr=self.lr, **self.optim_kwargs)
        
        if self.RL_loss.flag:
            if self.RL_loss.separate_opt:
                self.RL_opt = eval(self.OptimCls)(self.agent.model.parameters(),
                            lr=self.RL_loss.lr_ratio*self.lr, **self.optim_kwargs)
            else:
                self.RL_opt = self.opt

        if self.distributional:
            self.delta_z = (self.agent.V_max - self.agent.V_min) / (self.agent.n_atoms - 1)
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
        global InputBufferSamples
        InputBufferSamples = namedarraytuple("InputBufferSamples", field_names)
        
        input_buffer_example = InputBufferSamples(*(v for \
                                    k, v in sample_examples.items() \
                                    if k != "observation"))
        self.input_buffer = buffer_from_example(input_buffer_example,
                                                (self.max_size_T, self.B))
        if self.RL_loss.flag:
            # when calculate the RL loss, to also make sure the margine loss is guaranteed
            # self.data_label_index_buffer = buffer_from_example(0,
            #                             (self.max_size_T, self.B))  # (max_size_T, B)
            self.have_label_flag_buffer = buffer_from_example(False,
                                        (self.max_size_T, self.B))  # (max_size_T, B)
            if self.RL_loss.n_step > 1:
                self.input_samples_return_ = buffer_from_example(r,
                                            (self.max_size_T, self.B))  # (max_size_T B)
                self.input_samples_done_n = buffer_from_example(False,
                                            (self.max_size_T, self.B))  # (max_size_T, B)
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
        
        global RewardLabelBufferSamples
        RewardLabelBufferSamples = namedarraytuple("RewardLabelBufferSamples",
                                                   label_examples.keys())
        reward_label_buffer_example = RewardLabelBufferSamples(*(v for \
                                            k, v in label_examples.items()))
        
        self.label_buffer = buffer_from_example(reward_label_buffer_example,
                                                (self.label_capacity,))
        
        # if self.RL_label_loss.flag:
        #     pdb.set_trace()  # if cf_per_seg is not 1 then this buffer may include many repeated contents? 
        #     RL_example = dict()
        #     # TODO: although seems strange, currently let's use segment-based RL loss
        #     RL_example["observation"] = np.repeat(np.expand_dims(o[:],axis=0),
        #                                     repeats=self.size_segment, axis=0)  # shape: (size_seg, *obs_shape)
        #     RL_example["action"] = np.repeat([a], repeats=self.size_segment, axis=0)  # shape: (size_seg,)
        #     RL_example["oracle_action"] = np.repeat([a], repeats=self.size_segment, axis=0)  # shape: (size_seg,)
        #     RL_example["done"] = np.repeat([terminated], repeats=self.size_segment, axis=0)  # shape: (size_seg,)
        #     RL_example["CF_index"] = 0
        #     global RLBufferSamples
        #     RLBufferSamples = namedarraytuple("RLBufferSamples",
        #                                       RL_example.keys())
        #     RL_buffer_example = RLBufferSamples(*(v for \
        #                                         k, v in RL_example.items()))
        #     self.RL_buffer = buffer_from_example(RL_buffer_example,
        #                                          (self.RL_buffer_capacity,))
        #     self.RL_buffer_t = 0

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
        input_buffer_samples = InputBufferSamples(*(v \
                                for k, v in samples.items() \
                                if k != "observation"))
        T, B = get_leading_dims(input_buffer_samples, n_dim=2)  # samples.env.reward.shape[:2]
        self.T_itr_ls.append(T)
        assert B == self.B
        if t + T > self.max_size_T:  # Wrap.
            idxs = np.arange(t, t + T) % self.max_size_T
        else:
            idxs = slice(t, t + T)
        self.input_buffer[idxs] = input_buffer_samples
        if self.RL_loss.flag and self.RL_loss.n_step > 1:
            self.compute_returns(T)
            self.have_label_flag_buffer[idxs] = False  # for newly added data, it won't have label before get_label()
        
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
    
    """ From rlpyt.BaseNStepReturnBuffer """
    def compute_returns(self, T):  # T: the length/timesteps of newly added data
        """Compute the n-step returns using the new rewards just written into
        the buffer, but before the buffer cursor is advanced.
        Input ``T`` is the number of new timesteps which were just written (T>=1).
        Does nothing if `n-step==1`. 
        e.g. if 2-step return, t-1 is first return written here, 
        using reward at t-1 and new reward at t (up through t-1+T from t+T)."""
        if self.RL_loss.n_step == 1:
            return
        t = self.input_t
        s = self.input_buffer
        nm1 = self.RL_loss.n_step - 1
        if t - nm1 >= 0 and t + T <= self.max_size_T:  # No wrap (operate in-place).
            GT_reward = s.GT_reward[t - nm1: t + T]
            done = s.done[t - nm1: t + T]
            # NOTE: self.input_t to self.input_t+T are new appended data, so return_ and done_n can be updated only for (t-nm1: t-nm1+T)
            return_dest = self.input_samples_return_[t - nm1: t - nm1 + T]
            done_n_dest = self.input_samples_done_n[t - nm1: t - nm1 + T]
            discount_return_n_step(GT_reward, done,
                n_step=self.RL_loss.n_step, discount=self.gamma,
                return_dest=return_dest, done_n_dest=done_n_dest)
        else:
            idxs = np.arange(t - nm1, t + T) % self.max_size_T
            GT_reward = s.GT_reward[idxs]
            done = s.done[idxs]
            dest_idxs = idxs[:-nm1]
            return_, done_n = discount_return_n_step(GT_reward, done,
                n_step=self.RL_loss.n_step, discount=self.gamma)
            self.input_samples_return_[dest_idxs] = return_
            self.input_samples_done_n[dest_idxs] = done_n

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
        query_itr = min(len(self.T_itr_ls), self.query_recent_itr)
        min_index_T = self.len_inputs_T - np.sum(self.T_itr_ls[-query_itr:])
        # Batch = query_batch
        if use_seg:  # for top_seg_sampling
            assert query_batch is None
            assert self.B == 1
            batch_index = np.arange(min_index_T, max_index_T,
                                    step=self.size_segment).reshape(-1, 1)  # (query_batch, 1)
            query_batch = batch_index.shape[0]
        else:
            assert (max_index_T-min_index_T)*self.B > query_batch
            batch_index = min_index_T * self.B +\
                np.random.choice((max_index_T-min_index_T)*self.B,
                                  size=query_batch, replace=True).reshape(-1, 1)  # (query_batch, 1)
        
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

        return obs_t, act_t, oracle_act_t, oracle_act_prob_t, oracle_q_t, GT_reward_t,\
                 take_index_T.reshape(query_batch, self.size_segment),\
                 take_index_B.reshape(query_batch, self.size_segment)
    
    def put_queries(self, obs_t, act_t, oracle_act_t, early_advertising=False):
        # obs_t.shape: (query_batch * cf_per_seg * neighbor_size, *obs_shape) or (len_label, *obs_shape)
        # act_t & oracle_act_t.shape: (query_batch * cf_per_seg * neighbor_size,) or (len_label,)
        total_sample = obs_t.shape[0]
        next_index = self.label_t + total_sample  # new index in the query buffer after adding new queries
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

    def put_queries_with_RL(self, itr, obs_t, act_t, oracle_act_t,
                            take_index_T_selected, take_index_B_selected):
        # obs_t.shape: (query_batch * cf_per_seg * neighbor_size, *obs_shape) or (len_label, *obs_shape)
        # act_t & oracle_act_t.shape: (query_batch * cf_per_seg * neighbor_size,) or (len_label,)
        # take_index_T_selected, take_index_B_selected.shape (len_label,)
        total_sample = obs_t.shape[0]
        assert total_sample == take_index_T_selected.shape[0] == take_index_B_selected.shape[0]
        next_index = self.label_t + total_sample  # new index in the query buffer after adding new queries
        
        # NOTE: np.copyto(dest, src) is deepcopy for scalars
        if next_index >= self.label_capacity:
            assert not self.RL_loss.flag  # need to consider how to wrap
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
            
            # take_index_T_selected & take_index_T_selected is the index in self.input_buffer for those selected queries in this iteration
            # self.data_label_index_buffer[take_index_T_selected, take_index_B_selected] = np.arange(self.label_t, next_index)
            self.have_label_flag_buffer[take_index_T_selected, take_index_B_selected] = True
            
            self.label_t = next_index
    
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
                  oracle_q_t, GT_reward_t, take_index_T, take_index_B,
                  salient_oracle=False):
        # obs_t.shape: (query_batch, size_segment, *obs_shape)
        # act_t & oracle_act_t.shape: (query_batch, size_segment)
        # oracle_act_prob_t & oracle_q_t.shape: (query_batch, size_segment, action_ndim)
        # GT_reward_t.shape: (query_batch, size_seg)
        # take_index_B.shape & take_index_B.shape: (query_batch, size_segment)
        # NOTE: !!!! oracle_act_prob_t's minimal value should be larger than EPS!!! otherwise np.log will has errors!!!!
        segment_log_itr_path = os.path.join(self.segment_log_path, str(itr))
        os.makedirs(segment_log_itr_path, exist_ok=True)
        if self.oracle_type == 'oq':
            index_0 = np.arange(act_t.shape[0])
            index_1 = np.arange(act_t.shape[1])
            index_seg, index_batch = np.meshgrid(index_1, index_0)
            oracle_act_oracle_q = oracle_q_t[index_batch, index_seg, oracle_act_t]  # oracle's q for its own action, shape (query_batch, size_seg)
            act_oracle_q = oracle_q_t[index_batch, index_seg, act_t]  # oracle's q for current selected action, shape (query_batch, size_seg)
            
            q_diff = oracle_act_oracle_q - act_oracle_q  # q_diff are positive values, shape (query_batch, size_seg)
            # np.save(file=os.path.join(segment_log_itr_path, 'q_diff.npy'), arr=q_diff)
            
            target_oracle_index = np.argsort(q_diff, axis=-1)[:, -self.cf_per_seg:]  # ascending order, shape (query_batch, cf_per_seg)
            q_diff_selected = np.take_along_axis(q_diff,
                                                target_oracle_index,
                                                axis=1)  # (query_batch, cf_per_seg)
            take_index_T_selected = np.take_along_axis(take_index_T,
                                                target_oracle_index,
                                                axis=1)  # (query_batch, cf_per_seg)
            take_index_B_selected = np.take_along_axis(take_index_B,
                                                target_oracle_index,
                                                axis=1)  # (query_batch, cf_per_seg)
            if salient_oracle:  # select segments with larger q_diff
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
                take_index_T_selected = np.take_along_axis(take_index_T_selected,
                                                        q_diff_seg_selected[...,None], axis=0)  # (query_batch, cf_per_seg)
                take_index_B_selected = np.take_along_axis(take_index_B_selected,
                                                        q_diff_seg_selected[...,None], axis=0)  # (query_batch, cf_per_seg)
                oracle_act_oracle_q = np.take_along_axis(oracle_act_oracle_q,
                                            q_diff_seg_selected[...,None], axis=0)
                act_oracle_q = np.take_along_axis(act_oracle_q,
                                            q_diff_seg_selected[...,None], axis=0)
            assert q_diff_selected.shape == (self.query_batch, self.cf_per_seg)
        else:
            # TODO: maybe also consider r_hat, or even training agent's q_value?
            raise NotImplementedError
       
        # obs_t.shape: (len_label, *obs_shape)
        if self.ignore_small_qdiff.flag:
            kept_index = np.where(q_diff_selected>self.ignore_small_qdiff.thres)
            # q_diff_selected.shape = target_oracle_index.shape: (query_batch, cf_per_seg)
            len_label = kept_index[0].shape[0]
            kept_index_query = kept_index[0]  # shape: (len_label,)
            kept_index_timestep = target_oracle_index[kept_index[0], kept_index[1]]  # shape: (len_label,)
            # take_index_T_selected.shape: (query_batch, cf_per_seg) -> (len_label,)
            take_index_T_selected = take_index_T_selected[kept_index[0], kept_index[1]]  # shape: (len_label,)
            take_index_B_selected = take_index_B_selected[kept_index[0], kept_index[1]]  # shape: (len_label,)
            obs_t =  obs_t[kept_index_query, kept_index_timestep, ...].\
                        reshape(len_label, *self.obs_shape)
            # act_t.shape: (len_label,)
            act_t = act_t[kept_index_query, kept_index_timestep].\
                        reshape(len_label,)
            # oracle_act_t.shape: (len_label,)
            oracle_act_t = oracle_act_t[kept_index_query, kept_index_timestep].\
                            reshape(len_label,)
            
            # q_diff_t == q_diff_selected[kept_index] == q_diff_selected[kept_index[0], kept_index[1]]
            q_diff_t = q_diff[kept_index_query, kept_index_timestep].\
                        reshape(len_label,)
            # np.save(file=os.path.join(segment_log_itr_path, 'q_diff_t.npy'), arr=q_diff_t)
            self.q_diff_average = q_diff_t.mean()

            if self.log_q_path:
                oracle_q_t = oracle_q_t[kept_index_query, kept_index_timestep, ...].\
                            reshape(len_label, self.action_dim)
                # oracle_q_t.shape: (len_label, action_dim)
            else:
                oracle_q_t = None
        else:
            len_label = self.query_batch * self.cf_per_seg
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

            take_index_T_selected = take_index_T_selected.reshape(len_label,)
            take_index_B_selected = take_index_B_selected.reshape(len_label,)

            if self.log_q_path:
                oracle_q_t = np.take_along_axis(oracle_q_t,
                                                target_oracle_index[..., None],
                                                axis=1).\
                            reshape(len_label, self.action_dim)
            else:
                oracle_q_t = None
        
        self.label_itr_ls.append(len_label)
        if self.RL_loss.flag:
            return obs_t, act_t, oracle_act_t, oracle_q_t,\
                   take_index_T_selected, take_index_B_selected
        else:
            return obs_t, act_t, oracle_act_t, oracle_q_t

    def uniform_sampling(self, itr, salient_oracle=False, sample_multipler=1):
        # TODO: some CF label is provided on the same state
        # get queries
        obs_t, act_t, oracle_act_t, oracle_act_prob_t,\
            oracle_q_t, GT_reward_t,\
            take_index_T, take_index_B =  self.get_queries(
                query_batch=self.query_batch*sample_multipler)
        # obs_t.shape: (query_batch, cf_per_seg, *obs_shape)
        # take_index_T.shape = take_index_B.shape = (query_batch, size_segment)
        
        if self.RL_loss.flag:  # have done_t_seg
            # get labels
            obs_t, act_t, oracle_act_t, oracle_q_t,\
            take_index_T_selected, take_index_B_selected = \
                self.get_label(  # filter queries and 
                    itr, obs_t, act_t, oracle_act_t, oracle_act_prob_t,
                    oracle_q_t, GT_reward_t,
                    take_index_T, take_index_B,
                    salient_oracle=salient_oracle)
            # obs_t.shape: (len_label, *obs_shape)
            # take_index_T_selected.shape: (len_label, )
            self.put_queries_with_RL(itr, obs_t, act_t, oracle_act_t,
                            take_index_T_selected, take_index_B_selected)
        else:
            # get labels
            obs_t, act_t, oracle_act_t, oracle_q_t = self.get_label(  # filter queries and 
                itr, obs_t, act_t, oracle_act_t, oracle_act_prob_t,
                oracle_q_t, GT_reward_t,
                take_index_T, take_index_B,
                salient_oracle=salient_oracle)
            # obs_t.shape: (len_label, *obs_shape)
            self.put_queries(obs_t, act_t, oracle_act_t)

        if self.log_q_path:
            self.put_qlog(itr, obs_t, act_t, oracle_act_t, oracle_q_t)

        return obs_t.shape[0]  # query_batch * cf_per_seg
    
    def top_seg_sampling(self, itr):  # use_seg=True for self.get_queries(); salient_oracle=True for self.get_label()
        assert self.oracle_type == 'oq'
        # get queries
        obs_t, act_t, oracle_act_t, oracle_act_prob_t,\
            oracle_q_t, GT_reward_t,\
            take_index_T, take_index_B =  self.get_queries(
                query_batch=None, use_seg=True)
        # obs_t.shape: (query_batch, cf_per_seg, *obs_shape)
        
        # get labels
        if self.RL_loss.flag:  # have done_t_seg
            obs_t, act_t, oracle_act_t, oracle_q_t,\
            take_index_T_selected, take_index_B_selected = \
                self.get_label(  # filter queries and 
                    itr, obs_t, act_t, oracle_act_t, oracle_act_prob_t,
                    oracle_q_t, GT_reward_t, 
                    take_index_T, take_index_B,
                    salient_oracle=True)
            # obs_t.shape: (len_label, *obs_shape)
            # take_index_T_selected.shape: (len_label, )
            self.put_queries_with_RL(itr, obs_t, act_t, oracle_act_t,
                        take_index_T_selected, take_index_B_selected)
        else:
            # get labels
            obs_t, act_t, oracle_act_t, oracle_q_t = self.get_label(  # filter queries and 
                itr, obs_t, act_t, oracle_act_t, oracle_act_prob_t,
                oracle_q_t, GT_reward_t,
                take_index_T, take_index_B,
                salient_oracle=True)
            # obs_t.shape: (len_label, *obs_shape)
            self.put_queries(obs_t, act_t, oracle_act_t)

        if self.log_q_path:
            self.put_qlog(itr, obs_t, act_t, oracle_act_t, oracle_q_t)

        return obs_t.shape[0]  # query_batch * cf_per_seg

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
        grad_norms = []
        acc = 0.

        # max_len = self.label_capacity if self._label_buffer_full else self.label_t
        max_len = self.len_label
        total_batch_index = np.random.permutation(max_len)
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))  # NOTE: use 'cile', so all labeled data will be used to train reward predictor!
        # list_debug_loss1, list_debug_loss2 = [], []
        total = 0
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
            
            losses.append(loss.item())
            loss.backward()

            if self.clip_grad_norm is not None:
                grad_norms.append(torch.nn.utils.clip_grad_norm_(
                        self.agent.model.parameters(), self.clip_grad_norm).item())  # default l2 norm
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
        
        return acc, losses, grad_norms
    
    def RL_finetune(self, epoch):
        self.agent.train_mode(itr=1)
        if (epoch % self.RL_loss.update_tgt_interval) == 0:
            self.agent.update_target(tau=self.RL_loss.target_tau)
        if (self.RL_loss.separate_tgt) and\
            (self.RL_loss.separate_update_tgt_interval is not None) and\
            (epoch % self.RL_loss.separate_update_tgt_interval) == 0:
            self.agent.update_separate_target(tau=self.RL_loss.separate_target_tau)

        # acc for labels from expert
        CF_label_acc = 0.
        CF_label_cnt = 0
        # acc for labels from target model
        CF_tgt_acc = 0.
        CF_tgt_cnt = 0
        if self.RL_loss.sl_type in ['AL', 'ALD']:  # all label
            # acc for labels from label buffer
            CF_AL_acc = 0.
            CF_AL_cnt = 0

        cnt_label = 0  # the number of states that has a_E label from oracle

        rl_1_losses = []
        rl_n_losses = []
        sl_losses = []
        losses = []
        grad_norms = []

        data_itr = min(len(self.T_itr_ls), self.RL_loss.data_recent_itr)
        len_recent_sample = sum(self.T_itr_ls[-data_itr:])
        max_len = len_recent_sample - self.RL_loss.n_step

        assert max_len < self.max_size_T  # TODO: need to consider data wrap in self.input_buffer
        total_batch_index = self.len_inputs_T - len_recent_sample + \
                                np.random.permutation(max_len)
        RL_train_bc = self.RL_loss.train_bc_ratio * self.train_batch_size
        num_epochs = int(np.ceil(max_len / RL_train_bc))  # NOTE: use 'cile', so all labeled data will be used to train
        for epoch in range(num_epochs):  # NOTE: larger batch_size should use larger learning_rate, because #epochs will decrease
            self.RL_opt.zero_grad()
            
            if self.agent.model.noisy:  # For noisy net
                self.agent.model.head.reset_noise()
            
            last_index = (epoch + 1) * RL_train_bc
            if last_index > max_len:
                last_index = max_len
            
            idxs = total_batch_index[epoch * RL_train_bc: last_index]
            bs_epoch = idxs.shape[0]  # batch size in this epoch
            T_idxs = idxs
            B_idxs = np.zeros_like(idxs)  # because we assume input_buffer.B==1
            agent_inputs = self.extract_observation(T_idxs, B_idxs)
            agent_inputs_copy = torch.from_numpy(agent_inputs).float().to(self.device)
            agent_inputs = torch.from_numpy(agent_inputs).float().to(self.device)
            action = torch.from_numpy(self.input_buffer.action[T_idxs, B_idxs]).long().to(self.device)
            # for 1-step TD loss
            done = torch.from_numpy(self.input_buffer.done[T_idxs, B_idxs]).to(self.device)  # done.dtype = torch.bool
            GT_reward = torch.from_numpy(self.input_buffer.GT_reward[T_idxs, B_idxs]).float().to(self.device)
            target_next_T_idxs = T_idxs + 1
            assert np.all(target_next_T_idxs < self.max_size_T)  # TODO: for now, suppose the data buffer are unlimited. in the future, could consider use the current agent to explore and use RL on more data?
            target_next_agent_inputs = self.extract_observation(target_next_T_idxs, B_idxs)
            target_next_agent_inputs = torch.from_numpy(target_next_agent_inputs).float().to(self.device)
            
            if self.RL_loss.n_step > 1:
                # for n-step TD loss
                GT_return_ = torch.from_numpy(self.input_samples_return_[T_idxs, B_idxs]).float().to(self.device)
                done_n = torch.from_numpy(self.input_samples_done_n[T_idxs, B_idxs]).to(self.device)
                target_n_T_idxs = T_idxs + self.RL_loss.n_step
                assert np.all(target_n_T_idxs < self.max_size_T)  # TODO: for now, suppose the data buffer are unlimited. in the future, could consider use the current agent to explore and use RL on more data?
                target_n_agent_inputs = self.extract_observation(target_n_T_idxs, B_idxs)
                target_n_agent_inputs = torch.from_numpy(target_n_agent_inputs).float().to(self.device)
            else:
                target_n_agent_inputs = GT_return_ = done_n = None

            p_s = self.agent(agent_inputs, train=True)  # q_s_t.shape: (B, action_dim, n_atoms)
            if self.distributional:
                q_s = torch.tensordot(p_s, self.distri_z, dims=1)  # (B, action_dim)
            else:
                q_s = p_s
            
            if self.distributional:
                rl_1_loss, KL_1 = self.dist_rl_loss(n_step=1, ps=p_s, action=action,
                            done_n=done, return_=GT_reward, target_n_agent_inputs=target_next_agent_inputs)
                if self.RL_loss.n_step > 1:
                    rl_n_loss, KL_n = self.dist_rl_loss(n_step=self.RL_loss.n_step, ps=p_s, action=action,
                            done_n=done_n, return_=GT_return_, target_n_agent_inputs=target_n_agent_inputs)
                else:
                    rl_n_loss = 0.
            else:
                raise NotImplementedError
            
            with torch.no_grad():
                if self.distributional:
                    if self.RL_loss.separate_tgt:
                        tgt_p_s = self.agent.separate_target(agent_inputs_copy, train=True)  # q_s_t.shape: (B, action_dim, n_atoms)
                    else:
                        tgt_p_s = self.agent.target(agent_inputs_copy, train=True)  # q_s_t.shape: (B, action_dim, n_atoms)
                    # z = torch.linspace(self.agent.V_min, self.agent.V_max, self.agent.n_atoms)
                    tgt_q_s = torch.tensordot(tgt_p_s, self.distri_z, dims=1)  # (B, action_dim)
                else:
                    pdb.set_trace()  # check q_s_t.shape
                    if self.RL_loss.separate_tgt:
                        tgt_q_s = self.agent.separate_target(agent_inputs_copy, train=True)  # q_s_t.shape: (B, action_dim)
                    else:
                        tgt_q_s = self.agent.target(agent_inputs_copy, train=True)  # q_s_t.shape: (B, action_dim)
                oracle_act_tgt = torch.max(tgt_q_s.data, dim=-1)[1]  # tgt_max_q_a.shape: (agent_inputs.shape[0], )
                assert oracle_act_tgt.shape == (agent_inputs_copy.shape[0], )

            oracle_action = torch.from_numpy(self.input_buffer.oracle_act[T_idxs, B_idxs]).long().to(self.device)  # (B, )
            have_label_flag = torch.from_numpy(self.have_label_flag_buffer[T_idxs, B_idxs]).to(self.device)  # (B,)
            len_label = have_label_flag.sum().item()
            # TODO: maybe also mix some other loss from more CF labels
            if len_label > 0:
                cnt_label += len_label
                # pdb.set_trace()  # do not use cf_label_index in the following code? it's label index may not be correct
                # pdb.set_trace()  # check the following content
                # cf_label_index = torch.from_numpy(self.data_label_index_buffer[T_idxs, B_idxs]).long().to(self.device)
                # pdb.set_trace()  # 下面这个np.all 可能不太对，data_label_index_buffer是有默认0值的，如果CF正好在s0那就错了
                # print(f'{cf_label_index[cf_label_index>0].shape[0]==have_label_flag.sum()}')  # 除非有label是给在了input_buffer的第一个state上，否则这句话应该是成立的
                # print(f'{np.all(self.label_buffer.oracle_act[cf_label_index[cf_label_index>0]] == oracle_action[have_label_flag])}')
                
                oracle_act_label = oracle_action[have_label_flag]  # [have_label_flag.sum(),]
                # how to extract corresponding margine from label_buffer.margine is a problem TBD. margine_label = torch.from_numpy(self.label_buffer.margine[cf_label_index]).float().to(self.device)  # (B,)
                assert (len_label,) == oracle_act_label.shape

                label_qs = q_s[have_label_flag]  # label_qs shape [len_label, action_dim]. label_qs has gradient here. 

            if self.RL_loss.sl_type == 'label':
                if len_label > 0:
                    margine_tensor = torch.ones_like(oracle_act_label) * self.loss_margine
                    sl_loss =  self.finetune_loss(q_s=label_qs,
                                                oracle_act=oracle_act_label,
                                                loss_margine=margine_tensor)
                else:
                    sl_loss = torch.tensor(0.)
            elif self.RL_loss.sl_type in ['AL', 'ALD']:  # all label
                label_idxs = np.random.choice(self.len_label,
                                  size=bs_epoch, replace=True).reshape(-1,)  # (bc_epoch,)
                label_obs_t = self.label_buffer.observation[label_idxs]  # (bc_epoch, C, H, W)
                label_act_t = torch.from_numpy(self.label_buffer.action[label_idxs]).long().to(self.device)  # (bc_epoch,)
                label_oracle_act_t = torch.from_numpy(self.label_buffer.oracle_act[label_idxs]).long().to(self.device)  # (bc_epoch,)
                label_margine_t = torch.from_numpy(self.label_buffer.margine[label_idxs]).to(self.device)  # (bc_epoch,)
                label_obs_t = torch.from_numpy(label_obs_t).float().to(self.device)
                if self.distributional:
                    label_p_s = self.agent(label_obs_t, train=True)  # q_s_t.shape: (B, action_dim, n_atoms)
                    # z = torch.linspace(self.agent.V_min, self.agent.V_max, self.agent.n_atoms)
                    label_q_s = torch.tensordot(label_p_s, self.distri_z, dims=1)  # (B, action_dim)
                else:
                    pdb.set_trace()  # check q_s_t.shape
                    label_q_s = self.agent(label_obs_t, train=True)  # q_s_t.shape: (B, action_dim)
                
                # compute loss
                sl_loss = self.finetune_loss(q_s=label_q_s,
                                            oracle_act=label_oracle_act_t,
                                            loss_margine=label_margine_t)
                if self.RL_loss.sl_type == 'ALD':
                    sl_loss *= (self.len_label*1.0) / (max_len*1.0)
            elif self.RL_loss.sl_type == 'tgt':
                margine_tensor = torch.ones_like(oracle_act_tgt) * self.loss_margine
                sl_loss =  self.finetune_loss(q_s=q_s,  # q_s.shape (B, action_dim)
                                            oracle_act=oracle_act_tgt,  # oracle_act_tgt.shape: (B)
                                            loss_margine=margine_tensor)
            elif self.RL_loss.sl_type == 'tlEqu':
                margine_tensor = torch.ones_like(oracle_act_tgt) * self.loss_margine
                oracle_act_tlEqu = oracle_act_tgt.detach().clone()
                if len_label > 0:
                    oracle_act_tlEqu[have_label_flag] = oracle_act_label
                sl_loss =  self.finetune_loss(q_s=q_s,  # q_s.shape (B, action_dim)
                                            oracle_act=oracle_act_tlEqu,  # oracle_act_tgt.shape: (B)
                                            loss_margine=margine_tensor)
            elif self.RL_loss.sl_type is None:
                sl_loss = torch.tensor(0.)
            else:  # TODO: maybe consider combine both 'label' and 'tgt'
                raise NotImplementedError
            
            loss = rl_1_loss * self.RL_loss.one_step_weight \
                   + rl_n_loss * self.RL_loss.n_step_weight \
                   + sl_loss * self.RL_loss.sl_weight
            loss.backward()

            if self.clip_grad_norm is not None:
                grad_norms.append(torch.nn.utils.clip_grad_norm_(
                        self.agent.model.parameters(), self.clip_grad_norm).item())  # default l2 norm
            self.RL_opt.step()

            rl_1_losses.append(rl_1_loss.item())
            rl_n_losses.append(rl_n_loss.item())
            sl_losses.append(sl_loss.item())
            losses.append(loss.item())

            ## compute acc
            # acc for target modes's generated label
            pred_act = torch.max(q_s.data, dim=-1)[1]  # pred_act.shape: (agent_inputs.shape[0], )
            CF_tgt_cnt += pred_act.size(0)
            assert oracle_act_tgt.shape == pred_act.shape == (pred_act.shape[0], )
            correct_tgt = (pred_act == oracle_act_tgt).sum().item()  # count the number of samples that r_hat assign largest value for oracle_actions
            CF_tgt_acc += correct_tgt
            # acc for human's label
            if len_label > 0:
                label_pred_act = pred_act[have_label_flag]
                # label_pred_act_test = torch.max(label_qs.data, dim=-1)[1]  # tgt_max_q_a.shape: (len_label, )
                # assert torch.all(label_pred_act == label_pred_act_test)
                CF_label_cnt += label_pred_act.size(0)
                assert oracle_act_label.shape == label_pred_act.shape == (label_pred_act.shape[0], )
                correct_label = (label_pred_act == oracle_act_label).sum().item()  # count the number of samples that r_hat assign largest value for oracle_actions
                CF_label_acc += correct_label
            if self.RL_loss.sl_type in ['AL', 'ALD']:  # all label
                CF_AL_cnt += bs_epoch
                AL_pred_act = torch.max(label_q_s.data, dim=-1)[1]  # label_q_s.shape: (bs_epoch, action_dim); AL_pred_act.shape: (bs_epoch, )
                AL_correct_label = (AL_pred_act == label_oracle_act_t).sum().item()  # label_oracle_act_t.shape: (bs_epoch,). count the number of samples that r_hat assign largest value for oracle_actions
                CF_AL_acc += AL_correct_label
        
        rl_1_loss_avg = np.mean(np.array(rl_1_losses), axis=-1, keepdims=False)  # shape: (#ensemble,)
        rl_n_loss_avg = np.mean(np.array(rl_n_losses), axis=-1, keepdims=False)  # shape: (#ensemble,)
        sl_loss_avg = np.mean(np.array(sl_losses), axis=-1, keepdims=False)  # shape: (#ensemble,)
        loss_avg = np.mean(np.array(losses), axis=-1, keepdims=False)  # shape: (#ensemble,)
        grad_norms_avg = np.mean(np.array(grad_norms), axis=-1, keepdims=False)  # shape: (#ensemble,)
        CF_label_acc = CF_label_acc / CF_label_cnt
        CF_tgt_acc = CF_tgt_acc / CF_tgt_cnt
        # assert cnt_label == sum(self.label_itr_ls[-data_itr:])  # this assert is wrong, because some CF labels are given on the same states because segments can have overlaps
        if self.RL_loss.sl_type in ['AL', 'ALD']:  # all label
            CF_AL_acc = CF_AL_acc / CF_AL_cnt
        else:
            CF_AL_acc = None
        return CF_AL_acc, CF_label_acc, CF_tgt_acc,\
                rl_1_loss_avg, rl_n_loss_avg, sl_loss_avg, loss_avg,\
                grad_norms_avg

    def dist_rl_loss(self, n_step, ps, action,
                     done_n, return_, target_n_agent_inputs):
        # Makde 2-D tensor of contracted z_domain for each data point,
        # with zeros where next value should not be added.
        delta_z = self.delta_z
        z = self.distri_z  # [P']
        next_z = z * (self.gamma ** n_step)  # [P']
        next_z = torch.ger(1 - done_n.float(), next_z)  # [B,P']
        ret = return_.unsqueeze(1)  # [B,1]
        next_z = torch.clamp(ret + next_z, self.agent.V_min, self.agent.V_max)  # [B,P']

        z_bc = z.view(1, -1, 1)  # [1,P,1]
        next_z_bc = next_z.unsqueeze(1)  # [B,1,P']
        abs_diff_on_delta = abs(next_z_bc - z_bc) / delta_z
        projection_coeffs = torch.clamp(1 - abs_diff_on_delta, 0, 1)  # Most 0.
        # projection_coeffs is a 3-D tensor: [B,P,P']
        # dim-0: independent data entries
        # dim-1: base_z atoms (remains after projection)
        # dim-2: next_z atoms (summed in projection)

        with torch.no_grad():
            target_ps = self.agent.target(target_n_agent_inputs, train=True)  # [B,A,P'] train=True to avoid tensor.cpu()
            # TODO: do we really need double dqn here?
            if self.RL_loss.double_dqn:
                next_ps = self.agent(target_n_agent_inputs, train=True)  # [B,A,P'], here train=True to leave tensor on GPU device
                next_qs = torch.tensordot(next_ps, z, dims=1)  # [B,A]
                next_a = torch.argmax(next_qs, dim=-1)  # [B]
            else:
                target_qs = torch.tensordot(target_ps, z, dims=1)  # [B,A]
                next_a = torch.argmax(target_qs, dim=-1)  # [B]
            target_p_unproj = select_at_indexes(next_a, target_ps)  # [B,P']
            target_p_unproj = target_p_unproj.unsqueeze(1)  # [B,1,P']
            target_p = (target_p_unproj * projection_coeffs).sum(-1)  # [B,P]
        # ps = self.agent(*samples.agent_inputs)  # [B,A,P], obtained from self.loss()
        p = select_at_indexes(action, ps)  # [B,P]
        p = torch.clamp(p, EPS, 1)  # NaN-guard.
        losses = -torch.sum(target_p * torch.log(p), dim=1)  # Cross-entropy.

        target_p = torch.clamp(target_p, EPS, 1)
        KL_div = torch.sum(target_p *
            (torch.log(target_p) - torch.log(p.detach())), dim=1)
        KL_div = torch.clamp(KL_div, EPS, 1 / EPS)  # Avoid <0 from NaN-guard.

        loss = torch.mean(losses)

        return loss, KL_div
        
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
