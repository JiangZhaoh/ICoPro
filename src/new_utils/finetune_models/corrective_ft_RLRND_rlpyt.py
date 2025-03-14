import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pdb
import os
import matplotlib.pyplot as plt
import hydra
import pickle as pkl
import cv2

# try:  # for cluster
#     from new_utils.user_study_utils.interfaces import AtariUserApp
# except:
#     pass

from new_utils.new_agent.rainbow_replay_buffer import discount_return_n_step
from new_utils.tensor_utils import torchify_buffer, get_leading_dims, select_at_indexes

from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.buffer import buffer_from_example

SamplesToCFDataBuffer = namedarraytuple("SamplesToCFDataBuffer",
                            ["observation", "done", "action", "GT_reward",
                            "oracle_act", "oracle_act_prob", "oracle_q",
                            "human_img"])
# QueriesToRewardBuffer = namedarraytuple("QueriesToRewardBuffer",
#     ["observation", "action", "oracle_act"])
EPS = 1e-6  # (NaN-guard)

class CorrectiveRlpytFinetuneRLRNDModel:
    def __init__(self,
                 B,
                 obs_shape,
                 action_dim,
                 OptimCls,
                 optim_kwargs,
                 gamma,
                 steps_per_itr,
                 lr=3e-4,
                 query_batch=128,  # old name in PEBBLE is mb_size
                 train_batch_size=128,
                 size_segment=25,  # timesteps per segment
                 cf_per_seg=1,  # number of corrective feedback per segment
                 query_recent_itr=1,
                 max_size=None,  # max timesteps of trajectories for query sampling
                 label_capacity=None,  # "labeled" query buffer capacity, each query corresponds to a segment (old name in PEBBLE is capacity default 5e5)
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
                 log_path=None,
                 total_itr=None,
                 ignore_small_qdiff=None,
                 sampling_cfg=None,
                 use_ft=True,
                 tgt_in_ft=False,
                 RL_loss=None,
                 top_qdiff_cfg=None,
                 reset_opt=False,
                 cf_random=0.,
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
        self.reset_opt = reset_opt

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
        self.steps_per_itr = steps_per_itr

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
        elif loss_name == 'PV':
            self.finetune_loss = self.finetune_PV_loss
            self.init_loss_margine = loss_margine
            self.loss_margine = loss_margine  
            if self.loss_square:
                raise NotImplementedError# a useless flag to avoiding changing too much code
        else:
            raise NotImplementedError
        
        self.loss_name = loss_name
        self.margine_decay = margine_decay
        
        self.oracle_type = oracle_type
        assert self.oracle_type in ['oq', 'hm']
        self.softmax_tau = softmax_tau
        self.ckpt_path = ckpt_path

        self.distributional = distributional
        self.log_q_path = log_q_path
        if self.log_q_path:
            os.makedirs(self.log_q_path, exist_ok=True)
        self.log_path = log_path
        try:
            self.save_query_path = os.path.join(self.log_path, 'query_segments')  # human images
        except:
            pass
        if self.oracle_type == 'hm':
            os.makedirs(self.save_query_path, exist_ok=True)
        self.total_itr = total_itr

        self.ignore_small_qdiff = ignore_small_qdiff
        if self.ignore_small_qdiff.add_whole_seg == True:
            assert self.ignore_small_qdiff.flag == False

        self.gamma_exp = torch.tensor([self.gamma**k for k in range(self.size_segment)])
        self.discount_norm = torch.zeros_like(self.gamma_exp)
        self.discount_norm[0] = 1.
        for k in range(1, self.size_segment):
            self.discount_norm[k] = self.discount_norm[k - 1] + self.gamma_exp[k]
        self.np_gamma_exp = self.gamma_exp.detach().cpu().numpy()
        self.np_discount_norm = self.discount_norm.detach().cpu().numpy()
        self.gamma_exp = self.gamma_exp.float().to(self.device)
        self.discount_norm = self.discount_norm.float().to(self.device)
        
        self.sampling_cfg = sampling_cfg
        assert sampling_cfg.seg_type in ['u', 's']
        assert sampling_cfg.sampling_type in ['uni', 'topq']
        self.sampling_current_uni_ratio = 1  # default uni_ratio ('1' represents the initial uniform sampling case)
        if self.sampling_cfg.uni_ratio is not None:  # uniform sampling from a batch of segments
            if type(self.sampling_cfg.uni_ratio) != str:  # a constant value
                assert self.sampling_cfg.uni_ratio <= 1.
                assert self.sampling_cfg.uni_ratio > 0.
            else:
                assert not (tgt_in_ft.flag and tgt_in_ft.split_query)  # not implemented
                if self.sampling_cfg.uni_ratio in ['lin', 'cos']:
                    self.sampling_min_uni_ratio = (self.query_batch+1.0) \
                                / (np.floor((self.steps_per_itr - self.size_segment) // self.size_segment)*1.0)
                else:
                    self.sampling_min_uni_ratio = float(self.sampling_cfg.uni_ratio[3:])
                print(f'[sampling_min_uni_ratio]: {self.sampling_min_uni_ratio}; total_itr: {self.total_itr}')
            assert not (sampling_cfg.sampling_type == 'uni')  # ban this meaningless case
            if self.sampling_cfg.seg_type == 'u':
                # for seg_type == 'u', consider using sample_multipler then using uni_ratio
                assert self.sampling_cfg.sample_multipler is not None 
                assert self.sampling_cfg.sample_multipler >= 1
                if type(self.sampling_cfg.uni_ratio) != str:
                    assert self.sampling_cfg.sample_multipler * self.query_batch \
                            * self.sampling_cfg.uni_ratio >= self.query_batch
                else:
                    assert self.sampling_cfg.sample_multipler * self.query_batch \
                            * self.sampling_min_uni_ratio >= self.query_batch
            elif self.sampling_cfg.seg_type == 's':
                # for seg_type == 's', it has already cut all trajectories into segments and sample from them, so for this case only consider uni_ratio is enough
                if type(self.sampling_cfg.uni_ratio) != str:  # a constant value
                    assert np.floor((self.steps_per_itr - self.size_segment) // self.size_segment)\
                            * self.sampling_cfg.uni_ratio >= self.query_batch
                else:
                    assert np.floor((self.steps_per_itr - self.size_segment) // self.size_segment)\
                            * self.sampling_min_uni_ratio >= self.query_batch
            else:
                raise NotImplementedError
        
        self.use_ft = use_ft
        if not use_ft:
            assert tgt_in_ft.flag == False
        self.tgt_in_ft = tgt_in_ft
        self.ft_use_tgt = False
        if self.tgt_in_ft.flag:
            assert use_ft & RL_loss.flag
            assert (self.tgt_in_ft.bs_ratio == None) or\
                    (type(self.tgt_in_ft.bs_ratio) in [float, int])
        self.RL_loss = RL_loss  # TODO: for RL loss, currently suppose that we use the same data buffer as input_buffer, which only contains recently evaluated trajectories
        if self.RL_loss.flag:
            assert RL_loss.RL_recent_itr >= 1
            assert not self.margine_decay.flag  # if margine_decay, need to consider how to set the margine in RL_finetune
            assert not (self.RL_loss.tgt_label.same_data_RL \
                        and self.RL_loss.tgt_label.same_data_RL_target)
        
        self.top_qdiff_cfg = top_qdiff_cfg
        assert self.top_qdiff_cfg.type in [None, 'sum', 'max', 'mmSum']

        assert cf_random <= 1.
        self.cf_random = cf_random

        if self.RL_loss.use_RLIF_reward:
            assert self.RL_loss.flag
            assert self.RL_loss.use_reward
    
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
                assert self.reset_opt == True  # if separate_opt==True but reset_opt==False, then the internal states maintained inside Adam are almost wrong for the real statistics (i.e. the maintained 1-st and 2-nd oarder moment are not consistent with current gradient)
                self.RL_opt = eval(self.OptimCls)(self.agent.model.parameters(),
                            lr=self.RL_loss.lr_ratio*self.lr, **self.optim_kwargs)
            else:
                if self.reset_opt == True:  # reset_opt==True and RL_loss.separate_opt==False
                    assert self.RL_loss.lr_ratio == 1  # if not separate_opt==False, then can not using the same lr as ft
                self.RL_opt = self.opt

        if self.distributional:
            self.delta_z = (self.agent.V_max - self.agent.V_min) / (self.agent.n_atoms - 1)
            self.distri_z = torch.linspace(
                self.agent.V_min, self.agent.V_max, self.agent.n_atoms).\
                to(self.device)

    def initialize_buffer(self, agent, oracle, env,
                          check_label_path=None, segment_log_path=None,
                          action_names=None,
                          remove_frame_axis=False):
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
        if self.oracle_type == 'hm':
            from new_utils.user_study_utils.interfaces import AtariUserApp
            self.hm_interface_app = AtariUserApp
            sample_examples["human_img"] = env_info["human_img"][:]  # HWC

        field_names = [f for f in sample_examples.keys() if f not in ["observation", "human_img"]]
        global InputBufferSamples
        InputBufferSamples = namedarraytuple("InputBufferSamples", field_names)
        
        input_buffer_example = InputBufferSamples(*(v for \
                                    k, v in sample_examples.items() \
                                    if k not in ["observation", 'human_img']))
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
        self.remove_frame_axis = remove_frame_axis
        if self.remove_frame_axis:
            self.n_frames = n_frames = 1
            self.input_frames = buffer_from_example(o[:],  # avoid saving duplicated frames
                                (self.max_size_T + n_frames - 1, self.B))
        else:
            self.n_frames = n_frames = get_leading_dims(o[:], n_dim=1)[0]
            self.input_frames = buffer_from_example(o[0],  # avoid saving duplicated frames
                                (self.max_size_T + n_frames - 1, self.B))
        if self.oracle_type == 'hm':
            self.input_human_imgs = buffer_from_example(env_info["human_img"][:],  # HWC
                            (self.steps_per_itr+200, self.B))  # T,B,HWC
            # pdb.set_trace()  # remember to save the whole query segment and labeled CFs in this case
        print(f"[FT Buffer-Inputs] Frame-based buffer using {n_frames}-frame sequences.")
        
        # self.input_frames.shape: (self.max_size_T+n_frames-1, self.B, H, W), uint8
        self.input_new_frames = self.input_frames[n_frames - 1:]  # [self.max_size_T,B,H,W]
        
        self.input_t = 0
        self._input_buffer_full = False
        
        label_examples["observation"] = o[:]
        label_examples["action"] = a
        label_examples["oracle_act"] = oracle_a
        if self.loss_name in ['MargineLossDQfD', 'MargineLossMin0Fix', 'PV']:
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
        self.env_action_meaning = action_names
    
    def save_label_buffer(self, save_dir, itr, total_steps,
                          rewrite=False):
        save_path = 'label_buffer.pkl' if rewrite else \
                    f'label_buffer_{itr}_{total_steps}.pkl'
        with open(os.path.join(save_dir, save_path),
                  'wb') as file:  # wb will rewrite
            t_buffer = self.label_buffer[:self.len_label]
            t_dict = {k:v for k,v in t_buffer.items()}
            pkl.dump(t_dict, file)
    
    def reset_label_buffer(self, label_buffer):
        label_examples = dict()
        for k, v in label_buffer.items():
            label_examples[k] = v[0]
            len_buffer = v.shape[0]
        
        global RewardLabelBufferSamples
        RewardLabelBufferSamples = namedarraytuple("RewardLabelBufferSamples",
                                                   label_examples.keys())
        reward_label_buffer_example = RewardLabelBufferSamples(*(v for \
                                            k, v in label_examples.items()))
        self.label_capacity = len_buffer + 10
        self.label_buffer = buffer_from_example(reward_label_buffer_example,
                                                (self.label_capacity,))
        self.label_t = len_buffer
        self._label_buffer_full = False
        for k, v in label_buffer.items():
            try:
                eval(f'self.label_buffer.{k}')[:len_buffer] = label_buffer[k]
            except:
                eval(f'self.label_buffer.{k}')[:len_buffer] = eval(f'label_buffer.{k}')

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
    
    def get_uni_ratio(self, itr):
        if type(self.sampling_cfg.uni_ratio) != str:
            return self.sampling_cfg.uni_ratio
        elif self.sampling_cfg.uni_ratio.startswith('lin'):
            return 1.0 -\
                    ((itr+1-self.sampling_cfg.uni_itr)*1.0)/((self.total_itr-self.sampling_cfg.uni_itr)*1.0) \
                        * (1.0-self.sampling_min_uni_ratio)
        elif self.sampling_cfg.uni_ratio.startswith('cos'):
            raise NotImplementedError  # currently I think linear is the most basic solution for decay; cosine decay seems not necessary to test
        else:
            raise NotImplementedError

    def add_data(self, samples,
                 collected_timestep):
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
                                if k not in ["observation", "human_img"]))
        # T, B = get_leading_dims(input_buffer_samples, n_dim=2)  # samples.env.reward.shape[:2]
        buffer_T, B = get_leading_dims(input_buffer_samples, n_dim=2)  # samples.env.reward.shape[:2]
        T = collected_timestep
        self.T_itr_ls.append(T)
        assert B == self.B
        if t + T > self.max_size_T:  # Wrap.
            idxs = np.arange(t, t + T) % self.max_size_T
        else:
            idxs = slice(t, t + T)
        if T != buffer_T:  # the case: tgt_in_ft.split_query==True
            self.input_buffer[idxs] = input_buffer_samples[:T]
        else:
            self.input_buffer[idxs] = input_buffer_samples
        if self.RL_loss.flag and self.RL_loss.n_step > 1:
            if not self.RL_loss.use_RLIF_reward:
                self.compute_returns(T)
            else:
                self.input_buffer.GT_reward[idxs] = 0.
                self.input_samples_return_[idxs] = 0.
            self.have_label_flag_buffer[idxs] = False  # for newly added data, it won't have label before get_label()
        
        if not self._input_buffer_full and t + T >= self.max_size_T:
            self._input_buffer_full = True  # Only changes on first around.
        self.input_t = (t + T) % self.max_size_T

        if self.remove_frame_axis:
            assert samples.observation.ndim == 3 # (T, B, D)
            self.input_new_frames[idxs] = samples.observation[:T, :]
        else:
            assert samples.observation.ndim == 5 and\
                samples.observation.shape[2] == fm1 + 1 # (T, B, frame_stack, H, W)
            self.input_new_frames[idxs] = samples.observation[:T, :, -1]
        if self.oracle_type == 'hm':
            hm_img_idxs = np.arange(0, 0 + T) % self.max_size_T
            self.input_human_imgs[hm_img_idxs] = samples.human_img[:T, :]
        # self.samples_new_frames[idxs] = samples.observation[:, :, -1]
        # self.input_frames.shape: (size+fm1, B, H, W)

        if t == 0:  # Starting: write early frames
            for f in range(fm1):
                self.input_frames[f] = samples.observation[0, :, f]
        elif self.input_t <= t and fm1 > 0:  # Wrapped: copy any duplicate frames. In the case that T==self.max_size_T, self.input_t will == t
            self.input_frames[:fm1] = self.input_frames[-fm1:]
        # return T, idxs
    
    """ From rlpyt.BaseNStepReturnBuffer """
    def compute_returns(self, T, st_t=None):  # T: the length/timesteps of newly added data
        """Compute the n-step returns using the new rewards just written into
        the buffer, but before the buffer cursor is advanced.
        Input ``T`` is the number of new timesteps which were just written (T>=1).
        Does nothing if `n-step==1`. 
        e.g. if 2-step return, t-1 is first return written here, 
        using reward at t-1 and new reward at t (up through t-1+T from t+T)."""
        if self.RL_loss.n_step == 1:
            return
        if st_t is None:
            t = self.input_t
        else:
            t = st_t
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
        assert not self._label_buffer_full  # in previous experiments, we have tested that keeping all labels are beneficial.
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
        if self.remove_frame_axis:
            observation = np.concatenate([self.input_frames[t:t + self.n_frames, b]
                for t, b in zip(T_idxs, B_idxs)], axis=0)  # [B*n_frames, *obs_shape_for_one_frame]
        else:
            # np.stack will generate a new axis
            observation = np.stack([self.input_frames[t:t + self.n_frames, b]
                for t, b in zip(T_idxs, B_idxs)], axis=0)  # [B,n_frames,H,W]
        
        # Populate empty (zero) frames after environment done.
        for f in range(1, self.n_frames):
            # e.g. if done 1 step prior, all but newest frame go blank.
            b_blanks = np.where(self.input_buffer.done[T_idxs - f, B_idxs])[0]
            observation[b_blanks, :self.n_frames - f] = 0
        return observation

    def sampling(self, itr, logger, log_total_steps):
        if itr < self.sampling_cfg.uni_itr:
            cnt_labeled_queries = self.uniform_sampling(
                                        itr=itr, top_qdiff=False,
                                        logger=logger,
                                        log_total_steps=log_total_steps)
        else:
            if self.sampling_cfg.sampling_type == 'uni':
                cnt_labeled_queries = self.uniform_sampling(
                                            itr=itr, top_qdiff=False,
                                            logger=logger,
                                            log_total_steps=log_total_steps)
            elif self.sampling_cfg.sampling_type == 'topq':
                cnt_labeled_queries = self.uniform_sampling(
                                        itr=itr, top_qdiff=True,
                                        logger=logger,
                                        log_total_steps=log_total_steps)
            else:
                raise NotImplementedError
        return cnt_labeled_queries
    
    def uniform_sampling(self, itr, top_qdiff=False,
                         logger=None, log_total_steps=None):
        # get queries
        if self.sampling_cfg.seg_type == 's':
            query_batch=(self.query_batch if not top_qdiff\
                        else None)  # in get_queries, if query_batch==None, then all segments will be returned
        elif self.sampling_cfg.seg_type == 'u':
            query_batch=(self.query_batch if not top_qdiff\
                else self.query_batch*self.sampling_cfg.sample_multipler)
        else:
            raise NotImplementedError
        obs_t, act_t, oracle_act_t, oracle_act_prob_t,\
            oracle_q_t, GT_reward_t,\
            take_index_T, take_index_B,\
            min_index_T = self.get_queries(
                query_batch=query_batch)
        # obs_t.shape: (query_batch, size_segment, *obs_shape)
        # take_index_T.shape = take_index_B.shape = (query_batch, size_segment)        
        if self.RL_loss.flag:  # need tp return more information from get_label & put more information when put_queries
            # get labels
            obs_t, act_t, oracle_act_t, oracle_q_t,\
            take_index_T_selected, take_index_B_selected = \
                self.get_label(  # filter queries and 
                    itr, obs_t, act_t, oracle_act_t, oracle_act_prob_t,
                    oracle_q_t, GT_reward_t,
                    take_index_T, take_index_B,
                    top_qdiff=top_qdiff,
                    min_index_T=min_index_T
                    )
            # obs_t.shape: (len_label, *obs_shape)
            # take_index_T_selected.shape: (len_label, )
            self.put_queries_with_RL(obs_t, act_t, oracle_act_t,
                            take_index_T_selected, take_index_B_selected)
        else:
            # get labels
            obs_t, act_t, oracle_act_t, oracle_q_t = self.get_label(  # filter queries and 
                itr, obs_t, act_t, oracle_act_t, oracle_act_prob_t,
                oracle_q_t, GT_reward_t,
                take_index_T, take_index_B,
                top_qdiff=top_qdiff,
                min_index_T=min_index_T
                )
            # obs_t.shape: (len_label, *obs_shape)
            self.put_queries(obs_t, act_t, oracle_act_t)

        if self.log_q_path:
            self.put_qlog(itr, obs_t, act_t, oracle_act_t, oracle_q_t)
             
        return obs_t.shape[0]  # query_batch * cf_per_seg
    
    def get_queries(self, query_batch=None):
        # if query_batch is None, then for seg_type=='s', return all segment index
        # check shape. TODO: consider if done=True should be considered;
        # get train traj
        assert not self._input_buffer_full  # otherwise query_recent_itr is not accurate
        max_index_T = self.len_inputs_T - self.size_segment
        query_itr = min(len(self.T_itr_ls), self.query_recent_itr)
        min_index_T = self.len_inputs_T - np.sum(self.T_itr_ls[-query_itr:])
        # Batch = query_batch
        if self.sampling_cfg.seg_type == 's':
            assert self.B == 1
            batch_index = np.arange(min_index_T, max_index_T,
                                    step=self.size_segment).reshape(-1, 1)  # (query_batch, 1)
            cnt_seg = batch_index.shape[0]
            if query_batch is None:
                query_batch = cnt_seg
            else:
                selected_index = np.random.choice(cnt_seg,
                                  size=query_batch, replace=False)
                batch_index = batch_index[selected_index].reshape(query_batch, 1)
        elif self.sampling_cfg.seg_type == 'u':  # uniform
            assert (max_index_T-min_index_T)*self.B > query_batch
            assert query_batch is not None
            batch_index = min_index_T * self.B +\
                np.random.choice((max_index_T-min_index_T)*self.B,
                                  size=query_batch, replace=True).reshape(-1, 1)  # (query_batch, 1)
        else:
            raise NotImplementedError
        
        batch_index_T, batch_index_B = np.divmod(batch_index, self.B)  # (x // y, x % y), batch_index_B & batch_index_T.shape: (query_batch, 1)
        take_index_T = (batch_index_T + np.arange(0, self.size_segment)).reshape(-1)  # shape: (query_batch * size_segment, )
        take_index_B = batch_index_B.repeat(self.size_segment, axis=-1).reshape(-1)  # shape: (query_batch * size_segment, )

        # obs_t = self.input_frames[take_index_B, take_index_T, ...]  # (query_batch * size_segment, *obs_shape)
        obs_t = self.extract_observation(take_index_T, take_index_B)  # (query_batch * size_segment, *obs_shape)
        assert (obs_t.ndim == 1+len(self.obs_shape)) 
        assert (obs_t.shape[0] == query_batch * self.size_segment)
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
                 take_index_B.reshape(query_batch, self.size_segment),\
                 min_index_T
    
    def put_queries(self, obs_t, act_t, oracle_act_t):
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

    def put_queries_with_RL(self, obs_t, act_t, oracle_act_t,
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
            if self.RL_loss.use_RLIF_reward:
                self.input_buffer.GT_reward[take_index_T_selected, take_index_B_selected] = -1.
                self.compute_returns(self.T_itr_ls[-1], st_t=self.input_t-self.T_itr_ls[-1])

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
                  top_qdiff=False, min_index_T=None):
        # obs_t.shape: (query_batch, size_segment, *obs_shape)
        # act_t & oracle_act_t.shape: (query_batch, size_segment)
        # oracle_act_prob_t & oracle_q_t.shape: (query_batch, size_segment, action_ndim)
        # GT_reward_t.shape: (query_batch, size_seg)
        # take_index_B.shape & take_index_B.shape: (query_batch, size_segment)
        # NOTE: !!!! oracle_act_prob_t's minimal value should be larger than EPS!!! otherwise np.log will has errors!!!!
        # segment_log_itr_path = os.path.join(self.segment_log_path, str(itr))
        # os.makedirs(segment_log_itr_path, exist_ok=True)
        if self.oracle_type == 'oq':
            index_0 = np.arange(act_t.shape[0])
            index_1 = np.arange(act_t.shape[1])
            index_seg, index_batch = np.meshgrid(index_1, index_0)
            oracle_act_oracle_q = oracle_q_t[index_batch, index_seg, oracle_act_t]  # oracle's q for its own action, shape (query_batch, size_seg)
            act_oracle_q = oracle_q_t[index_batch, index_seg, act_t]  # oracle's q for current selected action, shape (query_batch, size_seg)
            
            q_diff = oracle_act_oracle_q - act_oracle_q  # q_diff are non-negative values, shape (query_batch, size_seg)
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
            if top_qdiff:  # select segments with larger q_diff
                raise NotImplementedError  # haven't consider add_whole_seg=True for this part
                pdb.set_trace()  # this part is better to be removed to XX_sampling(); and I think the code about target_oracle_index is wrong since it's only based on the top cf_per_seg states.
                pdb.set_trace()  # this part, MinMax normalization should use the global min & global max, instead MinMax from each segent
                if self.top_qdiff_cfg.type == 'sum':
                    q_diff_selected_pool = np.sum(q_diff_selected, axis=-1, keepdims=False)  # (self.query_batch*sampler_multiplier)
                elif self.top_qdiff_cfg.type == 'max':
                    q_diff_selected_pool = np.max(q_diff_selected, axis=-1, keepdims=False)  # (self.query_batch*sampler_multiplier)
                elif self.top_qdiff_cfg.type == 'mmSum':
                    q_diff_selected_min = np.min(q_diff_selected, axis=-1, keepdims=True)  # (self.query_batch*sampler_multiplier, 1)
                    q_diff_selected_max = np.max(q_diff_selected, axis=-1, keepdims=True)  # (self.query_batch*sampler_multiplier, 1)
                    q_diff_selected_mm = (q_diff_selected - q_diff_selected_min) / (q_diff_selected_max - q_diff_selected_min + 1e-8)
                    q_diff_selected_pool = np.sum(q_diff_selected_mm, axis=-1, keepdims=False)  # (self.query_batch*sampler_multiplier)
                else:
                    raise NotImplementedError
                if self.sampling_cfg.uni_ratio is None:
                    self.sampling_current_uni_ratio = -1
                    q_diff_seg_selected = np.argsort(q_diff_selected_pool, axis=-1)[-self.query_batch:]  # (self.query_batch)
                else:
                    self.sampling_current_uni_ratio = self.get_uni_ratio(itr=itr)
                    num_all_seg = q_diff_selected_pool.shape[0]
                    num_selected_seg = int(np.ceil(num_all_seg * \
                                                   self.sampling_current_uni_ratio))
                    assert num_selected_seg >= self.query_batch
                    assert num_selected_seg <= num_all_seg
                    q_diff_seg_selected_uni = np.argsort(q_diff_selected_pool, axis=-1)[-num_selected_seg:]  # (num_selected_seg)
                    q_diff_seg_selected = np.random.choice(q_diff_seg_selected_uni,
                                                size=self.query_batch, replace=False)  # (self.query_batch,)

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

            # obs_t.shape: (len_label, *obs_shape)
            if self.ignore_small_qdiff.flag == True:
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
                if self.ignore_small_qdiff.add_whole_seg == True:
                    # q_diff_selected.shape: (query_batch, cf_per_seg)
                    kept_single_index = np.where(q_diff_selected>self.ignore_small_qdiff.thres)  # return q_diff_selected.ndim array indicating the indeices of satisfied values
                    len_single_label = kept_single_index[0].shape[0]
                    kept_index_single_query = kept_single_index[0]  # shape: (len_single_label,)
                    kept_index_single_timestep = target_oracle_index[kept_single_index[0],
                                                                    kept_single_index[1]]  # shape: (len_single_label,)
                    # take_index_T_selected.shape: (query_batch, cf_per_seg)
                    # take_single_index_T_selected.shape: (query_batch, cf_per_seg) -> (len_single_label,)
                    take_single_index_T_selected = take_index_T_selected[kept_single_index[0],
                                                                        kept_single_index[1]].\
                                                        reshape(len_single_label,)  # shape: (len_label,)
                    take_single_index_B_selected = take_index_B_selected[kept_single_index[0],
                                                                        kept_single_index[1]].\
                                                        reshape(len_single_label,)  # shape: (len_label,)
                    single_obs_t = obs_t[kept_index_single_query, kept_index_single_timestep, ...].\
                                reshape(len_single_label, *self.obs_shape)
                    single_act_t = act_t[kept_index_single_query, kept_index_single_timestep].\
                                reshape(len_single_label,)
                    single_oracle_act_t = oracle_act_t[kept_index_single_query, kept_index_single_timestep].\
                                    reshape(len_single_label,)
                    single_q_diff_t = q_diff[kept_index_single_query, kept_index_single_timestep].\
                                reshape(len_single_label,)
                    
                    kept_whole_seg_index = np.where(
                        q_diff_selected.max(axis=-1, keepdims=False)<=self.ignore_small_qdiff.thres)
                    len_selected_seg = kept_whole_seg_index[0].shape[0]
                    len_selected_seg_states = len_selected_seg * self.size_segment
                    take_whole_seg_index_T_selected= take_index_T[kept_whole_seg_index].\
                                                    reshape(len_selected_seg_states,)  # (len_selected_seg*size_seg,)
                    take_whole_seg_index_B_selected = take_index_B[kept_whole_seg_index].\
                                                    reshape(len_selected_seg_states,)  # (len_selected_seg*size_seg,)
                    seg_obs_t = obs_t[kept_whole_seg_index, ...].\
                                reshape(len_selected_seg_states, *self.obs_shape)
                    seg_act_t = act_t[kept_whole_seg_index, :].\
                                reshape(len_selected_seg_states,)
                    seg_oracle_act_t = oracle_act_t[kept_whole_seg_index, :].\
                                    reshape(len_selected_seg_states,)
                    seg_q_diff_t = q_diff[kept_whole_seg_index, :].\
                                reshape(len_selected_seg_states,)
                    
                    len_label = len_single_label + len_selected_seg_states
                    take_index_T_selected = np.concatenate([take_single_index_T_selected,
                                                            take_whole_seg_index_T_selected])  # (len_laebl, )
                    take_index_B_selected = np.concatenate([take_single_index_B_selected,
                                                            take_whole_seg_index_B_selected])  # (len_laebl, )
                    assert take_index_T_selected.shape == (len_label,)
                    obs_t = np.concatenate([single_obs_t, seg_obs_t], axis=0)  # (len_laebl, *obs_shape)
                    act_t = np.concatenate([single_act_t, seg_act_t], axis=0)  # (len_laebl, )
                    oracle_act_t = np.concatenate([single_oracle_act_t, seg_oracle_act_t], axis=0)  # (len_laebl, )
                    q_diff_t = np.concatenate([single_q_diff_t, seg_q_diff_t], axis=0)  # (len_laebl, )
                    self.q_diff_average = q_diff_t.mean()

                    if self.log_q_path:
                        single_oracle_q_t = oracle_q_t[kept_index_single_query, kept_index_single_timestep, ...].\
                                    reshape(len_single_label, self.action_dim)
                        seg_oracle_q_t = oracle_q_t[kept_whole_seg_index, ...].\
                                    reshape(len_selected_seg_states, self.action_dim)
                        oracle_q_t = np.concatenate([single_oracle_q_t, seg_oracle_q_t], axis=0)
                        assert oracle_q_t.shape == (len_label, self.action_dim)
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
            if self.cf_random > 0.:
                random_action = np.random.randint(self.action_dim, size=oracle_act_t.shape)
                random_arr = np.random.rand(*oracle_act_t.shape)
                random_mask = random_arr < self.cf_random
                oracle_act_t = random_mask * random_action \
                            + (1 - random_mask) * oracle_act_t
        elif self.oracle_type == 'hm':  # human
            ## save q_diff as well
            index_0 = np.arange(act_t.shape[0])
            index_1 = np.arange(act_t.shape[1])
            index_seg, index_batch = np.meshgrid(index_1, index_0)
            oracle_act_oracle_q = oracle_q_t[index_batch, index_seg, oracle_act_t]  # oracle's q for its own action, shape (query_batch, size_seg)
            act_oracle_q = oracle_q_t[index_batch, index_seg, act_t]  # oracle's q for current selected action, shape (query_batch, size_seg)
            
            q_diff = oracle_act_oracle_q - act_oracle_q  # q_diff are non-negative values, shape (query_batch, size_seg)
            np.save(f'{self.save_query_path}/Oqdiff_Itr{itr}.npy', q_diff)
            # test_arr = np.load(f'{self.save_query_path}/Oqdiff_Itr{itr}.npy'), type(test_arr) = ndarray
            # q_diff_rank = np.argsort(np.argsort(q_diff, axis=-1), axis=-1)  # ascending order: larger rank<-->larger q_diff, shape (query_batch, cf_per_seg)
            take_index_T_hm = take_index_T - min_index_T
            assert np.all(take_index_T_hm >= 0)
            human_img_seg = self.input_human_imgs[take_index_T_hm, take_index_B]  # (query_batch, size_segment, HWC)
            human_img_shape_h, human_img_shape_w = (human_img_seg.shape[-3], human_img_seg.shape[-2])
            qbc = act_t.shape[0]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_hz = 10
            text_hight = 30
            text_font_size = 0.3
            text_action_pos = (1, 10)
            text_reward_pos = (1, 20)
            for id_seg in range(qbc):
                # save mp4 video
                seg_file_name = f'{self.save_query_path}/Itr{itr}_Seg{id_seg}.mp4'
                # obs_mp4_out = cv2.VideoWriter(f'{self.save_query_path}/Itr{itr}_Seg{id_seg}_obs.mp4',
                #         fourcc, float(video_hz),
                #         (84, 84))
                mp4_out = cv2.VideoWriter(seg_file_name,
                        fourcc, float(video_hz),
                        (human_img_shape_w, human_img_shape_h + text_hight))
                for id_frame in range(self.size_segment):
                    text_here = f"act_t: {self.env_action_meaning[act_t[id_seg][id_frame]]}"
                    color_here = (0, 0, 0)
                    save_img = cv2.putText(
                                            np.concatenate([
                                                (np.ones((text_hight, human_img_shape_w, 3))*255).astype(np.uint8), # white color is (255, 255, 255)
                                                human_img_seg[id_seg, id_frame][:, :, ::-1].copy(), # RGB to BGR, to be compatible with cv2
                                            ]), # RGB to BGR, to be compatible with cv2
                                            # f"a_t: {self.eval_envs[env_id].action_names[action[b]]}" if terminated_ls[b]==True else "Terminated",
                                            text_here,
                                            text_action_pos, cv2.FONT_HERSHEY_SIMPLEX, text_font_size,
                                            color_here, 1)  # position, font, fontsize, fontcolor, fontweight
                    if GT_reward_t[id_seg][id_frame] > 0:
                        reward_text_color = (0, 255, 0) # green
                    elif GT_reward_t[id_seg][id_frame] < 0:
                        reward_text_color = (0, 0, 255) # red
                    else:
                        reward_text_color = (0, 0, 0) # black
                    save_img = cv2.putText(save_img.copy(),
                                        f"[R(s_t,a_t)]Sign: {int(GT_reward_t[id_seg][id_frame]):d}",  # R_t+1
                                        text_reward_pos, cv2.FONT_HERSHEY_SIMPLEX, text_font_size,
                                        reward_text_color, 1)  # position, font, fontsize, fontcolor, fontweight
                    mp4_out.write(save_img.copy())
                    # obs_mp4_out.write(obs_t[id_seg][id_frame][-1][..., None].repeat(3,axis=-1).copy())
                mp4_out.release()
                # obs_mp4_out.release()
                cv2.destroyAllWindows()

            app = self.hm_interface_app(
                            window_title=f'Iteration {itr}, query batch size {qbc}',
                            human_img_seg=human_img_seg,
                            agent_act_seg=act_t,
                            action_names=self.env_action_meaning,
                        )
            app.MainLoop()

            human_CF_list = np.stack(app.Frame.CF_list)
            # human_CF_list[:, 0] is the id of segments in [0, qbc-1]
            # human_CF_list[:, 1] is the id of frames in a segment in [0, size_seg-1]
            # human_CF_list[:, 2] are provided CFs
            len_single_label = human_CF_list.shape[0]
            # kept_index_single_query = kept_single_index[0]  # shape: (len_single_label,)
            # kept_index_single_timestep = target_oracle_index[human_CF_list[:,0],
            #                                                 human_CF_list[:,1]]  # shape: (len_single_label,)
            # take_index_T_selected.shape: (query_batch, cf_per_seg)
            # take_single_index_T_selected.shape: (query_batch, cf_per_seg) -> (len_single_label,)
            # previous obs_t.shape = (qbc, size_segment)
            obs_t =  obs_t[human_CF_list[:, 0],
                            human_CF_list[:, 1]].\
                        reshape(len_single_label, *self.obs_shape)
            # act_t.shape: (query_batch * cf_per_seg,)
            act_t = act_t[human_CF_list[:, 0],
                            human_CF_list[:, 1]].\
                        reshape(len_single_label,)
            oracle_act_t = human_CF_list[:, 2].reshape(len_single_label,)
            oracle_q_t = None
            # # oracle_act_t.shape: (query_batch * cf_per_seg,)
            # oracle_act_t = np.take_along_axis(oracle_act_t,
            #                                 target_oracle_index,
            #                                 axis=1).\
            #                 reshape(len_single_label,)
            take_index_T_selected = take_index_T[human_CF_list[:, 0],
                                                 human_CF_list[:, 1]].\
                                                reshape(len_single_label,)  # shape: (len_label,)
            take_index_B_selected = take_index_B[human_CF_list[:, 0],
                                                 human_CF_list[:, 1]].\
                                                reshape(len_single_label,)  # shape: (len_label,)
            len_label = len_single_label
            self.q_diff_average = None
        else:
            # TODO: maybe also consider r_hat, or even training agent's q_value?
            raise NotImplementedError
        
        self.label_itr_ls.append(len_label)
        if self.RL_loss.flag:
            return obs_t, act_t, oracle_act_t, oracle_q_t,\
                   take_index_T_selected, take_index_B_selected
        else:
            return obs_t, act_t, oracle_act_t, oracle_q_t

    def finetune_PV_loss(self, q_s, oracle_act, agent_act, loss_margine):
        ## adapted from pvp.pvp_dqn.PVPDQN.train()
        # q_s.shape: (B, action_ndim)
        # oracle_act.shape = agent_act.shape: (B)
        assert q_s.ndim == 2
        assert q_s.shape[-1] == self.action_dim
        assert oracle_act.shape == agent_act.shape == (q_s.shape[0],)

        q_s_oa = torch.gather(input=q_s, dim=-1, index=oracle_act[...,None])  # r_hat_s_oa.shape: (B, 1)
        q_s_aa = torch.gather(input=q_s, dim=-1, index=agent_act[...,None])  # r_hat_s_aa.shape: (B, 1)
        no_overlap = (oracle_act != agent_act)  # no_overlap.shape: (B)
        pvp_loss = F.mse_loss(
                        q_s_oa,
                        torch.ones_like(q_s_oa)
                    ) + \
                    F.mse_loss(
                        no_overlap * q_s_aa,
                        no_overlap * -1. * torch.ones_like(q_s_aa)
                    )
        return pvp_loss.mean()  # tiny difference with PVP: PVP's implementation is tlEqu-tgt style, so smaller pvp_loss for a batch with less intervention; but here in our finetune-phase, this is not the case since we sample mini-batches from intervened data

    def finetune_margine_loss_DQfD(self, q_s, oracle_act, loss_margine,
                                   loss_weight=None, agent_act=None):
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
        if loss_weight is not None:
            # loss.shape: (B, 1); loss_weight.shape (B,); in this case, (loss*loss_weight).shape=(B, B)
            loss = loss.reshape(loss_weight.shape[0],)   # this reshape can also guarantee that loss_weight has the same batch_size with loss
            loss *= loss_weight
        return loss.mean()  # a scalar
    
    def finetune_margine_loss_min0_fix(self, q_s, oracle_act, loss_margine, agent_act=None):
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

    def finetune_exp_loss(self, q_s, oracle_act, loss_margine=None, agent_act=None):
        # q_s.shape: (B, action_ndim), oracle_act.shape: (B)
        loss = nn.CrossEntropyLoss()(q_s * self.exp_loss_beta,
                                     oracle_act.reshape(q_s.shape[0]))  # a scalar
        if self.loss_square:
            output_squared = q_s**2  # (B, action_dim)
            loss += self.loss_square_coef * output_squared.mean()  # output_squared.mean() is a scalar
        return loss

    @torch.no_grad()
    def tgt_label_test(self,):
        if self.tgt_in_ft.acc_test_include_new:  # (an unnatural case?) also include labels from the unseed new iteration
            max_len = self.len_label
        else:
            assert not self.tgt_in_ft.split_query  # in that case, self,label_itr_ls is not maintained correctly
            if len(self.label_itr_ls) == 1:
                return 0.
            else:
                max_len = sum(self.label_itr_ls[:-1])
        total_batch_index = np.random.permutation(max_len)
        batch_size = 2048
        total_sample = 0
        acc = 0
        num_epochs = int(np.ceil(max_len / batch_size))  # NOTE: use 'cile', so all labeled data will be used to train reward predictor!

        for epoch in range(num_epochs):  # NOTE: larger batch_size should use larger learning_rate, because #epochs will decrease
            last_index = (epoch + 1) * batch_size
            if last_index > max_len:
                last_index = max_len
            idxs = total_batch_index[epoch*batch_size : last_index]
            
            obs_t = self.label_buffer.observation[idxs]  # obs will be transferred to tensor in r_hat_s_member()
            oracle_act_t = torch.from_numpy(self.label_buffer.oracle_act[idxs]).long().to(self.device)  # (B,)
            obs_t = torch.from_numpy(obs_t).float().to(self.device)
            
            if self.distributional:
                tgt_p_s = self.agent.target(obs_t, train=True)  # q_s_t.shape: (B, action_dim, n_atoms)
                # z = torch.linspace(self.agent.V_min, self.agent.V_max, self.agent.n_atoms)
                tgt_q_s = torch.tensordot(tgt_p_s, self.distri_z, dims=1)  # (B, action_dim)
            else:
                pdb.set_trace()  # check q_s_t.shape
                tgt_q_s = self.agent.target(obs_t, train=True)  # q_s_t.shape: (B, action_dim)
            argmax_tgt_q_a = torch.max(tgt_q_s.data, dim=-1)[1]  # (B,)
            
            correct = (argmax_tgt_q_a == oracle_act_t).sum().item()  # count the number of samples that r_hat assign largest value for oracle_actions
            
            total_sample += oracle_act_t.size(0)
            acc += correct
        
        acc = acc / total_sample
        return acc
    
    def reset_opt_ft(self):
        if self.reset_opt:
            self.opt = eval(self.OptimCls)(self.agent.model.parameters(),
                            lr=self.lr, **self.optim_kwargs)
    
    def reset_opt_rl(self):
        if self.reset_opt:
            if self.RL_loss.separate_opt:
                self.RL_opt = eval(self.OptimCls)(self.agent.model.parameters(),
                                lr=self.RL_loss.lr_ratio*self.lr, **self.optim_kwargs)
            else:
                self.opt = eval(self.OptimCls)(self.agent.model.parameters(),
                            lr=self.lr, **self.optim_kwargs)
                self.RL_opt = self.opt
        
    def finetune(self, itr, ft_epoch):
        self.agent.train_mode(itr=1)
        # max_len = self.label_capacity if self._label_buffer_full else self.label_t
        max_human_len = self.len_label
        human_total_batch_index = np.random.permutation(max_human_len)
        if ft_epoch == 0:
            if self.tgt_in_ft.flag == True:
                # self.tgt_test_acc = self.tgt_label_test()
                # self.ft_use_tgt = (self.tgt_test_acc >= self.tgt_in_ft.acc_test_thres)
                # if self.tgt_in_ft.split_query \
                #     and (itr >= self.tgt_in_ft.st_itr):
                #     assert self.ft_use_tgt  # a sanity check, if split_query==True, then ft_use_tgt should have been modified in finetuneXXX.py
                self.ft_use_tgt = (itr >= self.tgt_in_ft.st_itr)
                self.prev_agent_tgt_act_acc = None
                self.prev_agent_human_act_acc = None
                if self.ft_use_tgt:
                    # if self.tgt_in_ft.tgt_upadte_interval is None:
                    # no matter what the value of self.tgt_in_ft.tgt_upadte_interval is, we should update target if ft_epoch == 0.
                    self.agent.update_target(tau=1.)
            else:
                self.ft_use_tgt = False
                self.tgt_test_acc = None
        else:
            if self.ft_use_tgt:
                if self.tgt_in_ft.tgt_upadte_interval is not None:
                    if (ft_epoch % self.tgt_in_ft.tgt_upadte_interval) == 0:
                        self.agent.update_target(tau=1.)
        
        if self.ft_use_tgt:
            tgt_data_itr = min(len(self.T_itr_ls), self.tgt_in_ft.data_recent_itr)
            if self.tgt_in_ft.split_query == True:
                assert self.tgt_in_ft.data_recent_itr == 1  # haven't consider other cases
            len_tgt_recent_sample = sum(self.T_itr_ls[-tgt_data_itr:])
            max_tgt_len = len_tgt_recent_sample

            if self.tgt_in_ft.bs_ratio == None:
                auto_human_bs_prop = max_human_len / (max_human_len + max_tgt_len)
                human_train_bs = min(1, int(self.train_batch_size * auto_human_bs_prop))
                tgt_train_bs = self.train_batch_size - human_train_bs
            else:
                human_train_bs = min(1, int(self.train_batch_size / (self.tgt_in_ft.bs_ratio + 1.0)))
                tgt_train_bs = self.train_batch_size - human_train_bs

            tgt_total_batch_index = self.input_t - len_tgt_recent_sample + \
                                np.random.permutation(max_tgt_len)
            tgt_total_batch_index %= self.max_size_T
        else:
            human_train_bs = self.train_batch_size
            tgt_train_bs = None
        self.ft_human_train_bs = human_train_bs
        self.ft_tgt_train_bs = tgt_train_bs

        losses = []
        grad_norms = []
        human_acc = 0.
        
        num_epochs = int(np.ceil(max_human_len/self.ft_human_train_bs))  # NOTE: use 'cile', so all labeled data will be used to train reward predictor!
        # list_debug_loss1, list_debug_loss2 = [], []
        total = 0
        if self.ft_use_tgt:
            cnt_tgt_states = 0.
            tgt_oracle_act_acc = 0.
            agent_match_tgt_acc = 0.
            agent_match_oracle_acc = 0.
            tgt_losses = []
            human_losses = []
        self.ft_w_human_ls = []
        self.ft_w_tgt_ls = []
        
        for epoch in range(num_epochs):  # NOTE: larger batch_size should use larger learning_rate, because #epochs will decrease
            self.opt.zero_grad()
            
            if self.agent.model.noisy:  # For noisy net
                self.agent.model.head.reset_noise()
            
            last_index = (epoch+1)*self.ft_human_train_bs
            if last_index > max_human_len:
                last_index = max_human_len
            
            idxs = human_total_batch_index[epoch*self.ft_human_train_bs:last_index]
            # obs_t = torch.from_numpy(self.label_buffer.observation[idxs]).float().to(self.device)
            obs_t = self.label_buffer.observation[idxs]  # NOTE: idxs is advanced indexing, so obs_t do not share the same memory with self.label_buffer.observation
            act_t = torch.from_numpy(self.label_buffer.action[idxs]).long().to(self.device)
            oracle_act_t = torch.from_numpy(self.label_buffer.oracle_act[idxs]).long().to(self.device)  # (B,)
            margine_t = torch.from_numpy(self.label_buffer.margine[idxs]).to(self.device)  # (B,)
            obs_t = torch.from_numpy(obs_t).float().to(self.device)  
            # NOTE: torch.from_numpy share the same memory with input, but .float() and .to(device) will create a new copy when it's declear a differnet dtype or device.
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
                                      agent_act=act_t)
            
            if self.ft_use_tgt:
                tgt_batch_size = self.ft_tgt_train_bs
                tgt_label_idxs = np.random.choice(tgt_total_batch_index,
                                    size=tgt_batch_size,
                                    replace=False).reshape(-1,)  # (tgt_batch_size,)

                tgt_B_idxs = np.zeros_like(tgt_label_idxs)  # because we assume input_buffer.B==1
                inputs_tgt = self.extract_observation(tgt_label_idxs, tgt_B_idxs)  # (bc_epoch, C, H, W)
                inputs_tgt = torch.from_numpy(inputs_tgt).float().to(self.device)

                tgt_oracle_act_t = torch.from_numpy(self.input_buffer.oracle_act[tgt_label_idxs, tgt_B_idxs]).long().to(self.device)  # (bc_epoch,)
                tgt_margine_t = torch.ones_like(tgt_oracle_act_t).to(self.device, dtype=margine_t.dtype) * self.loss_margine
                
                with torch.no_grad():
                    if self.distributional:
                        tgt_p_s = self.agent.target(inputs_tgt, train=True)  # q_s_t.shape: (B, action_dim, n_atoms)
                        # z = torch.linspace(self.agent.V_min, self.agent.V_max, self.agent.n_atoms)
                        tgt_q_s = torch.tensordot(tgt_p_s, self.distri_z, dims=1)  # (B, action_dim)
                    else:
                        pdb.set_trace()  # check q_s_t.shape
                        tgt_q_s = self.agent.target(inputs_tgt, train=True)  # q_s_t.shape: (B, action_dim)
                    pseudo_act_tgt = torch.max(tgt_q_s.data, dim=-1)[1]
                    assert tgt_oracle_act_t.shape == pseudo_act_tgt.shape
                    correct_tgt_oracle_act = (tgt_oracle_act_t == pseudo_act_tgt).sum().item()
                tgt_oracle_act_acc += correct_tgt_oracle_act
                cnt_tgt_states += tgt_oracle_act_t.shape[0]
                
                # compute loss
                if self.distributional:
                    agent_tgt_p_s = self.agent(inputs_tgt, train=True)  # q_s_t.shape: (B, action_dim, n_atoms)
                    # z = torch.linspace(self.agent.V_min, self.agent.V_max, self.agent.n_atoms)
                    agent_tgt_q_s = torch.tensordot(agent_tgt_p_s, self.distri_z, dims=1)  # (B, action_dim)
                else:
                    pdb.set_trace()  # check q_s_t.shape
                    agent_tgt_q_s = self.agent(inputs_tgt, train=True)  # q_s_t.shape: (B, action_dim)
                with torch.no_grad():
                    agent_act_4tgt = torch.max(agent_tgt_q_s.data, dim=-1)[1]
                    agent_act_4tgt_match_tgt = (agent_act_4tgt == pseudo_act_tgt).sum().item()
                    agent_act_4tgt_match_oracle = (agent_act_4tgt == tgt_oracle_act_t).sum().item()
                agent_match_tgt_acc += agent_act_4tgt_match_tgt
                agent_match_oracle_acc += agent_act_4tgt_match_oracle

                if self.tgt_in_ft.type == 'tgt':
                    sl_loss_tgt = self.finetune_loss(q_s=agent_tgt_q_s,
                                                oracle_act=pseudo_act_tgt,
                                                loss_margine=tgt_margine_t)
                elif self.tgt_in_ft.type == 'tlEqu':
                    tgt_oracle_action = torch.from_numpy(self.input_buffer.oracle_act[tgt_label_idxs, tgt_B_idxs]).long().to(self.device)  # (B, )
                    tgt_have_label_flag = torch.from_numpy(self.have_label_flag_buffer[tgt_label_idxs, tgt_B_idxs]).to(self.device)  # (B,)
                    len_tgt_label = tgt_have_label_flag.sum().item()
                    if len_tgt_label > 0:
                        # cnt_tgt_label += len_tgt_label
                        tgt_oracle_act_label = tgt_oracle_action[tgt_have_label_flag]  # [have_label_flag.sum(),]
                        pseudo_act_tgt[tgt_have_label_flag] = tgt_oracle_act_label
                    sl_loss_tgt = self.finetune_loss(q_s=agent_tgt_q_s,
                                                oracle_act=pseudo_act_tgt,
                                                loss_margine=tgt_margine_t)
                else:
                    raise NotImplementedError
                
                if self.tgt_in_ft.dynamic_weight == True:
                    if self.prev_agent_tgt_act_acc is not None:
                        # w_human = self.prev_agent_tgt_act_acc / (self.prev_agent_human_act_acc+self.prev_agent_tgt_act_acc)  # if tgt_acc higher than human, then w for human_loss should be larger; so w_human propotional to tgt_acc instead of human_acc
                        # w_tgt = 1.0 - w_human  # in this case, if prev_agent_human_act_acc==prev_agent_tgt_act_acc==0., w_tgt approximate 1
                        w_tgt = self.prev_agent_human_act_acc / (self.prev_agent_human_act_acc+self.prev_agent_tgt_act_acc+1e-15)  # if tgt_acc higher than human, then w for human_loss should be larger; so w_human propotional to tgt_acc instead of human_acc
                        w_human = 1.0 - w_tgt  # in this case, if prev_agent_human_act_acc==prev_agent_tgt_act_acc==0., w_human approximate 1
                    else:
                        w_human = 0.5
                        w_tgt = 0.5
                    sum_loss = w_human * loss + w_tgt * sl_loss_tgt
                else:
                    # loss += sl_loss_tgt
                    # loss /= 2.0
                    w_human = 0.5
                    w_tgt = 0.5
                    sum_loss = w_human * loss + w_tgt * sl_loss_tgt
                # self.ft_w_human = w_human
                # self.ft_w_tgt = w_tgt
                self.ft_w_human_ls.append(w_human)
                self.ft_w_tgt_ls.append(w_tgt)
                sum_loss.backward()
                losses.append(sum_loss.item())
                tgt_losses.append(sl_loss_tgt.item())
                human_losses.append(loss.item())
            else:
                self.ft_w_human_ls.append(1.)
                self.ft_w_tgt_ls.append(0.)
                loss.backward()
                losses.append(loss.item())

            if self.clip_grad_norm is not None:
                grad_norms.append(torch.nn.utils.clip_grad_norm_(
                        self.agent.model.parameters(), self.clip_grad_norm).item())  # default l2 norm
            self.opt.step()
            
            # compute acc
            max_q_a = torch.max(q_s.data, dim=-1)[1]
            # check max_q_a & oracle_act_t shape == (RL_train_bs (different batch_size for the last batch),)
            assert oracle_act_t.shape == max_q_a.shape == (oracle_act_t.shape[0], )
            human_correct = (max_q_a == oracle_act_t).sum().item()  # count the number of samples that r_hat assign largest value for oracle_actions
            human_acc += human_correct

        losses = np.mean(np.array(losses), axis=-1, keepdims=False)  # shape: (#ensemble,)
        grad_norms = np.mean(np.array(grad_norms), axis=-1, keepdims=False)  # shape: (#ensemble,)
        human_acc = human_acc / total
        
        if self.ft_use_tgt:
            tgt_oracle_act_acc /= cnt_tgt_states
            agent_match_tgt_acc /= cnt_tgt_states
            agent_match_oracle_acc /= cnt_tgt_states
            tgt_losses = np.mean(np.array(tgt_losses), axis=-1, keepdims=False)  # shape: (#ensemble,)
            human_losses = np.mean(np.array(human_losses), axis=-1, keepdims=False)  # shape: (#ensemble,)
            self.prev_agent_tgt_act_acc = agent_match_tgt_acc
            self.prev_agent_human_act_acc = human_acc
        else:
            tgt_oracle_act_acc = None
            agent_match_tgt_acc = None
            agent_match_oracle_acc = None
            tgt_losses = None
            human_losses = None
            self.prev_agent_tgt_act_acc = None
            self.prev_agent_human_act_acc = None  # in this case prev_agent_human_act_acc will not be used, so no need to assign the value of human_acc
        
        return human_acc, losses, grad_norms,\
               tgt_oracle_act_acc, agent_match_tgt_acc, agent_match_oracle_acc,\
               tgt_losses, human_losses
    
    def RL_finetune(self, upd_epoch):  # upd_epoch: update_epoch
        self.agent.train_mode(itr=1)
        if (upd_epoch % self.RL_loss.update_tgt_interval) == 0:
            self.agent.update_target(tau=self.RL_loss.target_tau)
        if (self.RL_loss.separate_tgt) and\
            (self.RL_loss.separate_update_tgt_interval is not None) and\
            (upd_epoch % self.RL_loss.separate_update_tgt_interval) == 0:
            self.agent.update_separate_target(tau=self.RL_loss.separate_target_tau)

        if self.RL_loss.human_label.sl_weight > 0:
            # acc for labels from expert
            CF_agent_pred_human_acc = 0.
            CF_agent_pred_human_cnt = 0
        if self.RL_loss.tgt_label.sl_weight > 0:
            # acc for labels from target model
            CF_agent_pred_tgt_acc = 0.
            CF_agent_pred_tgt_cnt = 0
            if self.RL_loss.tgt_label.RND_check.filter is not None:
                tgt_RND_confident_cnt = 0
                num_tgt_RND_state = 0
                num_wrong_doubt = 0
        if self.RL_loss.RND_label.sl_weight > 0:
            # acc for labels from pseudo labels from the target model
            pdb.set_trace()  # need to consider split the model used in tgt_label and RND_label?
            CF_agent_pred_RND_acc = 0.
            CF_agent_pred_RND_cnt = 0

        cnt_label = 0  # the number of states that has a_E label from oracle
        rl_1_losses = []
        rl_n_losses = []
        losses = []
        grad_norms = []
        sl_human_losses = []
        sl_tgt_losses = []
        sl_RND_losses = []
        
        data_itr = min(len(self.T_itr_ls), self.RL_loss.RL_recent_itr)
        len_recent_sample = sum(self.T_itr_ls[-data_itr:])
        max_len = len_recent_sample - self.RL_loss.n_step
        if (self.RL_loss.tgt_label.sl_weight > 0):
            if not (self.RL_loss.tgt_label.same_data_RL \
                    or self.RL_loss.tgt_label.same_data_RL_target):
                tgt_data_itr = min(len(self.T_itr_ls), self.RL_loss.tgt_label.data_recent_itr)
                len_tgt_recent_sample = sum(self.T_itr_ls[-tgt_data_itr:])
                max_tgt_len = len_tgt_recent_sample - self.RL_loss.n_step
            else:
                assert self.RL_loss.tgt_label.data_recent_itr is None
                assert self.RL_loss.tgt_label.bs_ratio is None
        if (self.RL_loss.RND_label.sl_weight > 0):
            if not self.RL_loss.RND_label.same_data_RL:
                RND_data_itr = min(len(self.T_itr_ls), self.RL_loss.RND_label.data_recent_itr)
                len_RND_recent_sample = sum(self.T_itr_ls[-RND_data_itr:])
                max_RND_len = len_RND_recent_sample - self.RL_loss.n_step
            else:
                assert self.RL_loss.RND_label.data_recent_itr is None
                assert self.RL_loss.RND_label.bs_ratio is None

        assert max_len < self.max_size_T  # TODO: need to consider data wrap in self.input_buffer
        if self.RL_loss.fix_data_index:
            total_batch_index = self.input_t - len_recent_sample + \
                                    np.random.permutation(max_len)
            total_batch_index %= self.max_size_T
        else:
            total_batch_index = self.len_inputs_T - len_recent_sample + \
                                    np.random.permutation(max_len)
        RL_train_bc = int(self.RL_loss.train_bs_ratio * self.train_batch_size)
        num_epochs = int(np.ceil(max_len / RL_train_bc))  # NOTE: use 'cile', so all labeled data will be used to train
        for epoch in range(num_epochs):  # NOTE: larger batch_size should use larger learning_rate, because #epochs will decrease
            self.RL_opt.zero_grad()
            
            if self.agent.model.noisy:  # For noisy net
                assert self.agent.model.head.network[-2].noise_override == False  # I remember we do not need to use noisy net
                self.agent.model.head.reset_noise()
            
            last_index = (epoch + 1) * RL_train_bc
            if last_index > max_len:
                last_index = max_len
            
            idxs = total_batch_index[epoch * RL_train_bc: last_index]
            RL_bs_epoch = idxs.shape[0]  # batch size in this epoch
            T_idxs = idxs
            B_idxs = np.zeros_like(idxs)  # because we assume input_buffer.B==1
            agent_inputs_buf = self.extract_observation(T_idxs, B_idxs)  # agent_inputs_buf: ndarray, uint8, [0, 255]

            ### RL loss
            agent_inputs_RL = torch.from_numpy(agent_inputs_buf).float().to(self.device)
            action = torch.from_numpy(self.input_buffer.action[T_idxs, B_idxs]).long().to(self.device)
            # for 1-step TD loss
            done = torch.from_numpy(self.input_buffer.done[T_idxs, B_idxs]).to(self.device)  # done.dtype = torch.bool
            if self.RL_loss.use_reward:
                GT_reward = torch.from_numpy(self.input_buffer.GT_reward[T_idxs, B_idxs]).float().to(self.device)
            else:
                GT_reward = torch.zeros_like(done).float().to(self.device)
            target_next_T_idxs = T_idxs + 1
            if self.RL_loss.fix_data_index:
                target_next_T_idxs %= self.max_size_T
            else:
                assert np.all(target_next_T_idxs < self.max_size_T)  # TODO: for now, suppose the data buffer are unlimited. in the future, could consider use the current agent to explore and use RL on more data?
            target_next_agent_inputs = self.extract_observation(target_next_T_idxs, B_idxs)
            target_next_agent_inputs = torch.from_numpy(target_next_agent_inputs).float().to(self.device)
            if self.RL_loss.n_step > 1:
                # for n-step TD loss
                if self.RL_loss.use_reward:
                    GT_return_ = torch.from_numpy(self.input_samples_return_[T_idxs, B_idxs]).float().to(self.device)
                else:
                    GT_return_ = torch.zeros_like(GT_reward).float().to(self.device)
                done_n = torch.from_numpy(self.input_samples_done_n[T_idxs, B_idxs]).to(self.device)
                target_n_T_idxs = T_idxs + self.RL_loss.n_step
                if self.RL_loss.fix_data_index:
                    target_n_T_idxs %= self.max_size_T
                else:
                    assert np.all(target_n_T_idxs < self.max_size_T)  # TODO: for now, suppose the data buffer are unlimited. in the future, could consider use the current agent to explore and use RL on more data?
                target_n_agent_inputs = self.extract_observation(target_n_T_idxs, B_idxs)
                target_n_agent_inputs = torch.from_numpy(target_n_agent_inputs).float().to(self.device)
            else:
                target_n_agent_inputs = GT_return_ = done_n = None

            RL_p_s = self.agent(agent_inputs_RL, train=True)  # q_s_t.shape: (B, action_dim, n_atoms)
            # !! NOTE: if agent.model.forward is using mul_(255.) instead of mul(255.), before self.agent(), agent_inputs_RL in range [0,255]; after: /=255.
            if self.distributional:
                RL_q_s = torch.tensordot(RL_p_s, self.distri_z, dims=1)  # (B, action_dim)
            else:
                RL_q_s = RL_p_s
            
            if self.distributional:
                rl_1_loss, KL_1 = self.dist_rl_loss(n_step=1, ps=RL_p_s, action=action,
                            done_n=done, return_=GT_reward, target_n_agent_inputs=target_next_agent_inputs)
                if self.RL_loss.n_step > 1:
                    rl_n_loss, KL_n = self.dist_rl_loss(n_step=self.RL_loss.n_step, ps=RL_p_s, action=action,
                            done_n=done_n, return_=GT_return_, target_n_agent_inputs=target_n_agent_inputs)
                else:
                    rl_n_loss = 0.
            else:
                raise NotImplementedError
            
            ### supervised loss
            ## sl: human labels
            if self.RL_loss.human_label.sl_weight > 0:
                human_bs_epoch = int(self.RL_loss.human_label.bs_ratio * RL_bs_epoch)
                label_idxs = np.random.choice(self.len_label,
                                    size=human_bs_epoch,
                                    replace=True).reshape(-1,)  # (bc_epoch,)
                label_obs_t = self.label_buffer.observation[label_idxs]  # (bc_epoch, C, H, W)
                # label_act_t = torch.from_numpy(self.label_buffer.action[label_idxs]).long().to(self.device)  # (bc_epoch,)
                label_oracle_act_t = torch.from_numpy(self.label_buffer.oracle_act[label_idxs]).long().to(self.device)  # (bc_epoch,)
                label_agent_act_t = torch.from_numpy(self.label_buffer.action[label_idxs]).long().to(self.device)  # (bc_epoch,)
                label_margine_t = torch.from_numpy(self.label_buffer.margine[label_idxs]).to(self.device)  # (bc_epoch,)
                label_obs_t = torch.from_numpy(label_obs_t).float().to(self.device)
                if self.distributional:
                    label_agent_p_s = self.agent(label_obs_t, train=True)  # q_s_t.shape: (B, action_dim, n_atoms)
                    # z = torch.linspace(self.agent.V_min, self.agent.V_max, self.agent.n_atoms)
                    label_agent_q_s = torch.tensordot(label_agent_p_s, self.distri_z, dims=1)  # (B, action_dim)
                else:
                    pdb.set_trace()  # check q_s_t.shape
                    label_agent_q_s = self.agent(label_obs_t, train=True)  # q_s_t.shape: (B, action_dim)
                # compute loss
                sl_loss_human = self.finetune_loss(q_s=label_agent_q_s,
                                            oracle_act=label_oracle_act_t,
                                            loss_margine=label_margine_t,
                                            agent_act=label_agent_act_t)
                
                # acc for human's label
                CF_agent_pred_human_cnt += human_bs_epoch
                agent_pred_human_act = torch.max(label_agent_q_s.data, dim=-1)[1]  # label_agent_q_s.shape: (bs_epoch, action_dim); agent_pred_human_act.shape: (bs_epoch, )
                CF_agent_pred_human_acc += (agent_pred_human_act == label_oracle_act_t).sum().item()  # label_oracle_act_t.shape: (bs_epoch,). count the number of samples that r_hat assign largest value for oracle_actions
            else:
                sl_loss_human = torch.tensor(0.)
            
            ## sl: tgt
            if self.RL_loss.tgt_label.sl_weight > 0:
                if self.RL_loss.tgt_label.same_data_RL:
                    tgt_bs_epoch = RL_bs_epoch
                    tgt_idxs = T_idxs
                    tgt_B_idxs = B_idxs
                    agent_inputs_tgt_buf = agent_inputs_buf
                elif self.RL_loss.tgt_label.same_data_RL_target:
                    tgt_bs_epoch = RL_bs_epoch
                    tgt_idxs_1 = T_idxs + 1
                    tgt_idxs_n = T_idxs + self.RL_loss.n_step
                    tgt_idxs = np.concatenate([tgt_idxs_1, tgt_idxs_n], axis=-1)
                    if self.RL_loss.fix_data_index:
                        tgt_idxs %= self.max_size_T
                    tgt_B_idxs = np.concatenate([B_idxs, B_idxs], axis=-1)
                    agent_inputs_tgt_buf = self.extract_observation(tgt_idxs, tgt_B_idxs)  # (bc_epoch, C, H, W)
                else:
                    tgt_bs_epoch = int(self.RL_loss.tgt_label.bs_ratio * RL_bs_epoch)
                    tgt_idxs = np.random.choice(max_tgt_len,
                                            size=tgt_bs_epoch,
                                            replace=True).reshape(-1,)  # (bc_epoch,)
                    tgt_B_idxs = np.zeros_like(tgt_idxs)  # because we assume input_buffer.B==1
                    agent_inputs_tgt_buf = self.extract_observation(tgt_idxs, tgt_B_idxs)  # (bc_epoch, C, H, W)
                agent_inputs_tgt = torch.from_numpy(agent_inputs_tgt_buf).float().to(self.device)

                with torch.no_grad():
                    if self.distributional:
                        if self.RL_loss.separate_tgt:
                            tgt_p_s = self.agent.separate_target(agent_inputs_tgt, train=True)  # q_s_t.shape: (B, action_dim, n_atoms)
                        else:
                            tgt_p_s = self.agent.target(agent_inputs_tgt, train=True)  # q_s_t.shape: (B, action_dim, n_atoms)
                        # z = torch.linspace(self.agent.V_min, self.agent.V_max, self.agent.n_atoms)
                        tgt_q_s = torch.tensordot(tgt_p_s, self.distri_z, dims=1)  # (B, action_dim)
                    else:
                        pdb.set_trace()  # check q_s_t.shape
                        if self.RL_loss.separate_tgt:
                            tgt_q_s = self.agent.separate_target(agent_inputs_tgt, train=True)  # q_s_t.shape: (B, action_dim)
                        else:
                            tgt_q_s = self.agent.target(agent_inputs_tgt, train=True)  # q_s_t.shape: (B, action_dim)
                    pseudo_act_tgt = torch.max(tgt_q_s.data, dim=-1)[1]  # tgt_max_q_a.shape: (agent_inputs_tgt.shape[0], )
                    assert pseudo_act_tgt.shape == (agent_inputs_tgt.shape[0], )

                if self.RL_loss.tgt_label.same_data_RL:
                    tgt_agent_q_s = RL_q_s
                else:
                    # case 1: self.RL_loss.tgt_label.same_data_RL_target
                    # case 2: (not self.RL_loss.tgt_label.same_data_RL) and (not self.RL_loss.tgt_label.same_data_RL_target)
                    # NOTE: agent_inputs_tgt.max <1. if with previous .mul_ inside agent.model.forward()
                    if self.distributional:
                        tgt_agent_p_s = self.agent(agent_inputs_tgt, train=True)  # q_s_t.shape: (B, action_dim, n_atoms)
                        tgt_agent_q_s = torch.tensordot(tgt_agent_p_s, self.distri_z, dims=1)  # (B, action_dim)
                    else:
                        tgt_agent_q_s = tgt_agent_p_s

                if self.RL_loss.tgt_label.type == 'tlEqu':
                    oracle_action = torch.from_numpy(self.input_buffer.oracle_act[tgt_idxs, tgt_B_idxs]).long().to(self.device)  # (B, )
                    have_label_flag = torch.from_numpy(self.have_label_flag_buffer[tgt_idxs, tgt_B_idxs]).to(self.device)  # (B,)
                    len_label = have_label_flag.sum().item()
                    # pseudo_act_tlEqu = pseudo_act_tgt.detach().clone()
                    if len_label > 0:
                        if self.RL_loss.tgt_label.RND_check.filter is not None:
                            num_wrong_doubt += (len_label - confident_flag[have_label_flag].sum().item())
                            confident_flag = torch.logical_or(confident_flag, have_label_flag)
                        cnt_label += len_label
                        oracle_act_label = oracle_action[have_label_flag]  # [have_label_flag.sum(),]
                        pseudo_act_tgt[have_label_flag] = oracle_act_label
                else:
                    assert self.RL_loss.tgt_label.type == 'tgt'
                
                margine_tensor = (torch.ones_like(pseudo_act_tgt) * self.loss_margine).to(self.device, dtype=torch.float32)
                if self.RL_loss.tgt_label.RND_check.filter is not None:
                    tgt_loss_weight = torch.ones_like(pseudo_act_tgt) * confident_flag +\
                            torch.ones_like(pseudo_act_tgt) * self.RL_loss.tgt_label.RND_check.diff_w * torch.logical_not(confident_flag)
                    tgt_loss_weight = tgt_loss_weight.to(self.device, dtype=margine_tensor.dtype)
                else:
                    tgt_loss_weight = None
                sl_loss_tgt =  self.finetune_loss(q_s=tgt_agent_q_s,  # q_s.shape (B, action_dim)
                                            oracle_act=pseudo_act_tgt,  # pseudo_act_tgt.shape: (B)
                                            loss_margine=margine_tensor,
                                            loss_weight=tgt_loss_weight)
                
                # acc for target modes's generated label
                agent_pred_tgt_act = torch.max(tgt_agent_q_s.data, dim=-1)[1]  # pred_act.shape: (agent_inputs_RL.shape[0], )
                CF_agent_pred_tgt_cnt += agent_pred_tgt_act.size(0)
                if self.RL_loss.tgt_label.same_data_RL_target:
                    assert pseudo_act_tgt.shape == agent_pred_tgt_act.shape == (agent_pred_tgt_act.shape[0],) == (tgt_bs_epoch*2,)
                else:
                    assert pseudo_act_tgt.shape == agent_pred_tgt_act.shape == (agent_pred_tgt_act.shape[0],) == (tgt_bs_epoch,)
                correct_agent_pred_tgt = (agent_pred_tgt_act == pseudo_act_tgt).sum().item()  # count the number of samples that r_hat assign largest value for oracle_actions
                CF_agent_pred_tgt_acc += correct_agent_pred_tgt
            else:
                sl_loss_tgt = torch.tensor(0.)
            
            sl_loss_RND = torch.tensor(0.)
            # NOTE: agent_inputs_RL, agent_inputs_tgt, agent_inputs_RND will not share the same memory
            
            loss = rl_1_loss * self.RL_loss.one_step_weight \
                   + rl_n_loss * self.RL_loss.n_step_weight \
                   + sl_loss_human * self.RL_loss.human_label.sl_weight \
                   + sl_loss_tgt * self.RL_loss.tgt_label.sl_weight \
                   + sl_loss_RND * self.RL_loss.RND_label.sl_weight
            loss.backward()

            if self.clip_grad_norm is not None:
                grad_norms.append(torch.nn.utils.clip_grad_norm_(
                        self.agent.model.parameters(), self.clip_grad_norm).item())  # default l2 norm
            self.RL_opt.step()

            rl_1_losses.append(rl_1_loss.item())
            rl_n_losses.append(rl_n_loss.item())
            sl_human_losses.append(sl_loss_human.item())
            sl_tgt_losses.append(sl_loss_tgt.item())
            sl_RND_losses.append(sl_loss_RND.item())
            losses.append(loss.item())

        rl_1_loss_avg = np.mean(np.array(rl_1_losses), axis=-1, keepdims=False)  # shape: (#ensemble,)
        rl_n_loss_avg = np.mean(np.array(rl_n_losses), axis=-1, keepdims=False)  # shape: (#ensemble,)
        sl_human_loss_avg = np.mean(np.array(sl_human_losses), axis=-1, keepdims=False)  # shape: (#ensemble,)
        sl_tgt_loss_avg = np.mean(np.array(sl_tgt_losses), axis=-1, keepdims=False)  # shape: (#ensemble,)
        sl_RND_loss_avg = np.mean(np.array(sl_RND_losses), axis=-1, keepdims=False)  # shape: (#ensemble,)
        loss_avg = np.mean(np.array(losses), axis=-1, keepdims=False)  # shape: (#ensemble,)
        grad_norms_avg = np.mean(np.array(grad_norms), axis=-1, keepdims=False)  # shape: (#ensemble,)
        # assert cnt_label == sum(self.label_itr_ls[-data_itr:])  # this assert is wrong, because some CF labels are given on the same states because segments can have overlaps
        
        if self.RL_loss.human_label.sl_weight > 0:
            CF_agent_pred_human_acc /= CF_agent_pred_human_cnt
        else:
            CF_agent_pred_human_acc = None
        
        tgt_RND_info = None
        if self.RL_loss.tgt_label.sl_weight > 0:
            CF_agent_pred_tgt_acc /= CF_agent_pred_tgt_cnt
            if self.RL_loss.tgt_label.RND_check.filter is not None:
                tgt_RND_confident_ratio = tgt_RND_confident_cnt / num_tgt_RND_state
                wrong_doubt_ratio = num_wrong_doubt / cnt_label
                tgt_RND_info = {
                    # 'tgt_RND_confident_cnt': tgt_RND_confident_cnt,
                    # 'num_wrong_doubt': num_wrong_doubt,
                    'tgt_RND_confident_ratio': tgt_RND_confident_ratio,
                    'wrong_doubt_ratio': wrong_doubt_ratio,
                }
        else:
            CF_agent_pred_tgt_acc = None
        
        if self.RL_loss.RND_label.sl_weight > 0:
            CF_agent_pred_RND_acc /= CF_agent_pred_RND_cnt
        else:
            CF_agent_pred_RND_acc = None
        
        return rl_1_loss_avg, rl_n_loss_avg, loss_avg, grad_norms_avg,\
                sl_human_loss_avg, sl_tgt_loss_avg, sl_RND_loss_avg,\
                CF_agent_pred_human_acc, CF_agent_pred_tgt_acc, CF_agent_pred_RND_acc,\
                tgt_RND_info

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
            human_img=samples.env.human_img,
        )

    def merge_T_itr_ls(self, merge_cnt):
        merged_T = sum(self.T_itr_ls[-merge_cnt:])
        del self.T_itr_ls[-merge_cnt:]  # if len(T_itr_ls)==merge_cnt, after 'del' -> T_itr_ls=[]
        self.T_itr_ls.append(merged_T)
