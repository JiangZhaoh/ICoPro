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

from collections import deque
from functools import partial
from new_utils.model_utils import weight_init
from new_utils.new_agent.encoder import cnn_mlp
from new_utils.tensor_utils import torchify_buffer, get_leading_dims, numpify_buffer
from new_utils.draw_utils import ax_plot_img, ax_plot_bar, ax_plot_heatmap
from new_utils.atari_env.wrapper import mask_img_score_func_, add_gaussian_noise
from new_utils.model_utils import count_parameters, norm_r_hat_s

from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.buffer import buffer_from_example

SamplesToCFRewardBuffer = namedarraytuple("SamplesToCFRewardBuffer",
                            ["observation", "done", "action",
                            "oracle_act", "oracle_act_prob", "oracle_q"])
# QueriesToRewardBuffer = namedarraytuple("QueriesToRewardBuffer",
#     ["observation", "action", "oracle_act"])

# TODO: check how to normalize the reward prediction to have a standard deviation?

class CorrectiveRlpytRewardModel:
    def __init__(self,
                 encoder_cfg,
                 B,
                #  episode_end_penalty,  # TODO: for corrective feedback, how to deal with episode end?
                 ensemble_size=3,
                 use_best_one=False,
                 reward_lr=3e-4,
                 query_batch=128,  # old name in PEBBLE is mb_size
                 is_traj_based=False,
                 query_traj=None,
                 train_batch_size=128,
                 size_segment=25,  # timesteps per segment
                 cf_per_seg=2,  # number of corrective feedback per segment
                 neighbor_size=1,  # for states around corrective actions, also assume using same optimal action
                 loss_margine=0.1,  # margine used to train reward model
                 max_size=100000,  # max timesteps of trajectories for query sampling
                 activation='tanh',
                 label_capacity=3000,  # "labeled" query buffer capacity, each query corresponds to a segment (old name in PEBBLE is capacity default 5e5)
                #  large_batch=1,  # some sampling methods need more samples for query selection
                 device='cuda',
                 init_type='orthogonal',
                 mask_img_score=True,
                 env_name=None,
                 loss_name='MargineLoss',
                 clip_grad_norm=None,
                 exp_loss_beta=1.,
                 loss_square=False,
                 loss_square_coef=1.,
                 gaussian_noise=None,
                 oracle_type='oe',  # {'oe': 'oracle_entropy', 'oq': 'oracle_q_value'}
                 softmax_tau=1.0,
                 ckpt_path=None,
                 reset=None,
                 normalize_reward_cfg=None,
                 ):
        self.device = device

        self.encoder_cfg = copy.deepcopy(encoder_cfg)
        # self.episode_end_penalty = episode_end_penalty
        # self.encoder_cfg.cnn_cfg['in_channels'] += encoder_cfg.action_dim  # frame_stack * img_channel + action_dim
        self.obs_shape = encoder_cfg.obs_shape  # (frame_stack, H, W)
        self.action_dim = encoder_cfg.action_dim  # number of available discrete actions
        self.max_entropy = -self.action_dim * ((1.0/self.action_dim) * np.log((1.0/self.action_dim)))
        print(f"[Reward Model] max_entropy {self.max_entropy}.")
        # self.model_input_shape = self.obs_shape[:]  # (frame_stack, H, W)
        # self.model_input_shape[0] += self.action_dim  # input shape for reward models: [frame_stack+action_dim, H, W]
        self.B = int(B)
        assert self.B == 1
        self.de = ensemble_size
        self.lr = reward_lr
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        # self.model = None
        assert max_size % self.B == 0
        self.max_size_T = int(max_size // self.B)
        self.max_size = max_size
        self.activation = activation
        self.loss_square = loss_square
        self.loss_square_coef = loss_square_coef
        self.size_segment = size_segment

        self.neighbor_size = neighbor_size
        assert self.neighbor_size == 1  # For larger value, need to debug index in functions like get_queries
        self.cf_per_seg = cf_per_seg
        self.loss_margine = loss_margine
        self.clip_grad_norm = clip_grad_norm

        self.label_capacity = int(label_capacity)  # "labeled" query buffer capacity
        # self.label_count = 0

        # Move buffer part to self.initialize()
        # self.buffer_obs = np.empty((self.label_capacity, *self.obs_shape), dtype=np.float32)
        # self.buffer_act = np.empty((self.label_capacity, 1), dtype=np.int32)
        # self.buffer_oracle_act = np.empty((self.label_capacity, 1), dtype=np.int32)
        # # self.buffer_oracle_act_prob = np.empty((self.label_capacity, self.action_dim), dtype=np.float32)
        # self.buffer_index = 0
        # self.buffer_full = False  # query buffer full

        self.ckpt_path = ckpt_path
        self.reset = reset
        self.init_type = init_type
        self.construct_ensemble(init_type=init_type)

        self.use_best_one = use_best_one
        if self.use_best_one:
            self.best_r_id = -1
        # self.inputs_obs = None  # all available trajectories
        # self.inputs_act = None
        # self.targets_oracle_act = None
        # self.targets_oracle_act_prob = None
        # self.raw_actions = []
        # self.img_inputs = []
        self.query_batch = query_batch  # reward batch size
        self.origin_query_batch = query_batch  # reward batch size may change according to the current training step
        self.origin_query_traj = query_traj
        self.total_traj = 0
        self.train_batch_size = train_batch_size
        self.running_means = []
        self.running_stds = []
        self.best_seg = []
        self.best_label = []
        self.best_action = []

        self.avg_train_step_return = deque([], maxlen=100000)  # as a reference to decide margin

        self.mask_img_score = mask_img_score
        self.env_name = env_name

        if loss_name == 'MargineLoss':
            self.reward_loss = self.reward_margine_loss
            if self.loss_square:
                raise NotImplementedError
        elif loss_name == 'MargineLossMin0':
            self.reward_loss = self.reward_margine_loss_min0
            if self.loss_square:
                raise NotImplementedError
        elif loss_name == 'MargineLossMin0Fix':
            self.reward_loss = self.reward_margine_loss_min0_fix
            if self.loss_square:
                raise NotImplementedError
        elif loss_name == 'Exp':
            self.reward_loss = self.reward_exp_loss
            self.exp_loss_beta = exp_loss_beta
        else:
            raise NotImplementedError
        
        self.use_gaussian_noise = gaussian_noise.flag
        if self.use_gaussian_noise:
            self.gaussian_amplitude = gaussian_noise.amplitude

        self.oracle_type = oracle_type
        self.softmax_tau = softmax_tau
        self.normalize_reward_cfg = normalize_reward_cfg
        self.is_traj_based = is_traj_based
        if is_traj_based:
            assert self.max_size >= 108000  # mask sure enough space to save a whole traj

    def initialize_buffer(self, agent, oracle, env, check_label_path=None):
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
        sample_examples["oracle_act"] = oracle_a
        sample_examples["oracle_act_prob"] = F.softmax(oracle_info.value, dim=-1)  # (action_dim,)
        sample_examples["oracle_q"] = oracle_info.value  # (action_dim,)

        if self.is_traj_based:
            field_names = [f for f in sample_examples.keys()]
        else:
            field_names = [f for f in sample_examples.keys() if f != "observation"]
        global RewardBufferSamples
        RewardBufferSamples = namedarraytuple("RewardBufferSamples", field_names)
        if self.is_traj_based:
            reward_buffer_example = RewardBufferSamples(*(v for \
                                    k, v in sample_examples.items()))
            
            self.input_buffer = buffer_from_example(reward_buffer_example,
                                                    (self.max_size,))
        else:
            reward_buffer_example = RewardBufferSamples(*(v for \
                                        k, v in sample_examples.items() \
                                        if k != "observation"))
            self.input_buffer = buffer_from_example(reward_buffer_example,
                                                    (self.max_size_T, self.B))
            # self.input_buffer.action&oracle_act.shape: (self.max_size_T, self.B), int64
            # self.input_buffer.oracle_act_prob.shape: (self.max_size_T, self.B, act_dim), float32
            self.n_frames = n_frames = get_leading_dims(o[:], n_dim=1)[0]
            print(f"[Reward Buffer-Inputs] Frame-based buffer using {n_frames}-frame sequences.")
            self.input_frames = buffer_from_example(o[0],  # avoid saving duplicated frames
                                    (self.max_size_T + n_frames - 1, self.B))
            # self.input_frames.shape: (self.max_size_T+n_frames-1, self.B, H, W), uint8
            self.input_new_frames = self.input_frames[n_frames - 1:]  # [self.max_size_T,B,H,W]
        self.input_t = 0
        self._input_buffer_full = False
        
        label_examples["observation"] = o[:]
        label_examples["action"] = a
        label_examples["oracle_act"] = oracle_a
        
        global RewardLabelBufferSamples
        RewardLabelBufferSamples = namedarraytuple("RewardLabelBufferSamples",
                                                   label_examples.keys())
        reward_label_buffer_example = RewardLabelBufferSamples(*(v for \
                                            k, v in label_examples.items()))
        
        self.label_buffer = buffer_from_example(reward_label_buffer_example,
                                                (self.label_capacity,))
        # self.label_buffer.observation.shape: (self.label_capacity, C, H, W), uint8
        # self.label_buffer.action&oracle_act.shape: (self.label_capacity,), int64
        self.label_t = 0
        self._label_buffer_full = False
        self.total_label_seg = 0  # the number of segments that have been labeled (queried)
        self.check_label_path = check_label_path
        if self.check_label_path:
            os.makedirs(self.check_label_path, exist_ok=True)
        self.env_action_meaning = env.unwrapped.get_action_meanings()

        # if self.mask_img_score:  # because ENV_SCORE_AREA is for (84*84) (moved to mask_img_score())
        #     assert o.shape[-1] == o.shape[-2] == 84

    def KCenterGreedy(self, obs, full_obs, num_new_sample):
        raise NotImplementedError
        selected_index = []
        current_index = list(range(obs.shape[0]))
        new_obs = obs
        new_full_obs = full_obs
        start_time = time.time()
        for count in range(num_new_sample):
            dist = self.compute_smallest_dist(new_obs, new_full_obs)
            max_index = torch.argmax(dist)
            max_index = max_index.item()
            
            if count == 0:
                selected_index.append(max_index)
            else:
                selected_index.append(current_index[max_index])
            current_index = current_index[0:max_index] + current_index[max_index+1:]
            
            new_obs = obs[current_index]
            new_full_obs = np.concatenate([
                full_obs, 
                obs[selected_index]], 
                axis=0)
        return selected_index

    def compute_smallest_dist(self, obs, full_obs):
        raise NotImplementedError
        obs = torch.from_numpy(obs).float()
        full_obs = torch.from_numpy(full_obs).float()
        batch_size = 100
        with torch.no_grad():
            total_dists = []
            for full_idx in range(len(obs) // batch_size + 1):
                full_start = full_idx * batch_size
                if full_start < len(obs):
                    full_end = (full_idx + 1) * batch_size
                    dists = []
                    for idx in range(len(full_obs) // batch_size + 1):
                        start = idx * batch_size
                        if start < len(full_obs):
                            end = (idx + 1) * batch_size
                            dist = torch.norm(
                                obs[full_start:full_end, None, :].to(self.device) - full_obs[None, start:end, :].to(self.device),
                                dim=-1, p=2
                            )
                            dists.append(dist)
                    dists = torch.cat(dists, dim=1)
                    small_dists = torch.torch.min(dists, dim=1).values
                    total_dists.append(small_dists)
                    
            total_dists = torch.cat(total_dists)
        return total_dists.unsqueeze(1)
    
    def change_batch(self, new_frac):
        self.query_batch = max(int(self.origin_query_batch * new_frac), 1)
    
    def set_batch(self, new_batch):
        self.query_batch = int(new_batch)

    def change_traj(self, new_frac):
        self.query_traj = max(int(self.origin_query_traj * new_frac), 1)
    
    def set_traj(self, new_traj):
        self.query_traj = int(new_traj)
    
    def construct_ensemble(self, init_type):
        if self.activation == 'tanh':
            output_mod = nn.Tanh()
            # self.min_reward = 0.0
        elif self.activation == 'sig':
            output_mod = nn.Sigmoid()
            # self.min_reward = -1.0
        elif self.activation == 'relu':
            output_mod = nn.ReLU()
            # self.min_reward = 0.0
        # elif self.activation == 'minmax':
        #     output_mod = None
        elif (self.activation is None) or (self.activation == 'null'):
            output_mod = None
        else:
            raise NotImplementedError
        
        if self.ckpt_path is not None:
            ckpt_name_ls_t = os.listdir(self.ckpt_path)
            ckpt_name_ls = ckpt_name_ls_t[:]
            for id_e, name in enumerate(ckpt_name_ls_t):
                name_id = name.split('_')[-1].split('.')[0]
                ckpt_name_ls[int(name_id)] = ckpt_name_ls_t[id_e]
        for i in range(self.de):
            model = cnn_mlp(obs_shape=self.obs_shape,
                            output_dim=self.action_dim,  # r_hat(s) outputs r_hat(s,a) for |A| actions
                            cnn_cfg=self.encoder_cfg.cnn_cfg,
                            mlp_cfg=self.encoder_cfg.mlp_cfg,
                            output_mod=output_mod,
                            nonlinearity=self.encoder_cfg.nonlinearity,
                            ).float().to(self.device)
            # For vector states:
            #  model = nn.Sequential(*gen_net(in_size=self.ds+self.da, 
            #                                out_size=1, H=256, n_layers=3, 
            #                                activation=self.activation)).float().to(self.device)
            if self.ckpt_path is None:
                for submodel in model:
                    submodel.apply(partial(weight_init, init_type=init_type))
            else:
                msg = model.load_state_dict(
                    torch.load('%s/%s' % (self.ckpt_path, ckpt_name_ls[i]),
                               map_location=self.device)
                )
                print(f"[REWARD MODEL]checkpoint load information for reward model{i}")
                print(msg)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())
        print(f'[Reward Model]: {model} on device: {self.device}')
        print(f"[Reward Model]: Initialized model with {count_parameters(model)} parameters")
        # TODO: add bn & (per-channel) dropout & l2 regularization
        self.opt = torch.optim.Adam(self.paramlst, lr = self.lr)
        if self.reset.flag:
            self.get_reset_list()

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
        assert not self.is_traj_based
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
        elif self.input_t <= t and fm1 > 0:  # Wrapped: copy any duplicate frames.
            self.input_frames[:fm1] = self.input_frames[-fm1:]
        # return T, idxs

    def add_demo_data(demo_agent):
        raise NotImplementedError

    def add_data_batch(self, obses, rewards):
        raise NotImplementedError
        num_env = obses.shape[0]
        for index in range(num_env):
            self.inputs.append(obses[index])
            self.targets.append(rewards[index])
        
    def get_rank_probability(self, x_1, x_2):
        raise NotImplementedError
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_member(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        
        return np.mean(probs, axis=0), np.std(probs, axis=0)
    
    def get_entropy(self, x_1, x_2):
        raise NotImplementedError
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_entropy(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.mean(probs, axis=0), np.std(probs, axis=0)

    def p_hat_member(self, x_1, x_2, member):
        raise NotImplementedError
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        
        # taking 0 index for probability x_1 > x_2
        return F.softmax(r_hat, dim=-1)[:,0]
    
    def p_hat_entropy(self, x_1, x_2, member):
        raise NotImplementedError
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        
        # H(p_hat) = - sum_x{p_hat(x) * log(p_hat(x))}, where p_hat(x)=softmax(x)
        ent = F.softmax(r_hat, dim=-1) * F.log_softmax(r_hat, dim=-1)
        ent = ent.sum(axis=-1).abs()
        return ent  # shape: (B,)

    def r_hat_s_member(self, obs, member, train=False):  # NOTE: for gradient back propagate
        # NOTE: before r_hat_s_member, img score in obs should have been masked
        # the network parameterizes r hat in eqn 1 from the paper
        assert obs.ndim == 4  # (B, C*frame_stack, H, W), torch.uint8
        
        if self.use_gaussian_noise and train:
            obs = add_gaussian_noise(img=obs, amplitude=self.gaussian_amplitude)
        
        if type(obs) == np.ndarray:
            obs = torch.from_numpy(obs).float().to(self.device)
        else:
            obs = obs.float().to(self.device)
        obs /= 255.0

        if self.mask_img_score:
            mask_img_score_func_(env_name=self.env_name, obs=obs)  # modified in place
        
        r_hat_s = self.ensemble[member](obs)  # (B, |A|) for atari
        assert r_hat_s.shape == (obs.shape[0], self.action_dim)
        r_hat_s = norm_r_hat_s(r_hat_s,
                     self.normalize_reward_cfg.train if train \
                        else self.normalize_reward_cfg.eval)
        return r_hat_s  # shape: (B, self.action_dim)

    def r_hat_sa(self, obs, act):  # NOTE: for data collection & evaluation & replay buffer relabeling, no grad
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        assert not self.ensemble[0].training
        assert obs.ndim == 4  # (B, C*frame_stack, H, W)
        assert act.ndim == 1  # (B,)
        # assert np.isscalar(x[1])
        # obs, act = np.array(x[0], dtype=float)[None, ...], np.array([x[1]])[None, ...]
        # obs /= 255.0
        r_hats_sa = []
        # moved into r_hat_s_member()
        # obs = obs.float().to(self.device)
        act = act.long().to(self.device)
        if self.use_best_one and self.best_r_id != -1:  # best_r_id == -1 means the reward model have not been trained
            member = self.best_r_id
            r_hat_s = self.r_hat_s_member(obs, member=member)  # r_hat_s.shape: (batch_size, action_dim)
            r_hat_sa = torch.gather(r_hat_s, dim=-1, index=act[...,None])  # r_hat_sa.shape: (batch_size. 1)
            assert r_hat_sa.shape == (r_hat_s.shape[0], 1) == (obs.shape[0], 1)
            return r_hat_sa.reshape(-1).detach().cpu().numpy()  # shape: (batch_size,)
        else:
            for member in range(self.de):
                r_hat_s = self.r_hat_s_member(obs, member=member)  # r_hat_s.shape: (batch_size, action_dim)
                r_hat_sa = torch.gather(r_hat_s, dim=-1, index=act[...,None])  # r_hat_sa.shape: (batch_size. 1)
                assert r_hat_sa.shape == (r_hat_s.shape[0], 1) == (obs.shape[0], 1)
                r_hats_sa.append(r_hat_sa.reshape(-1).detach().cpu().numpy())  # shape: (batch_size,)
            r_hats_sa = np.array(r_hats_sa)  # shape: (#ensemble, batch_size)
            return np.mean(r_hats_sa, axis=0, keepdims=False)  # shape: (batch_size,)
    
    # def r_hat_sa_batch(self, obs, act):  # NOTE: for , no grad
    #     # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
    #     # but I don't understand how the normalization should be happening right now :(
    #     # obs /= 255.0  # To be processed in r_hat_s_member
    #     r_hats = []
    #     # obs.shape: (replay_buffer.batch_this_itr, *obs_shape)
    #     # act.shape: (replay_buffer.batch_this_itr, )
    #     for member in range(self.de):
    #         r_hat_s = self.r_hat_s_member(obs, member=member)
    #         r_hat_sa = torch.gather(r_hat_s, dim=-1, index=act[...,None]).detach().cpu().numpy()  # (batch_size, action_dim)
    #         pdb.set_trace()  # TODO: check 
    #         r_hats.append(r_hat_sa)
    #     r_hats = np.array(r_hats)

    #     return np.mean(r_hats, axis=0)
    
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
    
    def get_train_acc(self):
        raise NotImplementedError
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.label_capacity if self.buffer_full else self.buffer_index
        total_batch_index = np.random.permutation(max_len)
        batch_size = 256
        num_epochs = int(np.ceil(max_len/batch_size))
        
        total = 0
        for epoch in range(num_epochs):
            last_index = (epoch+1)*batch_size
            if (epoch+1)*batch_size > max_len:
                last_index = max_len
                
            sa_t_1 = self.buffer_seg1[epoch*batch_size:last_index]
            sa_t_2 = self.buffer_seg2[epoch*batch_size:last_index]
            labels = self.buffer_label[epoch*batch_size:last_index]
            labels = torch.from_numpy(labels.flatten()).long().to(self.device)
            total += labels.size(0)
            for member in range(self.de):
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)                
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
        ensemble_acc = ensemble_acc / total
        return np.mean(ensemble_acc)
    
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

    def get_queries(self, query_batch):
        # check shape. TODO: consider if done=True should be considered;
        # get train traj
        max_index_T = self.len_inputs_T - self.size_segment - (self.neighbor_size // 2) * 2
        # Batch = query_batch
        assert (max_index_T * self.B) > query_batch
        batch_index = np.random.choice(max_index_T*self.B, size=query_batch, replace=True).reshape(-1, 1)  # (query_batch, 1)
        batch_index_T, batch_index_B = np.divmod(batch_index, self.B)  # (x // y, x % y), batch_index_B & batch_index_T.shape: (query_batch, 1)
        batch_index_T += (self.neighbor_size // 2) # batch_index_T.shape: (query_batch, 1)
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

        # obs_t = np.take(self.inputs_obs, take_index, axis=0) # (query_batch, size_segment, obs_shape)
        # act_t = np.take(self.inputs_act, take_index, axis=0) # (query_batch, size_segment, 1)
        # oracle_act_t = np.take(self.targets_oracle_act, take_index, axis=0) # (query_batch, size_segment, 1)
        # oracle_act_prob_t = np.take(self.targets_oracle_act_prob, take_index, axis=0) # (query_batch, size_segment, 1)

        return obs_t, act_t, oracle_act_t, oracle_act_prob_t, oracle_q_t

    def put_queries(self, obs_t, act_t, oracle_act_t, early_advertising=False):
        # obs_t.shape: (query_batch * cf_per_seg * neighbor_size, *obs_shape)
        # act_t & oracle_act_t.shape: (query_batch * cf_per_seg * neighbor_size,)
        
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

            remain = total_sample - (maximum_index)
            if remain > 0:  # if next_index exceed capacity, extra new queries will be added to the beginning of query buffer
                np.copyto(self.label_buffer.observation[0:remain], obs_t[maximum_index:])
                np.copyto(self.label_buffer.action[0:remain], act_t[maximum_index:])
                np.copyto(self.label_buffer.oracle_act[0:remain], oracle_act_t[maximum_index:])

            self.label_t = remain
        else:
            np.copyto(self.label_buffer.observation[self.label_t:next_index], obs_t)
            np.copyto(self.label_buffer.action[self.label_t:next_index], act_t)
            np.copyto(self.label_buffer.oracle_act[self.label_t:next_index], oracle_act_t)
            self.label_t = next_index
    
    def get_label(self, obs_t, act_t, oracle_act_t, oracle_act_prob_t, oracle_q_t):
        # obs_t.shape: (query_batch, size_segment, *obs_shape)
        # act_t & oracle_act_t.shape: (query_batch, size_segment)
        # oracle_act_prob_t & oracle_q_t.shape: (query_batch, size_segment, action_ndim)
        # NOTE: !!!! oracle_act_prob_t's minimal value should be larger than EPS!!! otherwise np.log will has errors!!!!
        if self.oracle_type == 'oe':
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
            # assert np.all(q_diff >= 0)  # to make sure the correctness of q and oracle_act. TODO: maybe remove this part after debugging?
            target_oracle_index = np.argsort(q_diff, axis=-1)[:, -self.cf_per_seg:]  # ascending order, shape (query_batch, cf_per_seg)
        else:
            # TODO: maybe also consider r_hat, or even training agent's q_value?
            raise NotImplementedError
        if (self.neighbor_size // 2) > 0:
            target_oracle_index_neighbor = np.tile(target_oracle_index,
                                                   reps=(1, self.neighbor_size))  # shape: (query_batch, cf_per_seg*neighbor_size)
            for delt in range(self.neighbor_size//2):
                target_oracle_index_neighbor[:, delt*2*self.cf_per_seg: (delt*2+1)*self.cf_per_seg] -= (delt+1)
                target_oracle_index_neighbor[:, (delt*2+1)*self.cf_per_seg: (delt*2+2)*self.cf_per_seg] += (delt+1)
            # TODO: if so, will have repeated index, should consider more about this case
            assert np.all(target_oracle_index_neighbor >= 0)  # having negative index indicates code bug
        else:
            target_oracle_index_neighbor = target_oracle_index[:]  # shape: (query_batch, cf_per_seg * neighbor_size)
        
        if self.check_label_path:
            # check selected (s,a) pair, check if low entropy = confidence about action selection
            query_cnt = obs_t.shape[0]
            for id_seg in range(query_cnt):
                widths = [10, 10, 10, 10, 10, 1, 1]  # 4 frame stack + 1 height bar + 2 entropy color
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
                    
                    ax_prob_bar = fig.add_subplot(spec[id_in_seg, -3])
                    if self.oracle_type == 'oe':
                        bar_height = oracle_act_prob_t[id_seg][id_in_seg]
                        bars = ax_plot_bar(ax=ax_prob_bar,
                                        xlabel=self.env_action_meaning,
                                        height=bar_height)
                    elif self.oracle_type == 'oq':
                        bar_height = oracle_q_t[id_seg][id_in_seg]  # can have negative values
                        value_min = bar_height.min()
                        bar_height_min0 = bar_height - value_min
                        bars = ax_plot_bar(ax=ax_prob_bar,
                                        xlabel=self.env_action_meaning,
                                        height=bar_height_min0,
                                        yvals=bar_height,
                                        top=bar_height_min0.max()+0.,
                                        )
                        
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
                
                # rects = [
                #     patches.Rectangle(xy=(0, 5 * y), width=5, height=5,
                #                       linewidth=10, edgecolor='r',
                #                       facecolor='none') \
                #     for y in target_oracle_index[id_seg]
                # ]
                # for rect in rects:
                #     ax_entropy.add_patch(rect)

                fig.tight_layout()
                fig.savefig(fname=os.path.join(self.check_label_path,
                                              f'{self.total_label_seg + id_seg}.png'),
                            bbox_inches='tight', pad_inches=0)
        
        # obs_t.shape: (query_batch * cf_per_seg * neighbor_size, *obs_shape)
        obs_t =  np.take_along_axis(obs_t,
                                    target_oracle_index_neighbor[...,None,None,None],
                                    axis=1).\
                    reshape(self.query_batch * self.cf_per_seg * self.neighbor_size,
                            *self.obs_shape)
        # act_t.shape: (query_batch * cf_per_seg * neighbor_size,)
        act_t = np.take_along_axis(act_t,
                                   target_oracle_index_neighbor,
                                   axis=1).\
                    reshape(self.query_batch * self.cf_per_seg * self.neighbor_size,)
        # oracle_act_t.shape: (query_batch * cf_per_seg * neighbor_size,)
        oracle_act_t = np.tile(np.take_along_axis(oracle_act_t,
                                                  target_oracle_index,
                                                  axis=1)\
                                ,reps=(1, self.neighbor_size)).\
                    reshape(self.query_batch * self.cf_per_seg * self.neighbor_size,)
        return obs_t, act_t, oracle_act_t
    
    def kcenter_sampling(self):
        raise NotImplementedError
        
        # get queries
        num_init = self.query_batch*self.large_batch
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            query_batch=num_init)
        
        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:,:,:self.ds]
        temp_sa_t_2 = sa_t_2[:,:,:self.ds]
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init, -1),  
                                  temp_sa_t_2.reshape(num_init, -1)], axis=1)
        
        max_len = self.label_capacity if self.buffer_full else self.buffer_index
        
        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),  
                                 tot_sa_2.reshape(max_len, -1)], axis=1)
        
        selected_index = self.KCenterGreedy(temp_sa, tot_sa, self.query_batch)

        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def kcenter_disagree_sampling(self):
        raise NotImplementedError
        
        num_init = self.query_batch*self.large_batch
        num_init_half = int(num_init*0.5)
        
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            query_batch=num_init)
        
        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:num_init_half]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        
        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:,:,:self.ds]
        temp_sa_t_2 = sa_t_2[:,:,:self.ds]
        
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init_half, -1),  
                                  temp_sa_t_2.reshape(num_init_half, -1)], axis=1)
        
        max_len = self.label_capacity if self.buffer_full else self.buffer_index
        
        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),  
                                 tot_sa_2.reshape(max_len, -1)], axis=1)
        
        selected_index = self.KCenterGreedy(temp_sa, tot_sa, self.query_batch)
        
        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def kcenter_entropy_sampling(self):
        raise NotImplementedError
        
        num_init = self.query_batch*self.large_batch
        num_init_half = int(num_init*0.5)
        
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            query_batch=num_init)
        
        
        # get final queries based on uncertainty
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        top_k_index = (-entropy).argsort()[:num_init_half]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        
        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:,:,:self.ds]
        temp_sa_t_2 = sa_t_2[:,:,:self.ds]
        
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init_half, -1),  
                                  temp_sa_t_2.reshape(num_init_half, -1)], axis=1)
        
        max_len = self.label_capacity if self.buffer_full else self.buffer_index
        
        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),  
                                 tot_sa_2.reshape(max_len, -1)], axis=1)
        
        selected_index = self.KCenterGreedy(temp_sa, tot_sa, self.query_batch)
        
        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def early_advertising_states_in_buffer(self):
        # extract all data from data buffer as (s, a_E)
        max_index_T = self.len_inputs_T
        take_index_T = np.arange(max_index_T).reshape(-1).\
                        repeat(repeats=self.B, axis=-1)  # (T,)->(T * B,): [0...T, 0...T, ...]
        take_index_B = np.arange(self.B).reshape(-1, 1).\
                        repeat(repeats=max_index_T, axis=-1).reshape(-1)  # (B,1)->(B,T)->(T*B,): [0...0, 1...1, ...]
        obs_t = self.extract_observation(take_index_T, take_index_B)\
                .reshape(max_index_T*self.B, *self.obs_shape)  # (max_index_T*B, *obs_shape)
        # NOTE: for np, this index will be selected by zip(take_index_T, take_index_B)
        act_t = self.input_buffer.action[take_index_T, take_index_B]\
                        .reshape(max_index_T*self.B,)  # (max_index_T*B,)
        oracle_act_t = self.input_buffer.oracle_act[take_index_T, take_index_B]\
                        .reshape(max_index_T*self.B,)  # (max_index_T*B,)
        self.put_queries(obs_t, act_t, oracle_act_t, early_advertising=True)
        
        return obs_t.shape[0]

    def traj_based_sampling(self, agent, oracle_agent, env, TrajInfoCls,
                            fb_ratio, fb_per_traj):
        # Sample trajectries from the agent in evaluation mode; give CF for states in this traj; append labels in label buffer
        # oracle_agent has already in eval mode
        # agent.reset()
        eps_this = agent.eval_mode(itr=1)
        print(f"[Traj_based_sampling] eps: {eps_this}")
        
        self.completed_infos = list()
        num_fb_per_traj_ls = []
        for id_t in range(self.query_traj):
            truncated = False

            observation_py = env.reset()[0][:]
            traj_infos = TrajInfoCls(total_lives=env.ale.lives())

            observation = buffer_from_example(observation_py, 1)  # (1, C*frame_stack, H, W)
            observation[0] = observation_py  # (1, C*frame_stack, H, W)
            obs_pyt = torchify_buffer((observation))  # shape: (1, C*frame_stack, H, W)
            ###### Collect trajectory
            tt = 0
            while not (truncated or env.need_reset):
                self.input_buffer.observation[tt] = observation[0] # slice 
                # NOTE: "env_buf.observation[t]" will not be modified after "observation" being updated
                # Agent inputs and outputs are torch tensors.
                # act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
                act_pyt, agent_info = agent.step(obs_pyt)  # NOTE: no_grad for agent.step
                oracle_act_pyt, oracle_act_info = oracle_agent.step(obs_pyt)
                # oracle_act_pyt.shape: (1,), oracle_act_info.p.shape: (1, action_dim)

                action = numpify_buffer(act_pyt)  # shape (1,)
                oracle_act = numpify_buffer(oracle_act_pyt)
                self.input_buffer.action[tt] = action
                self.input_buffer.oracle_act[tt] = oracle_act

                oracle_act_prob_pyt = F.softmax(
                    oracle_act_info.value / self.softmax_tau,
                    dim=-1)
                oracle_act_prob = numpify_buffer(oracle_act_prob_pyt)
                self.input_buffer.oracle_act_prob[tt] = oracle_act_prob

                oracle_q = numpify_buffer(oracle_act_info.value)
                self.input_buffer.oracle_q[tt] = oracle_q
                
                # Environment inputs and outputs are numpy arrays.
                o, r, terminated, truncated, env_info = env.step(action[0])
                traj_infos.step(reward=r,
                                r_hat=None,
                                raw_reward=env_info["raw_reward"],
                                terminated=terminated,
                                truncated=truncated,
                                need_reset=env.need_reset,
                                lives=env.ale.lives())
                if truncated or env.need_reset:
                    if truncated:
                        assert env.need_reset
                    self.completed_infos.append(traj_infos.terminate())
                # if terminated:
                #     self.agent.reset_one(idx=b)
                observation[0] = o[:]
                self.input_buffer.done[tt] = terminated
                tt += 1
            
            ###### Select timestep index that oracle give feedback
            if self.oracle_type == 'oe':
                oracle_act_prob_t = self.input_buffer.oracle_act_prob[:tt]
                assert oracle_act_prob_t.ndim == 2  # (tt, action_dim)
                pdb.set_trace()  # check shape
                assert np.min(oracle_act_prob_t) > 1e-11  # because use log to calculate entropy
                oracle_act_entropy = (-oracle_act_prob_t * np.log(oracle_act_prob_t))\
                                        .sum(axis=-1, keepdims=False)  # shape(tt, action_dim)
                # NOTE: (This argument may not true!) smaller entropy -> more confident about one state
                target_index_order = np.argsort(-oracle_act_entropy, axis=-1) # ascending order (so descending orger about entropy), shape (tt,)
                assert oracle_act_entropy.shape == target_index_order.shape == (tt,)
                max_cf_num = tt
            elif self.oracle_type == 'oq':
                oracle_q_t = self.input_buffer.oracle_q[:tt]  # (tt, action_dim)
                oracle_act_t = self.input_buffer.oracle_act[:tt]  # (tt,)
                act_t = self.input_buffer.action[:tt]  # (tt,)
                index_t = np.arange(oracle_act_t.shape[0])
                oracle_act_oracle_q = oracle_q_t[index_t, oracle_act_t]  # oracle's q for its own action, shape (tt,)
                act_oracle_q = oracle_q_t[index_t, act_t]  # oracle's q for current selected action, shape (tt,)
                q_diff = oracle_act_oracle_q - act_oracle_q  # q_diff are positive values, shape (tt,)
                target_index_order = np.argsort(q_diff, axis=-1)  # ascending order, shape (tt,)
                max_cf_num = np.sum(q_diff > 0)
                # NOTE: select actions with larger q_diff
            else:
                # TODO: maybe also consider r_hat, or even training agent's q_value?
                raise NotImplementedError
            
            if fb_ratio is not None:
                cf_num = int(max_cf_num * fb_ratio)
            elif fb_per_traj is not None:
                cf_num = min(tt, fb_per_traj)
            else:
                raise NotImplementedError
            
            cf_num = min(cf_num, max_cf_num)
            num_fb_per_traj_ls.append(cf_num)
            target_index = target_index_order[-cf_num:]

            ###### put label into buffer
            # obs_t.shape: (tt, *obs_shape)
            obs_t =  self.input_buffer.observation[:tt]
            # act_t.shape: (tt,)
            act_t = self.input_buffer.action[:tt]
            label_act_t = copy.deepcopy(act_t[...])  # (tt,)
            label_act_t[target_index] = self.input_buffer.oracle_act[target_index]
            self.put_queries(obs_t=obs_t, act_t=act_t, oracle_act_t=label_act_t)
            
        self.total_traj += self.query_traj
        return np.sum(num_fb_per_traj_ls)
    
    def uniform_sampling(self):
        # get queries
        obs_t, act_t, oracle_act_t, oracle_act_prob_t, oracle_q =  self.get_queries(
            query_batch=self.query_batch)
        # obs_t.shape: (query_batch, size_segment, *obs_shape)
        
        # get labels
        obs_t, act_t, oracle_act_t = self.get_label(  # filter queries and 
            obs_t, act_t, oracle_act_t, oracle_act_prob_t, oracle_q)
        # obs_t.shape: (query_batch, cf_per_seg*neighbor_size, *obs_shape)
        
        assert obs_t.shape[0] == act_t.shape[0] == oracle_act_t.shape[0] \
                == self.query_batch * self.cf_per_seg * self.neighbor_size
        
        self.put_queries(obs_t, act_t, oracle_act_t)
        assert obs_t.shape[0] == self.query_batch * self.cf_per_seg * self.neighbor_size

        return obs_t.shape[0]  # query_batch * cf_per_seg * neighbor_size
    
    def disagreement_sampling(self):
        raise NotImplementedError
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            query_batch=self.query_batch*self.large_batch)
        
        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = disagree.argsort(descending=True)[:self.query_batch]
        pdb.set_trace()
        # TODO: check index slice here
        r_t_1, sa_t_1[0], sa_t_1[1] = r_t_1[top_k_index], sa_t_1[0][top_k_index], sa_t_1[1][top_k_index]
        r_t_2, sa_t_2[0], sa_t_2[1] = r_t_2[top_k_index], sa_t_2[0][top_k_index], sa_t_2[1][top_k_index]
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def entropy_sampling(self):
        raise NotImplementedError
        pdb.set_trace()  # check shape here
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            query_batch=self.query_batch*self.large_batch)
        
        # get final queries based on uncertainty
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        
        # We need top K largest entropy
        top_k_index = entropy.argsort(descending=True)[:self.query_batch]  # argsort: default descending=False
        r_t_1, sa_t_1[0], sa_t_1[1] = r_t_1[top_k_index], sa_t_1[0][top_k_index], sa_t_1[1][top_k_index]
        r_t_2, sa_t_2[0], sa_t_2[1] = r_t_2[top_k_index], sa_t_2[0][top_k_index], sa_t_2[1][top_k_index]
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def reward_margine_loss(self, r_hat_s, oracle_act):
        # NOTE: output from the reward model is constrained by tanh in (0, 1)
        # r_hat_s.shape: (B, action_ndim), oracle_act.shape: (B, 1)
        assert r_hat_s.ndim == 2
        assert r_hat_s.shape[-1] == self.action_dim
        assert oracle_act.ndim == 1
        act_mask_ninf = torch.zeros_like(r_hat_s).scatter_(
                        dim=-1, index=oracle_act[...,None],
                        # src=-torch.inf)  # shape: (B, action_dim)
                        src=-torch.inf * torch.ones_like(oracle_act[...,None]))  # shape: (B, action_dim)
        act_mask_zero = torch.ones_like(r_hat_s).scatter_(
                        dim=-1, index=oracle_act[...,None],
                        # src=0.)  # shape: (B, action_dim)
                        src=torch.zeros_like(oracle_act[...,None], dtype=r_hat_s.dtype))  # shape: (B, action_dim)
        # TODO: check if there have problems with r_hat_s, why so many zeros in it?
        r_hat_s_oa = torch.gather(input=r_hat_s, dim=-1, index=oracle_act[...,None])  # r_hat_s_oa.shape: (B, 1)
        max_noa_r_hat = torch.max(act_mask_zero * r_hat_s + act_mask_ninf,
                                  dim=-1, keepdim=True)[0] # (max returns (values, indices))  max none-oracle-action reward
        # max_noa_r_hat.shape: (B, 1)
        assert r_hat_s_oa.shape == max_noa_r_hat.shape == (r_hat_s.shape[0], 1)
        # NOTE: bug here: loss_margine is a constant; 
        #       for case that r_hat(s,a_E) > r_hat(s,a), it still try to enlarge the difference bwtween a_E and a
        loss = max_noa_r_hat + self.loss_margine - r_hat_s_oa  # loss.shape: (B, 1)
        return loss.mean()  # a scalar
    
    def reward_margine_loss_min0(self, r_hat_s, oracle_act):
        # NOTE: this loss's min value is not 0...
        # NOTE: output from the reward model is constrained by tanh in (0, 1)
        # r_hat_s.shape: (B, action_ndim), oracle_act.shape: (B, 1)
        assert r_hat_s.ndim == 2
        assert r_hat_s.shape[-1] == self.action_dim
        assert oracle_act.ndim == 1
        act_mask_ninf = torch.zeros_like(r_hat_s).scatter_(
                        dim=-1, index=oracle_act[...,None],
                        # src=-torch.inf)  # shape: (B, action_dim)
                        src=-torch.inf * torch.ones_like(oracle_act[...,None]))  # shape: (B, action_dim)
        act_mask_zero = torch.ones_like(r_hat_s).scatter_(
                        dim=-1, index=oracle_act[...,None],
                        # src=0.)  # shape: (B, action_dim)
                        src=torch.zeros_like(oracle_act[...,None], dtype=r_hat_s.dtype))  # shape: (B, action_dim)
        r_hat_s_oa = torch.gather(input=r_hat_s, dim=-1, index=oracle_act[...,None])  # r_hat_s_oa.shape: (B, 1)
        max_noa_r_hat = torch.max(act_mask_zero * r_hat_s + act_mask_ninf,
                                  dim=-1, keepdim=True)[0] # (torch.max returns (values, indices))  max none-oracle-action reward
        # max_noa_r_hat.shape: (B, 1)
        # Assign margine=0 if the max non-oracle reward < oracle reward; else margine=self.loss_margine
        loss_margine = torch.where(max_noa_r_hat < r_hat_s_oa, 0, self.loss_margine)
        assert r_hat_s_oa.shape == max_noa_r_hat.shape == \
               loss_margine.shape == (r_hat_s.shape[0], 1)
        loss = max_noa_r_hat + loss_margine - r_hat_s_oa  # loss.shape: (B, 1)
        return loss.mean()  # a scalar

    def reward_margine_loss_min0_fix(self, r_hat_s, oracle_act):
        # NOTE: output from the reward model is constrained by tanh in (0, 1)
        # r_hat_s.shape: (B, action_ndim), oracle_act.shape: (B, 1)
        assert r_hat_s.ndim == 2
        assert r_hat_s.shape[-1] == self.action_dim
        assert oracle_act.ndim == 1
        r_hat_s_oa = torch.gather(input=r_hat_s, dim=-1, index=oracle_act[...,None])  # r_hat_s_oa.shape: (B, 1)
        max_r_hat_val, max_r_hat_id = torch.max(r_hat_s, dim=-1, keepdim=True) # (torch.max returns (values, indices))  max action reward
        # max_r_hat_val.shape == max_r_hat_id.sahpe: (B, 1)
        # Assign margine=0 if the max non-oracle reward < oracle reward; else margine=self.loss_margine
        loss_margine = torch.where(max_r_hat_id == oracle_act[...,None], 0, self.loss_margine)
        assert r_hat_s_oa.shape == max_r_hat_val.shape == \
               loss_margine.shape == (r_hat_s.shape[0], 1)
        loss = max_r_hat_val + loss_margine - r_hat_s_oa  # loss.shape: (B, 1)
        return loss.mean()  # a scalar

    def reward_exp_loss(self, r_hat_s, oracle_act):
        # TODO: think about the meaning of larger/smaller beta value
        # r_hat_s.shape: (B, action_ndim), oracle_act.shape: (B, 1)
        loss = nn.CrossEntropyLoss()(r_hat_s * self.exp_loss_beta,
                                     oracle_act.reshape(r_hat_s.shape[0]))  # a scalar
        if self.loss_square:
            output_squared = r_hat_s**2  # (B, action_dim)
            loss += self.loss_square_coef * output_squared.mean()  # output_squared.mean() is a scalar
        return loss
    
    def model_reset(self,):
        print(f'[REWARD MODEL] model reset with {self.reset.type}')
        # Fior bn, weight->1, bias->0, running_mean->0, running_var->1, num_batches_tracked->0
        if self.reset.type == 'rnd':
            for id_e in range(self.de):
                for id_l in self.reset_mlp_ls:
                    self.ensemble[id_e][1][id_l].apply(
                        partial(weight_init, init_type=self.init_type, trival=True))
                for id_l in self.reset_cnn_ls:
                    self.ensemble[id_e][0].conv[id_l].apply(
                        partial(weight_init, init_type=self.init_type, trival=True))
        elif self.reset.type == 'init':
            for id_e in range(self.de):
                for name_l in self.reset_ls:
                    # for bn, running_mean & running_var & num_batches_tracked will also be reset
                    self.ensemble[id_e].state_dict()[name_l][...] = \
                        self.init_model_dict[id_e][name_l]
        else:
            raise NotImplementedError

    def get_reset_list(self,):
        reset_ls = []
        reset_mlp_ls = []
        reset_cnn_ls = []
        layers_in_mlp = (self.encoder_cfg.mlp_cfg.dropout > 0) \
                        + (self.encoder_cfg.mlp_cfg.norm_type is not None) \
                        + 2  # 1 for activation, 1 for Linear
        layers_in_cnn = (self.encoder_cfg.cnn_cfg.dropout > 0) \
                        + (self.encoder_cfg.cnn_cfg.channel_dropout > 0) \
                        + (self.encoder_cfg.cnn_cfg.norm_type is not None) \
                        + 2  # 1 for activation, 1 for Linear
        assert not self.encoder_cfg.cnn_cfg.use_maxpool  # haven't check it
        for idx in range(self.encoder_cfg.mlp_cfg.hidden_depth + 1,
                         self.encoder_cfg.mlp_cfg.hidden_depth - self.reset.mlp_layers + 1, -1):
            # nn.Linear
            layer_id = (idx - 1) * layers_in_mlp
            reset_ls.append(f'1.{layer_id}.weight')
            reset_ls.append(f'1.{layer_id}.bias')
            reset_mlp_ls.append(layer_id)
            if idx == self.encoder_cfg.mlp_cfg.hidden_depth + 1:  # For mlp, the last layer is output
                continue
            layer_id += 2  # 1 for Linear, 1 for activation
            # norm_layer
            if self.encoder_cfg.mlp_cfg.norm_type is not None:
                assert self.encoder_cfg.mlp_cfg.norm_type == 'bn'  # the following names are for bn; for other norm_type, need to check again
                reset_ls.append(f'1.{layer_id}.weight')
                reset_ls.append(f'1.{layer_id}.bias')
                reset_ls.append(f'1.{layer_id}.running_mean')
                reset_ls.append(f'1.{layer_id}.running_var')
                reset_ls.append(f'1.{layer_id}.num_batches_tracked')
                reset_mlp_ls.append(layer_id)
                layer_id += 1
            # dropout layer
            if self.encoder_cfg.mlp_cfg.dropout > 0:
                layer_id += 1
        for idx in range(len(self.encoder_cfg.cnn_cfg.channels),
                         len(self.encoder_cfg.cnn_cfg.channels) - self.reset.cnn_layers, -1):
            layer_id = (idx - 1) * layers_in_cnn
            # Conv2d
            reset_ls.append(f'0.conv.{layer_id}.weight')
            reset_ls.append(f'0.conv.{layer_id}.bias')
            reset_cnn_ls.append(layer_id)
            layer_id += 2  # 1 for Conv2d, 1 for activation
            if self.encoder_cfg.cnn_cfg.norm_type is not None:
                assert self.encoder_cfg.cnn_cfg.norm_type == 'bn'  # the following names are for bn; for other norm_type, need to check again
                reset_ls.append(f'0.conv.{layer_id}.weight')
                reset_ls.append(f'0.conv.{layer_id}.bias')
                reset_ls.append(f'0.conv.{layer_id}.running_mean')
                reset_ls.append(f'0.conv.{layer_id}.running_var')
                reset_ls.append(f'0.conv.{layer_id}.num_batches_tracked')
                reset_cnn_ls.append(layer_id)
                layer_id += 1
            if self.encoder_cfg.cnn_cfg.dropout > 0:
                layer_id += 1
            if self.encoder_cfg.cnn_cfg.channel_dropout > 0:
                layer_id += 1
            assert layer_id % layers_in_cnn == 0
        self.reset_ls = reset_ls
        self.reset_mlp_ls = reset_mlp_ls
        self.reset_cnn_ls = reset_cnn_ls

    def train_reward(self):
        assert self.ensemble[0].training
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_grad_norms = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        # max_len = self.label_capacity if self._label_buffer_full else self.label_t
        max_len = self.len_label
        total_batch_index = []
        for _ in range(self.de):
            # TODO: If the states in this buffer is not diverse enough, the permutation can also be hard to optimize?
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))  # NOTE: use 'cile', so all labeled data will be used to train reward predictor!
        # list_debug_loss1, list_debug_loss2 = [], []
        total = 0
        
        for epoch in range(num_epochs):  # NOTE: larger batch_size should use larger learning_rate, because #epochs will decrease
            self.opt.zero_grad()
            loss = 0.0
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
            
            for member in range(self.de):
                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                # obs_t = torch.from_numpy(self.label_buffer.observation[idxs]).float().to(self.device)
                obs_t = self.label_buffer.observation[idxs]  # obs will be transferred to tensor in r_hat_s_member()
                act_t = torch.from_numpy(self.label_buffer.action[idxs]).long().to(self.device)
                oracle_act_t = torch.from_numpy(self.label_buffer.oracle_act[idxs]).long().to(self.device)
                r_hat_s_t = self.r_hat_s_member(obs_t, member=member, train=True)  # r_hat_s_t.shape: (batch_size, action_dim)
                
                if member == 0:
                    total += oracle_act_t.size(0)

                # compute loss
                curr_loss = self.reward_loss(r_hat_s=r_hat_s_t,
                                             oracle_act=oracle_act_t)
                loss += curr_loss  # TODO: can we learn some weights for different reward_predictor??
                ensemble_losses[member].append(curr_loss.item())
                
                # compute acc
                max_r_hat_a = torch.max(r_hat_s_t.data, dim=-1)[1]
                # check max_r_hat_a & oracle_act_t shape == (self.train_batch_size (different batch_size for the last batch),)
                assert oracle_act_t.shape == max_r_hat_a.shape == (oracle_act_t.shape[0], )
                correct = (max_r_hat_a == oracle_act_t).sum().item()  # count the number of samples that r_hat assign largest value for oracle_actions
                ensemble_acc[member] += correct

            loss.backward()
            if self.clip_grad_norm is not None:
                for idx, model in enumerate(self.ensemble):  # NOTE: not sure how much extra computational cost will be
                    ensemble_grad_norms[idx].append(torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.clip_grad_norm).item())  # default l2 norm
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
        ensemble_losses = np.mean(np.array(ensemble_losses), axis=-1, keepdims=False)  # shape: (#ensemble,)
        ensemble_grad_norms = np.mean(np.array(ensemble_grad_norms), axis=-1, keepdims=False)  # shape: (#ensemble,)
        ensemble_acc = ensemble_acc / total
        
        if self.use_best_one:
            self.best_r_id = np.argmax(ensemble_acc)
        
        return ensemble_acc, ensemble_losses, ensemble_grad_norms

    def samples_to_reward_buffer(self, samples):
        # NOTE: data's order must be the same with initialize_buffer()'s sample_examples 
        #       Otherwise a serious bug will occur in add_data: e.g. samples.done be saved in input_buffer.done
        return SamplesToCFRewardBuffer(
            observation=samples.env.observation,
            done=samples.env.done,
            action=samples.agent.action,
            oracle_act=samples.oracle_act,
            oracle_act_prob=samples.oracle_act_prob,
            oracle_q=samples.oracle_q,
        )
    
    def eval(self):
        for member in range(self.de):
            self.ensemble[member].eval()

    def train(self,):
        for member in range(self.de):
            self.ensemble[member].train()