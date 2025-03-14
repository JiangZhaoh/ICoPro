import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
import pdb
import os

from collections import deque
from functools import partial

from new_utils.model_utils import weight_init, concat_sa
from new_utils.new_agent.encoder import cnn_mlp
from new_utils.tensor_utils import torchify_buffer, numpify_buffer, get_leading_dims
from new_utils.atari_env.wrapper import mask_img_score_func_, add_gaussian_noise
from new_utils.model_utils import count_parameters

from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.buffer import buffer_from_example

# TODO: check how to normalize the reward prediction to have a standard deviation?
SamplesToPRewardBuffer = namedarraytuple("SamplesToPRewardBuffer",
    ["observation", "done", "action", "GT_reward"])  # save ground-truth reward here

class PEBBLEAtariRlpytRewardModel:
    def __init__(self,
                 encoder_cfg,
                 B,
                 episode_end_penalty,
                 ensemble_size=3,
                 reward_lr=3e-4,
                 query_batch = 128,  # old name in PEBBLE is mb_size
                 train_batch_size=128,
                 size_segment=1,  # timesteps per segment
                 max_size=100000,  # max timesteps of trajectories for query sampling
                 activation='tanh',
                 label_capacity=3000,  # "labeled" query buffer capacity, each query corresponds to a segment (old name in PEBBLE is capacity default 5e5)
                 large_batch=1,  # some sampling methods need more samples for query selection
                 label_margin=0.0,
                 teacher_beta=-1,  # coefficient for summation of segment reward
                 teacher_gamma=1,  # assign higher weight for steps close to the end of segments
                 teacher_eps_mistake=0,  # eps for assigning wrong labels to queries 
                 teacher_eps_skip=0,
                 teacher_eps_equal=0,
                 device='cuda',
                 init_type='orthogonal',
                 mask_img_score=True,
                 env_name=None,
                 clip_grad_norm=None,
                 loss_square=False,
                 loss_square_coef=1.,
                 gaussian_noise=None,
                 use_demo_pretrain=False,
                 total_demo_t=None,
                 ckpt_path=None,
                 reset=None,
                 ):
        self.device = device

        self.encoder_cfg = copy.deepcopy(encoder_cfg)
        self.episode_end_penalty = episode_end_penalty
        # self.encoder_cfg.cnn_cfg['in_channels'] += encoder_cfg.action_dim  # frame_stack * img_channel + action_dim
        self.obs_shape = encoder_cfg.obs_shape  # (frame_stack, H, W)
        self.action_dim = encoder_cfg.action_dim  # number of available discrete actions
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
        # self.max_size = max_size
        assert max_size % self.B == 0
        self.max_size_T = int(max_size // self.B)  # For input data buffer
        self.activation = activation
        self.loss_square = loss_square
        self.loss_square_coef = loss_square_coef
        self.clip_grad_norm = clip_grad_norm
        self.size_segment = size_segment

        self.label_capacity = int(label_capacity)  # "labeled" query buffer capacity
        # self.label_count = 0

        # Move buffer part to self.initialize()
        # self.buffer_seg1_obs = np.empty((self.label_capacity, size_segment, *self.obs_shape), dtype=np.float32)
        # self.buffer_seg1_act = np.empty((self.label_capacity, size_segment, 1), dtype=np.int32)
        # self.buffer_seg2_obs = np.empty((self.label_capacity, size_segment, *self.obs_shape), dtype=np.float32)
        # self.buffer_seg2_act = np.empty((self.label_capacity, size_segment, 1), dtype=np.int32)

        # self.buffer_label = np.empty((self.label_capacity, 1), dtype=np.float32)
        # self.buffer_index = 0
        # self.buffer_full = False  # query buffer full
        
        self.ckpt_path = ckpt_path
        self.reset = reset
        self.init_type = init_type
        self.construct_ensemble(init_type=init_type)
        # self.inputs_obs = None  # all available trajectories
        # self.inputs_act = None
        # self.targets = None
        # self.raw_actions = []
        # self.img_inputs = []
        self.query_batch = query_batch  # reward batch size
        self.origin_query_batch = query_batch  # reward batch size may change according to the current training step
        self.train_batch_size = train_batch_size
        self.CEloss = nn.CrossEntropyLoss()
        self.running_means = []
        self.running_stds = []
        self.best_seg = []
        self.best_label = []
        self.best_action = []
        self.large_batch = large_batch
        
        # new teacher
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.teacher_eps_equal = teacher_eps_equal
        self.teacher_eps_skip = teacher_eps_skip
        self.teacher_thres_skip = 0
        self.teacher_thres_equal = 0
        
        self.label_margin = label_margin  # TODO: how to use label_margin?
        self.label_target = 1 - 2*self.label_margin  # TODO: how to use label_target?

        self.avg_train_step_return = deque([], maxlen=100000)  # as a reference to decide margin

        self.mask_img_score = mask_img_score
        self.env_name = env_name

        self.use_gaussian_noise = gaussian_noise.flag
        if self.use_gaussian_noise:
            self.gaussian_amplitude = gaussian_noise.amplitude

        self.use_demo_pretrain = use_demo_pretrain
        self.total_demo_t = total_demo_t

    def initialize_buffer(self, agent, env, check_label_path=None):
        sample_examples = dict()
        label_sa_examples = dict()  # state & action
        label_p_examples = dict()  # preference
        # From get_example_outputs(agent, env, examples)
        env.reset()
        a = env.action_space.sample()
        o, r, terminated, truncated, env_info = env.step(a)
        r = np.asarray(r, dtype="float32")  # Must match torch float dtype here.
        # agent.reset()
        agent_inputs = torchify_buffer(o)
        a, agent_info = agent.step(agent_inputs)
        
        sample_examples["observation"] = o[:]
        sample_examples["done"] = terminated
        sample_examples["action"] = a
        sample_examples["GT_reward"] = r
       
        field_names = [f for f in sample_examples.keys() if f != "observation"]
        global PRewardBufferSamples
        PRewardBufferSamples = namedarraytuple("PRewardBufferSamples",
                                                        field_names)
        reward_buffer_example = PRewardBufferSamples(*(v for \
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

        if self.use_demo_pretrain:
            self.demo_buffer = buffer_from_example(reward_buffer_example,
                                                    (self.total_demo_t, 1))
            self.demo_frames = buffer_from_example(o[0],  # avoid saving duplicated frames
                                    (self.total_demo_t + n_frames - 1, 1))
            self.demo_new_frames = self.demo_frames[n_frames - 1:]  # [self.total_demo_t,1,H,W]
        
        self.input_t = 0
        self._input_buffer_full = False
        
        label_sa_examples["observation"] = o[:]  # (frame_stack, H, W)
        label_sa_examples["action"] = a
        label_p_examples["pref"] = 0
        
        # TODO: if want to use less memory for label buffer, could optimize buffer for observation to avoid saving repeated frames
        #       but then it may need significant time cost to extract observation by indexing?
        global PRewardLabelSABufferSamples
        PRewardLabelSABufferSamples = namedarraytuple("PRewardLabelSABufferSamples",
                                                   label_sa_examples.keys())
        reward_label_sa_buffer_example = PRewardLabelSABufferSamples(*(v for \
                                            k, v in label_sa_examples.items()))
        global PRewardLabelPBufferSamples
        PRewardLabelPBufferSamples = namedarraytuple("PRewardLabelPBufferSamples",
                                                   label_p_examples.keys())
        reward_label_p_buffer_example = PRewardLabelPBufferSamples(*(v for \
                                            k, v in label_p_examples.items()))
        
        # TODO: For img input, this part can be optimized specially for n_frame
        # self.labelbuffer_saX.observation.shape: (label_capacity, size_segment, Frame, H, W), uint8
        # self.labelbuffer_saX.action.shape: (label_capacity, size_segment), int64
        # NOTE: the two buffer will not share memory
        self.label_buffer_sa1 = buffer_from_example(reward_label_sa_buffer_example,
                                                (self.label_capacity, self.size_segment))
        self.label_buffer_sa2 = buffer_from_example(reward_label_sa_buffer_example,
                                                (self.label_capacity, self.size_segment))
        # self.label_buffer_p.pref.shape: (label_capacity,), int64
        self.label_buffer_p = buffer_from_example(reward_label_p_buffer_example,
                                                (self.label_capacity,))
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
    
    def softXEnt_loss(self, input, target):
        pdb.set_trace()  # TODO: check shape & do not know what's the difference btw this loss and CELoss()
        # input.shape = target.shape: (B, 2)
        logprobs = torch.nn.functional.log_softmax(input, dim = 1)  # logprobs.shape: (B, 2)
        return  -(target * logprobs).sum() / input.shape[0]
    
    def change_batch(self, new_frac):
        self.query_batch = max(int(self.origin_query_batch * new_frac), 1)
    
    def set_batch(self, new_batch):
        self.query_batch = int(new_batch)
    
    # def set_teacher_thres_skip(self, new_margin):
    #     # Only train with queries that have at least one segment performs excellent
    #     self.teacher_thres_skip = new_margin * self.teacher_eps_skip
        
    # def set_teacher_thres_equal(self, new_margin):
    #     # If two segments with reward summation < teacher_thres_equal,
    #     # They will be regarded as equally preferred
    #     self.teacher_thres_equal = new_margin * self.teacher_eps_equal
        
    def construct_ensemble(self, init_type):
        if self.activation == 'tanh':
            output_mod = nn.Tanh()
        elif self.activation == 'sig':
            output_mod = nn.Sigmoid()
        elif self.activation == 'relu':
            output_mod = nn.ReLU()
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
        print(f'%%%%%% reward model: {model}')
        print(f"[Reward Model]: Initialized model with {count_parameters(model)} parameters")
        self.opt = torch.optim.Adam(self.paramlst, lr = self.lr)
        if self.reset.flag:
            self.get_reset_list()

    # def add_data(self, obs, act, rew, terminated):
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
        done_penalty = torch.where(condition=samples.done, 
                                   input=torch.tensor([1.0 * self.episode_end_penalty]),
                                   other=torch.tensor([0.]))  # shape (T, B)
        samples.GT_reward[...] -= done_penalty
        
        t, fm1 = self.input_t, self.n_frames - 1
        reward_buffer_samples = PRewardBufferSamples(*(v \
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
        # self.input_frames[idxs] = samples.observation[:, :, -1]
        self.input_new_frames[idxs] = samples.observation[:, :, -1]
        if t == 0:  # Starting: write early frames
            for f in range(fm1):
                self.input_frames[f] = samples.observation[0, :, f]
        elif self.input_t <= t and fm1 > 0:  # Wrapped: copy any duplicate frames.
            self.input_frames[:fm1] = self.input_frames[-fm1:]

    def add_demo_data(self, agent, agent_eps, env):
        obs, _ = env.reset()
        obs_buf = obs[:]
        obs_pyt = torchify_buffer(obs_buf)  # share self.agent_inputs's memory
        
        # agent.reset()
        eps_return = agent.eval_mode(itr=1, eps=agent_eps)
        assert eps_return == agent_eps

        demo_t = 0
        truncated = False

        reward_return_ls = []
        traj_len_ls = []
        reward_return = traj_len = 0
        while demo_t < self.total_demo_t:
            if truncated or env.need_reset:
                obs, _ = env.reset()  # (C, H, W)
                obs_buf[...] = obs[:]
                reward_return_ls.append(reward_return)
                reward_return = 0
                traj_len_ls.append(traj_len)
                traj_len = 0
            act_pyt, agent_info = agent.step(obs_pyt)  # act_pyt.shape: (#eval_envs,), torch.int64
            next_obs, r, terminated, truncated, env_info = env.step(act_pyt)
            act = numpify_buffer(act_pyt)  # shape (B,)
            
            if demo_t == 0:
                self.demo_frames[:self.n_frames, 0, ...] = obs_buf[...]
            else:
                self.demo_new_frames[demo_t, 0, ...] = obs_buf[-1]
            self.demo_buffer.action[demo_t, 0] = act
            self.demo_buffer.GT_reward[demo_t, 0] = r
            self.demo_buffer.done[demo_t, 0] = terminated

            reward_return += r
            traj_len += 1
            obs_buf[...] = next_obs[:]
            demo_t += 1
        return reward_return_ls, traj_len_ls

    def add_data_batch(self, obses, rewards):
        raise NotImplementedError
        num_env = obses.shape[0]
        for index in range(num_env):
            self.inputs.append(obses[index])
            self.targets.append(rewards[index])
    
    def get_rank_probability(self, x_1, x_2):
        pdb.set_trace()
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_member(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        
        return np.mean(probs, axis=0), np.std(probs, axis=0)
    
    def get_entropy(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_entropy(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)  # (#ensemble, B)
        return np.mean(probs, axis=0), np.std(probs, axis=0)  # (B,)

    def p_hat_member(self, x_1, x_2, member):
        pdb.set_trace()
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
        # x_[0]: obs.shape: (B, size_seg, *obs_shape)
        # x_[1]: act.shape: (B, size_seg)
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            r_hat1 = self.r_hat_sa_member(x_1, member=member)  # (B, size_seg)
            r_hat2 = self.r_hat_sa_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1, keepdims=True)  # (B, 1)
            r_hat2 = r_hat2.sum(axis=1, keepdims=True)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)  # (B, 2)
        
        # H(p_hat) = - sum_x{p_hat(x) * log(p_hat(x))}, where p_hat(x)=softmax(x)
        ent = F.softmax(r_hat, dim=-1) * F.log_softmax(r_hat, dim=-1)  # (B, 2)
        ent = ent.sum(axis=-1, keepdims=False).abs()  # (B,)
        return ent  # shape: (B,)

    def r_hat_sa_member(self, x, member, return_r_s=False, train=False):
        # the network parameterizes r hat in eqn 1 from the paper
        obs, act = x[0], x[1]
        if act.ndim == 2:  # (B, size_seg)
            B = act.shape[0]
            obs = obs.reshape(B * self.size_segment, *self.obs_shape)
            act = act.reshape(B * self.size_segment,)
        else:  # (B)
            assert act.ndim == 1
            B = None

        if self.use_gaussian_noise and train:  # because use np.random.normal, we add the noise before transferring to torch.Tensor
            obs = add_gaussian_noise(img=obs, amplitude=self.gaussian_amplitude)
        
        if type(obs) == np.ndarray:
            obs = torch.from_numpy(obs).float().to(self.device)
            act = torch.from_numpy(act).long().to(self.device)
        else:
            obs = obs.float().to(self.device)
            act = act.long().to(self.device)
        obs /= 255.0

        if self.mask_img_score:
            mask_img_score_func_(env_name=self.env_name, obs=obs)  # modified in place
        
        r_hat_s = self.ensemble[member](obs)
        assert r_hat_s.shape == (obs.shape[0], self.action_dim)
        r_hat_sa = torch.gather(r_hat_s, dim=-1, index=act[...,None]) # r_hat_sa.shape: (B, 1)
        assert r_hat_sa.shape == (r_hat_s.shape[0], 1)
        if B is not None:
            r_hat_sa = r_hat_sa.reshape(B, self.size_segment)
        if return_r_s:
            return r_hat_sa, r_hat_s
        else:
            return r_hat_sa

    def r_hat_sa(self, obs, act):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        assert not self.ensemble[0].training
        assert obs.ndim == 4  # (B, C*frame_stack, H, W)
        assert act.ndim == 1  # (B,)
        r_hats_sa = []
        # move this part in r_hat_sa_member()
        # obs = obs.float().to(self.device)
        # act = act.long().to(self.device)
        for member in range(self.de):
            r_hat_sa = self.r_hat_sa_member((obs, act), member=member).detach().cpu().numpy()
            r_hats_sa.append(r_hat_sa.reshape(-1))
        r_hats_sa = np.array(r_hats_sa)  # shape: (#ensemble, batch_size)
        return np.mean(r_hats_sa, axis=0, keepdims=False)  # shape: (batch_size,)
    
    def r_hat_batch(self, x):
        raise NotImplementedError
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        obs, act = np.array(x[0], dtype=float), np.array(x[1])
        obs /= 255.0
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member((obs, act), member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)

        return np.mean(r_hats, axis=0)
    
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
    
    def extract_demo_observation(self, T_idxs):
        # same function as extract_observation bur with different buffer
        observation = np.stack([self.demo_frames[t:t + self.n_frames, 0]
                                for t in T_idxs], axis=0)  # [B,C,H,W]
        # Populate empty (zero) frames after environment done.
        for f in range(1, self.n_frames):
            # e.g. if done 1 step prior, all but newest frame go blank.
            b_blanks = np.where(self.demo_buffer.done[T_idxs - f, 0])[0]
            observation[b_blanks, :self.n_frames - f] = 0
        return observation
    
    def get_queries(self, query_batch):  # TODO: consider whether should reshape data t0 query_batch*size_segment*...
        # max_len: total number of collected trajectories
        # if len(self.inputs[-1]) < len_traj:
        #     max_len = max_len - 1
        
        # get train traj
        # train_inputs = np.array(self.inputs[:max_len])
        # train_targets = np.array(self.targets[:max_len])
        # NOTE: query_batch*size_segment
        max_index_T = self.len_inputs_T - self.size_segment
        # Batch = query_batch
        batch_index_2 = np.random.choice(max_index_T*self.B, size=query_batch, replace=True).reshape(-1, 1)  # (query_batch, 1)
        batch_index_2_T, batch_index_2_B = np.divmod(batch_index_2, self.B)  # (x // y, x % y), batch_index_B & batch_index_T.shape: (query_batch, 1)
        take_index_2_T = (batch_index_2_T + np.arange(0, self.size_segment)).reshape(-1)  # shape: (query_batch * size_segment, )
        take_index_2_B = batch_index_2_B.repeat(self.size_segment, axis=-1).reshape(-1)  # shape: (query_batch * size_segment, )
        # obs_t_2 = np.take(self.inputs_obs, take_index_2, axis=0) # (query_batch, size_segment, obs_shape)
        obs_t_2 = self.extract_observation(take_index_2_T, take_index_2_B)  # (query_batch * size_segment, *obs_shape)
        assert (obs_t_2.ndim == 4) and (obs_t_2.shape[0] == query_batch * self.size_segment)
        obs_t_2 = obs_t_2.reshape(query_batch, self.size_segment, *self.obs_shape)  # (query_batch, size_segment, *obs_shape)
        # act_t_2 = np.take(self.inputs_act, take_index_2, axis=0) # (query_batch, size_segment, 1)
        act_t_2 = self.input_buffer.action[take_index_2_T, take_index_2_B]  # (query_batch * size_segment, )
        act_t_2 = act_t_2.reshape(query_batch, self.size_segment)  # (query_batch, size_segment)
        # r_t_2 = np.take(self.targets, take_index_2, axis=0) # (query_batch, size_segment, 1)
        r_t_2 = self.input_buffer.GT_reward[take_index_2_T, take_index_2_B]  # (query_batch * size_segment, )
        r_t_2 = r_t_2.reshape(query_batch, self.size_segment)  # (query_batch, size_segment)

        batch_index_1 = np.random.choice(max_index_T*self.B, size=query_batch, replace=True).reshape(-1, 1)  # (query_batch, 1)
        batch_index_1_T, batch_index_1_B = np.divmod(batch_index_1, self.B)  # (x // y, x % y), batch_index_B & batch_index_T.shape: (query_batch, 1)
        take_index_1_T = (batch_index_1_T + np.arange(0, self.size_segment)).reshape(-1)  # shape: (query_batch * size_segment, )
        take_index_1_B = batch_index_1_B.repeat(self.size_segment, axis=-1).reshape(-1)  # shape: (query_batch * size_segment, )
        # obs_t_2 = np.take(self.inputs_obs, take_index_2, axis=0) # (query_batch, size_segment, obs_shape)
        obs_t_1 = self.extract_observation(take_index_1_T, take_index_1_B)  # (query_batch * size_segment, *obs_shape)
        # assert (obs_t_1.ndim == 4) and (obs_t_1.shape[0] == query_batch * self.size_segment)
        obs_t_1 = obs_t_1.reshape(query_batch, self.size_segment, *self.obs_shape)  # (query_batch, size_segment, *obs_shape)
        # act_t_2 = np.take(self.inputs_act, take_index_2, axis=0) # (query_batch, size_segment, 1)
        act_t_1 = self.input_buffer.action[take_index_1_T, take_index_1_B]  # (query_batch * size_segment, )
        act_t_1 = act_t_1.reshape(query_batch, self.size_segment)  # (query_batch, size_segment)
        # r_t_2 = np.take(self.targets, take_index_2, axis=0) # (query_batch, size_segment, 1)
        r_t_1 = self.input_buffer.GT_reward[take_index_1_T, take_index_1_B]  # (query_batch * size_segment, )
        r_t_1 = r_t_1.reshape(query_batch, self.size_segment)  # (query_batch, size_segment)

        # batch_index_1 = np.random.choice(max_index_T, size=query_batch, replace=True).reshape(-1, 1)  # (query_batch, 1)
        # take_index_1 = batch_index_1 + np.arange(0, self.size_segment).reshape(1, -1)  # shape: (query_batch, size_segment)
        # obs_t_1 = np.take(self.inputs_obs, take_index_1, axis=0) # (query_batch, size_segment, obs_shape)
        # act_t_1 = np.take(self.inputs_act, take_index_1, axis=0) # (query_batch, size_segment)
        # r_t_1 = np.take(self.targets, take_index_1, axis=0) # (query_batch, size_segment)
        # obs_t.shape: (B, size_segment, frame_stack, H, W), act_t.shape: (query_batch, size_segment)
        sa_t_1 = (obs_t_1, act_t_1)
        sa_t_2 = (obs_t_2, act_t_2)
        return sa_t_1, sa_t_2, r_t_1, r_t_2
    
    def get_queries_pretrain(self, num_feedback):
        max_index_data_T = self.len_inputs_T - self.size_segment
        max_index_demo_T = self.total_demo_t - self.size_segment
        
        # extract segments from the randomly collected data at the beginning of policy training
        batch_index_rnd = np.random.choice(max_index_data_T*self.B, size=num_feedback, replace=True).reshape(-1, 1)  # (query_batch, 1)
        batch_index_rnd_T, batch_index_rnd_B = np.divmod(batch_index_rnd, self.B)  # (x // y, x % y), batch_index_B & batch_index_T.shape: (query_batch, 1)
        take_index_rnd_T = (batch_index_rnd_T + np.arange(0, self.size_segment)).reshape(-1)  # shape: (query_batch * size_segment, )
        take_index_rnd_B = batch_index_rnd_B.repeat(self.size_segment, axis=-1).reshape(-1)  # shape: (query_batch * size_segment, )
        # obs_t_2 = np.take(self.inputs_obs, take_index_2, axis=0) # (query_batch, size_segment, obs_shape)
        obs_t_rnd = self.extract_observation(take_index_rnd_T, take_index_rnd_B)  # (query_batch * size_segment, *obs_shape)
        assert (obs_t_rnd.ndim == 4) and (obs_t_rnd.shape[0] == num_feedback * self.size_segment)
        obs_t_rnd = obs_t_rnd.reshape(num_feedback, self.size_segment, *self.obs_shape)  # (query_batch, size_segment, *obs_shape)
        # act_t_2 = np.take(self.inputs_act, take_index_2, axis=0) # (query_batch, size_segment, 1)
        act_t_rnd = self.input_buffer.action[take_index_rnd_T, take_index_rnd_B]  # (query_batch * size_segment, )
        act_t_rnd = act_t_rnd.reshape(num_feedback, self.size_segment)  # (query_batch, size_segment)
        # r_t_2 = np.take(self.targets, take_index_2, axis=0) # (query_batch, size_segment, 1)
        r_t_rnd = self.input_buffer.GT_reward[take_index_rnd_T, take_index_rnd_B]  # (query_batch * size_segment, )
        r_t_rnd = r_t_rnd.reshape(num_feedback, self.size_segment)  # (query_batch, size_segment)

        batch_index_demo = np.random.choice(max_index_demo_T*1, size=num_feedback, replace=True).reshape(-1, 1)  # (query_batch, 1)
        # batch_index_1_T, batch_index_1_B = np.divmod(batch_index_1, self.B)  # (x // y, x % y), batch_index_B & batch_index_T.shape: (query_batch, 1)
        take_index_demo = (batch_index_demo + np.arange(0, self.size_segment)).reshape(-1)  # shape: (query_batch * size_segment, )
        # take_index_1_B = batch_index_1_B.repeat(self.size_segment, axis=-1).reshape(-1)  # shape: (query_batch * size_segment, )
        # obs_t_2 = np.take(self.inputs_obs, take_index_2, axis=0) # (query_batch, size_segment, obs_shape)
        obs_t_demo = self.extract_demo_observation(take_index_demo)  # (query_batch * size_segment, *obs_shape)
        # assert (obs_t_1.ndim == 4) and (obs_t_1.shape[0] == query_batch * self.size_segment)
        obs_t_demo = obs_t_demo.reshape(num_feedback, self.size_segment, *self.obs_shape)  # (query_batch, size_segment, *obs_shape)
        # act_t_2 = np.take(self.inputs_act, take_index_2, axis=0) # (query_batch, size_segment, 1)
        act_t_demo = self.demo_buffer.action[take_index_demo, 0]  # (query_batch * size_segment, )
        act_t_demo = act_t_demo.reshape(num_feedback, self.size_segment)  # (query_batch, size_segment)
        # r_t_2 = np.take(self.targets, take_index_2, axis=0) # (query_batch, size_segment, 1)
        r_t_demo = self.demo_buffer.GT_reward[take_index_demo, 0]  # (query_batch * size_segment, )
        r_t_demo = r_t_demo.reshape(num_feedback, self.size_segment)  # (query_batch, size_segment)

        sa_t_demo = (obs_t_demo, act_t_demo)
        sa_t_rnd = (obs_t_rnd, act_t_rnd)
        return sa_t_demo, sa_t_rnd, r_t_demo, r_t_rnd

    def put_queries(self, sa_t_1, sa_t_2, labels):
        # states, sa_t[0].shape: (B, size_segment, *obs_shape)
        # action, sa_t[1].shape: (B, size_segment)
        # labels.shape: (B,)
        assert sa_t_1[0].shape[0] == sa_t_1[1].shape[0] == \
               sa_t_2[0].shape[0] == sa_t_2[1].shape[0] == \
               labels.shape[0]
        total_sample = labels.shape[0]
        # next_index = self.buffer_index + total_sample  # new index in the query buffer after adding new queries
        next_index = self.label_t + total_sample  # new index in the query buffer after adding new queries
        if next_index >= self.label_capacity:
            # self.buffer_full = True
            self._label_buffer_full = True
            maximum_index = self.label_capacity - self.label_t
            np.copyto(self.label_buffer_sa1.observation[self.label_t:self.label_capacity], sa_t_1[0][:maximum_index])
            np.copyto(self.label_buffer_sa1.action[self.label_t:self.label_capacity], sa_t_1[1][:maximum_index])
            np.copyto(self.label_buffer_sa2.observation[self.label_t:self.label_capacity], sa_t_2[0][:maximum_index])
            np.copyto(self.label_buffer_sa2.action[self.label_t:self.label_capacity], sa_t_2[1][:maximum_index])
            np.copyto(self.label_buffer_p.pref[self.label_t:self.label_capacity], labels[:maximum_index])
            remain = total_sample - (maximum_index)
            if remain > 0:  # if next_index exceed capacity, extra new queries will be added to the beginning of query buffer
                np.copyto(self.label_buffer_sa1.observation[0:remain], sa_t_1[0][maximum_index:])
                np.copyto(self.label_buffer_sa1.action[0:remain], sa_t_1[1][maximum_index:])
                np.copyto(self.label_buffer_sa2.observation[0:remain], sa_t_2[0][maximum_index:])
                np.copyto(self.label_buffer_sa2.action[0:remain], sa_t_2[1][maximum_index:])
                np.copyto(self.label_buffer_p.pref[0:remain], labels[maximum_index:])
            self.label_t = remain
        else:
            np.copyto(self.label_buffer_sa1.observation[self.label_t:next_index], sa_t_1[0])
            np.copyto(self.label_buffer_sa1.action[self.label_t:next_index], sa_t_1[1])
            np.copyto(self.label_buffer_sa2.observation[self.label_t:next_index], sa_t_2[0])
            np.copyto(self.label_buffer_sa2.action[self.label_t:next_index], sa_t_2[1])
            np.copyto(self.label_buffer_p.pref[self.label_t:next_index], labels)
            self.label_t = next_index
    
    def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2):
        # r_t.shape: (query_batch, size_segment)
        # sa_t.shape: 0-state: (query_batch, size_segment, frame_stack, H, W), 
        #             1-action: (query_batch, size_segment)
        sum_r_t_1 = np.sum(r_t_1, axis=-1, keepdims=False)  # shape: (query_batch,)
        sum_r_t_2 = np.sum(r_t_2, axis=-1, keepdims=False)  # shape: (query_batch,)
        
        # skip the queries that do not show significant difference between segment pairs
        if self.teacher_thres_skip > 0:
            raise NotImplementedError
            max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
            max_index = (max_r_t > self.teacher_thres_skip).reshape(-1)
            if sum(max_index) == 0:
                return None, None, None, None, []

            sa_t_1[0] = sa_t_1[0][max_index]  # 0: obs
            sa_t_1[1] = sa_t_1[1][max_index]  # 1: act
            sa_t_2[0] = sa_t_2[0][max_index]
            sa_t_2[1] = sa_t_2[1][max_index]
            r_t_1 = r_t_1[max_index]
            r_t_2 = r_t_2[max_index]
            sum_r_t_1 = np.sum(r_t_1, axis=1, keepdims=False)  # shape: (#labels,)
            sum_r_t_2 = np.sum(r_t_2, axis=1, keepdims=False)
        
        # equally preferable
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_thres_equal).reshape(-1)
        
        # perfectly rational
        seg_size = r_t_1.shape[1]
        temp_r_t_1 = r_t_1.copy()  # values will not affect each other
        temp_r_t_2 = r_t_2.copy()
        for index in range(seg_size-1):
            temp_r_t_1[:,:index+1] *= self.teacher_gamma
            temp_r_t_2[:,:index+1] *= self.teacher_gamma
        sum_r_t_1 = np.sum(temp_r_t_1, axis=1, keepdims=False)  # (query_batch, )
        sum_r_t_2 = np.sum(temp_r_t_2, axis=1, keepdims=False)  # (query_batch, )
        
        # if sum_r_t_1 < sum_r_t_2, rational_labels = 1 * (True) = 1
        # if sum_r_t_1 >= sum_r_t_2, rational_labels = 1 * (False) = 0
        # so labels is the indices where segments are preferred
        rational_labels = 1 * (sum_r_t_1 < sum_r_t_2)  # shape (B,)
        if self.teacher_beta > 0: # Bradley-Terry rational model
            r_hat = torch.cat([torch.Tensor(sum_r_t_1).reshape(-1, 1),
                               torch.Tensor(sum_r_t_2).reshape(-1, 1)], axis=-1)  # (# selected queries, 2)
            r_hat = r_hat * self.teacher_beta
            ent = F.softmax(r_hat, dim=-1)[:, 1]  # ent.shape: (# selected queries), means P(seg_2 > seg_1)
            labels = torch.bernoulli(ent).int().numpy().reshape(-1)  # this distribution will return 1 given input probability
            pdb.set_trace()  # check labels.shape
        else:
            labels = rational_labels
        
        # making a mistake
        len_labels = labels.shape[0]
        # self.label_count += len_labels  # one label corresponds to comparing 2 segments
        rand_num = np.random.rand(len_labels)  # from a uniform distribution over [0, 1)
        noise_index = rand_num <= self.teacher_eps_mistake
        labels[noise_index] = 1 - labels[noise_index]

        # equally preferable
        labels[margin_index] = -1
        
        return sa_t_1, sa_t_2, r_t_1, r_t_2, labels
    
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
    
    def pretrain_sampling(self, num_feedback):
        # get queries
        sa_t_demo, sa_t_rnd, r_t_demo, r_t_rnd = self.get_queries_pretrain(num_feedback)
        
        # get labels
        sa_t_demo, sa_t_rnd, r_t_demo, r_t_rnd, labels = self.get_label(  # filter queries and 
            sa_t_demo, sa_t_rnd, r_t_demo, r_t_rnd)  # label: 0: demo >= rnd; 1: demo < rnd

        previous_label_cnt = self.len_label
        self.put_queries(sa_t_demo, sa_t_rnd, labels)
        now_label_cnt = self.len_label

        assert len(labels) == num_feedback == now_label_cnt - previous_label_cnt
        return num_feedback

    def uniform_sampling(self):
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            query_batch=self.query_batch)

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(  # filter queries and 
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def disagreement_sampling(self):
        pdb.set_trace()
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            query_batch=self.query_batch*self.large_batch)
        
        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = disagree.argsort(descending=True)[:self.query_batch]
        
        # TODO: check index slice here
        pdb.set_trace()
        r_t_1, sa_t_1[0], sa_t_1[1] = \
            r_t_1[top_k_index], sa_t_1[0][top_k_index], sa_t_1[1][top_k_index]
        r_t_2, sa_t_2[0], sa_t_2[1] = \
            r_t_2[top_k_index], sa_t_2[0][top_k_index], sa_t_2[1][top_k_index]
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def entropy_sampling(self):
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            query_batch=self.query_batch*self.large_batch)
        
        # get final queries based on uncertainty
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)  # (B,)
        
        # We need top K largest entropy
        top_k_index = (-entropy).argsort(axis=-1)[:self.query_batch]  # np.argsort: sort in ascending order
        r_t_1, obs_1, act_1 = \
            r_t_1[top_k_index], sa_t_1[0][top_k_index], sa_t_1[1][top_k_index]
        sa_t_1 = (obs_1, act_1)
        r_t_2, obs_2, act_2 = \
            r_t_2[top_k_index], sa_t_2[0][top_k_index], sa_t_2[1][top_k_index]
        sa_t_2 = (obs_2, act_2)
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def model_reset(self):
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
                    self.ensemble[id_e].state_dict()[name_l][...] = \
                        self.init_model_dict[id_e][name_l]
        else:
            raise NotImplementedError

    def get_reset_list(self):
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
        
        max_len = self.len_label
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))  # NOTE: use 'cile', so all labeled data will be used to train reward predictor!
        # list_debug_loss1, list_debug_loss2 = [], []
        total = 0
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
            
            for member in range(self.de):
                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                # move the torch.Tensor part in r_hat_sa_member()
                # obs_t_1 = torch.from_numpy(self.label_buffer_sa1.observation[idxs]).float().to(self.device)
                # act_t_1 = torch.from_numpy(self.label_buffer_sa1.action[idxs]).long().to(self.device)
                obs_t_1 = self.label_buffer_sa1.observation[idxs]
                act_t_1 = self.label_buffer_sa1.action[idxs]
                sa_t_1 = (obs_t_1, act_t_1)
                # move the torch.Tensor part in r_hat_sa_member()
                # obs_t_2 = torch.from_numpy(self.label_buffer_sa2.observation[idxs]).float().to(self.device)
                # act_t_2 = torch.from_numpy(self.label_buffer_sa2.action[idxs]).long().to(self.device)
                obs_t_2 = self.label_buffer_sa2.observation[idxs]
                act_t_2 = self.label_buffer_sa2.action[idxs]
                sa_t_2 = (obs_t_2, act_t_2)
                # labels = self.buffer_label[idxs]
                labels = torch.from_numpy(self.label_buffer_p.pref[idxs]).long().to(self.device)
                
                if member == 0:
                    total += labels.size(0)
                
                # get logits
                r_hat1, r_hat_s1 = self.r_hat_sa_member(sa_t_1, member=member,
                                                        return_r_s=True, train=True)  # (B, size_seg)
                r_hat2, r_hat_s2 = self.r_hat_sa_member(sa_t_2, member=member,
                                                        return_r_s=True, train=True)  # (B, size_seg)
                r_hat1 = r_hat1.sum(axis=1, keepdim=True)  # (B, 1)
                r_hat2 = r_hat2.sum(axis=1, keepdim=True)  # (B, 1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)  # shape (B, 2)
                assert (r_hat.ndim == 2) and (r_hat.shape[-1] == 2)
                # compute loss
                # NOTE: For nn.CrossEntropyLoss,
                # The input is expected to contain the unnormalized logits for each class
                #    (which do not need to be positive or sum to 1, in general).
                curr_loss = self.CEloss(r_hat, labels)
                if self.loss_square:
                    output_squared1 = r_hat_s1**2  # (B, action_dim)
                    output_squared2 = r_hat_s2**2  # (B, action_dim)
                    output_squared = (output_squared1 + output_squared2) * 0.5
                    curr_loss += self.loss_square_coef * output_squared.mean()  # output_squared.mean() is a scalar
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                assert labels.shape == predicted.shape == (r_hat.shape[0], )
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
            loss.backward()
            if self.clip_grad_norm is not None:
                for idx, model in enumerate(self.ensemble):  # NOTE: not sure how much extra computational cost will be
                    ensemble_grad_norms[idx].append(torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.clip_grad_norm).item())  # default l2 norm
            self.opt.step()
        ensemble_losses = np.mean(np.array(ensemble_losses), axis=-1, keepdims=False)  # shape: (#ensemble,)
        ensemble_grad_norms = np.mean(np.array(ensemble_grad_norms), axis=-1, keepdims=False)  # shape: (#ensemble,)
        ensemble_acc = ensemble_acc / total
        
        return ensemble_acc, ensemble_losses, ensemble_grad_norms  # nsemble_XXX.shape: (B, )
    
    def train_soft_reward(self):
        raise NotImplementedError
        pdb.set_trace()  # TODO: why have 2 train_XXX_reward?
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        # max_len = self.label_capacity if self.buffer_full else self.buffer_index
        max_len = self.len_label
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        total = 0
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
            
            for member in range(self.de):
                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                obs_t_1 = self.buffer_seg1_obs[idxs]
                act_t_1 = self.buffer_seg1_act[idxs]
                sa_t_1 = (obs_t_1, act_t_1)
                obs_t_2 = self.buffer_seg2_obs[idxs]
                act_t_2 = self.buffer_seg2_act[idxs]
                sa_t_2 = (obs_t_2, act_t_2)
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(self.device)
                
                if member == 0:
                    total += labels.size(0)
                
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                uniform_index = labels == -1
                labels[uniform_index] = 0
                pdb.set_trace()
                # TODO: check here, I think the correct code should be:
                # target_onehot = torch.zeros_like(r_hat) + labels * self.label_target
                target_onehot = torch.zeros_like(r_hat).scatter(  # TODO: the index value seems not correct?
                    dim=1, 
                    index=labels.unsqueeze(1),
                    src=self.label_target)
                target_onehot += self.label_margin
                if sum(uniform_index) > 0:
                    target_onehot[uniform_index] = 0.5
                curr_loss = self.softXEnt_loss(r_hat, target_onehot)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
            loss.backward()
            self.opt.step()
        
        ensemble_acc = ensemble_acc / total
        
        return ensemble_acc
    
    def samples_to_reward_buffer(self, samples):
        # TODO: think about how to save r_hat & reward
        return SamplesToPRewardBuffer(
            observation=samples.env.observation,
            done=samples.env.done,
            action=samples.agent.action,
            GT_reward=samples.env.reward,
        )
    
    def eval(self):
        for member in range(self.de):
            self.ensemble[member].eval()

    def train(self,):
        for member in range(self.de):
            self.ensemble[member].train()
