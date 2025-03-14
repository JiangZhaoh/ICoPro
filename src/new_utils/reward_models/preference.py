import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
import pdb

from collections import deque
from functools import partial

from new_utils.model_utils import weight_init, concat_sa
from new_utils.new_agent.encoder import cnn_mlp

# TODO: check how to normalize the reward prediction to have a standard deviation?


class PEBBLEAtariRewardModel:
    def __init__(self,
                 encoder_cfg,
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
                 init_type='orthogonal'):
        self.device = device

        self.encoder_cfg = copy.deepcopy(encoder_cfg)
        self.episode_end_penalty = episode_end_penalty
        # self.encoder_cfg.cnn_cfg['in_channels'] += encoder_cfg.action_dim  # frame_stack * img_channel + action_dim
        self.obs_shape = encoder_cfg.obs_shape  # (frame_stack, H, W)
        self.action_dim = encoder_cfg.action_dim  # number of available discrete actions
        # self.model_input_shape = self.obs_shape[:]  # (frame_stack, H, W)
        # self.model_input_shape[0] += self.action_dim  # input shape for reward models: [frame_stack+action_dim, H, W]
        
        self.de = ensemble_size
        self.lr = reward_lr
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        # self.model = None
        self.max_size = max_size
        self.activation = activation
        self.size_segment = size_segment

        self.label_capacity = int(label_capacity)  # "labeled" query buffer capacity
        self.label_count = 0

        self.buffer_seg1_obs = np.empty((self.label_capacity, size_segment, *self.obs_shape), dtype=np.float32)
        self.buffer_seg1_act = np.empty((self.label_capacity, size_segment, 1), dtype=np.int32)
        self.buffer_seg2_obs = np.empty((self.label_capacity, size_segment, *self.obs_shape), dtype=np.float32)
        self.buffer_seg2_act = np.empty((self.label_capacity, size_segment, 1), dtype=np.int32)

        self.buffer_label = np.empty((self.label_capacity, 1), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False  # query buffer full
        
        self.construct_ensemble(init_type=init_type)
        self.inputs_obs = None  # all available trajectories
        self.inputs_act = None
        self.targets = None
        self.raw_actions = []
        self.img_inputs = []
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
        self.query_batch = int(self.origin_query_batch*new_frac)
    
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
        else:
            raise NotImplementedError
        
        for i in range(self.de):
            model = cnn_mlp(obs_shape=self.obs_shape,
                            output_dim=self.action_dim,  # r_hat(s) outputs r_hat(s,a) for |A| actions
                            cnn_cfg=self.encoder_cfg.cnn_cfg,
                            mlp_cfg=self.encoder_cfg.mlp_cfg,
                            output_mod=output_mod
                            ).float().to(self.device)
            # For vector states:
            #  model = nn.Sequential(*gen_net(in_size=self.ds+self.da, 
            #                                out_size=1, H=256, n_layers=3, 
            #                                activation=self.activation)).float().to(self.device)
            for submodel in model:
                submodel.apply(partial(weight_init, init_type=init_type))

            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())
        print(f'%%%%%% reward model: {model}')
        self.opt = torch.optim.Adam(self.paramlst, lr = self.lr)

    def add_data(self, obs, act, rew, terminated):
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
        obs = np.array(obs, dtype=float)[None, ...]  # (1, C, H, W)
        obs /= 255.0
        act = np.array(act).reshape(1, 1)
        r_t = rew
        if terminated:
            r_t -= self.episode_end_penalty
        self.avg_train_step_return.append(r_t)
        # reshape to batch_size=1
        # flat_input = sa_t.reshape(1, self.da+self.ds)
        r_t = np.array(r_t)
        flat_target = r_t.reshape(1, 1)

        if self.inputs_obs is None:
            self.inputs_obs = obs
            self.inputs_act = act
            self.targets = flat_target
        else:
            self.inputs_obs = np.concatenate([self.inputs_obs, obs], axis=0)
            self.inputs_act = np.concatenate([self.inputs_act, act], axis=0)
            self.targets = np.concatenate([self.targets, flat_target], axis=0)

        if self.inputs_obs.shape[0] > self.max_size:  # NOTE: in PEBBLE, max_size = #traj, so the total timesteps should be max_size*len(traj_i)
            self.inputs_obs = self.inputs_obs[1:, ...]
            self.inputs_act = self.inputs_act[1:, ...]
            self.targets = self.targets[1:, ...]

    def add_data_batch(self, obses, rewards):
        raise NotImplementedError
        num_env = obses.shape[0]
        for index in range(num_env):
            self.inputs.append(obses[index])
            self.targets.append(rewards[index])
        
    def get_rank_probability(self, x_1, x_2):
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
        probs = np.array(probs)
        return np.mean(probs, axis=0), np.std(probs, axis=0)

    def p_hat_member(self, x_1, x_2, member):
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

    def r_hat_member(self, x, member):
        # the network parameterizes r hat in eqn 1 from the paper
        obs, act = x[0], x[1]
        if act.ndim == 3:
            B, len_seg = act.shape[0], act.shape[1]
            obs = obs.reshape(B*len_seg, *self.obs_shape)
            act = act.reshape(B*len_seg, 1)
        else:
            B = len_seg = None
            assert act.ndim == 2
        assert obs.max() <= 1.0
        input = concat_sa(obs, act)  # shape: (B(*len_seg), frame_stack, H, W)
        output = self.ensemble[member](torch.from_numpy(input).float().to(self.device))  # (B, |A|) for atari
        if len(act.shape) < len(obs.shape):  # Atari
            res = torch.gather(output, dim=1, index=torch.from_numpy(act.astype(np.int64)).to(self.device))
            if B is not None:
                res = res.reshape(B, len_seg, 1)
        else:
            res = output  # shape (B, 1)
        return res

    def r_hat(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        assert np.isscalar(x[1])
        obs, act = np.array(x[0], dtype=float)[None, ...], np.array([x[1]])[None, ...]
        obs /= 255.0
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member((obs, act), member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)
    
    def r_hat_batch(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        obs, act = np.array(x[0], dtype=float), np.array(x[1])
        obs /= 255.0
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member((obs, act), member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)

        return np.mean(r_hats, axis=0)
    
    def save(self, model_dir, step):
        for member in range(self.de):
            torch.save(
                self.ensemble[member].state_dict(), '%s/reward_model_%s_%s.pt' % (model_dir, step, member)
            )
    
    def load(self, model_dir, step):
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load('%s/reward_model_%s_%s.pt' % (model_dir, step, member))
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
    
    def get_queries(self, query_batch):  # TODO: consider whether should reshape data t0 query_batch*size_segment*...
        # max_len: total number of collected trajectories
        # len_traj: length of each trajectory # TODO: does every trajectory has the same len_traj?
        # TODO: consider variable-length trajs for Atari
        # len_traj, max_len = len(self.inputs[0]), len(self.inputs)
        img_t_1, img_t_2 = None, None
        
        # if len(self.inputs[-1]) < len_traj:
        #     max_len = max_len - 1
        
        # get train traj
        # train_inputs = np.array(self.inputs[:max_len])
        # train_targets = np.array(self.targets[:max_len])
        # NOTE: query_batch*size_segment
        max_index = len(self.inputs_obs) - self.size_segment 
        # Batch = query_batch
        batch_index_2 = np.random.choice(max_index, size=query_batch, replace=True).reshape(-1, 1)  # (query_batch, 1)
        take_index_2 = batch_index_2 + np.arange(0, self.size_segment).reshape(1, -1)  # shape: (query_batch, size_segment)
        obs_t_2 = np.take(self.inputs_obs, take_index_2, axis=0) # (query_batch, size_segment, obs_shape)
        act_t_2 = np.take(self.inputs_act, take_index_2, axis=0) # (query_batch, size_segment, 1)
        r_t_2 = np.take(self.targets, take_index_2, axis=0) # (query_batch, size_segment, 1)

        batch_index_1 = np.random.choice(max_index, size=query_batch, replace=True).reshape(-1, 1)  # (query_batch, 1)
        take_index_1 = batch_index_1 + np.arange(0, self.size_segment).reshape(1, -1)  # shape: (query_batch, size_segment)
        obs_t_1 = np.take(self.inputs_obs, take_index_1, axis=0) # (query_batch, size_segment, obs_shape)
        act_t_1 = np.take(self.inputs_act, take_index_1, axis=0) # (query_batch, size_segment, 1)
        r_t_1 = np.take(self.targets, take_index_1, axis=0) # (query_batch, size_segment, 1)
        # obs_t.shape: (B, size_segment, frame_stack, H, W), act_t.shape: (query_batch, size_segment, 1)
        sa_t_1 = (obs_t_1, act_t_1)
        sa_t_2 = (obs_t_2, act_t_2)
        return sa_t_1, sa_t_2, r_t_1, r_t_2

    def put_queries(self, sa_t_1, sa_t_2, labels):
        total_sample = sa_t_1[0].shape[0]
        next_index = self.buffer_index + total_sample  # new index in the query buffer after adding new queries
        if next_index >= self.label_capacity:
            self.buffer_full = True
            maximum_index = self.label_capacity - self.buffer_index
            np.copyto(self.buffer_seg1_obs[self.buffer_index:self.label_capacity], sa_t_1[0][:maximum_index])
            np.copyto(self.buffer_seg1_act[self.buffer_index:self.label_capacity], sa_t_1[1][:maximum_index])
            np.copyto(self.buffer_seg2_obs[self.buffer_index:self.label_capacity], sa_t_2[0][:maximum_index])
            np.copyto(self.buffer_seg2_act[self.buffer_index:self.label_capacity], sa_t_2[1][:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.label_capacity], labels[:maximum_index])

            remain = total_sample - (maximum_index)
            if remain > 0:  # if next_index exceed capacity, extra new queries will be added to the beginning of query buffer
                np.copyto(self.buffer_seg1_obs[0:remain], sa_t_1[0][maximum_index:])
                np.copyto(self.buffer_seg1_act[0:remain], sa_t_1[1][maximum_index:])
                np.copyto(self.buffer_seg2_obs[0:remain], sa_t_2[0][maximum_index:])
                np.copyto(self.buffer_seg2_act[0:remain], sa_t_2[1][maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])

            self.buffer_index = remain
        else:
            np.copyto(self.buffer_seg1_obs[self.buffer_index:next_index], sa_t_1[0])
            np.copyto(self.buffer_seg1_act[self.buffer_index:next_index], sa_t_1[1])
            np.copyto(self.buffer_seg2_obs[self.buffer_index:next_index], sa_t_2[0])
            np.copyto(self.buffer_seg2_act[self.buffer_index:next_index], sa_t_2[1])
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            self.buffer_index = next_index
    
    def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2):
        # r_t.shape: (B, size_segment, 1)
        # sa_t.shape: 0: (B, size_segment, frame_stack, H, W), 1: (B, size_segment, 1)
        sum_r_t_1 = np.sum(r_t_1, axis=1)  # shape: (query_batch, 1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)  # shape: (query_batch, 1)
        
        # skip the queries that do not show significant difference between segment pairs
        if self.teacher_thres_skip > 0:
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
            sum_r_t_1 = np.sum(r_t_1, axis=1)  # shape: (#labels,)
            sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # equally preferable
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_thres_equal).reshape(-1)
        
        # perfectly rational
        seg_size = r_t_1.shape[1]
        temp_r_t_1 = r_t_1.copy()
        temp_r_t_2 = r_t_2.copy()
        for index in range(seg_size-1):
            temp_r_t_1[:,:index+1] *= self.teacher_gamma
            temp_r_t_2[:,:index+1] *= self.teacher_gamma
        sum_r_t_1 = np.sum(temp_r_t_1, axis=1)
        sum_r_t_2 = np.sum(temp_r_t_2, axis=1)
        
        rational_labels = 1*(sum_r_t_1 < sum_r_t_2)  # shape (B, 1)
        if self.teacher_beta > 0: # Bradley-Terry rational model
            r_hat = torch.cat([torch.Tensor(sum_r_t_1),
                               torch.Tensor(sum_r_t_2)], axis=-1)  # (# selected queries, 2)
            r_hat = r_hat*self.teacher_beta
            ent = F.softmax(r_hat, dim=-1)[:, 1]  # ent.shape: (# selected queries), means P(seg_2 > seg_1)
            labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1)  # this distribution will return 1 given input probability
        else:
            labels = rational_labels
        
        # making a mistake
        len_labels = labels.shape[0]
        self.label_count += len_labels
        rand_num = np.random.rand(len_labels)
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
    
    def train_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.label_capacity if self.buffer_full else self.buffer_index
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
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)  # shape (B, 2)

                # compute loss
                curr_loss = self.CEloss(r_hat, labels)
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
    
    def train_soft_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.label_capacity if self.buffer_full else self.buffer_index
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
