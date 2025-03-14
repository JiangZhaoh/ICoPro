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

class CorrectiveRewardModel:
    def __init__(self,
                 encoder_cfg,
                #  episode_end_penalty,  # TODO: for corrective feedback, how to deal with episode end?
                 ensemble_size=3,
                 reward_lr=3e-4,
                 query_batch = 128,  # old name in PEBBLE is mb_size
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
                 init_type='orthogonal'):
        self.device = device

        self.encoder_cfg = copy.deepcopy(encoder_cfg)
        # self.episode_end_penalty = episode_end_penalty
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

        self.neighbor_size = neighbor_size
        self.cf_per_seg = cf_per_seg
        self.loss_margine = loss_margine

        self.label_capacity = int(label_capacity)  # "labeled" query buffer capacity
        self.label_count = 0

        self.buffer_obs = np.empty((self.label_capacity, *self.obs_shape), dtype=np.float32)
        self.buffer_act = np.empty((self.label_capacity, 1), dtype=np.int32)
        self.buffer_oracle_act = np.empty((self.label_capacity, 1), dtype=np.int32)
        # self.buffer_oracle_act_prob = np.empty((self.label_capacity, self.action_dim), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False  # query buffer full
        
        self.construct_ensemble(init_type=init_type)
        self.inputs_obs = None  # all available trajectories
        self.inputs_act = None
        self.targets_oracle_act = None
        self.targets_oracle_act_prob = None
        self.raw_actions = []
        self.img_inputs = []
        self.query_batch = query_batch  # reward batch size
        self.origin_query_batch = query_batch  # reward batch size may change according to the current training step
        self.train_batch_size = train_batch_size
        self.running_means = []
        self.running_stds = []
        self.best_seg = []
        self.best_label = []
        self.best_action = []

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
    
    def change_batch(self, new_frac):
        self.query_batch = int(self.origin_query_batch*new_frac)
    
    def set_batch(self, new_batch):
        self.query_batch = int(new_batch)
    
    def construct_ensemble(self, init_type):
        if self.activation == 'tanh':
            output_mod = nn.Tanh()
            self.min_reward = 0.0
        elif self.activation == 'sig':
            output_mod = nn.Sigmoid()
            self.min_reward = -1.0
        elif self.activation == 'relu':
            output_mod = nn.ReLU()
            self.min_reward = 0.0
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

    def add_data(self, obs, act, oracle_act, oracle_act_prob, terminated):
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
        pdb.set_trace()  # TODO: check oracle_Act & oracle_act_prob.shape, dtype
        obs = np.array(obs, dtype=float)[None, ...]  # (1, C, H, W)
        obs /= 255.0
        act = np.array(act).reshape(1, 1)
        oracle_act = np.array(oracle_act).reshape(1, 1)
        oracle_act_prob = np.array(oracle_act_prob).reshape(1, self.action_dim)
        # TODO: consider more details about end of episodes
        self.avg_train_step_return.append(r_t)
        # reshape to batch_size=1
        # flat_input = sa_t.reshape(1, self.da+self.ds)
        r_t = np.array(r_t)
        flat_target = r_t.reshape(1, 1)

        if self.inputs_obs is None:
            self.inputs_obs = obs
            self.inputs_act = act
            self.targets_oracle_act = oracle_act
            self.targets_oracle_act_prob = oracle_act_prob
        else:
            self.inputs_obs = np.concatenate([self.inputs_obs, obs], axis=0)
            self.inputs_act = np.concatenate([self.inputs_act, act], axis=0)
            self.targets_oracle_act = np.concatenate([self.targets_oracle_act, oracle_act], axis=0)
            self.targets_oracle_act_prob = np.concatenate([self.targets_oracle_act_prob, oracle_act_prob], axis=0)

        if self.inputs_obs.shape[0] > self.max_size:  # NOTE: in PEBBLE, max_size = #traj, so the total timesteps should be max_size*len(traj_i)
            self.inputs_obs = self.inputs_obs[1:, ...]
            self.inputs_act = self.inputs_act[1:, ...]
            self.targets_oracle_act = self.targets_oracle_act[1:, ...]
            self.targets_oracle_act_prob = self.targets_oracle_act_prob[1:, ...]

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

    def r_hat_member(self, obs, member):
        # the network parameterizes r hat in eqn 1 from the paper
        pdb.set_trace()  # check obs.shape
        assert obs.ndim == 4  # (B, C*frame_stack, H, W)
        assert obs.max() <= 1.0
        r_hat_s = self.ensemble[member](torch.from_numpy(obs).float().to(self.device))  # (B, |A|) for atari
        assert r_hat_s.shape == (obs.shape[0], self.action_dim)
        return r_hat_s  # shape: (B, self.action_dim)

    def r_hat(self, x):
        raise NotImplementedError
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
    
    def get_queries(self, query_batch):
        # get train traj
        max_index = len(self.inputs_obs) - self.size_segment - (self.neighbor_size // 2) * 2
        # Batch = query_batch
        batch_index = np.random.choice(max_index, size=query_batch, replace=True).reshape(-1, 1)\
                        + self.neighbor_size // 2  # (query_batch, 1)
        take_index = batch_index + np.arange(0, self.size_segment).reshape(1, -1)  # shape: (query_batch, size_segment)
        obs_t = np.take(self.inputs_obs, take_index, axis=0) # (query_batch, size_segment, obs_shape)
        act_t = np.take(self.inputs_act, take_index, axis=0) # (query_batch, size_segment, 1)
        oracle_act_t = np.take(self.targets_oracle_act, take_index, axis=0) # (query_batch, size_segment, 1)
        oracle_act_prob_t = np.take(self.targets_oracle_act_prob, take_index, axis=0) # (query_batch, size_segment, 1)

        return obs_t, act_t, oracle_act_t, oracle_act_prob_t

    def put_queries(self, obs_t, act_t, oracle_act_t):
        pdb.set_trace()  # check shape & dtype
        assert obs_t[0] == self.query_batch * self.cf_per_seg * self.neighbor_size
        # obs_t.shape: (query_batch*cf_per_seg * neighbor_size)
        total_sample = obs_t.shape[0]
        next_index = self.buffer_index + total_sample  # new index in the query buffer after adding new queries
        if next_index >= self.label_capacity:
            self.buffer_full = True
            maximum_index = self.label_capacity - self.buffer_index
            np.copyto(self.buffer_obs[self.buffer_index:self.label_capacity], obs_t[:maximum_index])
            np.copyto(self.buffer_act[self.buffer_index:self.label_capacity], act_t[:maximum_index])
            np.copyto(self.buffer_oracle_act[self.buffer_index:self.label_capacity], oracle_act_t[:maximum_index])

            remain = total_sample - (maximum_index)
            if remain > 0:  # if next_index exceed capacity, extra new queries will be added to the beginning of query buffer
                np.copyto(self.buffer_obs[0:remain], obs_t[maximum_index:])
                np.copyto(self.buffer_act[0:remain], act_t[maximum_index:])
                np.copyto(self.buffer_oracle_act[0:remain], oracle_act_t[maximum_index:])

            self.buffer_index = remain
        else:
            np.copyto(self.buffer_obs[self.buffer_index:next_index], obs_t[0])
            np.copyto(self.buffer_act[self.buffer_index:next_index], act_t[1])
            np.copyto(self.buffer_oracle_act[self.buffer_index:next_index], oracle_act_t[1])
            self.buffer_index = next_index
    
    def get_label(self, obs_t, act_t, oracle_act_t, oracle_act_prob_t):
        # obs_t.shape: (query_batch, size_segment, *obs_shape)
        # act_t & oracle_act_t.shape: (query_batch, size_segment, 1)
        # oracle_act_prob_t.shape: (query_batch, size_segment, action_ndim)
        oracle_act_entropy = (-oracle_act_prob_t * np.log(oracle_act_prob_t)).sum(axis=-1, keepdims=False)  # shape(query_batch, size_segment)
        target_oracle_index = np.argsort(oracle_act_entropy, axis=-1)[:, -self.cf_per_seg:]  # ascending order, shape (query_batch, cf_per_seg)
        pdb.set_trace()  # check entropy, and shape of target_oracle_index
        if self.neighbor_size // 2 > 0:
            target_oracle_index_neighbor = np.tile(target_oracle_index, reps=(1, self.neighbor_size))
            for delt in range(self.neighbor_size//2):
                target_oracle_index_neighbor[:, delt*2*self.cf_per_seg: (delt*2+1)*self.cf_per_seg] -= delt
                target_oracle_index_neighbor[:, (delt*2+1)*self.cf_per_seg: (delt*2+2)*self.cf_per_seg] += delt
                # shape: (query_batch, cf_per_seg*neighbor_size)
        
        pdb.set_trace()  # check target_oracle_act.shape: (query_batch, cf_per_seg, 1), target_oracle_act_prob.shape(query_batch, cf_per_seg, action_ndim)
        
        # obs_t.shape: (query_batch*cf_per_seg*neighbor_size, *obs_shape)
        obs_t = np.take(obs_t, target_oracle_index_neighbor, axis=0).\
                    reshape(self.query_batch * self.cf_per_seg * self.neighbor_size, *self.obs_shape)  
        # act_t.shape: (query_batch, cf_per_seg*neighbor_size, 1)
        act_t = np.take(act_t, target_oracle_index_neighbor, axis=0).\
                    reshape(self.query_batch * self.cf_per_seg * self.neighbor_size, 1)
        # oracle_act_t.shape: (query_batch, cf_per_seg*neighbor_size, 1)
        oracle_act_t = np.take(oracle_act_t, target_oracle_index, axis=0).\
                    tile(reps=(1, self.neighbor_size, 1)).\
                    reshape(self.query_batch * self.cf_per_seg * self.neighbor_size, 1)
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
    
    def uniform_sampling(self):
        # get queries
        obs_t, act_t, oracle_act_t, oracle_act_prob_t =  self.get_queries(
            query_batch=self.query_batch)
        # obs_t.shape: (query_batch, size_segment, *obs_shape)
        
        # get labels
        obs_t, act_t, oracle_act_t = self.get_label(  # filter queries and 
            obs_t, act_t, oracle_act_t, oracle_act_prob_t)
        # obs_t.shape: (query_batch, cf_per_seg*neighbor_size, *obs_shape)
        
        self.put_queries(obs_t, act_t, oracle_act_t)
        
        return obs_t.shape[0]  # query_batch*cf_per_seg*neighbor_size
    
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
        assert r_hat_s.shape[-1] == self.action_ndim
        assert oracle_act.ndim == 2
        assert oracle_act.shape[-1] == 1
        pdb.set_trace()  # TODO: check mask
        act_mask = torch.ones_like(r_hat_s).scatter_(
                        dim=-1, index=oracle_act,
                        src=-torch.inf * torch.ones_like(oracle_act))
        r_hat_s_oa = torch.gather(input=r_hat_s, dim=-1, index=oracle_act)
        # TODO: check max_noa_reward.shape == (B, 1), and its gradient function
        max_noa_r_hat = torch.max(act_mask * r_hat_s, dim=-1)[0] # max none-oracle-action reward
        assert max_noa_r_hat.shape == (r_hat_s.shape[0], 1)
        assert r_hat_s_oa.shape == max_noa_r_hat.shape
        loss = max_noa_r_hat + self.loss_margin - r_hat_s_oa
        return loss.mean()  # a scalar

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
                obs_t = self.buffer_obs[idxs]
                act_t = self.buffer_act[idxs]
                oracle_act_t = self.buffer_oracle_act[idxs]
                r_hat_s_t = self.r_hat_member(obs_t, member=member)
                
                if member == 0:
                    total += oracle_act_t.size(0)
                
                # compute loss
                curr_loss = self.reward_margine_loss(r_hat_s=r_hat_s_t,
                                                     oracle_act=oracle_act_t)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                # compute acc
                max_r_hat_a = torch.max(r_hat_s_t.data, dim=-1)[1]
                pdb.set_trace()  # check max_r_hat_a & oracle_act_t shape; check ==
                assert oracle_act_t.shape == max_r_hat_a.shape == (oracle_act_t.sape[0], 1)
                correct = (max_r_hat_a == oracle_act_t).sum().item()  # count the number of samples that r_hat assign largest value for oracle_actions
                ensemble_acc[member] += correct
                
            loss.backward()
            self.opt.step()
        
        ensemble_acc = ensemble_acc / total
        
        return ensemble_acc
