import numpy as np
import hydra
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from old_utils.agent.sac import SACAgent
from old_utils.utils import mlp, to_np, soft_update_params

from .encoder import cnn_mlp
from ..model_utils import weight_init


class DiscreteActor(nn.Module):
    def __init__(self, init_type, obs_type,
                 model_cfg):
        super().__init__()
        self.obs_shape = model_cfg['obs_shape']
        self.action_dim = model_cfg['action_dim']
        self.obs_type = obs_type
        if self.obs_type == 'rom':
            self.pi = mlp(input_dim=np.prod(model_cfg['obs_shape']),
                          hidden_dim=model_cfg.mlp_cfg['hidden_dim'],
                          output_dim=model_cfg['action_dim'],
                          hidden_depth=model_cfg.mlp_cfg['hidden_depth'],
                          output_mod=nn.Softmax(dim=-1))
        elif self.obs_type == 'img':
            self.pi = cnn_mlp(obs_shape=model_cfg['obs_shape'],
                              output_dim=model_cfg['action_dim'],
                              cnn_cfg=model_cfg['cnn_cfg'],
                              mlp_cfg=model_cfg['mlp_cfg'],
                              output_mod=nn.Softmax(dim=-1))
        else:
            raise NotImplementedError

        self.outputs = dict()
        self.apply(partial(weight_init, init_type=init_type))

    def forward(self, obs):
        """ Code is modified from Deep-Reinforcement-Learning_algorithms-with-Pytorch.SACDiscrete"""
        # obs.shape: (B, frame_stack, H, W)
        assert torch.max(obs).item() <= 1.0
        assert obs.ndim == 4
        action_probabilities = self.pi(obs)
        self.outputs['action_probabilities'] = action_probabilities
        # max_probability_action = torch.argmax(action_probabilities, dim=-1)
        assert action_probabilities.size()[-1] == self.action_dim, "Actor output the wrong size"
        return action_probabilities
    
    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)
        
        for i, m in enumerate(self.pi):  # TODO: check what are saved in log_param
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)


class DiscreteDoubleQCritic(nn.Module):
    def __init__(self, init_type, obs_type,
                 model_cfg):
                #  obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.obs_type = obs_type
        if self.obs_type == 'rom':
            self.Q1 = mlp(input_dim=np.prod(model_cfg['obs_shape']),
                          hidden_dim=model_cfg.mlp_cfg['hidden_dim'],
                          output_dim=model_cfg['action_dim'],
                          hidden_depth=model_cfg.mlp_cfg['hidden_depth'])
            self.Q2 = mlp(input_dim=np.prod(model_cfg['obs_shape']),
                          hidden_dim=model_cfg.mlp_cfg['hidden_dim'],
                          output_dim=model_cfg['action_dim'],
                          hidden_depth=model_cfg.mlp_cfg['hidden_depth'])
        elif self.obs_type == 'img':  # TODO: don't they need to share lower level CNN?
            self.Q1 = cnn_mlp(obs_shape=model_cfg['obs_shape'],
                              output_dim=model_cfg['action_dim'],
                              cnn_cfg=model_cfg['cnn_cfg'],
                              mlp_cfg=model_cfg['mlp_cfg'])
            self.Q2 = cnn_mlp(obs_shape=model_cfg['obs_shape'],
                              output_dim=model_cfg['action_dim'],
                              cnn_cfg=model_cfg['cnn_cfg'],
                              mlp_cfg=model_cfg['mlp_cfg'])
        else:
            raise NotImplementedError

        self.outputs = dict()
        self.apply(partial(weight_init, init_type=init_type))

    def forward(self, obs):
        # obs.shape: (B, frame_stack, H, W); data_range: [0, 1]
        assert torch.max(obs) <= 1.0
        q1 = self.Q1(obs)
        q2 = self.Q2(obs)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2
    
    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)


class SACAgentDiscrete(SACAgent):
    def __init__(self, obs_type, obs_shape, action_dim, device, discount, batch_size,
                 critic_cfg, critic_tau, critic_target_update_frequency, critic_lr, critic_betas,
                 actor_cfg, actor_update_frequency, actor_lr, actor_betas,
                 init_temperature, learnable_temperature, alpha_lr, alpha_betas,
                 ):
        super().__init__(obs_shape=obs_shape, action_dim=action_dim, action_range=None, device=device,
                         critic_cfg=critic_cfg, actor_cfg=actor_cfg, discount=discount, init_temperature=init_temperature,
                         alpha_lr=alpha_lr, alpha_betas=alpha_betas, actor_lr=actor_lr, actor_betas=actor_betas,
                         actor_update_frequency=actor_update_frequency, critic_lr=critic_lr, critic_betas=critic_betas,
                         critic_tau=critic_tau, critic_target_update_frequency=critic_target_update_frequency,
                         batch_size=batch_size, learnable_temperature=learnable_temperature,
                         normalize_state_entropy=None)

        # TODO: check obs_dim for img and rom
        self.obs_type = obs_type
        if obs_type == 'rom':
            assert len(self.obs_shape) == 2
        elif obs_type == 'img':
            assert len(self.obs_shape) == 3
        else:
            raise NotImplementedError

        # used in update_state_ent function, for unsupervised exploration
        # TODO: consider unsupervised exploration later on
        # self.s_ent_stats = utils.TorchRunningMeanStd(shape=[1], device=device)
        # self.normalize_state_entropy = normalize_state_entropy

        # for continuous action space: set target entropy to -|A|
        # -np.log((1.0 / self.action_dim)) * 0.98 is the heuristic used in the original SAC-discrete paper
        # TODO: check this value further. Check how to derive the loss for alpha
        self.target_entropy = -np.log((1.0 / self.action_dim)) * 0.98  # set the max possible entropy as the target entropy
        print(f'******* action_dim:{self.action_dim}, target_entropy: {self.target_entropy} ******')

    def act(self, obs, sample=False, return_dist=False):
        obs = torch.FloatTensor(np.array(obs)/255.0).to(self.device)
        assert obs.ndim == len(self.obs_shape)  # no batch dimension
        obs = obs.unsqueeze(0)
        
        if len(self.obs_shape) == 2:  # ROM: shape (frame_stack, rom_dim)
            action_prob = self.actor(torch.flatten(obs, start_dim=-2, end_dim=-1))
        elif len(self.obs_shape) == 3:  # grey img: shape (frame_stack, H, W)
            action_prob = self.actor(obs)
        else:
            raise NotImplementedError

        if sample:
            action_dist = torch.distributions.Categorical(action_prob)
            action = action_dist.sample()
        else:
            action = torch.argmax(action_prob, dim=-1)
        assert action.ndim == 1 and action.shape[0] == 1
        if return_dist:
            return action.item(), action_prob
        else:
            return action.item()

    def update_critic(self, obs, action, reward, next_obs, 
                      not_done, logger, step, print_flag=True):
        # next_obs.shape: (B, frame_stack, rom_dim) or (B, frame_stack, H, W)
        # next_action_prob.shape: (B, #actions)
        if len(self.obs_shape) == 2:
            next_action_prob  = self.actor(torch.flatten(next_obs, -2, -1))
        elif len(self.obs_shape) == 3:
            next_action_prob  = self.actor(next_obs)
        else:
            raise NotImplementedError

        # next_action_dist = torch.distributions.Categorical(next_action_prob)
        # next_action = next_action_dist.sample()  # next_action.shape: (B, 1)

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = next_action_prob == 0.0
        z = z.float() * 1e-8
        log_prob = torch.log(next_action_prob + z)  # log_prob.shape: (B, #actions)

        target_Q1, target_Q2 = self.critic_target(next_obs)  # Q.shape: (B, #actions)
        min_target_Q_entropy = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_prob  # min_target_Q_entropy.shape: (B, #actions)
        target_V = torch.einsum('ba,ba->b', next_action_prob, min_target_Q_entropy).unsqueeze(-1)  # target_V.shape: (B, 1)
        target_Q = reward + (not_done * self.discount * target_V)  # target_Q.shape: (B, 1)
        target_Q = target_Q.detach()  # NOTE: important!!! target_Q should not back propagate gradients!

        current_Q1_all, current_Q2_all = self.critic(obs)  # Q_all.shape: (B, #action)
        current_Q1 = current_Q1_all.gather(1, action.long())  # Q.shape: (B, 1)
        current_Q2 = current_Q2_all.gather(1, action.long())

        critic_loss = F.mse_loss(current_Q1, target_Q)\
                    + F.mse_loss(current_Q2, target_Q)
        
        if print_flag:
            logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, logger, step, print_flag=False):
        # next_obs.shape: (B, frame_stack, rom_dim) or (B, frame_stack, H, W)
        # action_prob.shape: (B, #action)
        if len(self.obs_shape) == 2:
            action_prob  = self.actor(torch.flatten(obs, -2, -1))
        if len(self.obs_shape) == 3:
            action_prob  = self.actor(obs)
        else:
            raise NotImplementedError

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_prob == 0.0
        z = z.float() * 1e-8
        log_prob = torch.log(action_prob + z)  # log_prob.shape: (B, #actions)

        actor_Q1_all, actor_Q2_all = self.critic(obs)  # Q_all.shape: (B, #action)
        min_actor_Q_all = torch.min(actor_Q1_all, actor_Q2_all)  # Q_all.shape: (B, #action)
        actor_loss = torch.einsum('ba,ba->b', action_prob, (self.alpha.detach() * log_prob - min_actor_Q_all))  # (B)
        actor_loss = actor_loss.mean()  # averaged over batches

        if print_flag:
            logger.log('train_actor/loss', actor_loss, step)
            logger.log('train_actor/target_entropy', self.target_entropy, step)
            logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        if self.learnable_temperature:
            alpha_loss = (self.alpha *
                          (- torch.einsum('ba,ba->b', action_prob, log_prob)
                           - self.target_entropy).detach()  # TODO: check how to set \bar{H}; in Deep-Reinforcement-Learning-Algorithms-with-Pytorch, why do they use log_alpha instead of alpha?
                         ).mean()
            if print_flag:
                logger.log('train_alpha/loss', alpha_loss, step)
                logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update_after_reset(self, replay_buffer, logger, step, gradient_update=1, policy_update=True):
        raise NotImplementedError
    
    def update_state_ent(self, replay_buffer, logger, step, gradient_update=1, K=5):
        raise NotImplementedError
    
    def update_critic_state_ent(
        self, obs, full_obs, action, next_obs, not_done, logger,
        step, K=5, print_flag=True):
        # NOTE: special critic update function for unsupervised exploration
        raise NotImplementedError
    # code that same with SACAgent

    # def train(self, training=True):
    #     self.training = training
    #     self.actor.train(training)
    #     self.critic.train(training)

    # @property
    # def alpha(self):
    #     return self.log_alpha.exp()

    def update(self, replay_buffer, logger, step, gradient_update=1):
        for index in range(gradient_update):
            obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
                self.batch_size)
            # obs/next_obs.shape: (B, frame_stack, H, W)
            assert obs.requires_grad == next_obs.requires_grad == False  # I think the /=255.0 operation should not be included in gradient propagation part
            obs /= 255.0
            next_obs /= 255.0

            print_flag = False
            if index == gradient_update -1:
                logger.log('train/batch_reward', reward.mean(), step)
                print_flag = True
                
            self.update_critic(obs, action, reward, next_obs, not_done_no_max,
                            logger, step, print_flag)

            if step % self.actor_update_frequency == 0:
                self.update_actor_and_alpha(obs, logger, step, print_flag)

        if step % self.critic_target_update_frequency == 0:
            soft_update_params(self.critic, self.critic_target,
                               self.critic_tau)
