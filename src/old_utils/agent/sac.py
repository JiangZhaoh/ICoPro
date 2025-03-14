import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import utils
import hydra
import pdb

from .abc_agent import Agent

def compute_state_entropy(obs, full_obs, k):
    batch_size = 500
    with torch.no_grad():
        dists = []
        for idx in range(len(full_obs) // batch_size + 1):
            start = idx * batch_size
            end = (idx + 1) * batch_size
            dist = torch.norm(
                obs[:, None, :] - full_obs[None, start:end, :], dim=-1, p=2
            )
            dists.append(dist)

        dists = torch.cat(dists, dim=1)
        knn_dists = torch.kthvalue(dists, k=k + 1, dim=1).values
        state_entropy = knn_dists
    return state_entropy.unsqueeze(1)

class SACAgent(Agent):
    """SAC algorithm."""
    def __init__(self, obs_shape, action_dim, action_range, device, critic_cfg,
                 actor_cfg, discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature,
                 normalize_state_entropy=True):
        super().__init__()

        self.device = torch.device(device)
        self.discount = discount
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.action_range = action_range
        self.obs_shape = obs_shape
        self.obs_dim = np.prod(obs_shape)  # TODO: check obs_dim for img and rom

        self.actor_cfg = actor_cfg
        self.actor_update_frequency = actor_update_frequency
        self.actor_lr = actor_lr
        self.actor_betas = actor_betas

        self.critic_cfg = critic_cfg
        self.critic_target_update_frequency = critic_target_update_frequency
        self.critic_tau = critic_tau
        self.critic_lr = critic_lr
        self.critic_betas = critic_betas

        self.init_temperature = init_temperature
        self.learnable_temperature = learnable_temperature
        self.alpha_lr = alpha_lr
        self.alpha_betas = alpha_betas

        self.s_ent_stats = utils.TorchRunningMeanStd(shape=[1], device=device)
        self.normalize_state_entropy = normalize_state_entropy

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)
        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True

        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr,
            betas=actor_betas)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr,
            betas=critic_betas)
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=alpha_lr,
            betas=alpha_betas)
        
        # change mode
        self.train()
        self.critic_target.train()  # here declare train() for critic_target specially because self.train() only declare for self.critic and self.actor 
    
    def reset_critic(self):
        self.critic = hydra.utils.instantiate(self.critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(self.critic_cfg).to(
            self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            betas=self.critic_betas)
    
    def reset_actor(self):
        # reset log_alpha
        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=self.alpha_lr,
            betas=self.alpha_betas)
        
        # reset actor
        self.actor = hydra.utils.instantiate(self.actor_cfg).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.actor_lr,
            betas=self.actor_betas)
        
    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):  # TODO: why we need to use exp to transfer log_alpha to alpha, instead record alpha directly? Maybe to make back propagation easier? 
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs,
                      not_done, logger, step, print_flag=True):
        # next_obs.shape: (B, obs_dim)
        ## To calculate V_{\bar\theta}(s_{t+1}), a_{t+1} will be sampled from \pi_
        dist = self.actor(next_obs)  # dist.scale.shape=dist.loc.shape: (B, action_dim)
        # TODO: check how SquashedNormal.mean() be used
        # TODO: why only sample once to calculate V(s_{t+1})? How about sample more than one time?
        next_action = dist.rsample()  # next_action.shape: (B, action_dim); rsample(): sampled using reparameterisation trick
        # dist.log_prob(next_action).shape: (B, action_dim); log_prob.shape: (B, 1);
        # NOTE: sum log_prob across all action dims to calculates log(pi(a_t|s_t))
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)  # Q.shape: (B, 1)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()  # NOTE: important!!! target_Q should not back propagate gradients!

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        
        if print_flag:
            logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic.log(logger, step)
        
    def update_critic_state_ent(
        self, obs, full_obs, action, next_obs, not_done, logger,
        step, K=5, print_flag=True):
        # NOTE: special critic update function for unsupervised exploration
        
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        
        # compute state entropy
        state_entropy = compute_state_entropy(obs, full_obs, k=K)
        if print_flag:
            logger.log("train_critic/entropy", state_entropy.mean(), step)
            logger.log("train_critic/entropy_max", state_entropy.max(), step)
            logger.log("train_critic/entropy_min", state_entropy.min(), step)
        
        self.s_ent_stats.update(state_entropy)
        norm_state_entropy = state_entropy / self.s_ent_stats.std
        
        if print_flag:
            logger.log("train_critic/norm_entropy", norm_state_entropy.mean(), step)
            logger.log("train_critic/norm_entropy_max", norm_state_entropy.max(), step)
            logger.log("train_critic/norm_entropy_min", norm_state_entropy.min(), step)
        
        if self.normalize_state_entropy:
            state_entropy = norm_state_entropy
        
        target_Q = state_entropy + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        
        if print_flag:
            logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic_target.state_dict(), '%s/critic_target_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step, device):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step), map_location=device)
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step), map_location=device)
        )
        self.critic_target.load_state_dict(
            torch.load('%s/critic_target_%s.pt' % (model_dir, step), map_location=device)
        )

    def update_actor_and_alpha(self, obs, logger, step, print_flag=False):
        # obs.shape: (B, obs_dim)
        dist = self.actor(obs)  # dist.scale.shape=dist.loc.shape: (B, action_dim)
        action = dist.rsample()  # action.shape: (B, action_dim)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)  # log_prob.shape: (B, 1)
        actor_Q1, actor_Q2 = self.critic(obs, action)  # Q.shape: (B, 1)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        # NOTE: here self.alpha use detach(), do not update self.alpha here
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
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
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()  
            if print_flag:
                logger.log('train_alpha/loss', alpha_loss, step)
                logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, logger, step, gradient_update=1):
        for index in range(gradient_update):
            obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
                self.batch_size)

            print_flag = False
            if index == gradient_update -1:
                logger.log('train/batch_reward', reward.mean(), step)
                print_flag = True
                
            self.update_critic(obs, action, reward, next_obs, not_done_no_max,
                               logger, step, print_flag)

            if step % self.actor_update_frequency == 0:
                self.update_actor_and_alpha(obs, logger, step, print_flag)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)
            
    def update_after_reset(self, replay_buffer, logger, step, gradient_update=1, policy_update=True):
        for index in range(gradient_update):
            obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
                self.batch_size)

            print_flag = False
            if index == gradient_update -1:
                logger.log('train/batch_reward', reward.mean(), step)
                print_flag = True
                
            self.update_critic(obs, action, reward, next_obs, not_done_no_max,
                               logger, step, print_flag)

            if index % self.actor_update_frequency == 0 and policy_update:
                self.update_actor_and_alpha(obs, logger, step, print_flag)

            if index % self.critic_target_update_frequency == 0:
                utils.soft_update_params(self.critic, self.critic_target,
                                         self.critic_tau)
            
    def update_state_ent(self, replay_buffer, logger, step, gradient_update=1, K=5):
        # For unsupervised exploration
        for index in range(gradient_update):
            obs, full_obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample_state_ent(
                self.batch_size)

            print_flag = False
            if index == gradient_update -1:
                logger.log('train/batch_reward', reward.mean(), step)
                print_flag = True
                
            self.update_critic_state_ent(
                obs, full_obs, action, next_obs, not_done_no_max,
                logger, step, K=K, print_flag=print_flag)

            if step % self.actor_update_frequency == 0:
                self.update_actor_and_alpha(obs, logger, step, print_flag)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)