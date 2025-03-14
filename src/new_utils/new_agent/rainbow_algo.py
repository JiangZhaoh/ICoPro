import torch
import pdb
from collections import namedtuple

from rlpyt.utils.collections import namedarraytuple

from new_utils.new_agent.rainbow_replay_buffer import PrioritizedReplayFrameBuffer, UniformReplayFrameBuffer
from new_utils.tensor_utils import select_at_indexes

EPS = 1e-6  # (NaN-guard)
# ModelSamplesToBuffer = namedarraytuple("SamplesToBuffer",
#     ["observation", "action", "reward", "done", "value"])
OptInfo = namedtuple("OptInfo", ["loss", "gradNorm", "tdAbsErr"])
SamplesToBuffer = namedarraytuple("SamplesToBuffer",
    ["observation", "action", "reward", "done"])
SamplesToBuffer2R = namedarraytuple("SamplesToBuffer2R",
    ["observation", "action", "reward_GT", "reward_hat", "done"])

class CategoricalDQN:
    """
    Distributional DQN with fixed probability bins for the Q-value of each
    action, a.k.a. categorical.
    """
    def __init__(
            self,
            # From SPRCategoricalDQN
            distributional,
            # From DQN
            discount,
            batch_size,  # batch size to train algo (not necessarily equal to sampler_t * sampler_b)
            min_steps_learn,
            replay_size,
            replay_ratio,  # data_consumption / data_generation.
            target_update_tau,
            target_update_interval,  # 312 * 32 = 1e4 env steps.
            n_step_return,
            learning_rate,
            OptimCls,
            optim_kwargs,
            initial_optim_state_dict,
            clip_grad_norm,
            eps_steps,  # STILL IN ALGO (to convert to itr).
            double_dqn,
            prioritized_replay,
            pri_alpha,
            pri_beta_init,
            pri_beta_final,
            pri_beta_steps,
            replay_relabel_batch_size,
            obs_shape,
            delta_clip=1.,
            default_priority=None,
            # ReplayBufferCls=None,  # Leave None to select by above options.
            # updates_per_sync,  # For async mode only.
            # From CategoricalDQN
            V_min=-10,
            V_max=10,
            r_hat_GT_coef=None,
            use_potential=None,
    ):
        # TODO: log statistics during training; e.g. opt_info_fields in spr.algo
        # From DQN
        if optim_kwargs is None:
            optim_kwargs = dict(eps=0.01 / batch_size)
        if default_priority is None:
            default_priority = delta_clip
        self._batch_size = batch_size
        del batch_size  # Property.
        self.replay_relabel_batch_size = replay_relabel_batch_size
        self.obs_shape = obs_shape
        
        # save__init__args(locals())  # TODO: define self.xxx
        self.discount = discount
        self.replay_size = replay_size
        self.replay_ratio = replay_ratio
        self.target_update_tau = target_update_tau
        self.target_update_interval = target_update_interval
        self.n_step_return = n_step_return

        self.learning_rate = learning_rate
        self.OptimCls = OptimCls
        self.optim_kwargs = optim_kwargs
        self.initial_optim_state_dict = initial_optim_state_dict
        self.clip_grad_norm = clip_grad_norm

        self.min_steps_learn = min_steps_learn
        self.eps_steps = eps_steps

        self.double_dqn = double_dqn
        self.prioritized_replay = prioritized_replay
        self.pri_alpha = pri_alpha
        self.pri_beta_init = pri_beta_init
        self.pri_beta_final = pri_beta_final
        self.pri_beta_steps = pri_beta_steps
        self.default_priority = default_priority

        self.update_counter = 0
        # From CategoricalDQN
        self.V_min = V_min
        self.V_max = V_max
        assert abs(self.V_max - abs(self.V_min)) <= EPS

        self.distributional = distributional
        if self.distributional:
            assert abs(V_min) == V_max == 10  # since in model.py.from_categorical(), limit=10
        if "eps" not in self.optim_kwargs:  # Assume optim.Adam
            self.optim_kwargs["eps"] = 0.01 / self.batch_size

        # From SPR.algo
        if not distributional:
            self.rl_loss = self.dqn_rl_loss
        else:
            self.rl_loss = self.dist_rl_loss

        self.r_hat_GT_coef = r_hat_GT_coef  # if not None, r = GT + coef*r_hat
        self.use_potential = use_potential

    """ From DQN """
    def initialize(self,
                    agent,
                    n_itr,
                    batch_spec,  # batch_spec.size
                    mid_batch_reset,
                    examples,
                    # world_size,
                    rank,
                    remove_frame_axis=False,
                    ):
        # From DQN.initialize()
        self.agent = agent
        self.n_itr = n_itr
        self.sampler_bs = batch_spec.size
        self.mid_batch_reset = mid_batch_reset
        self.updates_per_optimize = max(1, round(self.replay_ratio * self.sampler_bs /
            self.batch_size))
        print(f"------ From sampler batch size {self.sampler_bs}, training "
            f"batch size {self.batch_size}, and replay ratio "
            f"{self.replay_ratio}, computed {self.updates_per_optimize} "
            f"updates per iteration. ------")
        self.min_itr_learn = int(self.min_steps_learn // self.sampler_bs)
        eps_itr_max = max(1, int(self.eps_steps // self.sampler_bs))
        agent.set_epsilon_itr_min_max(self.min_itr_learn, eps_itr_max)
        self.initialize_replay_buffer(examples=examples,
                                      batch_spec=batch_spec,
                                      remove_frame_axis=remove_frame_axis)
        self.optim_initialize(rank)
        # From CategoricalDQN.initialize()
        self.agent.give_V_min_max(self.V_min, self.V_max)

    def initialize_replay_buffer(self, examples, batch_spec,
                                async_=False, remove_frame_axis=False):
        """
        From rlpyt
        Allocates replay buffer using examples and with the fields in `SamplesToBuffer`
        namedarraytuple.  Uses frame-wise buffers, so that only unique frames are stored,
        using less memory (usual observations are 4 most recent frames, with only newest
        frame distince from previous observation).
        """
        # example_to_buffer = self.examples_to_buffer(examples)
        if self.r_hat_GT_coef is None:
            example_to_buffer = SamplesToBuffer(
                observation=examples["observation"],
                action=examples["action"],
                reward=examples["reward"],
                done=examples["done"],
            )
        else:
            example_to_buffer = SamplesToBuffer2R(
                observation=examples["observation"],
                action=examples["action"],
                reward_GT=examples["reward"],
                reward_hat=examples["reward"],  # NOTE: modifying example_to_buffer.reward_GT will affect example_to_buffer.reward_hat
                done=examples["done"],
            )
        replay_kwargs = dict(
            example=example_to_buffer,
            size=self.replay_size,
            B=batch_spec.B,
            discount=self.discount,
            n_step_return=self.n_step_return,
            relabel_batch_size=self.replay_relabel_batch_size,
            obs_shape=self.obs_shape,
            r_hat_GT_coef=self.r_hat_GT_coef,
            use_potential=self.use_potential,
            remove_frame_axis=remove_frame_axis,
        )
        if self.prioritized_replay:
            replay_kwargs.update(dict(
                alpha=self.pri_alpha,
                beta=self.pri_beta_init,
                default_priority=self.default_priority,
            ))
            ReplayCls = PrioritizedReplayFrameBuffer
        else:
            ReplayCls = UniformReplayFrameBuffer
        
        self.replay_buffer = ReplayCls(**replay_kwargs)
    
    def optim_initialize(self, rank=0):
        """Called in initilize or by async runner after forking sampler."""
        self.rank = rank
        # From SPR.algo
        self.optimizer = eval(self.OptimCls)(self.agent.model.parameters(),
            lr=self.learning_rate, **self.optim_kwargs)
        self.model = self.agent.model
        if self.initial_optim_state_dict is not None:
            self.optimizer.load_state_dict(self.initial_optim_state_dict)
        if self.prioritized_replay:
            self.pri_beta_itr = max(1, self.pri_beta_steps // self.sampler_bs)
    
    def samples_to_buffer(self, samples, use_r_hat):
        """Defines how to add data from sampler into the replay buffer. Called
        in optimize_agent() if samples are provided to that method.  In
        asynchronous mode, will be called in the memory_copier process."""
        if self.r_hat_GT_coef is None:
            return SamplesToBuffer(
                observation=samples.env.observation,
                action=samples.agent.action,
                reward=samples.r_hat if use_r_hat else samples.env.reward,
                done=samples.env.done,
                # value=samples.agent.agent_info.p,  # in SPR, they use 'agent_info.p' to save 'value' (i.e. Q(s,a)) ...
            )
        else:
            assert use_r_hat
            return SamplesToBuffer2R(
                observation=samples.env.observation,
                action=samples.agent.action,
                reward_GT=samples.env.reward,
                reward_hat=samples.r_hat,
                done=samples.env.done,
                # value=samples.agent.agent_info.p,  # in SPR, they use 'agent_info.p' to save 'value' (i.e. Q(s,a)) ...
            )
    
    # def optimize_agent(self, itr, samples=None):
    def optimize_agent(self, itr):
        """
        Extracts the needed fields from input samples and stores them in the 
        replay buffer.  Then samples from the replay buffer to train the agent
        by gradient updates (with the number of updates determined by replay
        ratio, sampler batch size, and training batch size).  If using prioritized
        replay, updates the priorities for sampled training batches.
        """
        # Move replay_buffer part to workspace.run
        # if samples is not None:
        #     samples_to_buffer = self.samples_to_buffer(samples)
        #     self.replay_buffer.append_samples(samples_to_buffer)
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        if itr < self.min_itr_learn:
            return opt_info
        for _ in range(self.updates_per_optimize):
            samples_from_replay = self.replay_buffer.sample_batch(self.batch_size)
            self.optimizer.zero_grad()
            loss, td_abs_errors = self.loss(samples_from_replay)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.agent.parameters(), self.clip_grad_norm)  # default l2 norm
            self.optimizer.step()
            if self.prioritized_replay:
                self.replay_buffer.update_batch_priorities(td_abs_errors)
            opt_info.loss.append(loss.item())
            opt_info.gradNorm.append(grad_norm.clone().detach().item())  # following torch's recommendation
            opt_info.tdAbsErr.extend(td_abs_errors[::8].numpy())  # Downsample.
            self.update_counter += 1
            if self.update_counter % self.target_update_interval == 0:
                self.agent.update_target(self.target_update_tau)
        self.update_itr_hyperparams(itr)
        return opt_info
    
    def update_itr_hyperparams(self, itr):
        # EPS NOW IN AGENT.
        # if itr <= self.eps_itr:  # Epsilon can be vector-valued.
        #     prog = min(1, max(0, itr - self.min_itr_learn) /
        #       (self.eps_itr - self.min_itr_learn))
        #     new_eps = prog * self.eps_final + (1 - prog) * self.eps_init
        #     self.agent.set_sample_epsilon_greedy(new_eps)
        if self.prioritized_replay and itr <= self.pri_beta_itr:
            prog = min(1, max(0, itr - self.min_itr_learn) /
                (self.pri_beta_itr - self.min_itr_learn))
            new_beta = (prog * self.pri_beta_final +
                (1 - prog) * self.pri_beta_init)
            self.replay_buffer.set_beta(new_beta)
    
    """ From SPRCategoricalDQN """
    def loss(self, samples):
        if self.model.noisy:  # For noisy net
            self.model.head.reset_noise()
        # log_pred_ps, pred_rew, spr_loss\
        ps = self.agent(  # [B,A,P]
            observation=samples.agent_inputs.observation.to(self.agent.device))

        rl_loss, KL = self.rl_loss(ps, samples)
        if self.prioritized_replay:
            # weights = samples.is_weights
            # RL losses are no longer scaled in the c51 function
            rl_loss = rl_loss * samples.is_weights

        return rl_loss.mean(), KL

    def dist_rl_loss(self, ps, samples):
        """
        Computes the Distributional Q-learning loss, based on projecting the
        discounted rewards + target Q-distribution into the current Q-domain,
        with cross-entropy loss.  

        Returns loss and KL-divergence-errors for use in prioritization.
        """
        delta_z = (self.V_max - self.V_min) / (self.agent.n_atoms - 1)
        z = torch.linspace(self.V_min, self.V_max, self.agent.n_atoms)
        # Makde 2-D tensor of contracted z_domain for each data point,
        # with zeros where next value should not be added.
        next_z = z * (self.discount ** self.n_step_return)  # [P']
        next_z = torch.ger(1 - samples.done_n.float(), next_z)  # [B,P']
        ret = samples.return_.unsqueeze(1)  # [B,1]
        next_z = torch.clamp(ret + next_z, self.V_min, self.V_max)  # [B,P']

        z_bc = z.view(1, -1, 1)  # [1,P,1]
        next_z_bc = next_z.unsqueeze(1)  # [B,1,P']
        abs_diff_on_delta = abs(next_z_bc - z_bc) / delta_z
        projection_coeffs = torch.clamp(1 - abs_diff_on_delta, 0, 1)  # Most 0.
        # projection_coeffs is a 3-D tensor: [B,P,P']
        # dim-0: independent data entries
        # dim-1: base_z atoms (remains after projection)
        # dim-2: next_z atoms (summed in projection)

        with torch.no_grad():
            target_ps = self.agent.target(*samples.target_inputs, train=False)  # [B,A,P']
            if self.double_dqn:
                next_ps = self.agent(*samples.target_inputs)  # [B,A,P']
                next_qs = torch.tensordot(next_ps, z, dims=1)  # [B,A]
                next_a = torch.argmax(next_qs, dim=-1)  # [B]
            else:
                target_qs = torch.tensordot(target_ps, z, dims=1)  # [B,A]
                next_a = torch.argmax(target_qs, dim=-1)  # [B]
            target_p_unproj = select_at_indexes(next_a, target_ps)  # [B,P']
            target_p_unproj = target_p_unproj.unsqueeze(1)  # [B,1,P']
            target_p = (target_p_unproj * projection_coeffs).sum(-1)  # [B,P]
        # ps = self.agent(*samples.agent_inputs)  # [B,A,P], obtained from self.loss()
        p = select_at_indexes(samples.action, ps)  # [B,P]
        p = torch.clamp(p, EPS, 1)  # NaN-guard.
        losses = -torch.sum(target_p * torch.log(p), dim=1)  # Cross-entropy.

        if self.prioritized_replay:
            losses *= samples.is_weights

        target_p = torch.clamp(target_p, EPS, 1)
        KL_div = torch.sum(target_p *
            (torch.log(target_p) - torch.log(p.detach())), dim=1)
        KL_div = torch.clamp(KL_div, EPS, 1 / EPS)  # Avoid <0 from NaN-guard.

        if not self.mid_batch_reset:
            valid = valid_from_done(samples.done)
            loss = valid_mean(losses, valid)
            KL_div *= valid
        else:
            loss = torch.mean(losses)

        return loss, KL_div

    def dqn_rl_loss(self, qs, samples):
        """
        From rlpyt.algos.dqn.loss() & SPR.algo.dqn_rl_loss()
        Computes the Q-learning loss, based on: 0.5 * (Q - target_Q) ^ 2.
        Implements regular DQN or Double-DQN for computing target_Q values
        using the agent's target network.  Computes the Huber loss using 
        ``delta_clip``, or if ``None``, uses MSE.  When using prioritized
        replay, multiplies losses by importance sample weights.

        Input ``samples`` have leading batch dimension [B,..] (but not time).

        Calls the agent to compute forward pass on training inputs, and calls
        ``agent.target()`` to compute target values.

        Returns loss and TD-absolute-errors for use in prioritization.

        Warning: 
            If not using mid_batch_reset, the sampler will only reset environments
            between iterations, so some samples in the replay buffer will be
            invalid.  This case is not supported here currently.
        """
        # qs = self.agent(*samples.agent_inputs)
        q = select_at_indexes(samples.action, qs).cpu()
        with torch.no_grad():
            target_qs = self.agent.target(*samples.target_inputs)
            if self.double_dqn:
                next_qs = self.agent(*samples.target_inputs)
                next_a = torch.argmax(next_qs, dim=-1)
                target_q = select_at_indexes(next_a, target_qs)
            else:
                target_q = torch.max(target_qs, dim=-1).values

            disc_target_q = (self.discount ** self.n_step_return) * target_q
            y = samples.return_ + (1 - samples.done_n.float()) * disc_target_q

        delta = y - q
        losses = 0.5 * delta ** 2
        abs_delta = abs(delta)
        if self.delta_clip > 0:  # Huber loss.
            b = self.delta_clip * (abs_delta - self.delta_clip / 2)
            losses = torch.where(abs_delta <= self.delta_clip, losses, b)
        # if self.prioritized_replay:
        #     losses *= samples.is_weights
        td_abs_errors = abs_delta.detach()
        if self.delta_clip is not None:
            td_abs_errors = torch.clamp(td_abs_errors, 0, self.delta_clip)
        if not self.mid_batch_reset:  # self.mid_batch_reset==False
            # FIXME: I think this is wrong, because the first "done" sample
            # is valid, but here there is no [T] dim, so there's no way to
            # know if a "done" sample is the first "done" in the sequence.
            raise NotImplementedError
            # valid = valid_from_done(samples.done)
            # loss = valid_mean(losses, valid)
            # td_abs_errors *= valid
        else:
            loss = torch.mean(losses)

        return loss, td_abs_errors
    """ From RlAlgorithm """
    def optim_state_dict(self):
        """Return the optimizer state dict (e.g. Adam); overwrite if using
        multiple optimizers."""
        return self.optimizer.state_dict()

    def load_optim_state_dict(self, state_dict):
        """Load an optimizer state dict; should expect the format returned
        from ``optim_state_dict().``"""
        self.optimizer.load_state_dict(state_dict)

    @property
    def batch_size(self):
        return self._batch_size  # For logging at least.