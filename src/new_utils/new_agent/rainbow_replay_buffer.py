import pdb
import math
import torch
import numpy as np

from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.buffer import buffer_from_example
from rlpyt.replays.sum_tree import SumTree, AsyncSumTree

from new_utils.new_agent.rainbow_agent import AgentInputs
from new_utils.tensor_utils import zeros, torchify_buffer, numpify_buffer, get_leading_dims

EPS = 1e-6

SamplesFromReplay = namedarraytuple("SamplesFromReplay",
    ["agent_inputs", "action", "return_", "done", "done_n", "target_inputs"])
SamplesFromReplayPri = namedarraytuple("SamplesFromReplayPri",
    SamplesFromReplay._fields + ("is_weights",))  # is = importance sampling


def discount_return_n_step(reward, done, n_step, discount, return_dest=None,
        done_n_dest=None, do_truncated=False):
    """
    From rlpyt.algos.utils
    Time-major inputs, optional other dimension: [T], [T,B], etc.  Computes
    n-step discounted returns within the timeframe of the of given rewards. If
    `do_truncated==False`, then only compute at time-steps with full n-step
    future rewards are provided (i.e. not at last n-steps--output shape will
    change!).  Returns n-step returns as well as n-step done signals, which is
    True if `done=True` at any future time before the n-step target bootstrap
    would apply (bootstrap in the algo, not here)."""
    rlen = reward.shape[0]  # rlen=T+n_step-1, T=sampler_t
    if do_truncated == False:
        rlen -= (n_step - 1)  # rlen=T=sampler_t
    # NOTE: if return_dest/done_n_dest != None, they are given buffer to save results
    return_ = return_dest if return_dest is not None else zeros(
        (rlen,) + reward.shape[1:], dtype=reward.dtype)
    done_n = done_n_dest if done_n_dest is not None else zeros(
        (rlen,) + reward.shape[1:], dtype=done.dtype)
    return_[:] = reward[:rlen]  # 1-step return is current reward.
    done_n[:] = done[:rlen]  # True at time t if done any time by t + n - 1
    is_torch = isinstance(done, torch.Tensor)
    if is_torch:
        done_dtype = done.dtype
        done_n = done_n.type(reward.dtype)
        done = done.type(reward.dtype)
    if n_step > 1:
        if do_truncated:  # rlen=T+n_step-1=return_/done_n.length
            for n in range(1, n_step):
                return_[:-n] += (discount ** n) * reward[n:n + rlen] * (1 - done_n[:-n])
                done_n[:-n] = np.maximum(done_n[:-n], done[n:n + rlen])
        else:
            for n in range(1, n_step):  # rlen=T=return_/done_n.length
                return_ += (discount ** n) * reward[n:n + rlen] * (1 - done_n)
                done_n[:] = np.maximum(done_n, done[n:n + rlen])  # Supports tensors.
    if is_torch:
        done_n = done_n.type(done_dtype)
    return return_, done_n


class NStepFrameBuffer:
    """ From rlpyt.BaseReplayBuffer """
    async_ = False

    """Special method for re-assembling observations from frames."""
    def __init__(self,
                example,
                size,
                B,
                discount,
                n_step_return,
                relabel_batch_size,
                obs_shape,
                r_hat_GT_coef,
                use_potential,
                remove_frame_axis=False):
        # From rlpyt.FrameBufferMixin
        field_names = [f for f in example._fields if f != "observation"]
        global BufferSamples
        BufferSamples = namedarraytuple("BufferSamples", field_names)
        buffer_example = BufferSamples(*(v for k, v in example.items()
            if k != "observation"))
        
        # super().__init__(example=buffer_example, **kwargs)
        ## From rlpyt.BaseNStepReturnBuffer.__init__
        self.total_T = total_T = math.ceil(size / B)  # larger than original size
        self.B = B
        self.size = total_T * B
        print(f'[Replay Buffer] total_T: {self.total_T}, B: {self.B}, size: {self.size}')
        self.discount = discount
        self.n_step_return = n_step_return
        self.obs_shape = obs_shape
        self.relabel_batch_size = relabel_batch_size
        assert self.relabel_batch_size % self.B == 0  # make relabel easier
        
        self.r_hat_GT_coef = r_hat_GT_coef
        self.use_potential = use_potential
        assert not (self.use_potential.ahead and self.use_potential.back)  # can only have one flag equal to True
        if self.use_potential.ahead:
            self.pot_coef = use_potential.discount
            assert self.r_hat_GT_coef is not None
        elif self.use_potential.back:
            self.pot_coef = 1.0 / use_potential.discount
            assert self.r_hat_GT_coef is not None
        
        self.t = 0  # Cursor (in T dimension).
        self.samples = buffer_from_example(buffer_example, (total_T, B),
                    share_memory=self.async_)  # initial contents are 0/False
        # NOTE: modifying self.samples.reward_GT will NOT affect self.samples.reward_hat
        if n_step_return > 1:
            if r_hat_GT_coef is None:
                self.samples_return_ = buffer_from_example(buffer_example.reward, (total_T, B),
                    share_memory=self.async_)
            else:
                assert r_hat_GT_coef >= 0.
                self.samples_return_ = buffer_from_example(
                    buffer_example.reward_GT + self.r_hat_GT_coef * buffer_example.reward_hat,
                    (total_T, B),
                    share_memory=self.async_)
            self.samples_done_n = buffer_from_example(buffer_example.done, (total_T, B),
                share_memory=self.async_)
        else:
            if r_hat_GT_coef is None:
                self.samples_return_ = self.samples.reward
            else:
                assert r_hat_GT_coef >= 0.
                self.samples_return_ = self.samples.reward_GT + \
                                       self.r_hat_GT_coef * self.samples.reward_hat
            self.samples_done_n = self.samples.done
        self._buffer_full = False
        self.off_backward = n_step_return  # Current invalid samples.
        self.off_forward = 1  # i.e. current cursor, prev_action overwritten.
        self.remove_frame_axis = remove_frame_axis

        # Back to rlpyt.FrameBufferMixin
        # Equivalent to image.shape[0] if observation is image array (C,H,W):
        if len(self.obs_shape) == 1:
            self.n_frames = n_frames = 1
            self.samples_frames = buffer_from_example(example.observation,
                                (self.total_T + n_frames - 1, self.B),
                                share_memory=self.async_)  # [T, D]
            print(f"[Replay Buffer] no frame stacked.")
            self.off_forward = max(self.off_forward, n_frames - 1)
            assert self.off_forward == 1  # NOTE: for other setting, check carefully about off_forward
        else:
            assert len(self.obs_shape) == 3
            self.n_frames = n_frames = get_leading_dims(example.observation,
                n_dim=1)[0]
            self.samples_frames = buffer_from_example(example.observation[0],
                                (self.total_T + n_frames - 1, self.B),
                                share_memory=self.async_)  # [T+n_frames-1,B,H,W]
            print(f"[Replay Buffer] Frame-based buffer using {n_frames}-frame sequences.")
            self.off_forward = max(self.off_forward, n_frames - 1)
            assert self.off_forward == n_frames - 1 == 3  # NOTE: for other setting, check carefully about off_forward

        # frames: oldest stored at t; duplicate n_frames - 1 beginning & end.
        # new_frames: shifted so newest stored at t; no duplication.
        self.samples_new_frames = self.samples_frames[n_frames - 1:]  # [T,B,H,W]
        
    def append_samples(self, samples):
        """Appends all samples except for the `observation` as normal.
        Only the new frame in each observation is recorded."""
        # From rlpyt.FrameBufferMixin
        t, fm1 = self.t, self.n_frames - 1
        buffer_samples = BufferSamples(*(v for k, v in samples.items()
            if k != "observation"))
        
        # T, idxs = super().append_samples(buffer_samples)
        ## From rlpyt.BaseNStepReturnBuffer
        T, B = get_leading_dims(buffer_samples, n_dim=2)  # samples.env.reward.shape[:2]
        assert B == self.B
        t = self.t
        if t + T > self.total_T:  # Wrap.
            idxs = np.arange(t, t + T) % self.total_T
        else:
            idxs = slice(t, t + T)
        self.samples[idxs] = buffer_samples
        self.compute_returns(T)
        if not self._buffer_full and t + T >= self.total_T:
            self._buffer_full = True  # Only changes on first around.
        self.t = (t + T) % self.total_T

        # back to rlpyt.FrameBufferMixin
        if samples.observation.ndim == 5:  # (T, B, frame_stack, H, W)
            self.samples_new_frames[idxs] = samples.observation[:, :, -1]
            if t == 0:  # Starting: write early frames
                for f in range(fm1):
                    self.samples_frames[f] = samples.observation[0, :, f]
            elif self.t < t and fm1 > 0:  # Wrapped: copy any duplicate frames.
                self.samples_frames[:fm1] = self.samples_frames[-fm1:]
        elif samples.observation.ndim == 3:  # (T, B, D)
            assert self.remove_frame_axis == True  # for the highway case
            assert fm1 == 0
            self.samples_new_frames[idxs] = samples.observation[:, :, :]
            if t == 0:  # Starting: write early frames
                for f in range(fm1):
                    raise SyntaxError  # this line should not be reached for this case
                    self.samples_frames[f] = samples.observation[0, :, f]
            elif self.t < t and fm1 > 0:  # Wrapped: copy any duplicate frames.
                raise SyntaxError  # this line should not be reached for this case
                self.samples_frames[:fm1] = self.samples_frames[-fm1:]
        else:
            raise NotImplementedError
        return T, idxs

    """ From rlpyt.BaseNStepReturnBuffer """
    def compute_returns(self, T, st=None):  # T: sampler_t
        """Compute the n-step returns using the new rewards just written into
        the buffer, but before the buffer cursor is advanced.  Input ``T`` is
        the number of new timesteps which were just written (T>=1).
        Does nothing if `n-step==1`. e.g. if 2-step return, t-1
        is first return written here, using reward at t-1 and new reward at t
        (up through t-1+T from t+T)."""
        if self.n_step_return == 1:
            if self.r_hat_GT_coef is not None:
                if self.use_potential.ahead:
                    pdb.set_trace()  # check contents in next_hat
                    next_hat = np.concat((s.reward_hat[1:], [[0.]]), axis=0)
                    # next_done = np.concat((s.done[1:], [[True]]), axis=0)
                    F_pot = self.pot_coef * next_hat * s.done - s.reward_hat
                elif self.use_potential.back:
                    pdb.set_trace()  # check contents in prev_hat, prev_done
                    prev_hat = np.concat(([[0.]], s.reward_hat[:-1]), axis=0)
                    prev_done = np.concat((s.done[-1], s.done[:-1]), axis=0)
                    F_pot = s.reward_hat - self.pot_coef * prev_hat * prev_done
                else:
                    self.samples_return_ = self.samples.reward_GT + \
                                    self.r_hat_GT_coef * self.samples.reward_hat
            return  # return = reward, done_n = done
        s = self.samples
        if st is None:
            t = self.t
        else:
            t = st
        nm1 = self.n_step_return - 1
        if t - nm1 >= 0 and t + T <= self.total_T:  # No wrap (operate in-place).
            if self.r_hat_GT_coef is None:
                reward = s.reward[t - nm1:t + T]
            else:
                if self.use_potential.ahead:  # potential look-ahead
                    if t + T + 1 > self.total_T:
                        # because t + T <= self.total_T, therefore if t + T + 1 > self.total_T < 0, t + T + 1 == self.total_T + 1
                        ahead_idx = np.arange(t - nm1 + 1, t + T + 1) % self.total_T
                        F_pot = self.pot_coef * s.reward_hat[ahead_idx] * s.done[t-nm1: t+T] \
                            - s.reward_hat[t - nm1:t + T] 
                    else:
                        F_pot = self.pot_coef * s.reward_hat[t-nm1+1: t+T+1] * s.done[t-nm1: t+T] \
                            - s.reward_hat[t - nm1:t + T] 
                    reward = s.reward_GT[t - nm1:t + T] + self.r_hat_GT_coef * F_pot
                elif self.use_potential.back:  # potential look-back
                    if t - nm1 - 1 < 0:
                        # because t - nm1 >= 0, therefore if t - nm1 - 1 < 0, t - nm1 - 1 == -1
                        back_idx = np.arange(t - nm1 - 1, t + T - 1) % self.total_T
                        F_pot = s.reward_hat[t - nm1:t + T] \
                            - self.pot_coef * s.reward_hat[back_idx] * s.done[back_idx]
                    else:
                        F_pot = s.reward_hat[t - nm1:t + T] \
                            - self.pot_coef * s.reward_hat[t - nm1 - 1:t + T - 1] * s.done[t - nm1 - 1:t + T - 1]
                    reward = s.reward_GT[t - nm1:t + T] + self.r_hat_GT_coef * F_pot
                else:
                    reward = s.reward_GT[t - nm1:t + T] + \
                            self.r_hat_GT_coef * s.reward_hat[t - nm1:t + T]
            done = s.done[t - nm1:t + T]
            # NOTE: self.t to self.t+T are new appended data, so return_ and done_n can be updated only for (t-nm1: t-nm1+T)
            return_dest = self.samples_return_[t - nm1: t - nm1 + T]
            done_n_dest = self.samples_done_n[t - nm1: t - nm1 + T]
            discount_return_n_step(reward, done, n_step=self.n_step_return,
                discount=self.discount, return_dest=return_dest,
                done_n_dest=done_n_dest)
        else:  # Wrap (copies); Let it (wrongly) wrap at first call.
            idxs = np.arange(t - nm1, t + T) % self.total_T
            if self.r_hat_GT_coef is None:
                reward = s.reward[idxs]
            else:
                if self.use_potential.ahead:  # potential look-ahead
                    ahead_idx = np.arange(t - nm1 + 1, t + T + 1) % self.total_T
                    F_pot = self.pot_coef * s.reward_hat[ahead_idx] * s.done[idxs]\
                                - s.reward_hat[idxs]
                    reward = s.reward_GT[idxs] + self.r_hat_GT_coef * F_pot
                elif self.use_potential.back:  # potential look-back
                    back_idx = np.arange(t - nm1 - 1, t + T - 1) % self.total_T
                    F_pot = s.reward_hat[idxs] \
                        - self.pot_coef * s.reward_hat[back_idx] * s.done[back_idx]  # suppose Phi(s_{-1},a_{-1})=0, so * s.done[back_idx]
                    reward = s.reward_GT[idxs] + self.r_hat_GT_coef * F_pot
                else:
                    reward = s.reward_GT[idxs] + \
                            self.r_hat_GT_coef * s.reward_hat[idxs]
            done = s.done[idxs]
            dest_idxs = idxs[:-nm1]
            return_, done_n = discount_return_n_step(reward, done,
                n_step=self.n_step_return, discount=self.discount)
            self.samples_return_[dest_idxs] = return_
            self.samples_done_n[dest_idxs] = done_n

    """ From rlpyt.NStepReturnBuffer """
    def extract_batch(self, T_idxs, B_idxs):
        """From buffer locations `[T_idxs,B_idxs]`, extract data needed for
        training, including target values at `T_idxs + n_step_return`.  Returns
        namedarraytuple of torch tensors (see file for all fields).  Each tensor
        has leading batch dimension ``len(T_idxs)==len(B_idxs)``, but individual
        samples are drawn, so no leading time dimension."""
        s = self.samples
        target_T_idxs = (T_idxs + self.n_step_return) % self.total_T
        batch = SamplesFromReplay(
            agent_inputs=AgentInputs(
                observation=self.extract_observation(T_idxs, B_idxs),
                # prev_action=s.action[T_idxs - 1, B_idxs],
                # prev_reward=s.reward[T_idxs - 1, B_idxs],
            ),
            action=s.action[T_idxs, B_idxs],
            return_=self.samples_return_[T_idxs, B_idxs],
            done=self.samples.done[T_idxs, B_idxs],
            done_n=self.samples_done_n[T_idxs, B_idxs],
            target_inputs=AgentInputs(
                observation=self.extract_observation(target_T_idxs, B_idxs),
                # prev_action=s.action[target_T_idxs - 1, B_idxs],
                # prev_reward=s.reward[target_T_idxs - 1, B_idxs],
            ),
        )
        # t_news = np.where(s.done[T_idxs - 1, B_idxs])[0]
        # batch.agent_inputs.prev_action[t_news] = 0
        # batch.agent_inputs.prev_reward[t_news] = 0
        return torchify_buffer(batch)

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
            observation = np.concatenate([self.samples_frames[t:t + self.n_frames, b]
                for t, b in zip(T_idxs, B_idxs)], axis=0)  # [B*n_frames, *obs_shape_for_one_frame]
        else:
            # np.stack will generate a new axis
            observation = np.stack([self.samples_frames[t:t + self.n_frames, b]
                for t, b in zip(T_idxs, B_idxs)], axis=0)  # [B,n_frames,H,W]
        # Populate empty (zero) frames after environment done.
        for f in range(1, self.n_frames):
            # e.g. if done 1 step prior, all but newest frame go blank.
            b_blanks = np.where(self.samples.done[T_idxs - f, B_idxs])[0]
            observation[b_blanks, :self.n_frames - f] = 0
        return observation

    def relabel_with_predictor(self, predictor):  # predictor = (reward predictor)
        batch_size_T = self.relabel_batch_size // self.B

        len_T = self.total_T if self._buffer_full else self.t
        total_iter = int(len_T // batch_size_T)

        if len_T > batch_size_T * total_iter:
            total_iter += 1

        for index in range(total_iter):
            last_index_T = (index + 1) * batch_size_T
            last_index_T = len_T if last_index_T > len_T else last_index_T
            batch_this_itr = last_index_T - index * batch_size_T

            # obses = self.samples_frames[index * batch_size_T: last_index_T]
            take_index_T = np.arange(index * batch_size_T, last_index_T).\
                            reshape(-1, 1).repeat(self.B, axis=-1).\
                            reshape(-1)  # shape: (batch_this_itr, 1) -> (batch_this_itr, self.B) -> (batch_this_itr*self.B,)
            assert take_index_T.shape == (batch_this_itr * self.B, )
            take_index_B = np.arange(self.B).reshape(1, -1).\
                            repeat(batch_this_itr, axis=0).reshape(-1)  # shape: (1, self.B) -> (batch_this_itr, self.B) -> (batch_this_itr*self.B, )
            assert take_index_B.shape == take_index_T.shape
            
            obses = self.extract_observation(take_index_T, take_index_B)  # obses.shape: (batch_this_itr * self.B, *self.obs_shape)
            assert obses.shape == (batch_this_itr * self.B, *self.obs_shape)
            actions = self.samples.action[index * batch_size_T : last_index_T].\
                        reshape(-1)  # actions.shape: (batch_this_itr * self.B,)
            assert actions.shape == (batch_this_itr * self.B, )

            obses_pyt = torch.from_numpy(obses)
            act_pyt = torch.from_numpy(actions)
            pred_reward = predictor.r_hat_sa(obses_pyt, act_pyt)  # pred_reward.shape: (batch_this_itr * self.B,)
            assert pred_reward.shape == (batch_this_itr * self.B,)
            pred_reward = pred_reward.reshape(batch_this_itr, self.B)
            if self.r_hat_GT_coef is None:
                self.samples.reward[index*batch_size_T: last_index_T, :] = pred_reward
            else:
                self.samples.reward_hat[index*batch_size_T: last_index_T, :] = pred_reward
        
        # relabel return_ values
        self.compute_returns(T=self.total_T if self._buffer_full else self.t, st=0)

class PrioritizedReplay:
    """
    From rlpyt.PrioritizedReplay
    Prioritized experience replay using sum-tree prioritization.

    The priority tree must configure at instantiation if priorities will be
    input with samples in ``append_samples()``, by parameter
    ``input_priorities=True``, else the default value will be applied to all
    new samples.
    """

    def __init__(self,
                alpha,
                beta,
                default_priority,
                unique=False,
                input_priorities=False,
                input_priority_shift=0,
                **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.default_priority = default_priority  # TODO: If have time, may check how to /why set default_priority this value.
        self.unique = unique
        self.input_priorities = input_priorities
        self.input_priority_shift = input_priority_shift
        self.init_priority_tree()

    def init_priority_tree(self):
        """Organized here for clean inheritance."""
        SumTreeCls = AsyncSumTree if self.async_ else SumTree
        self.priority_tree = SumTreeCls(
            T=self.total_T,
            B=self.B,
            off_backward=self.off_backward,
            off_forward=self.off_forward,
            default_value=self.default_priority ** self.alpha,
            enable_input_priorities=self.input_priorities,
            input_priority_shift=self.input_priority_shift,
        )

    def set_beta(self, beta):
        self.beta = beta

    def append_samples(self, samples):
        """Looks for ``samples.priorities``; if not found, uses default priority.  Writes
        samples using super class's ``append_samples``, and advances matching cursor in
        priority tree.
        """
        if hasattr(samples, "priorities"):
            priorities = samples.priorities ** self.alpha
            samples = samples.samples
        else:
            priorities = None
        T, idxs = super().append_samples(samples)  # NStepFrameBuffer.append_samples
        self.priority_tree.advance(T, priorities=priorities)  # Progress priority_tree cursor.
        return T, idxs

    def sample_batch(self, batch_B):  # batch_B = algo.batch_size
        """Calls on the priority tree to generate random samples.  Returns
        samples data and normalized importance-sampling weights:
        ``is_weights=priorities ** -beta``
        """
        (T_idxs, B_idxs), priorities = self.priority_tree.sample(batch_B,
            unique=self.unique)
        batch = self.extract_batch(T_idxs, B_idxs)
        is_weights = (1. / (priorities + EPS)) ** self.beta  # Unnormalized.
        is_weights /= max(is_weights)  # Normalize.
        is_weights = torchify_buffer(is_weights).float()
        return SamplesFromReplayPri(*batch, is_weights=is_weights)

    def update_batch_priorities(self, priorities):
        """Takes in new priorities (i.e. from the algorithm after a training
        step) and sends them to priority tree as ``priorities ** alpha``; the
        tree internally remembers which indexes were sampled for this batch.
        """
        priorities = numpify_buffer(priorities)
        self.priority_tree.update_batch_priorities(priorities ** self.alpha)

    # def relabel_priority(self):
    #     raise NotImplementedError
    
    # def relabel_with_predictor(self, predictor):  # predictor = (reward predictor)
    #     super.relabel_with_predictor(predictor)
    #     pdb.set_trace()  # TODO: make sure we can step into here
    #     self.relabel_priority()


class UniformReplay:
    """
    From rlpyt.UniformReplay
    Replay of individual samples by uniform random selection."""

    def sample_batch(self, batch_B):
        """Randomly select desired batch size of samples to return, uses
        ``sample_idxs()`` and ``extract_batch()``."""
        T_idxs, B_idxs = self.sample_idxs(batch_B)
        return self.extract_batch(T_idxs, B_idxs)

    def sample_idxs(self, batch_B):
        """Randomly choose the indexes of data to return using
        ``np.random.randint()``.  Disallow samples within certain proximity to
        the current cursor which hold invalid data.
        """
        t, b, f = self.t, self.off_backward, self.off_forward
        high = self.total_T - b - f if self._buffer_full else t - b
        low = 0 if self._buffer_full else f
        T_idxs = np.random.randint(low=low, high=high, size=(batch_B,))
        T_idxs[T_idxs >= t - b] += min(t, b) + f  # min for invalid high t.
        B_idxs = np.random.randint(low=0, high=self.B, size=(batch_B,))
        return T_idxs, B_idxs
    

class UniformReplayFrameBuffer(UniformReplay, NStepFrameBuffer):
    pass


class PrioritizedReplayFrameBuffer(PrioritizedReplay, NStepFrameBuffer):
    pass
