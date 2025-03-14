'''
Rainbow based on rlpyt implementation.
'''
import torch
import torch.nn as nn
import pdb
import hydra
import numpy as np

from rlpyt.utils.collections import namedarraytuple, namedtuple

from new_utils.tensor_utils import buffer_to, to_onehot, from_onehot,\
                                     valid_mean, np_mp_array
from new_utils.model_utils import update_state_dict

# TODO: maybe it's not necessary to use namedarraytuple
AgentInfo = namedarraytuple("AgentInfo", ["value"])
# AgentInputs = namedarraytuple("AgentInputs",
#     ["observation", "prev_action", "prev_reward"])
AgentInputs = namedarraytuple("AgentInputs", ["observation"])
# pdb.set_trace()  # TODO: check where should use AgentInputs to wrap observation
AgentStep = namedarraytuple("AgentStep",
                            ["action", "agent_info"])
OptInfo = namedtuple("OptInfo", ["loss", "gradNorm", "tdAbsErr"])
SamplesToBuffer = namedarraytuple("SamplesToBuffer",
    ["observation", "action", "reward", "done"])


class CategoricalEpsilonGreedy:
    """For epsilon-greedy exploration from state-action Q-values."""
    def __init__(self,
                 dim,  # action dim
                 epsilon=1.,
                 z=None,
                 dtype=torch.long, 
                 onehot_dtype=torch.float):
        # attributes for discrete distribution
        self._dim = dim  # action dim
        self.dtype = dtype
        self.onehot_dtype = onehot_dtype
        # attributes for epsilon greedy
        self._epsilon = epsilon
        self.z = z

    def sample(self, p, z=None):
        """Input p to be shaped [T,B,A,P] or [B,A,P],
        A: number of actions, P: number of atoms.
        Optional input z is domain of atom-values, shaped [P].
        Vector epsilon of length B will apply across Batch dimension."""
        pdb.set_trace()  # check p.shape and q.shape and dims in tensordot (Seems never use this part)
        q = torch.tensordot(p, z or self.z, dims=1)  # dims=1 means only dot sum over the one last dimension
        arg_select = torch.argmax(q, dim=-1)
        mask = torch.rand(arg_select.shape) < self._epsilon
        arg_rand = torch.randint(low=0, high=q.shape[-1], size=(mask.sum(),))
        arg_select[mask] = arg_rand
        return arg_select

    def set_z(self, z):
        """Assign vector of bin locations, distributional domain."""
        self.z = z
    
    @property
    def epsilon(self):
        return self._epsilon

    def set_epsilon(self, epsilon):
        """Assign value for epsilon (can be vector)."""
        self._epsilon = epsilon

    """Methods from DiscreteMixin class"""
    @property
    def dim(self):
        return self._dim
    
    def to_onehot(self, indexes, dtype=None):
        """Convert from integer indexes to one-hot, preserving leading dimensions."""
        return to_onehot(indexes, self._dim, dtype=dtype or self.onehot_dtype)

    def from_onehot(self, onehot, dtype=None):
        """Convert from one-hot to integer indexes, preserving leading dimensions."""
        return from_onehot(onehot, dtpye=dtype or self.dtype)
    
    """Methods from Distribution class."""
    def entropy(self, dist_info):
        """
        Compute entropy of distributions contained in ``dist_info``; should
        maintain any leading dimensions.
        """
        raise NotImplementedError
    
    def perplexity(self, dist_info):
        """Exponential of the entropy, maybe useful for logging."""
        return torch.exp(self.entropy(dist_info))
    
    def mean_entropy(self, dist_info, valid=None):
        """In case some sophisticated mean is needed (e.g. internally
        ignoring select parts of action space), can override."""
        return valid_mean(self.entropy(dist_info), valid)

    def mean_perplexity(self, dist_info, valid=None):
        """Exponential of the entropy, maybe useful for logging."""
        return valid_mean(self.perplexity(dist_info), valid)


class ActionSelection(nn.Module):

    def __init__(self, network, distribution, device="cpu"):
        super().__init__()
        self.network = network
        self.epsilon = distribution._epsilon
        self.device = device
        self.first_call = True

    def to_device(self, device):
        self.device = device

    @torch.no_grad()
    def run(self, obs, potential_model=None):
        # while len(obs.shape) <= 4:  # in SPR, to simulate T=B=1
        #     obs.unsqueeze_(0)
        # assert len(obs.shape) >= 3
        # obs = obs.to(self.device).float() / 255.

        value = self.network.select_action(obs.to(self.device).float())
        if potential_model is not None:
            assert potential_model.ensemble[0].training == False
            potential_ls = []
            for member in range(potential_model.de):
                r_hat_s = potential_model.r_hat_s_member(obs, member=member)  # r_hat_s.shape: (#obs, action_dim)
                potential_ls.append(r_hat_s[None])  # add an extra dim for #ens
            potential = torch.mean(torch.cat(potential_ls, dim=0), dim=0, keepdims=False)  # shape: (#ensemble, #obs, action_dim) -> (#obs, action_dim)
            value += potential
        action = self.select_action(value)
        # TODO: strange, can not understand why spr use this hack
        # Stupid, stupid hack because rlpyt does _not_ handle batch_b=1 well.
        # if self.first_call:
        #     action = action.squeeze()
        #     self.first_call = False
        # return action, value.squeeze()
        return action, value

    def select_action(self, value):
        """Input can be shaped [T,B,Q] or [B,Q], and vector epsilon of length
        B will apply across the Batch dimension (same epsilon for all T)."""
        arg_select = torch.argmax(value, dim=-1)  # shape=(1,) will be reduced to a scalar
        mask = torch.rand(arg_select.shape, device=value.device) < self.epsilon
        if mask.shape == ():  # scalar
            if mask:
                arg_rand = torch.randint(low=0, high=value.shape[-1], size=(), device=value.device)
                arg_select = arg_rand
        else:
            arg_rand = torch.randint(low=0, high=value.shape[-1], size=(mask.sum(),), device=value.device)
            arg_select[mask] = arg_rand
        return arg_select


class CatDqnAgent:
    """
    rlpyt.AtariCatDqnAgent + SPRAgent
    Standard agent for DQN algorithms with epsilon-greedy exploration.  
    """

    def __init__(
                self,
                model,
                # ModelCls=None,
                # model_kwargs=None,
                n_atoms=51,
                eps_eval=0.001,
                # initial_model_state_dict=None,
                # If use noisy nets, the following params are useless
                eps_init=1,
                eps_final=0.01,
                eps_final_min=None,  # Give < eps_final for vector epsilon.
                eps_itr_min=50,  # Algo may overwrite.
                eps_itr_max=1000,
                ckpt_path=None,
                separate_tgt=False,
                 ):
        """
        NOTE: from BaseAgent.__init__() & EpsilonGreedyAgentMixin.__init__()
        Arguments are saved but no model initialization occurs.

        Args:
            ModelCls: The model class to be used.
            model_kwargs (optional): Any keyword arguments to pass when instantiating the model.
            initial_model_state_dict (optional): Initial model parameter values.
        """
        """Agent part"""
        # self.ModelCls = ModelCls
        # self.model_kwargs = model_kwargs
        self.model = hydra.utils.instantiate(model)
        self.target_model = hydra.utils.instantiate(model)
        self.separate_tgt = separate_tgt
        if separate_tgt:
            self.separate_target_model = hydra.utils.instantiate(model)

        self.ckpt_path = ckpt_path

        # self.model = None  # type: torch.nn.Module
        self.shared_model = None
        self.distribution = None
        self.device = torch.device("cpu")
        self._mode = None
        
        # For categorical DQN
        # self.n_atoms = self.model_kwargs["n_atoms"] = n_atoms  # this logic has been configured in rainbow.yaml
        self.n_atoms = n_atoms

        # For epsilon greedy
        self.eps_eval = eps_eval
        self.eps_init = eps_init
        self.eps_final = eps_final

        self.eps_final_min = eps_final_min
        self._eps_final_scalar = eps_final  # In case multiple vec_eps calls.
        self._eps_init_scalar = eps_init

        self.eps_itr_min = eps_itr_min
        self.eps_itr_max = eps_itr_max
        self._eps_itr_min_max = np_mp_array(2, "int")  # Shared memory for CpuSampler
        self._eps_itr_min_max[0] = eps_itr_min if eps_itr_min else 0
        self._eps_itr_min_max[1] = eps_itr_max if eps_itr_max else self._eps_itr_min_max[0]

        self.potential_model=None

    def initialize(self, 
                    # env_spaces,
                    action_dim,
                    global_B=1,
                    env_ranks=None):
        """
        Instantiates the neural net model(s) according to the environment
        interfaces.  

        Uses shared memory as needed--e.g. in CpuSampler, workers have a copy
        of the agent for action-selection.  The workers automatically hold
        up-to-date parameters in ``model``, because they exist in shared
        memory, constructed here before worker processes fork. Agents with
        additional model components (beyond ``self.model``) for
        action-selection should extend this method to share those, as well.

        Typically called in the sampler during startup.

        Args:
            env_spaces: passed to ``make_env_to_model_kwargs()``, typically namedtuple of 'observation' and 'action'.
            share_memory (bool): whether to use shared memory for model parameters.
        """
        # self.env_model_kwargs = self.make_env_to_model_kwargs(env_spaces)
        # pdb.set_trace()  # TODO: check sac_discrete for model initialization
        # self.model = self.ModelCls(**self.env_model_kwargs,
        #     **self.model_kwargs)

        if self.ckpt_path is not None:
            ckpt = torch.load('%s' % (self.ckpt_path), map_location=self.device)
            msg = self.model.load_state_dict(ckpt, strict=False)
            print(f"{'*'*10}checkpoint load information for a previous agent model{'*'*10}")
            print(msg)
        # TODO: optimizer's dict?

        # self.env_spaces = env_spaces
        self.action_dim = action_dim
        self.target_model.load_state_dict(self.model.state_dict())
        if self.separate_tgt:
            self.separate_target_model.load_state_dict(self.model.state_dict())

        # self.distribution = EpsilonGreedy(dim=env_spaces.action.n)
        if env_ranks is not None:  # For EpsilonGreedy
            self.make_vec_eps(global_B, env_ranks)
        
        self.distribution = CategoricalEpsilonGreedy(dim=action_dim,
            z=torch.linspace(-1, 1, self.n_atoms))  # z placeholder for init. will be updates in set_z
        
        self.search = ActionSelection(self.model, self.distribution)

    def __call__(self, observation, train=False):
        """Returns Q-values for states/observations (with grad)."""
        model_inputs = buffer_to((observation), device=self.device)
        # observation.shape: (batch_size, frame_Stack, H, W), [0,255], torch.uint8
        if train:
            return self.model(model_inputs)
        else:
            return self.model(model_inputs).cpu()

    @torch.no_grad()
    def step(self, observation):
        """
        From SPR.agent
        Computes Q-values for states/observations and selects actions by
        epsilon-greedy. (no grad)"""
        # NOTE: SPR split model.step() and model.select_action(), 
        #       because select_action() only  receive obs as model inputs
        #       without prev_action & prev_reward.
        action, value = self.search.run(observation.to(self.search.device),
                                        potential_model=self.potential_model)
        value = value.cpu()  # value.shape: (action_dim)
        action = action.cpu()  # action.shape: ()
        assert value.shape[-1] == self.action_dim

        agent_info = AgentInfo(value=value)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    # def target(self, observation, prev_action, prev_reward):
    def target(self, observation, train=True):
        """Returns the target Q-values for states/observations."""
        # prev_action = self.distribution.to_onehot(prev_action)
        # model_inputs = buffer_to((observation, prev_action, prev_reward),
        model_inputs = buffer_to((observation),
            device=self.device)
        target_q = self.target_model(model_inputs)
        if train:
            return target_q
        else:
            return target_q.cpu()
        
    def separate_target(self, observation, train=True):
        """Returns the target Q-values for states/observations."""
        # prev_action = self.distribution.to_onehot(prev_action)
        # model_inputs = buffer_to((observation, prev_action, prev_reward),
        model_inputs = buffer_to((observation),
            device=self.device)
        target_q = self.separate_target_model(model_inputs)
        if train:
            return target_q
        else:
            return target_q.cpu()

    def update_target(self, tau=1):
        """Copies the model parameters into the target model."""
        update_state_dict(self.target_model, self.model.state_dict(), tau)

    def update_separate_target(self, tau=1):
        """Copies the model parameters into the target model."""
        update_state_dict(self.separate_target_model, self.model.state_dict(), tau)

    def train_mode(self, itr):
        """Go into training mode (e.g. see PyTorch's ``Module.train()``)."""
        self.model.train()
        self._mode = "train"

        # From SPRAgent
        self.search.network.head.set_sampling(True)
        self.itr = itr

    def sample_mode(self, itr, verbose=False):
        """Extend method to set epsilon for sampling (including annealing)."""
        self.model.eval()
        self._mode = "sample"

        itr_min = self._eps_itr_min_max[0]  # Shared memory for CpuSampler
        itr_max = self._eps_itr_min_max[1]
        if itr <= itr_max:
            prog = min(1, max(0, itr - itr_min) / (itr_max - itr_min))
            self.eps_sample = prog * self.eps_final + (1 - prog) * self.eps_init
            if itr % (itr_max // 10) == 0 or itr == itr_max:
                print(f"Agent at itr {itr}, sample eps {self.eps_sample}"
                      f" (min itr: {itr_min}, max_itr: {itr_max})")
        self.distribution.set_epsilon(self.eps_sample)

        if verbose:
            print(f"sample_mode: Agent at itr {itr}, eps_sample {self.eps_sample}")
        # From SPRAgent
        self.search.epsilon = self.distribution.epsilon
        self.search.network.head.set_sampling(True)
        self.itr = itr

    def eval_mode(self, itr, eps=None, verbose=False):
        """Extend method to set epsilon for evaluation, using 1 for
        pre-training eval."""
        self.model.eval()
        self._mode = "eval"

        eps_this_eval = self.eps_eval if itr > 0 else 1.
        if eps is not None:
            eps_this_eval = eps
        if verbose:
            print(f"eval_mode: Agent at itr {itr}, eval eps {eps_this_eval}")
        self.distribution.set_epsilon(eps_this_eval)
        
        # From SPRAgent
        self.search.epsilon = self.distribution.epsilon
        self.search.network.head.set_sampling(False)
        self.itr = itr
        return eps_this_eval

    def to_device(self, device):
        """Moves the model to the specified cuda device, if not ``None``.  If
        sharing memory, instantiates a new model to preserve the shared (CPU)
        model.  Agents with additional model components (beyond
        ``self.model``) for action-selection or for use during training should
        extend this method to move those to the device, as well.

        Typically called in the runner during startup.
        """
        self.device = device
        assert self.device is not None
        self.model.to(self.device)
        self.target_model.to(self.device)
        if self.separate_tgt:
            self.separate_target_model.to(self.device)
        self.search.to_device(self.device)
        self.search.network = self.model
        print(f"------ Move agent model & target model on device: {self.device}. ------")
    
    def parameters(self):
        """Parameters to be optimized (overwrite in subclass if multiple models)."""
        return self.model.parameters()

    def state_dict(self):
        if self.separate_tgt:
            return dict(model=self.model.state_dict(),
                    target=self.target_model.state_dict(),
                    separate_target=self.separate_target_model.state_dict())
        else:
            return dict(model=self.model.state_dict(),
                        target=self.target_model.state_dict())
    
    """GreedyEpsilonMixin"""
    def collector_initialize(self, global_B=1, env_ranks=None):
        """For vector-valued epsilon, the agent inside the sampler worker process
        must initialize with its own epsilon values."""
        if env_ranks is not None:
            self.make_vec_eps(global_B, env_ranks)
    
    def make_vec_eps(self, global_B, env_ranks):
        """Construct log-spaced epsilon values and select local assignments
        from the global number of sampler environment instances (for SyncRl
        and AsyncRl)."""
        if (self.eps_final_min is not None and
                self.eps_final_min != self._eps_final_scalar):  # vector epsilon.
            if self.alternating:  # In FF case, sampler sets agent.alternating.
                raise NotImplementedError
                assert global_B % 2 == 0
                global_B = global_B // 2  # Env pairs will share epsilon.
                env_ranks = list(set([i // 2 for i in env_ranks]))
            self.eps_init = self._eps_init_scalar * torch.ones(len(env_ranks))
            global_eps_final = torch.logspace(
                torch.log10(torch.tensor(self.eps_final_min)),
                torch.log10(torch.tensor(self._eps_final_scalar)),
                global_B)
            self.eps_final = global_eps_final[env_ranks]
        self.eps_sample = self.eps_init

    def set_epsilon_itr_min_max(self, eps_itr_min, eps_itr_max):
        # Beginning and end of linear ramp down of epsilon.
        print(f"Agent setting min/max epsilon itrs: {eps_itr_min}, "
            f"{eps_itr_max}")
        self.eps_itr_min = eps_itr_min
        self.eps_itr_max = eps_itr_max
        self._eps_itr_min_max[0] = eps_itr_min  # Shared memory for CpuSampler
        self._eps_itr_min_max[1] = eps_itr_max
        assert eps_itr_max > eps_itr_min

    def set_sample_epsilon_greedy(self, epsilon):
        self.distribution.set_epsilon(epsilon)

    """From CatDqnAgent"""
    def give_V_min_max(self, V_min, V_max):
        self.V_min = V_min
        self.V_max = V_max
        self.distribution.set_z(torch.linspace(V_min, V_max, self.n_atoms))

    # """From AtariMixin"""
    # def make_env_to_model_kwargs(self, env_spaces):
    #     return dict(image_shape=env_spaces.observation.shape,
    #                 output_size=env_spaces.action.n)

    # """ Other trival functions from BaseAgent """
    # def reset_one(self, idx):  
    #     pass

    """ Other functions"""
    def save(self, model_dir, step):
        torch.save(
            self.search.network.state_dict(), '%s/ckpt_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step, device):
        ckpt = torch.load('%s/ckpt_%s.pt' % (model_dir, step),
                           map_location=device)
        msg = self.search.network.load_state_dict(ckpt, strict=False)
        print(f"{'*'*10}checkpoint load information for pretrained agent model{'*'*10}")
        print(msg)

    def config_potential(self, potential_model):
        self.potential_model = potential_model