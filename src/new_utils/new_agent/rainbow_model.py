import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.utils import scale_grad

from new_utils.model_utils import count_parameters


class AtariCatDqnModel(nn.Module):
    """2D conlutional network feeding into MLP with ``n_atoms`` outputs
    per action, representing a discrete probability distribution of Q-values."""

    def __init__(
            self,
            obs_shape,  # env_spaces.observation.shape, (frame_stack, H, W) for greyscaled image
            action_dim,  # env_spaces.action.n
            n_atoms,
            dueling,
            noisy,  # True for noisy_net
            noisy_std_init,
            distributional,
            dqn_head_hidden_size,
            conv_cfg,
            V_limit=10,
            ):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()
        self.dueling = dueling
        self.noisy = noisy
        self.distributional = distributional
        if not self.distributional:
            assert n_atoms == 1
        self.n_atoms = 1 if not self.distributional else n_atoms
        self.dqn_head_hidden_size = dqn_head_hidden_size
        self.output_size = action_dim
        
        if conv_cfg.get('is_mlp', False) == False:  # since is_mlp is a new hyper-param, so we use get() to return a default value for older config files of oracle agents' ckpt
            self.is_cnn = True
            c, h, w = obs_shape
            self.conv = Conv2dModel(  # TODO: pass conv's hyperparameters from hydra
                in_channels=c,
                channels=conv_cfg.channels,
                kernel_sizes=conv_cfg.kernel_sizes,
                strides=conv_cfg.strides,
                paddings=conv_cfg.paddings,
                use_maxpool=conv_cfg.use_maxpool,
                dropout=conv_cfg.dropout,
            )
            fake_input = torch.zeros(1, *obs_shape)  # fake_input.shape=(1, frame_stack, H, W)
            fake_output = self.conv(fake_input)  # fake_output.shape=(1, last_channel, 4, 4)
            # self.hidden_size = fake_output.shape[1]
            self.head_inchannels = fake_output.shape[1]
            self.head_inpixels = fake_output.shape[-1] * fake_output.shape[-2]
            self.hidden_flatten_dim = 3
        else:
            self.is_cnn = False
            (d,) = obs_shape
            hidden_sizes = []
            if conv_cfg.hidden_size_1 is not None:
                hidden_sizes.append(conv_cfg.hidden_size_1)
            if conv_cfg.hidden_size_2 is not None:
                assert conv_cfg.hidden_size_1 is not None
                hidden_sizes.append(conv_cfg.hidden_size_2)
            self.mlp = MlpModel(
                input_size=d,
                hidden_sizes=hidden_sizes,
            )
            fake_input = torch.zeros(1, *obs_shape)  # fake_input.shape=(1, frame_stack, D)
            fake_output = self.mlp(fake_input)  # fake_output.shape=(1, hidden_sizes[-1])
            self.head_inchannels = 1
            self.head_inpixels = fake_output.shape[-1]
            self.hidden_flatten_dim = 1

        if dueling:
            self.head = DistributionalDuelingHeadModel(
                input_channels=self.head_inchannels,
                input_pixels=self.head_inpixels,
                output_size=self.output_size,
                n_atoms=self.n_atoms,
                noisy=self.noisy,
                noisy_std_init=noisy_std_init,
                hidden_size=self.dqn_head_hidden_size,
                hidden_flatten_dim=self.hidden_flatten_dim,
                )
        else:
            self.head = DistributionalHeadModel(
                input_channels=self.head_inchannels,
                input_pixels=self.head_inpixels,
                output_size=self.output_size,
                n_atoms=self.n_atoms,
                noisy=self.noisy,
                noisy_std_init=noisy_std_init,
                hidden_size=self.dqn_head_hidden_size,
                hidden_flatten_dim=self.hidden_flatten_dim,
                )
            
        self.V_limit = V_limit

        print(f"------ Initialized model with {count_parameters(self)} parameters ------")

    """ From SPRCatDqnModel """
    # def stem_forward(self, img):
    #     """Returns the output of convolutional layers."""
    #     # Infer (presence of) leading dimensions: [T,B], [B], or [].
    #     lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

    #     conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.
    #     return conv_out

    # def head_forward(self,
    #                  conv_out,
    #                  logits=False):
    #     lead_dim, T, B, img_shape = infer_leading_dims(conv_out, 3)
    #     p = self.head(conv_out)

    #     if self.distributional:
    #         if logits:
    #             p = F.log_softmax(p, dim=-1)
    #         else:
    #             p = F.softmax(p, dim=-1)
    #     else:
    #         p = p.squeeze(-1)

    #     # Restore leading dimensions: [T,B], [B], or [], as input.
    #     p = restore_leading_dims(p, lead_dim, T, B)
    #     return p
    
    def forward(self, observation):
        # if train:
        #     # From SPRCatDqnModel
        #     if observation.ndim != 5:
        #         pdb.set_trace()  # if ndim!=5, then we should use flatten(1,2)
        #     input_obs = observation.flatten(1, 2)  # (batch_size, T*B, H, W)
        #     latent = self.stem_forward(input_obs)
        #     log_pred_ps = self.head_forward(latent, logits=True)
        #     return log_pred_ps
        # else:
        """
        From rlpyt.CatDqnModel
        Returns the probability masses ``num_atoms x num_actions`` for the Q-values
        for each state/observation, using softmax output nonlinearity."""
        if self.is_cnn:
            img = observation.type(torch.float)  # Expect torch.uint8 inputs
            assert img.max().item() > 1.  # normally, observation is normalized inside this function
            # img = img.mul_(1. / 255.)  # (!! Do not use inplace!! easy to get covert bug outside!!)
            img = img.mul(1. / 255.)  # From [0-255] to [0-1], in place.

            # Infer (presence of) leading dimensions: [T,B], [B], or [].
            lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

            conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.
            assert conv_out.shape[0] == T * B and \
                    conv_out.shape[1] == self.head_inchannels and \
                    conv_out.shape[2] * conv_out.shape[3] == self.head_inpixels
            # conv_out.shape: (T*B, self.head_inchannels, sqrt(self.head_inpixels), sqrt(self.head_inpixels))
            # p = self.head(conv_out.view(T * B, -1))
            p = self.head(conv_out)
        else:
            # Infer (presence of) leading dimensions: [T,B], [B], or [].
            obs = observation.type(torch.float).clone()  # use clone to avoid memory sharing because observation.dtype already be float32
            lead_dim, T, B, vec_shape = infer_leading_dims(observation, 1)
            mlp_out = self.mlp(obs.view(T * B, *vec_shape))  # Fold if T dimension.
            assert mlp_out.shape[0] == T * B and \
                    mlp_out.shape[1] == self.head_inchannels * self.head_inpixels
            # mlp_out.shape: (T*B,self.head_inchannels * self.head_inpixels)
            # p = self.head(conv_out.view(T * B, -1))
            p = self.head(mlp_out)
        # TODO: For reward model, maybe we don't need distributional?
        if self.distributional:
            p = F.softmax(p, dim=-1)  # softmax over n_atoms
            # p.shape: (T*B, action_dim(=self.output_size), n_atoms)
        else:
            p = p.squeeze(-1)  # p.shape: (T*B, action_dim(=self.output_size))
        assert (p.shape[-1] == self.n_atoms) and (p.shape[-2] == self.output_size)
        # Restore leading dimensions: [T,B], [B], or [], as input.
        p = restore_leading_dims(p, lead_dim, T, B)
        return p
    
    def select_action(self, obs):
        value = self.forward(obs)  # NOTE: value has been normalized via softmax in self.forward
        if self.distributional:
            # value.shape: (*lead_dim, action_dim, n_atoms)
            # NOTE: limit==|V_min|==V_max; since self.forward has already use softmax, here use_logits=False
            value = from_categorical(value, use_logits=False, limit=self.V_limit)
            # value.shape: (*lead_dim, action_dim(=output_size))
            assert value.shape[-1] == self.output_size  # value.shape: (B, action_dim)
        return value


def from_categorical(distribution, limit=300, use_logits=True):
    distribution = distribution.float()  # Avoid any fp16 shenanigans
    if use_logits:
        distribution = torch.softmax(distribution, -1)
    num_atoms = distribution.shape[-1]
    weights = torch.linspace(-limit, limit, num_atoms, device=distribution.device).float()
    return distribution @ weights


class Conv2dModel(nn.Module):
    """2-D Convolutional model component, with option for max-pooling vs
    downsampling for strides > 1.  Requires number of input channels, but
    not input shape.  Uses ``torch.nn.Conv2d``.
    """

    def __init__(
            self,
            in_channels,
            channels,
            kernel_sizes,
            strides,
            paddings=None,
            nonlinearity=torch.nn.ReLU,  # Module, not Functional.
            use_maxpool=False,  # if True: convs use stride 1, maxpool downsample.
            # head_sizes=None,  # Put an MLP head on top.
            dropout=0.,
            ):
        super().__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [in_channels] + channels[:-1]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            maxp_strides = strides
            strides = ones
        else:
            maxp_strides = ones
        conv_layers = [torch.nn.Conv2d(in_channels=ic, out_channels=oc,
            kernel_size=k, stride=s, padding=p) for (ic, oc, k, s, p) in
            zip(in_channels, channels, kernel_sizes, strides, paddings)]
        sequence = list()
        for conv_layer, maxp_stride in zip(conv_layers, maxp_strides):
            sequence.extend([conv_layer, nonlinearity()])
            if dropout > 0:
                sequence.append(nn.Dropout(dropout))
            if maxp_stride > 1:
                sequence.append(torch.nn.MaxPool2d(maxp_stride))  # No padding.
        self.conv = torch.nn.Sequential(*sequence)

    def forward(self, input):
        """Computes the convolution stack on the input; assumes correct shape
        already: [B,C,H,W]."""
        assert input.max() <= 1.0
        # print(f'[Conv Input]: min: {input.min().item()}, max: {input.max().item()}')
        return self.conv(input)


class MlpModel(torch.nn.Module):
    """Multilayer Perceptron with last layer linear.

    Args:
        input_size (int): number of inputs
        hidden_sizes (list): can be empty list for none (linear model).
        output_size: linear layer at output, or if ``None``, the last hidden size will be the output size and will have nonlinearity applied
        nonlinearity: torch nonlinearity Module (not Functional).
    """

    def __init__(
            self,
            input_size,
            hidden_sizes,  # Can be empty list or None for none.
            output_size=None,  # if None, last layer has nonlinearity applied.
            nonlinearity=torch.nn.ReLU,  # Module, not Functional.
            ):
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        elif hidden_sizes is None:
            hidden_sizes = []
        hidden_layers = [torch.nn.Linear(n_in, n_out) for n_in, n_out in
            zip([input_size] + hidden_sizes[:-1], hidden_sizes)]
        sequence = list()
        for layer in hidden_layers:
            sequence.extend([layer, nonlinearity()])
        if output_size is not None:
            last_size = hidden_sizes[-1] if hidden_sizes else input_size
            sequence.append(torch.nn.Linear(last_size, output_size))
        self.model = torch.nn.Sequential(*sequence)
        self._output_size = (hidden_sizes[-1] if output_size is None
            else output_size)

    def forward(self, input):
        """Compute the model on the input, assuming input shape [B,input_size]."""
        return self.model(input)

    @property
    def output_size(self):
        """Retuns the output size of the model."""
        return self._output_size


class DistributionalHeadModel(nn.Module):
    """An MLP head which reshapes output to [B, output_size, n_atoms]."""

    def __init__(
                self,
                input_channels,
                input_pixels,
                output_size,
                n_atoms,
                noisy,
                noisy_std_init,
                hidden_flatten_dim,
                hidden_size=256,
                ):
        super().__init__()
        if noisy:
            self.linears = [NoisyLinear(input_channels*input_pixels, hidden_size, std_init=noisy_std_init),
                            NoisyLinear(hidden_size, output_size * n_atoms, std_init=noisy_std_init)]
        else:
            self.linears = [nn.Linear(input_channels*input_pixels, hidden_size),
                            nn.Linear(hidden_size, output_size * n_atoms)]
        self.hidden_flatten_dim = hidden_flatten_dim
        layers = [nn.Flatten(-self.hidden_flatten_dim, -1),
                  self.linears[0],
                  nn.ReLU(),
                  self.linears[1]]
        self.network = nn.Sequential(*layers)
        self._output_size = output_size
        self._n_atoms = n_atoms

        # pdb.set_trace() # TODO: initialize model weights explicitly?

    def forward(self, input):
        return self.network(input).view(-1, self._output_size, self._n_atoms)
    
    def reset_noise(self):  # TODO: check where is this function be called in spr
        for module in self.linears:
            module.reset_noise()

    def set_sampling(self, sampling):  # TODO: check where is this function be called in spr
        for module in self.linears:
            module.sampling = sampling

    def set_noise_override(self, noise_override):
        for module in self.linears:
            module.noise_override = noise_override


class DistributionalDuelingHeadModel(nn.Module):
    """Model component for Dueling Distributional (Categorical) DQN, like
    ``DuelingHeadModel``, but handles `n_atoms` outputs for each state-action
    Q-value distribution.
    """

    def __init__(
            self,
            input_channels,
            input_pixels,
            output_size,
            n_atoms,
            noisy,
            noisy_std_init,
            hidden_flatten_dim,
            hidden_size=256,
            grad_scale=2 ** (-1 / 2),
            ):
        super().__init__()
        if noisy:
            self.linears = [NoisyLinear(input_pixels * input_channels, hidden_size, std_init=noisy_std_init),
                            NoisyLinear(hidden_size, output_size * n_atoms, std_init=noisy_std_init),
                            NoisyLinear(input_pixels * input_channels, hidden_size, std_init=noisy_std_init),
                            NoisyLinear(hidden_size, n_atoms, std_init=noisy_std_init)
                            ]
        else:
            self.linears = [nn.Linear(input_pixels * input_channels, hidden_size),
                            nn.Linear(hidden_size, output_size * n_atoms),
                            nn.Linear(input_pixels * input_channels, hidden_size),
                            nn.Linear(hidden_size, n_atoms)
                            ]
        self.hidden_flatten_dim = hidden_flatten_dim
        self.advantage_layers = [nn.Flatten(-self.hidden_flatten_dim, -1),  # cnn: flatten (c, h, w); mlp: (frame_stack, d)
                                 self.linears[0],
                                 nn.ReLU(),
                                 self.linears[1]]
        self.value_layers = [nn.Flatten(-self.hidden_flatten_dim, -1),  # cnn: flatten (c, h, w); mlp: (frame_stack, d)
                             self.linears[2],
                             nn.ReLU(),
                             self.linears[3]]
        self.advantage_hidden = nn.Sequential(*self.advantage_layers[:3])
        self.advantage_out = self.advantage_layers[3]
        self.advantage_bias = torch.nn.Parameter(torch.zeros(n_atoms), requires_grad=True)
        self.value = nn.Sequential(*self.value_layers)
        self.network = self.advantage_hidden  # TODO: why??
        self._grad_scale = grad_scale
        self._output_size = output_size
        self._n_atoms = n_atoms

    def forward(self, input):
        x = scale_grad(input, self._grad_scale)
        advantage = self.advantage(x)
        value = self.value(x).view(-1, 1, self._n_atoms)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def advantage(self, input):
        x = self.advantage_hidden(input)
        x = self.advantage_out(x)
        x = x.view(-1, self._output_size, self._n_atoms)
        return x + self.advantage_bias
    
    def reset_noise(self):  # TODO: check where is this function be called in spr
        for module in self.linears:
            module.reset_noise()

    def set_sampling(self, sampling):  # TODO: check where is this function be called in spr
        for module in self.linears:
            module.sampling = sampling

    def set_noise_override(self, noise_override):
        for module in self.linears:
            module.noise_override = noise_override

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.1, bias=True):
        super(NoisyLinear, self).__init__()
        self.bias = bias
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.sampling = True
        self.noise_override = None
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features), requires_grad=bias)
        self.bias_sigma = nn.Parameter(torch.empty(out_features), requires_grad=bias)
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        if not self.bias:
            self.bias_mu.fill_(0)
            self.bias_sigma.fill_(0)
        else:
            self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
            self.bias_mu.data.uniform_(-mu_range, mu_range)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        # Self.training alone isn't a good-enough check, since we may need to
        # activate .eval() during sampling even when we want to use noise
        # (due to batchnorm, dropout, or similar).
        # The extra "sampling" flag serves to override this behavior and causes
        # noise to be used even when .eval() has been called.
        if self.noise_override is None:
            use_noise = self.training or self.sampling
        else:
            use_noise = self.noise_override
        if use_noise:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)
