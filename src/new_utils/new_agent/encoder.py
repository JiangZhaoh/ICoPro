import numpy as np
import pdb

import torch
import torch.nn as nn
from functools import partial

from old_utils.utils import mlp, init_normalization

class Conv2dModelAdv(nn.Module):
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
            paddings,
            use_maxpool,  # if True: convs use stride 1, maxpool downsample.
            dropout=0.,
            channel_dropout=0.,
            nonlinearity='leakyRelu',  # Module, not Functional.
            norm_type=None,
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
        conv_layers = [nn.Conv2d(in_channels=ic, out_channels=oc,
            kernel_size=k, stride=s, padding=p) for (ic, oc, k, s, p) in
            zip(in_channels, channels, kernel_sizes, strides, paddings)]
        sequence = list()
        if nonlinearity == 'leakyRelu':
            nonlinearity = partial(torch.nn.LeakyReLU, inplace=True)
        elif nonlinearity == 'Relu':
            nonlinearity = partial(torch.nn.ReLU, inplace=True)
        else:
            raise NotImplementedError
        
        id_layer = 0
        for conv_layer, maxp_stride in zip(conv_layers, maxp_strides):
            sequence.extend([conv_layer, nonlinearity()])
            if norm_type is not None:
                sequence.append(init_normalization(channels[id_layer], norm_type))
            if dropout > 0:
                sequence.append(nn.Dropout(p=dropout))  # here p = probability of an element to be zero-ed.
            if channel_dropout > 0:
                sequence.append(nn.Dropout2d(p=channel_dropout))
            if maxp_stride > 1:
                sequence.append(nn.MaxPool2d(maxp_stride))  # No padding.
            id_layer += 1
        sequence.append(nn.Flatten(start_dim=1, end_dim=-1))  # BN should before flatten
        self.conv = nn.Sequential(*sequence)

        # self.conv.apply(weight_init)

    def forward(self, input):
        """Computes the convolution stack on the input; assumes correct shape
        already: [B,C,H,W]."""
        # B = input.shape[0]
        assert len(input.shape) >= 4  # Dropout2d can not work correctly without the batch dimension
        return self.conv(input)  # shape: (B, latent_dim_size)


def cnn_mlp(obs_shape, output_dim, cnn_cfg, mlp_cfg, output_mod=None,
            nonlinearity='leakyRelu'):
    cnn_cfg['nonlinearity'] = nonlinearity
    cnn_encoder = Conv2dModelAdv(**cnn_cfg)

    fake_B = 1
    fake_input = torch.rand((fake_B, *obs_shape))
    with torch.no_grad():
        fake_output = cnn_encoder(fake_input)
    assert fake_output.ndim == 2  # shape (fake_B, latent_dim)

    flatten_dim = fake_output.shape[-1]

    mlp_head = mlp(input_dim=flatten_dim, output_dim=output_dim,
                   hidden_dim=mlp_cfg['hidden_dim'],
                   hidden_depth=mlp_cfg['hidden_depth'],
                   output_mod=output_mod,  # a separate activation function for output layer
                   nonlinearity=nonlinearity,
                   dropout=mlp_cfg['dropout'],
                   norm_type=mlp_cfg['norm_type'],
                   )
    
    mods = [cnn_encoder, mlp_head]
    trunk = nn.Sequential(*mods)

    return trunk
