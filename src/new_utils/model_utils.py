import torch
import torch.nn as nn
import pdb
import numpy as np

def weight_init(m, init_type='orthogonal', trival=False):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        if init_type == 'kaiming_linear':
            nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
        elif init_type == 'kaiming_lrelu':
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        elif init_type == 'orthogonal':  # original initialization method used in BPref
            nn.init.orthogonal_(m.weight.data)
        else:
            raise NotImplementedError
        nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.reset_parameters()
    else:
        if trival:
            # NOTE: this message will show for nn.Sequential, but in fact submodules inside has been initialized. 
            print(f'******** no special initialization for {m} ********')
            # raise NotImplementedError  # Normally, activation function do not need to be initialized

def concat_sa(obs, act):
    # obs.shape: (B, frame_stack * img_channel, H, W)
    assert len(act.shape) == 2  # (B, 1)
    assert act.shape[-1] == 1
    assert np.max(obs) <= 1.0
    if len(act.shape) < len(obs.shape):  # img obs and scalar act
        ### an old solution that calculate r_hat(s, a)
        # action_onehot = np.zeros((act.shape[0],  # batch
        #                           self.action_dim,
        #                           obs.shape[-2],
        #                           obs.shape[-1]))
        # action_onehot[:, act, :, :] = 1
        # sa_t = np.concatenate([obs, action_onehot], axis=1)  # TODO: check dim
        sa_t = obs
    else:  # both obs and act are vectors
        sa_t = np.concatenate([obs, act], axis=-1)
    return sa_t

def update_state_dict(model, state_dict, tau=1):
    """Update the state dict of ``model`` using the input ``state_dict``, which
    must match format.  ``tau==1`` applies hard update, copying the values, ``0<tau<1``
    applies soft update: ``tau * new + (1 - tau) * old``.
    """
    if tau == 1:
        model.load_state_dict(state_dict)
    elif tau > 0:
        update_sd = {k: tau * state_dict[k] + (1 - tau) * v
            for k, v in model.state_dict().items()}
        model.load_state_dict(update_sd)

def count_parameters(model):
    print(model)
    total_params = 0
    total_params_grad = 0
    for n, p in model.named_parameters():
        print(f'{n}, requires_grad:{p.requires_grad}, shape:{p.shape}, numel:{p.numel()}')
        total_params += p.numel()
        total_params_grad += p.numel() if p.requires_grad else 0

    print(f'total_params: {total_params}, total_params_grad: {total_params_grad}')
    return total_params_grad

def norm_r_hat_s(r_hat_s, norm_type):
    assert len(r_hat_s.shape) == 2  # (B, |A|)
    if norm_type is None:
        return r_hat_s
    elif norm_type == 'm2':  # min_max_scale
        min_val = torch.min(r_hat_s, dim=1, keepdim=True)[0]
        max_val = torch.max(r_hat_s, dim=1, keepdim=True)[0]
        normd = ((r_hat_s - min_val) / (max_val - min_val + 1e-8)) * 2.0 - 1.0  # scaled to [-1, 1]
        return normd
    elif norm_type == 'm2a':  # min_max_scale
        min_val, max_val = torch.aminmax(r_hat_s, dim=1, keepdim=True)  # this function can not be back-propagated
        normd = ((r_hat_s - min_val) / (max_val - min_val + 1e-8)) * 2.0 - 1.0  # scaled to [-1, 1]
        return normd
    else:
        raise NotImplementedError
    # TODO: maybe consider other normalization methods?
    # elif norm_type == 'l2':  # l2 normalize