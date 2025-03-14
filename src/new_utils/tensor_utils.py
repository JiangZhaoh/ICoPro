import torch
import multiprocessing as mp
import ctypes
import numpy as np
import pdb
import gymnasium

from rlpyt.utils.buffer import np_mp_array_spawn


def buffer_to(buffer_, device=None):
    """Send contents of ``buffer_`` to specified device (contents must be
    torch tensors.). ``buffer_`` can be an arbitrary structure of tuples,
    namedtuples, namedarraytuples, NamedTuples and NamedArrayTuples, and a
    new, matching structure will be returned."""
    if buffer_ is None:
        return
    if isinstance(buffer_, torch.Tensor):
        return buffer_.to(device)
    elif isinstance(buffer_, np.ndarray):
        raise TypeError("Cannot move numpy array to device.")
    contents = tuple(buffer_to(b, device=device) for b in buffer_)
    if type(buffer_) is tuple:
        return contents
    return buffer_._make(contents)  # For NamedTuples/NamedArrayTuples


def np_mp_array(shape, dtype):
    """Allocate a numpy array on OS shared memory."""
    if mp.get_start_method() == "spawn":
        return np_mp_array_spawn(shape, dtype)
    size = int(np.prod(shape))
    nbytes = size * np.dtype(dtype).itemsize
    mp_array = mp.RawArray(ctypes.c_char, nbytes)
    return np.frombuffer(mp_array, dtype=dtype, count=size).reshape(shape)


def to_onehot(indexes, num, dtype=None):
    """Converts integer values in multi-dimensional tensor ``indexes``
    to one-hot values of size ``num``; expanded in an additional
    trailing dimension."""
    if dtype is None:
        dtype = indexes.dtype
    onehot = torch.zeros(indexes.shape + (num,),
        dtype=dtype, device=indexes.device)
    onehot.scatter_(-1, indexes.unsqueeze(-1).type(torch.long), 1)
    return onehot


def from_onehot(onehot, dim=-1, dtype=None):
    """Argmax over trailing dimension of tensor ``onehot``. Optional return
    dtype specification."""
    indexes = torch.argmax(onehot, dim=dim)
    if dtype is not None:
        indexes = indexes.type(dtype)
    return indexes


def valid_mean(tensor, valid=None, dim=None):
    """Mean of ``tensor``, accounting for optional mask ``valid``,
    optionally along a dimension."""
    dim = () if dim is None else dim
    if valid is None:
        return tensor.mean(dim=dim)
    valid = valid.type(tensor.dtype)  # Convert as needed.
    return (tensor * valid).sum(dim=dim) / valid.sum(dim=dim)


def zeros(shape, dtype):
    """Attempt to return torch tensor of zeros, or if numpy dtype provided,
    return numpy array or zeros."""
    try:
        return torch.zeros(shape, dtype=dtype)
    except TypeError:
        return np.zeros(shape, dtype=dtype)


def select_at_indexes(indexes, tensor):
    """Returns the contents of ``tensor`` at the multi-dimensional integer
    array ``indexes``. Leading dimensions of ``tensor`` must match the
    dimensions of ``indexes``.
    """
    dim = len(indexes.shape)
    assert indexes.shape == tensor.shape[:dim]
    num = indexes.numel()
    t_flat = tensor.view((num,) + tensor.shape[dim:])
    # pdb.set_trace()  # TODO: do not understand what happened inside this part.
    s_flat = t_flat[torch.arange(num), indexes.view(-1)]
    return s_flat.view(tensor.shape[:dim] + tensor.shape[dim + 1:])


def torchify_buffer(buffer_):
    """Convert contents of ``buffer_`` from numpy arrays to torch tensors.
    ``buffer_`` can be an arbitrary structure of tuples, namedtuples,
    namedarraytuples, NamedTuples, and NamedArrayTuples, and a new, matching
    structure will be returned. ``None`` fields remain ``None``, and torch
    tensors are left alone."""
    if buffer_ is None:
        return
    if isinstance(buffer_, gymnasium.wrappers.frame_stack.LazyFrames):
        buffer_ = buffer_[:]
    if isinstance(buffer_, np.ndarray):
        return torch.from_numpy(buffer_)
    elif isinstance(buffer_, torch.Tensor):
        return buffer_
    contents = tuple(torchify_buffer(b) for b in buffer_)
    if type(buffer_) is tuple:  # tuple, namedtuple instantiate differently.
        return contents
    return buffer_._make(contents)


def numpify_buffer(buffer_):
    """Convert contents of ``buffer_`` from torch tensors to numpy arrays.
    ``buffer_`` can be an arbitrary structure of tuples, namedtuples,
    namedarraytuples, NamedTuples, and NamedArrayTuples, and a new, matching
    structure will be returned. ``None`` fields remain ``None``, and numpy
    arrays are left alone."""
    if buffer_ is None:
        return
    if isinstance(buffer_, gymnasium.wrappers.frame_stack.LazyFrames):
        buffer_ = buffer_[:]
    if isinstance(buffer_, torch.Tensor):
        return buffer_.cpu().numpy()
    elif isinstance(buffer_, np.ndarray):
        return buffer_
    contents = tuple(numpify_buffer(b) for b in buffer_)
    if type(buffer_) is tuple:
        return contents
    return buffer_._make(contents)


def get_leading_dims(buffer_, n_dim=1):
    """Return the ``n_dim`` number of leading dimensions of the contents of
    ``buffer_``. Checks to make sure the leading dimensions match for all
    tensors/arrays, except ignores ``None`` fields.
    """
    if buffer_ is None:
        return
    if isinstance(buffer_, gymnasium.wrappers.frame_stack.LazyFrames):
        buffer_ = buffer_[:]
    if isinstance(buffer_, (torch.Tensor, np.ndarray)):
        return buffer_.shape[:n_dim]
    contents = tuple(get_leading_dims(b, n_dim) for b in buffer_ if b is not None)
    if not len(set(contents)) == 1:
        raise ValueError(f"Found mismatched leading dimensions: {contents}")
    return contents[0]
