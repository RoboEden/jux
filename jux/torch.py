import jax
import jax.dlpack

try:
    import torch
    import torch.utils.dlpack
except ImportError:
    pass


def to_torch(x: jax.Array) -> 'torch.Tensor':
    """
    Convert a jax array to a torch tensor. The new torch tensor shares memory
    with the jax array. So, any modification in torch side will be reflected in
    jax side.

    Args:
        x (jax.Array): input jax array

    Returns:
        torch.Tensor: output torch tensor
    """
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x))


def from_torch(x: 'torch.Tensor') -> jax.Array:
    """
    Convert a torch tensor to a jax array. It seems that jax will copy the array
    data, so the new jax tensor does not share memory with the torch one.

    Args:
        x (torch.Tensor): input torch tensor

    Returns:
        jax.Array: output jax array
    """
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x))
