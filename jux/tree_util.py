from functools import partial
from typing import List, Sequence, TypeVar

import jax
import jax.numpy as jnp
from jax import Array

T = TypeVar("T")


def map_to_aval(pytree):
    return jax.tree_map(lambda x: x.aval, pytree)


def batch_into_leaf(seq: Sequence[T], axis=0) -> T:
    """Transpose a batch of pytrees into a pytree of batched leaves.

    Args:
        seq: a sequence of pytrees with identical structure.
        axis (int, optional): the axis of leaves to be batch dimension, usually 0 or -1. Defaults to 0.

    Returns:
        pytree: a pytree with the same structure as seq[0], but leaves has one more dimension for batch.
    """
    assert len(seq) > 0
    stack = partial(jnp.stack, axis=axis)
    return jax.tree_map(lambda *xs: stack(xs), *seq)


def batch_out_of_leaf(tree: T, axis=0) -> List[T]:
    """Untranspose a pytree of batched leaves into a batch of pytrees.

    Args:
        tree: a pytree.
        axis (int, optional): the batch dimension in leaves, usually 0 or -1. Defaults to 0.

    Returns:
        list: a list of pytree with the same structure as tree, but the batch dimension in leaves is removed.
    """
    leaves, structure = jax.tree_util.tree_flatten(tree)

    # move batch dimension to the front
    if axis != 0:
        leaves = [jnp.moveaxis(x, axis, 0) for x in leaves]

    n_batch = leaves[0].shape[0]

    # turn leaves into a batch of pytrees
    leaves = [[x for x in batched_leaf] for batched_leaf in leaves]
    tree = jax.tree_util.tree_unflatten(structure, leaves)

    # unbatch pytrees
    transposed = jax.tree_util.tree_transpose(
        outer_treedef=structure,
        inner_treedef=jax.tree_util.tree_structure([0] * n_batch),
        pytree_to_transpose=tree,
    )
    return transposed


def concat_in_leaf(seq: Sequence[T], axis=0) -> T:
    """Concatenate a batch of pytrees into a pytree of concatenated leaves.

    Args:
        seq: a sequence of pytrees with identical structure.
        axis (int, optional): the axis of leaves to be concatenated (usually batch dimension, such as 0 or -1). Defaults to 0.

    Returns:
        pytree: a pytree with the same structure as seq[0].
    """
    assert len(seq) > 0
    concat = partial(jnp.concatenate, axis=axis)
    return jax.tree_map(lambda *xs: concat(xs), *seq)


def tree_where(cond: Array, true: T, false: T) -> T:
    """A version of jnp.where that works on pytrees.

    Args:
        cond: a pytree of bools.
        true: a pytree.
        false: a pytree has same structure as true.

    Returns:
        pytree: a pytree with the same structure as true and false.
    """

    def _where(t, f):
        new_cond_shape = cond.shape + (1, ) * (max(len(t.shape), len(f.shape)) - len(cond.shape))
        return jnp.where(cond.reshape(new_cond_shape), t, f)

    return jax.tree_util.tree_map(_where, true, false)
