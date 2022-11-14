from functools import partial

import jax
import jax.numpy as jnp


def batch_into_leaf(iterable, axis=0):
    """Transpose a batch of pytrees into a pytree of batched leaves.

    Args:
        iterable: a sequence of pytrees with identical structure.
        axis (int, optional): the axis of leaves to be batch dimension, usually 0 or -1. Defaults to 0.

    Returns:
        pytree: a pytree with the same structure as iterable[0], but leaves has one more dimension for batch.
    """
    stack = partial(jnp.stack, axis=axis)
    return jax.tree_map(lambda *xs: stack(xs), *iterable)


def batch_out_of_leaf(tree, axis=0):
    """Untranspose a pytree of batched leaves into a batch of pytrees.

    Args:
        tree: a pytree.
        axis (int, optional): the batch dimension in leaves, usually 0 or -1. Defaults to 0.

    Returns:
        list: a list of pytree with the same structure as tree, but the batch dimension in leaves is removed.
    """
    jax.tree_util.tree_flatten(tree)
    leaves, structure = jax.tree_util.tree_flatten(tree)

    # move batch dimension to the front
    leaves = [jnp.moveaxis(x, axis, 0) for x in leaves]

    n_batch = leaves[0].shape[0]

    # turn leaves into a batch of pytrees
    tree = jax.tree_util.tree_unflatten(structure, leaves)
    # leaves = [[x for x in batched_leaf] for batched_leaf in leaves]

    # unbatch pytrees
    transposed = jax.tree_util.tree_transpose(
        outer_treedef=structure,
        inner_treedef=[0] * n_batch,
        pytree_to_transpose=tree,
    )
    return transposed
