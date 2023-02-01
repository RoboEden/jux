from functools import partial

import jax
import jax.numpy as jnp


def color2code(color):
    code_dtype = jnp.dtype(f"i{color.dtype.itemsize*2}")
    return color.view(code_dtype).squeeze()


def code2color(code):
    assert code.dtype.itemsize % 2 == 0
    color_dtype = jnp.dtype(f"i{code.dtype.itemsize//2}")
    return code.view(color_dtype).reshape(code.shape + (2, ))


@jax.jit
def flood_fill(mask: jax.Array) -> jax.Array:
    """
    A flood fill algorithm that returns the color of each cell.

    Args:
        mask (Array): bool[H, W] that stores the barriers on the map. False
            means no barrier, True means barrier.

    Returns:
        int[H, W, 2]: the color of each cell. For all cells with the same color,
            they are connected. The color is represented by the index of the
            cell that is the smallest in each connected component.
    """
    H, W = mask.shape
    map_size = jnp.array([H, W], dtype=jnp.int16)

    # 1. prepare neighbor index list
    ij = jnp.mgrid[:H, :W].astype(jnp.int16).transpose(1, 2, 0)  # int[H, W, 2]
    ij = ij[:, :, None, :]
    delta_ij = jnp.array(
        [  # int[5, 2]
            [0, 0],
            [-1, 0],
            [0, 1],
            [1, 0],
            [0, -1],
        ],
        dtype=ij.dtype,
    )
    neighbor_ij = delta_ij[None, None, :, :] + ij  # int[H, W, 5, 2]

    # handle map boundary.
    neighbor_ij = jnp.clip(neighbor_ij, a_min=0, a_max=(map_size - 1))  # int[H, W, 5, 2]

    # barrier only connects itself.
    neighbor_ij = jnp.where(mask[..., None, None], ij, neighbor_ij)  # int[H, W, 5, 2]

    # no cell connects to barriers.
    neighbor_is_mask = mask[neighbor_ij[..., 0], neighbor_ij[..., 1]]  # int[H, W, 5]
    neighbor_ij = jnp.where(neighbor_is_mask[..., None], ij, neighbor_ij)  # int[5, H, W, 2]

    # 2.run the flood fill algorithm
    color = _flood_fill(neighbor_ij)

    return color


def _flood_fill(neighbor_ij: jax.Array) -> jax.Array:
    """
    A flood fill algorithm that returns the color of each cell.

    Args:
        neighbor_ij (Array): int[H, W, N, 2] that stores the index of the N
            neighbors of each cell. If the number of a cell is less than N, you
            may pad it with its only index. It has no harm to let one cell
            adjacent to itself. It is user's responsibility to make sure that
            every connected component is strongly connected. If there is a path
            from a to b through the neighbor list in `neighbor_ij`, then there
            must be a path from b to a.

    Returns:
        int[H, W, 2]: the color of each cell. For all cells with the same color,
            they are connected. The color is represented by the index of the
            cell that is the smallest in each connected component.
    """

    def _cond(args):
        color, new_color = args
        return (color != new_color).any()

    def _body(args):
        color, new_color = args
        color = new_color
        # jax.debug.print("{}", color2code(color))
        neighbor_color = color[neighbor_ij[..., 0], neighbor_ij[..., 1]]  # int[H, W, 5, 2]
        min_idx = jnp.argmin(color2code(neighbor_color), axis=-1)  # int[H, W]
        new_color = neighbor_color[ij[..., 0], ij[..., 1], min_idx]  # int[H, W, 2]

        code = color2code(new_color)  # int[H, W]
        new_color = code2color(code.at[color[..., 0], color[..., 1]].min(code))  # int[H, W, 2]
        new_color = new_color[new_color[..., 0], new_color[..., 1]]
        return color, new_color

    H, W = neighbor_ij.shape[:2]
    ij = jnp.mgrid[:H, :W].astype(neighbor_ij.dtype).transpose(1, 2, 0)  # int[H, W, 2]
    color, new_color = (ij - 1, ij)  # make sure color != new_color, so the loop will start
    color, _ = jax.lax.while_loop(_cond, _body, (color, new_color))
    return color
    '''
    # This is the same as the above loop
    color = ij-1 # make sure color != new_color, so the loop will start
    new_color = ij # this actually the color in first iteration
    while (color != new_color).any():
        color = new_color
        neighbor_color = color[:, neighbor_ij[:, 0], neighbor_ij[:, 1]]
        min_idx = jnp.argmin(neighbor_color[0] * W + neighbor_color[1], axis=0)
        new_color = neighbor_color[:, min_idx, ij[0], ij[1]]

        new_color = code2color(color2code(new_color).at[color[0], color[1]].min(color2code(new_color)))
        new_color = new_color[:, new_color[0], new_color[1]]

        print(f"outer:\n", color2code(new_color), sep="")
    '''


@jax.jit
def component_sum(data, color):
    """
    Compute the sum of data within each component.

    Args:
        data: broadcastable to int[H, W]. the data to be summed.
        color: int[H, W, 2]. the color of each cell. Calculated by `flood_fill()` or
        `_flood_fill()`.

    Returns:
        [H, W]: the sum of data within the component each cell is located.

    Example:
        ```python
        >>> mask = jnp.array([[0, 0, 1],
        ...                   [0, 1, 0],
        ...                   [1, 0, 0]])
        >>> color = flood_fill(mask)
        >>> component_sum(1, color) # calculate the number of cells in each component
        DeviceArray([[3, 3, 1],
                    [3, 1, 3],
                    [1, 3, 3]], dtype=int32)
        ```
    """
    cmp_sum = jnp.zeros(color.shape[:2], jnp.asarray(data).dtype)
    cmp_sum = cmp_sum.at[color[..., 0], color[..., 1]].add(data)
    cmp_sum = cmp_sum[color[..., 0], color[..., 1]]
    return cmp_sum


@jax.jit
def boundary_sum(data, color, mask):
    """
    Compute the sum of data at component boundary.

    Args:
        data: broadcastable to int[H, W]. the data to be summed.
        color: int[H, W, 2]. the color of each cell. Calculated by `flood_fill(mask)`.
        mask: bool[H, W] that stores the barriers on the map. False
            means no barrier, True means barrier.

    Returns:
        [H, W]: the sum of data in the boundary of the component each cell is
        located. Component boundary refers to the barriers that are adjacent to the
        component but not in the component.


    Example:
        ```python
        >>> mask = jnp.array([
        ...     [1, 0, 0],
        ...     [0, 1, 0],
        ...     [0, 0, 1]
        ... ], dtype=jnp.bool_)
        >>> color = flood_fill(mask)
        >>> boundary_sum(1, color, mask) # calculate the boundary size
        DeviceArray([[0, 3, 3],
                    [3, 0, 3],
                    [3, 3, 0]], dtype=int32)
        ```
    """
    H, W = mask.shape

    ij = jnp.mgrid[:H, :W].astype(jnp.int16).transpose(1, 2, 0)  # int[H, W, 2]
    ij = ij[:, :, None, :]
    delta_ij = jnp.array(  # int[4, 2]
        [
            [-1, 0],
            [0, 1],
            [1, 0],
            [0, -1],
        ], dtype=ij.dtype)
    neighbor_ij = delta_ij[None, None] + ij  # int[H, W, 4, 2]

    # handle map boundary.
    INT_MAX = jnp.iinfo(ij.dtype).max
    neighbor_ij = neighbor_ij.at[0, :, 0, :].set(INT_MAX)
    neighbor_ij = neighbor_ij.at[:, W - 1, 1, :].set(INT_MAX)
    neighbor_ij = neighbor_ij.at[H - 1, :, 2, :].set(INT_MAX)
    neighbor_ij = neighbor_ij.at[:, 0, 3, :].set(INT_MAX)

    # non-barriers connect to nothing.
    neighbor_ij = jnp.where(mask[:, :, None, None], neighbor_ij, INT_MAX)

    # get the neighbor color
    neighbor_color = color.at[neighbor_ij[..., 0], neighbor_ij[..., 1]]\
                          .get(mode='fill', fill_value=INT_MAX)  # int[H, W, 4, 2]

    # remove duplicated neighbor color
    unique_vmap = jax.vmap(
        jax.vmap(
            partial(jnp.unique, return_index=True, axis=0, size=4, fill_value=INT_MAX),
            in_axes=0,
            out_axes=0,
        ),
        in_axes=0,
        out_axes=0,
    )
    color_code = color2code(neighbor_color)  # int[H, W, 4]
    unique_neighbor_color, unique_idx = unique_vmap(color_code)  # (int[H, W, 4], int[H, W, 4])
    unique_idx = jnp.where(unique_neighbor_color == INT_MAX, INT_MAX, unique_idx)
    unique_neighbor_color = neighbor_color.at[ij[..., 0],ij[..., 1], unique_idx, :] \
                                          .get(mode='fill', fill_value=INT_MAX)  # int[H, W, 4, 2]

    # accumulate the boundary data to the root of each component
    boundary_sum = jnp.zeros(color.shape[:2], jnp.asarray(data).dtype)
    boundary_sum = boundary_sum.at[unique_neighbor_color[..., 0], unique_neighbor_color[..., 1]].add(data[..., None])
    boundary_sum = boundary_sum[color[..., 0], color[..., 1]]
    return boundary_sum


if __name__ == '__main__':

    mask = jnp.array(
        [
            # [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            # [1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            # [0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
            # [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

            # [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            # [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            # [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            # [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            # [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
            # [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

            # [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            # [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            # [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            # [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            # [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            # [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            # [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            # [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            # [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            # [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        ],
        dtype=jnp.bool_,
    )

    color = flood_fill(mask)
    data = jnp.arange(mask.shape[1]).repeat(mask.shape[0]).reshape(mask.shape)

    cmp_sum = component_sum(data, color)
    cmp_cnt = component_sum(1, color)
    cmp_mean = cmp_sum / cmp_cnt

    bdr_sum = boundary_sum(data, color, mask)
    bdr_cnt = boundary_sum(1, color, mask)
    bdr_mean = bdr_sum / bdr_cnt


def sync():
    jnp.array(0).block_until_ready()
