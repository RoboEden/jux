from functools import partial

import jax
import jax.numpy as jnp

INT32_MAX = jnp.iinfo(jnp.int32).max


def color2code(color):
    W = color.shape[-1]
    return color[..., 0, :, :] * W + color[..., 1, :, :]


def code2color(code):
    W = code.shape[-1]
    return jnp.array([code // W, code % W])


@jax.jit
def flood_fill(mask):
    H, W = mask.shape
    map_size = jnp.array([H, W], dtype=jnp.int32)

    ij = jnp.mgrid[:H, :W]
    delta_ij = jnp.array([
        [0, 0],
        [-1, 0],
        [0, 1],
        [1, 0],
        [0, -1],
    ])  # int[2, H, W]
    neighbor_ij = delta_ij[..., None, None] + ij[None, ...]  # int[5, 2, H, W]

    # handle map boundary.
    neighbor_ij = jnp.clip(neighbor_ij, a_min=0, a_max=(map_size - 1)[..., None, None])  # int[5, 2, H, W]

    # barrier only connects itself.
    neighbor_ij = jnp.where(mask[None], ij, neighbor_ij)

    # on cell connects to barriers.
    neighbor_is_mask = mask[neighbor_ij[:, 0], neighbor_ij[:, 1]]  # int[5, H, W]
    neighbor_ij = jnp.where(neighbor_is_mask[:, None], ij, neighbor_ij)  # int[5, 2, H, W]

    def _cond(args):
        color, new_color = args
        return (color != new_color).any()

    def _body(args):
        color, new_color = args
        color = new_color
        # jax.debug.print("{}", color2code(color))
        neighbor_color = color[:, neighbor_ij[:, 0], neighbor_ij[:, 1]]  # int[2, 5, H, W]
        min_idx = jnp.argmin(neighbor_color[0] * W + neighbor_color[1], axis=0)
        new_color = neighbor_color[:, min_idx, ij[0], ij[1]]  # int[2, H, W]

        new_color = code2color(color2code(new_color).at[color[0], color[1]].min(color2code(new_color)))
        new_color = new_color[:, new_color[0], new_color[1]]
        return color, new_color

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
    cmp_sum = jnp.zeros(color.shape[-2:], jnp.asarray(data).dtype)
    cmp_sum = cmp_sum.at[color[0], color[1]].add(data)
    cmp_sum = cmp_sum[color[0], color[1]]
    return cmp_sum


@jax.jit
def boundary_sum(data, color, mask):
    H, W = mask.shape
    map_size = jnp.array([H, W], dtype=jnp.int32)

    ij = jnp.mgrid[:H, :W]
    delta_ij = jnp.array([
        [-1, 0],
        [0, 1],
        [1, 0],
        [0, -1],
    ])  # int[2, H, W]
    neighbor_ij = delta_ij[..., None, None] + ij[None, ...]  # int[4, 2, H, W]

    # handle map boundary.
    neighbor_ij = neighbor_ij.at[0, :, 0, :].set(INT32_MAX)
    neighbor_ij = neighbor_ij.at[1, :, :, W - 1].set(INT32_MAX)
    neighbor_ij = neighbor_ij.at[2, :, H - 1, :].set(INT32_MAX)
    neighbor_ij = neighbor_ij.at[3, :, :, 0].set(INT32_MAX)

    # non-barriers connect to nothing.
    neighbor_ij = jnp.where(mask, neighbor_ij, INT32_MAX)

    # get the neighbor color
    neighbor_color = color.at[:, neighbor_ij[:, 0], neighbor_ij[:, 1]].get(mode='fill',
                                                                           fill_value=INT32_MAX)  # int[2, 4, H, W]
    neighbor_color = jnp.transpose(neighbor_color, (1, 0, 2, 3))  # int[4, 2, H, W]

    # remove duplicated neighbor color
    unique_vmap = jax.vmap(
        jax.vmap(
            partial(jnp.unique, return_index=True, axis=0, size=4, fill_value=INT32_MAX),
            in_axes=-1,
            out_axes=-1,
        ),
        in_axes=-1,
        out_axes=-1,
    )
    color_code = color2code(neighbor_color)  # int[4, H, W]
    unique_neighbor_color, unique_idx = unique_vmap(color_code)  # (int[4, H, W], int[4, H, W])
    unique_idx = jnp.where(unique_neighbor_color == INT32_MAX, INT32_MAX, unique_idx)
    unique_neighbor_color = neighbor_color.at[unique_idx, :, ij[None, 0],
                                              ij[None, 1]].get(mode='fill', fill_value=INT32_MAX)  # int[4, H, W, 2]

    # accumulate the boundary data to the root of each component
    boundary_sum = jnp.zeros(color.shape[-2:], jnp.asarray(data).dtype)
    boundary_sum = boundary_sum.at[unique_neighbor_color[..., 0], unique_neighbor_color[..., 1]].add(data)
    boundary_sum = boundary_sum[color[0], color[1]]
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
