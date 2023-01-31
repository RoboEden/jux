from functools import partial
from typing import NamedTuple, Optional, Type

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from jax.numpy.fft import fft
from luxai_s2.map_generator import GameMap as LuxGameMap
from luxai_s2.map_generator.visualize import viz as lux_viz

from jux.config import EnvConfig, JuxBufferConfig
from jux.map_generator.flood import boundary_sum, component_sum, flood_fill
from jux.map_generator.generator_config import CaveConfig, MapDistributionType, MapType, MountainConfig
from jux.map_generator.symnoise import SymmetryNoise, SymmetryType, symmetrize
from jux.utils import INT32_MAX


class GameMap(NamedTuple):
    rubble: jnp.int8  # int8[height, width]
    ice: jnp.bool_  # bool[height, width]
    ore: jnp.bool_  # bool[height, width]
    symmetry: SymmetryType  # jnp.int8

    @property
    def width(self) -> int:
        return self.rubble.shape[1]

    @property
    def height(self) -> int:
        return self.rubble.shape[0]

    @staticmethod
    def new(rubble: Array, ice: Array, ore: Array, symmetry: SymmetryType) -> "GameMap":
        return GameMap(
            GameMap.__annotations__['rubble'](rubble),
            GameMap.__annotations__['ice'](ice),
            GameMap.__annotations__['ore'](ore),
            jnp.int8(symmetry),
        )

    @staticmethod
    def random_map(seed: jnp.int32 = None,
                   map_type: Optional[MapType] = None,
                   symmetry: Optional[SymmetryType] = None,
                   width: jnp.int32 = None,
                   height: jnp.int32 = None,
                   map_distribution_type: jnp.int32 = None) -> "GameMap":
        noise = SymmetryNoise(seed=seed, symmetry=symmetry, octaves=3)
        map_rand = lax.switch(map_type, [
            partial(cave, width, height),
            partial(craters, width, height),
            partial(island, width, height),
            partial(mountain, width, height),
        ], symmetry, noise, map_distribution_type)
        return GameMap.new(
            map_rand.rubble,
            map_rand.ice,
            map_rand.ore,
            map_rand.symmetry,
        )

    @classmethod
    def from_lux(cls: Type['GameMap'], lux_map: LuxGameMap) -> "GameMap":
        rubble = jnp.array(lux_map.rubble.astype(np.int32))
        ice = jnp.array(lux_map.ice != 0)
        ore = jnp.array(lux_map.ore != 0)

        return cls.new(
            rubble,
            ice,
            ore,
            symmetry=SymmetryType.from_lux(lux_map.symmetry),
        )

    def to_lux(self) -> LuxGameMap:
        rubble = np.array(self.rubble, dtype=np.int32)
        ice = np.array(self.ice, dtype=np.int32)
        ore = np.array(self.ore, dtype=np.int32)

        return LuxGameMap(rubble, ice, ore, SymmetryType.to_lux(self.symmetry))

    def __eq__(self, __o: 'GameMap') -> bool:
        if not isinstance(__o, GameMap):
            return False
        return ((self.width == __o.width) & (self.height == __o.height) & (self.symmetry == __o.symmetry)
                & jnp.array_equal(self.rubble, __o.rubble)
                & jnp.array_equal(self.ice, __o.ice)
                & jnp.array_equal(self.ore, __o.ore))


def maximum_filter(matrix, window_dimensions=(4, 4)):
    height, width = matrix.shape
    matrix11 = jnp.flip(matrix).T
    matrix12 = jnp.flipud(matrix)
    matrix13 = matrix.T
    matrix21 = jnp.fliplr(matrix)
    matrix22 = matrix
    matrix23 = jnp.fliplr(matrix)
    matrix31 = matrix.T
    matrix32 = jnp.flipud(matrix)
    matrix33 = jnp.flip(matrix).T
    matrix = jnp.array([[matrix11, matrix12, matrix13], [matrix21, matrix22, matrix23], [matrix31, matrix32, matrix33]])
    matrix = jnp.transpose(matrix, axes=(0, 2, 1, 3)).reshape(3 * height, 3 * width)
    matrix = jnp.flip(matrix).T
    matrix = lax.reduce_window(
        operand=matrix,
        window_dimensions=window_dimensions,
        window_strides=(1, 1),
        init_value=0.0,
        padding="same",
        computation=jnp.maximum,
    )
    matrix = jnp.flip(matrix).T
    matrix = matrix[height:2 * height, width:2 * width]
    return matrix


def cave(
    width: jnp.int32,
    height: jnp.int32,
    symmetry: SymmetryType,
    noise: SymmetryNoise,
    map_distribution_type: MapDistributionType,
) -> "GameMap":
    config: CaveConfig = CaveConfig.new(map_distribution_type=map_distribution_type)
    seed = noise.seed
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    mask = jax.random.randint(key=subkey, minval=0, maxval=3, shape=(height, width))
    mask = (mask >= 1).astype(jnp.int32)
    # mask = symmetrize(mask, symmetry)

    # Build clumps of ones (will be interior of caves)
    for i in range(3):
        mask = jax.scipy.signal.convolve2d(
            mask,
            jnp.ones(shape=(3, 3)),
            mode="same",
            boundary="fill",
            fillvalue=0,
        ) // 6
    mask = 1 - mask

    # Create cave wall
    mask = (mask + maximum_filter(mask)).astype(jnp.int32)
    # fix bug where filter will cause it to be not symmetric...
    # mask = symmetrize(mask, symmetry)

    # Make some noisy rubble
    x = jnp.linspace(0, 1, width)
    y = jnp.linspace(0, 1, height)
    rubble = noise.noise(x, y) * 50 + 50
    rubble = jnp.round(rubble)
    rubble = rubble * (mask != 1) + (rubble // 5) * (mask == 1)  # Cave walls
    rubble = rubble * (mask != 0)  # Interior of cave

    # Make some noisy ice, most ice is on cave edges
    ice = noise.noise(x, y + 100)
    ice = ice * (mask <= 1)
    ice = ice * (mask != 0)
    mid_mask = (ice > jnp.percentile(ice, config.ice_mid_range[0])) & (ice < jnp.percentile(
        ice, config.ice_mid_range[1]))
    high_mask = (ice > jnp.percentile(ice, config.ice_high_range[0])) & (ice < jnp.percentile(
        ice, config.ice_high_range[1]))
    ice = mid_mask | high_mask

    # Make some noisy ore, most ore is outside caves
    ore = noise.noise(x, y - 100)
    ore = ore * (mask != 1)
    ore = ore * (mask != 0)
    mid_mask = (ore > jnp.percentile(ore, config.ore_mid_range[0])) & (ore < jnp.percentile(
        ore, config.ore_mid_range[1]))
    high_mask = (ore > jnp.percentile(ore, config.ore_high_range[0])) & (ore < jnp.percentile(
        ore, config.ore_high_range[1]))
    ore = mid_mask | high_mask

    return GameMap(
        rubble=rubble,
        ice=ice,
        ore=ore,
        symmetry=symmetry,
    )


def craters(
    width: jnp.int32,
    height: jnp.int32,
    symmetry: SymmetryType,
    noise: SymmetryNoise,
    map_distribution_type: MapDistributionType,
) -> "GameMap":
    min_craters = jnp.maximum(2, width * height // 1000)
    max_craters = jnp.maximum(4, width * height // 500)
    seed = noise.seed
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    num_craters = jax.random.randint(key=subkey, shape=(), minval=min_craters, maxval=max_craters + 1)

    # Mask = how many craters have hit the spot. When it symmetrizes, it will divide by 2.
    mask = jnp.zeros((height, width))
    x = jnp.linspace(0, 1, width)
    y = jnp.linspace(0, 1, height)
    crater_noise = noise.noise(x, y, frequency=10) * 0.5 + 0.75  # Don't want perfectly circular craters.
    xx, yy = jnp.mgrid[:int(width), :int(height)]

    # ice should be around edges of crater
    ice_mask = jnp.zeros((height, width))

    def _body_func(i, val):
        key, mask, ice_mask = val
        key, subkey = jax.random.split(key)
        cx = jax.random.randint(key=subkey, shape=(1, ), minval=0, maxval=width)
        key, subkey = jax.random.split(key)
        cy = jax.random.randint(key=subkey, shape=(1, ), minval=0, maxval=height)
        key, subkey = jax.random.split(key)
        cr_max = jnp.minimum(width, height) // 4
        cr = jax.random.randint(key=subkey, shape=(1, ), minval=3, maxval=cr_max)
        # cr = jax.random.randint(key=subkey, shape=(1, ), minval=3, maxval=height // 4)
        c = (xx - cx)**2 + (yy - cy)**2
        c = c.T * crater_noise
        mask = mask + (c < cr**2)
        edge = jnp.logical_and(c >= cr**2, c < 2 * cr**2)
        ice_mask = (~edge) * ice_mask + edge * 2
        return key, mask, ice_mask

    key, mask, ice_mask = lax.fori_loop(
        lower=0,
        upper=num_craters,
        body_fun=_body_func,
        init_val=(key, mask, ice_mask),
    )

    mask = mask.astype(jnp.float32)

    # mask = symmetrize(mask, symmetry)
    # ice_mask = symmetrize(ice_mask, symmetry)

    rubble = (jnp.minimum(mask, 1) * 90 + 10) * noise.noise(x, y + 100, frequency=3)
    ice = noise.noise(x, y - 100)
    ice = ice * (ice_mask != 0)
    ice = ice * (ice >= jnp.percentile(ice, 95))
    ice = jnp.round(50 * ice + 50).astype(jnp.bool_)

    ore = jnp.minimum(mask, 1) * noise.noise(x, y + 100, frequency=3)
    ore = ore * (ore >= jnp.percentile(ore, 95))
    ore = jnp.round(50 * ore + 50).astype(jnp.bool_)
    return GameMap(
        rubble=rubble,
        ice=ice,
        ore=ore,
        symmetry=symmetry,
    )


def dctn1(matrix, axes=(
    0,
    1,
)):
    for ax in axes:
        size = matrix.shape[ax]
        matrix = jnp.moveaxis(matrix, ax, 0)
        matrix = jnp.concatenate((matrix, matrix[1:-1, ...][::-1, ...]), axis=0)
        matrix = fft(matrix, axis=0).real[:size, ...]
        matrix = jnp.moveaxis(matrix, 0, ax)
    return matrix


idctn1 = lambda matrix, axes=(
    0,
    1,
): dctn1(matrix, axes=axes) / jnp.prod(jnp.array([2 * (matrix.shape[ax] - 1) for ax in axes]))


def solve_poisson(f):
    """
    Solves the Poisson equation
        ∇²p = f
    using the finite difference method with Neumann boundary conditions.
    References: https://elonen.iki.fi/code/misc-notes/neumann-cosine/
                https://scicomp.stackexchange.com/questions/12913/poisson-equation-with-neumann-boundary-conditions
    """
    nx, ny = f.shape

    # Transform to DCT space
    dct = dctn1(f)

    # Divide by magical factors
    cx = jnp.cos(jnp.pi * jnp.arange(nx) / (nx - 1))
    cy = jnp.cos(jnp.pi * jnp.arange(ny) / (ny - 1))
    f = cx[:, None] + cy[None, :] - 2

    dct = jnp.divide(dct, f)
    dct = jnp.nan_to_num(dct, copy=True, nan=0.0, posinf=0.0, neginf=0.0)

    # Return to normal space
    potential = idctn1(dct)
    return potential / 2


def convolve2d_reflect(matrix, kernel):
    height, width = matrix.shape
    matrix11 = jnp.flip(matrix).T
    matrix12 = jnp.flipud(matrix)
    matrix13 = matrix.T
    matrix21 = jnp.fliplr(matrix)
    matrix22 = matrix
    matrix23 = jnp.fliplr(matrix)
    matrix31 = matrix.T
    matrix32 = jnp.flipud(matrix)
    matrix33 = jnp.flip(matrix).T
    matrix = jnp.array([[matrix11, matrix12, matrix13], [matrix21, matrix22, matrix23], [matrix31, matrix32, matrix33]])
    matrix = jnp.transpose(matrix, axes=(0, 2, 1, 3)).reshape(3 * height, 3 * width)
    matrix = jax.scipy.signal.convolve2d(in1=matrix, in2=kernel, mode="same", boundary="fill", fillvalue=0)
    matrix = matrix[height:2 * height, width:2 * width]
    return matrix


def mountain(
    width: jnp.int32,
    height: jnp.int32,
    symmetry: SymmetryType,
    noise: SymmetryNoise,
    map_distribution_type: MapDistributionType,
) -> "GameMap":
    config: MountainConfig = MountainConfig.new(map_distribution_type=map_distribution_type)
    seed = noise.seed
    f = jnp.zeros((height, width))

    # Sprinkle a few mountains on the map.
    BOUND_MOUNTAINS = 100  # max(4, width * height // 375)

    min_mountains = jnp.maximum(2, width * height // 750)
    max_mountains = jnp.maximum(4, width * height // 375)

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    corr_mask = jnp.arange(start=0, stop=BOUND_MOUNTAINS)
    corr_mask = (((corr_mask < max_mountains) & (corr_mask >= min_mountains)
                  & jax.random.randint(key=subkey, shape=(1, BOUND_MOUNTAINS), minval=0, maxval=2)) |
                 (corr_mask < min_mountains))

    corr_mask = jnp.ones(shape=(2, 1), dtype=jnp.bool_) * corr_mask
    corr_mask = corr_mask.astype(jnp.bool_)

    key, subkey = jax.random.split(key)
    random_coor = jnp.floor(
        jax.random.uniform(key=subkey, shape=(2, BOUND_MOUNTAINS), minval=0, maxval=1) *
        jnp.array([[width], [height]])).astype(jnp.int32)
    random_coor = random_coor * corr_mask + (~corr_mask) * INT32_MAX
    y, x = random_coor[1, :], random_coor[0, :]
    f = f.at[y, x].add(-1, mode="drop")

    # f = symmetrize(f, symmetry)

    # mask will be floats in [0, 1], where 0 = no mountain, 1 = tallest peak
    mask = solve_poisson(f)
    # mask = symmetrize(mask, symmetry)  # in case of floating point errors
    x = jnp.linspace(0, 1, width)
    y = jnp.linspace(0, 1, height)
    mask = mask * (5 + noise.noise(x, y, frequency=3))
    mask = mask - jnp.amin(mask)

    # Find the valleys
    Lap = convolve2d_reflect(mask, jnp.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])) / 3  # Laplacian
    Dxx = convolve2d_reflect(mask, jnp.array([[
        1,
    ], [
        -2,
    ], [
        1,
    ]]))
    Dyy = convolve2d_reflect(mask, jnp.array([
        [1, -2, 1],
    ]))
    Dxy = convolve2d_reflect(mask, jnp.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]))
    det = 16 * Dxx * Dyy - Dxy**2  # Hessian determinant
    det = jnp.csingle(det)  # complex

    cond = Lap * (2 * Lap - jnp.sqrt(4 * Lap**2 - det)) / det - 0.25  # ratio of eigenvalues
    cond = jnp.abs(cond)  # should already be real except for floating point errors
    cond = jnp.maximum(cond, 1 / cond)
    # cond = symmetrize(cond, symmetry)  # for floating point errors

    bdry = jnp.abs(cond > 20) & (f == 0)
    color = flood_fill(bdry)
    cmp_sum = component_sum(mask, color)
    cmp_cnt = component_sum(1, color)
    cmp_mean = cmp_sum / cmp_cnt

    bdr_sum = boundary_sum(mask, color, bdry)
    bdr_cnt = boundary_sum(1, color, bdry)
    bdr_mean = bdr_sum / (bdr_cnt + jnp.finfo(jnp.float32).tiny)
    mask_mask = ~(bdry) & (cmp_mean < bdr_mean * 2) & (cmp_cnt * 5 <= width * height)
    mask = mask * (~mask_mask)
    mask = mask * (~bdry)

    positive_amin = jnp.amin(jnp.where(mask > 0, mask, jnp.inf))
    mask = jnp.where(mask > 0, mask - positive_amin, mask)
    mask = mask - jnp.amin(mask)
    mask = mask / jnp.amax(mask)

    rubble = (100 * mask).round()
    ice = (100 * mask)
    high_mask = (ice > jnp.percentile(ice, config.ice_high_range[0])) & (ice < jnp.percentile(
        ice, config.ice_high_range[1]))
    mid_mask = (ice > jnp.percentile(ice, config.ice_mid_range[0])) & (ice < jnp.percentile(
        ice, config.ice_mid_range[1]))
    low_mask = (ice > jnp.percentile(ice, config.ice_low_range[0])) & (ice < jnp.percentile(
        ice, config.ice_low_range[1]))
    ice = low_mask | mid_mask | high_mask
    ore = (100 * mask)

    mid_mask = (ore > jnp.percentile(ore, config.ore_mid_range[0])) & (ore < jnp.percentile(
        ore, config.ore_mid_range[1]))
    low_mask = (ore > jnp.percentile(ore, config.ore_low_range[0])) & (ore < jnp.percentile(
        ore, config.ore_low_range[1]))

    ore = low_mask | mid_mask

    return GameMap(
        rubble=rubble,
        ice=ice,
        ore=ore,
        symmetry=symmetry,
    )


def convolve2d_fill(matrix: Array, kernel: Array, fillvalue: jnp.float32):
    height_k, width_k = kernel.shape
    height, width = matrix.shape
    matrix_full = jnp.ones(shape=(height + height_k - 1, width + width_k - 1)) * fillvalue
    matrix_full = matrix_full.at[height_k // 2:-(height_k // 2) // 2, width_k // 2:-(width_k // 2) // 2].set(matrix)
    matrix_full = jax.scipy.signal.convolve2d(
        in1=matrix_full,
        in2=kernel,
        mode="same",
        boundary="fill",
        fillvalue=0,
    )
    matrix_full = matrix_full[height_k // 2:-(height_k // 2) // 2, width_k // 2:-(width_k // 2) // 2]
    return matrix_full


def island(
    width: jnp.int32,
    height: jnp.int32,
    symmetry: SymmetryType,
    noise: SymmetryNoise,
    map_distribution_type: MapDistributionType,
) -> "GameMap":
    # at the end, 0 = island, 1 = sea (of rubble)
    seed = noise.seed
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    mask = jax.random.randint(key=subkey, minval=0, maxval=4, shape=(height, width))
    mask = (mask >= 1) * 1

    def _cond_func(val):
        s, mask = val
        return jnp.sum(mask == 0) != s

    def _body_func(val):
        s, mask = val
        s = jnp.sum(mask == 0)
        mask = convolve2d_fill(
            matrix=mask,
            kernel=jnp.ones(shape=(3, 3)),
            fillvalue=1,
        )
        mask = jnp.int32(mask) // 6
        return s, mask

    s = -1
    s, mask = lax.while_loop(_cond_func, _body_func, (s, mask))

    # Shift every spot on the map a little, making the islands nosier.
    x = jnp.linspace(0, 1, width)
    y = jnp.linspace(0, 1, height)
    noise_dx = noise.noise(x, y, frequency=10) * 4 - 2
    noise_dy = noise.noise(x, y + 100, frequency=10) * 4 - 2

    xx, yy = jnp.mgrid[:int(width), :int(height)]
    new_xx = (xx + jnp.round(noise_dx)).astype(jnp.int32)
    new_xx = jnp.maximum(0, jnp.minimum(width - 1, new_xx))
    new_yy = (yy + jnp.round(noise_dy)).astype(jnp.int32)
    new_yy = jnp.maximum(0, jnp.minimum(height - 1, new_yy))

    mask = mask.at[yy, xx].set(mask[new_yy, new_xx])

    # mask = symmetrize(mask, symmetry)

    rubble = noise.noise(x, y - 100, frequency=3) * 50 + 50
    zero_mask = mask == 0
    rubble = rubble // 20 * zero_mask + rubble * (~zero_mask)

    # Unsure what to do about ice, ore right now. Place in pockets on islands?
    ice = noise.noise(x, y + 200, frequency=10)**2 * 100
    ice = ice * (~(ice < 50))
    ice = ice * (~(mask != 0))
    ice = ice.astype(jnp.bool_)

    ore = noise.noise(x, y - 200, frequency=10)**2 * 100
    ore = ore * (~(ore < 50))
    ore = ore * (~(mask != 0))
    ore = ore.astype(jnp.bool_)
    return GameMap(
        rubble=rubble,
        ice=ice,
        ore=ore,
        symmetry=symmetry,
    )
