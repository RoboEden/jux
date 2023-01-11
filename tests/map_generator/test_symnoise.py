import chex
import jax
import numpy as np
from jax import numpy as jnp
from luxai_s2.map_generator.symnoise import SymmetricNoise as LuxSymmetricNoise
from luxai_s2.map_generator.symnoise import symmetrize as lux_symmetrize

from jux.map_generator.symnoise import SymmetryNoise, SymmetryType, symmetrize


class TestSymnoise(chex.TestCase):

    @chex.variants(with_jit=True, without_jit=True, with_device=True, without_device=True)
    def test_symmetrize(self):
        seed = 42
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)

        width = 5
        height = 5
        matrix_jax = jax.random.randint(subkey, shape=(height, width), minval=0, maxval=10)
        jux_symmetrize = self.variant(symmetrize, static_argnames=())

        matrix_np = np.array(matrix_jax)
        lux_symmetrize(matrix_np, "horizontal")
        matrix_lux_symmetrize = matrix_np
        assert jnp.allclose(
            matrix_lux_symmetrize,
            jux_symmetrize(matrix_jax, SymmetryType.HORIZONTAL).astype(jnp.int32),
            rtol=1e-05,
        )

    @chex.variants(with_jit=True, without_jit=True, with_device=True, without_device=True)
    def test_noise(self):
        seed = 42
        octaves = 3
        size = 5
        x = jnp.linspace(start=0, stop=1, num=size)
        y = jnp.linspace(start=0, stop=1, num=size)

        symmetry_noise_jux = SymmetryNoise(
            seed=seed,
            octaves=octaves,
            symmetry=SymmetryType.HORIZONTAL,
        )
        symmetry_noise_lux = LuxSymmetricNoise(
            seed=seed,
            octaves=octaves,
            symmetry="horizontal",
        )
        noise_func = self.variant(symmetry_noise_jux.noise)

        noise_jux = noise_func(x, y)
        noise_lux = symmetry_noise_lux.noise(x, y)

        # jax.debug.print("noise_jux: {noise}", noise=noise_jux)
        # jax.debug.print("noise_lux: {noise}", noise=noise_lux)
        assert jnp.allclose(noise_jux, noise_lux)
