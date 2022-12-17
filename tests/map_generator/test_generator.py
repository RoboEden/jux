import os

import chex
import numpy as np

from jux.config import JuxBufferConfig
from jux.map_generator.generator import GameMap, LuxGameMap, MapType
from jux.map_generator.symnoise import SymmetryType

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.50"


def lux_game_map_eq(a: LuxGameMap, b: LuxGameMap) -> bool:
    return (a.height == b.height and a.width == b.width and a.symmetry == b.symmetry
            and np.array_equal(a.rubble, b.rubble) and np.array_equal(a.ice, b.ice) and np.array_equal(a.ore, b.ore))


class TestGameMap(chex.TestCase):

    def test_init(self):
        buf_cfg = JuxBufferConfig()

        map_type = np.random.choice(["Cave", "Mountain"])
        symmetry = np.random.choice(["horizontal", "vertical"])

        lux_game_map = LuxGameMap.random_map(
            map_type=map_type,
            symmetry=symmetry,
            width=buf_cfg.MAP_SIZE,
            height=buf_cfg.MAP_SIZE,
        )
        game_map = GameMap.from_lux(lux_game_map)

        assert lux_game_map_eq(game_map.to_lux(), lux_game_map)
        assert game_map == GameMap.from_lux((game_map.to_lux()))

    @chex.variants(with_jit=True, without_jit=True, with_device=True, without_device=True)
    def test_map(self):
        seed = 42
        map_type = MapType.MOUNTAIN
        symmetry = SymmetryType.HORIZONTAL
        width = 48
        height = 48
        # config.update("jax_disable_jit", True)
        map_generator = self.variant(
            GameMap.random_map,
            static_argnames=("width", "height"),
        )
        map_grid = map_generator(
            seed=seed,
            map_type=map_type,
            symmetry=symmetry,
            width=width,
            height=height,
        )
