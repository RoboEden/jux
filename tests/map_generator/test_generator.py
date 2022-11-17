import numpy as np

from jux.config import JuxBufferConfig
from jux.map_generator.generator import GameMap, LuxGameMap


def lux_game_map_eq(a: LuxGameMap, b: LuxGameMap) -> bool:
    return (a.height == b.height and a.width == b.width and a.symmetry == b.symmetry
            and np.array_equal(a.rubble, b.rubble) and np.array_equal(a.ice, b.ice) and np.array_equal(a.ore, b.ore))


class TestGameMap:

    def test_init(self):
        buf_cfg = JuxBufferConfig()

        map_type = np.random.choice(["Cave", "Mountain"])
        symmetry = np.random.choice(["horizontal", "vertical"])

        lux_game_map = LuxGameMap.random_map(
            map_type=map_type,
            symmetry=symmetry,
            width=buf_cfg.MAX_MAP_SIZE,
            height=buf_cfg.MAX_MAP_SIZE,
        )
        game_map = GameMap.from_lux(lux_game_map, buf_cfg)

        assert lux_game_map_eq(game_map.to_lux(), lux_game_map)
        assert game_map == GameMap.from_lux((game_map.to_lux()), buf_cfg)
