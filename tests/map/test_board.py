import numpy as np
from luxai2022.env import LuxAI2022

from jux.config import EnvConfig, JuxBufferConfig
from jux.map.board import Board, LuxBoard

from ..map_generator.test_generator import lux_game_map_eq


def lux_board_eq(a: LuxBoard, b: LuxBoard) -> bool:
    a_spawns0 = np.unique(a.spawns['player_0'], axis=0)
    b_spawns0 = np.unique(b.spawns['player_0'], axis=0)
    a_spawns1 = np.unique(a.spawns['player_1'], axis=0)
    b_spawns1 = np.unique(b.spawns['player_1'], axis=0)

    return (a.height == b.height and a.width == b.width and a.seed == b.seed
            and a.factories_per_team == b.factories_per_team and (a.lichen == b.lichen).all()
            and (a.lichen_strains == b.lichen_strains).all() and a.units_map == b.units_map
            and a.factory_map == b.factory_map and (a.factory_occupancy_map == b.factory_occupancy_map).all()
            and lux_game_map_eq(a.map, b.map) and (a_spawns0 == b_spawns0).all() and (a_spawns1 == b_spawns1).all()
            and (a.spawn_masks['player_0'] == b.spawn_masks['player_0']).all()
            and (a.spawn_masks['player_1'] == b.spawn_masks['player_1']).all())


class TestBoard:

    def test_from_to_lux(self):
        buf_cfg = JuxBufferConfig()

        lux = LuxAI2022()
        lux.reset()
        lux.step(dict(
            player_0=dict(faction='AlphaStrike', bid=0),
            player_1=dict(faction='AlphaStrike', bid=0),
        ))

        lux_board = lux.state.board
        jux_board = Board.from_lux(lux_board, buf_cfg)

        jux_board_from_lux = Board.from_lux(
            jux_board.to_lux(EnvConfig.from_lux(lux.env_cfg), lux.state.factories, lux.state.units),
            buf_cfg,
        )
        assert jux_board == jux_board_from_lux

        lux_board_from_jux = jux_board.to_lux(EnvConfig.from_lux(lux.env_cfg), lux.state.factories, lux.state.units)
        assert lux_board_eq(lux_board, lux_board_from_jux)
