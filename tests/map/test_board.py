import chex
import numpy as np
from luxai_s2.env import LuxAI_S2

from jux.config import EnvConfig, JuxBufferConfig
from jux.map.board import Board, LuxBoard

from ..map_generator.test_generator import lux_game_map_eq


def lux_board_eq(a: LuxBoard, b: LuxBoard) -> bool:

    return (a.height == b.height and a.width == b.width and a.seed == b.seed
            and a.factories_per_team == b.factories_per_team and np.array_equal(a.lichen, b.lichen)
            and np.array_equal(a.lichen_strains, b.lichen_strains) and a.units_map == b.units_map
            and a.factory_map == b.factory_map and np.array_equal(a.factory_occupancy_map, b.factory_occupancy_map)
            and lux_game_map_eq(a.map, b.map) and np.array_equal(a.valid_spawns_mask, b.valid_spawns_mask))


class TestBoard(chex.TestCase):

    def test_from_to_lux(self):
        buf_cfg = JuxBufferConfig()

        lux = LuxAI_S2()
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

    @chex.variants(with_jit=True, without_jit=True, with_device=True, without_device=True)
    def test(self):
        new_board = self.variant(Board.new, static_argnames=["buf_cfg"])
        board = new_board(seed=42, env_cfg=EnvConfig(), buf_cfg=JuxBufferConfig())
