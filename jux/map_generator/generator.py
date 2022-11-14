from enum import IntEnum
from typing import NamedTuple

from jax import Array
from luxai2022.map_generator import GameMap as LuxGameMap


class SymmetryType(IntEnum):
    HORIZONTAL = 0
    VERTICAL = 1
    ROTATIONAL = 2
    ANTI_DIAG = 3
    DIAG = 4


class GameMap(NamedTuple):
    rubble: Array  # int[height, width]
    ice: Array  # int[height, width]
    ore: Array  # int[height, width]
    symmetry: SymmetryType
    width: int
    height: int

    @classmethod
    def from_lux(cls, lux_map: LuxGameMap) -> "GameMap":
        # TODO
        pass

    def to_lux(self) -> LuxGameMap:
        # TODO
        pass
