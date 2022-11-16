from enum import IntEnum
from typing import NamedTuple, Type

import jax.numpy as jnp
import numpy as np
from jax import Array
from luxai2022.map_generator import GameMap as LuxGameMap

from jux.config import EnvConfig, JuxBufferConfig


class SymmetryType(IntEnum):
    HORIZONTAL = 0
    VERTICAL = 1
    ROTATIONAL = 2
    ANTI_DIAG = 3
    DIAG = 4

    @classmethod
    def from_lux(cls, lux_symmetry: str) -> "SymmetryType":
        idx = ["horizontal", "vertical", "rotational", "/", "\\"].index(lux_symmetry)
        return cls(idx)

    def to_lux(self) -> str:
        return ["horizontal", "vertical", "rotational", "/", "\\"][self]


class GameMap(NamedTuple):
    rubble: Array  # int[height, width]
    ice: Array  # bool[height, width]
    ore: Array  # bool[height, width]
    symmetry: SymmetryType
    width: int
    height: int

    @classmethod
    def from_lux(cls: Type['GameMap'], lux_map: LuxGameMap, buf_cfg: JuxBufferConfig) -> "GameMap":
        buf_size = (buf_cfg.MAX_MAP_SIZE, buf_cfg.MAX_MAP_SIZE)
        height, width = lux_map.height, lux_map.width

        rubble = jnp.empty(buf_size, dtype=jnp.float32).at[:height, :width].set(lux_map.rubble)
        ice = jnp.empty(buf_size, dtype=jnp.bool_).at[:height, :width].set(lux_map.ice != 0)
        ore = jnp.empty(buf_size, dtype=jnp.bool_).at[:height, :width].set(lux_map.ore != 0)

        return cls(
            rubble,
            ice,
            ore,
            symmetry=SymmetryType.from_lux(lux_map.symmetry),
            width=width,
            height=height,
        )

    def to_lux(self) -> LuxGameMap:
        width, height = self.width, self.height

        rubble = np.array(self.rubble[:height, :width], dtype=np.int32)
        ice = np.array(self.ice[:height, :width], dtype=np.int32)
        ore = np.array(self.ore[:height, :width], dtype=np.int32)

        return LuxGameMap(rubble, ice, ore, SymmetryType.to_lux(self.symmetry))

    def __eq__(self, __o: object) -> bool:
        width, height = self.width, self.height
        return (self.width == __o.width and self.height == __o.height and self.symmetry == __o.symmetry
                and jnp.array_equal(self.rubble[:height, :width], __o.rubble[:height, :width])
                and jnp.array_equal(self.ice[:height, :width], __o.ice[:height, :width])
                and jnp.array_equal(self.ore[:height, :width], __o.ore[:height, :width]))
