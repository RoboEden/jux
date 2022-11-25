from enum import IntEnum
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array
from luxai2022.map.position import Position as LuxPosition

INT32_MAX = jnp.iinfo(jnp.int32).max


class Position(NamedTuple):
    pos: Array = jnp.full((2, ), fill_value=INT32_MAX, dtype=jnp.int32)  # int32[..., 2]

    @property
    def x(self) -> int:
        return self.pos[..., 0]

    @property
    def y(self) -> int:
        return self.pos[..., 1]

    @classmethod
    def from_lux(cls, lux_pos: LuxPosition) -> "Position":
        return cls(jnp.array(lux_pos.pos, dtype=jnp.int32))

    def to_lux(self) -> LuxPosition:
        return LuxPosition(np.array(self.pos))

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Position) and np.array_equal(self.pos, __o.pos)

    def __add__(self, other: "Position") -> "Position":
        return Position(self.pos + other.pos)

    def __sub__(self, other: "Position") -> "Position":
        return Position(self.pos - other.pos)


class Direction(IntEnum):
    """Direction enum"""
    CENTER = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


direct2delta_xy = jnp.array([
    [0, 0],  # stay
    [-1, 0],  # up
    [0, 1],  # right
    [1, 0],  # down
    [0, -1],  # left
])
