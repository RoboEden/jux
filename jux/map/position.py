from enum import IntEnum
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array
from luxai_s2.map.position import Position as LuxPosition

from jux.utils import INT8_MAX, INT32_MAX


class Position(NamedTuple):
    pos: jnp.int8 = jnp.full((2, ), fill_value=INT8_MAX, dtype=jnp.int8)  # int8[..., 2]

    @property
    def x(self) -> int:
        return self.pos[..., 0]

    @property
    def y(self) -> int:
        return self.pos[..., 1]

    @staticmethod
    def dtype() -> jnp.dtype:
        return Position.__annotations__['pos']

    @classmethod
    def new(cls, pos: Array):
        return cls(pos.astype(Position.__annotations__['pos']))

    @classmethod
    def from_lux(cls, lux_pos: LuxPosition) -> "Position":
        return cls(jnp.array(lux_pos.pos, dtype=jnp.int8))

    def to_lux(self) -> LuxPosition:
        return LuxPosition(np.array(self.pos))

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Position):
            return False
        return jnp.array_equal(self.pos, __o.pos)

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


direct2delta_xy = jnp.array(
    [
        [0, 0],  # stay
        [0, -1],  # up
        [1, 0],  # right
        [0, 1],  # down
        [-1, 0],  # left
    ],
    dtype=Position.dtype())
