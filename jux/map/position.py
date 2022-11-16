from enum import IntEnum
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


class Position(NamedTuple):
    pos: Array  # int32[..., 2]

    @classmethod
    def new(cls):
        return cls(jnp.zeros((2, ), dtype=jnp.int32))

    @property
    def x(self) -> int:
        return self.pos[..., 0]

    @property
    def y(self) -> int:
        return self.pos[..., 1]


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
