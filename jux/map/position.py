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
