from typing import NamedTuple

import jax
from jax import Array
from luxai2022.state import State as LuxState

from jux.config import EnvConfig
from jux.factory import Factory
from jux.team import Team
from jux.unit import Unit


class Board(NamedTuple):
    pass


class JuxState(NamedTuple):
    seed: jax.random.KeyArray  # the seed for reproducibility
    rng_state: jax.random.KeyArray  # current rng state
    env_steps: int
    board: Board
    weather_schedule: Array
    units: Unit
    factories: Factory
    teams: Team
    n_units: int
    n_factories: int
    env_cfg: EnvConfig

    @property
    def global_id(self):
        return self.n_units + self.n_factories

    @classmethod
    def from_lux(cls, lux_state: LuxState) -> "JuxState":
        # TODO
        pass

    def to_lux(self) -> LuxState:
        # TODO
        pass
