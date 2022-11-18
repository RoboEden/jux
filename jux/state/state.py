from typing import Dict, List, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from luxai2022.state import State as LuxState

from jux.config import EnvConfig, JuxBufferConfig
from jux.factory import Factory, LuxFactory
from jux.map import Board
from jux.team import LuxTeam, Team
from jux.tree_util import batch_into_leaf, batch_out_of_leaf, concat_in_leaf
from jux.unit import LuxUnit, Unit


class State(NamedTuple):
    env_cfg: EnvConfig

    seed: int  # the seed for reproducibility
    rng_state: jax.random.KeyArray  # current rng state

    env_steps: int
    board: Board
    weather_schedule: Array

    units: Unit  # Unit[2, MAX_N_UNITS]
    n_units: Array  # int[2]

    factories: Factory  # Factory[2, MAX_N_FACTORIES]
    n_factories: Array  # int[2]

    teams: Team  # Team[2]

    @property
    def global_id(self):
        return self.n_units[0] + self.n_units[1] + self.n_factories[0] + self.n_factories[1]

    @classmethod
    def from_lux(cls, lux_state: LuxState, buf_cfg: JuxBufferConfig) -> "State":
        env_cfg = EnvConfig.from_lux(lux_state.env_cfg)

        # convert units
        units: Tuple[List[Unit], List[Unit]] = (
            [Unit.from_lux(unit, env_cfg) for unit in lux_state.units['player_0'].values()],
            [Unit.from_lux(unit, env_cfg) for unit in lux_state.units['player_1'].values()],
        )
        units[0].sort(key=lambda unit: unit.unit_id)  # sort
        units[1].sort(key=lambda unit: unit.unit_id)
        n_units = [
            len(lux_state.units['player_0']),
            len(lux_state.units['player_1']),
        ]
        empty_unit = Unit.new(0, 0, 0, env_cfg)
        empty_unit = jax.tree_util.tree_map(lambda x: jnp.array(x)[None, ...], empty_unit)
        padding_units = (  # padding to length of buf_cfg.max_units
            jax.tree_util.tree_map(lambda x: x.repeat(buf_cfg.MAX_N_UNITS - n_units[0], axis=0), empty_unit),
            jax.tree_util.tree_map(lambda x: x.repeat(buf_cfg.MAX_N_UNITS - n_units[1], axis=0), empty_unit),
        )
        units: Unit = batch_into_leaf([  # batch into leaf
            concat_in_leaf([batch_into_leaf(units[0]), padding_units[0]]) if n_units[0] > 0 else padding_units[0],
            concat_in_leaf([batch_into_leaf(units[1]), padding_units[1]]) if n_units[1] > 0 else padding_units[1],
        ])
        n_units = jnp.array(n_units)

        # convert factories
        factories: Tuple[List[Factory], List[Factory]] = (
            [Factory.from_lux(fac) for fac in lux_state.factories['player_0'].values()],
            [Factory.from_lux(fac) for fac in lux_state.factories['player_1'].values()],
        )
        factories[0].sort(key=lambda fac: fac.unit_id)  # sort
        factories[1].sort(key=lambda fac: fac.unit_id)
        n_factories = [
            len(lux_state.factories['player_0']),
            len(lux_state.factories['player_1']),
        ]
        f = Factory()
        factories = (  # padding to length of buf_cfg.max_factories
            factories[0] + [f] * (buf_cfg.MAX_N_FACTORIES - n_factories[0]),
            factories[1] + [f] * (buf_cfg.MAX_N_FACTORIES - n_factories[1]),
        )
        factories: Factory = batch_into_leaf([  # batch into leaf
            batch_into_leaf(factories[0]),
            batch_into_leaf(factories[1]),
        ])
        n_factories = jnp.array(n_factories)

        teams: List[Team] = [Team.from_lux(team, buf_cfg) for team in lux_state.teams.values()]
        teams.sort(key=lambda team: team.team_id)
        teams: Team = batch_into_leaf(teams)

        return State(
            env_cfg=env_cfg,
            seed=lux_state.seed,
            rng_state=jax.random.PRNGKey(lux_state.seed),
            env_steps=lux_state.env_steps,
            board=Board.from_lux(lux_state.board, buf_cfg),
            weather_schedule=jnp.array(lux_state.weather_schedule),
            units=units,
            n_units=n_units,
            factories=factories,
            n_factories=n_factories,
            teams=teams,
        )

    def to_lux(self) -> LuxState:
        lux_env_cfg = self.env_cfg.to_lux()

        # convert teams
        lux_teams: List[Team] = batch_out_of_leaf(self.teams)
        lux_teams: Dict[str, LuxTeam] = {f"player_{team.team_id}": team.to_lux() for team in lux_teams}

        # convert units
        def _to_lux_units(units: Unit, n_unit: int) -> Dict[str, LuxUnit]:
            units: List[Unit] = batch_out_of_leaf(units)[:n_unit]
            units: List[LuxUnit] = [u.to_lux(lux_teams, lux_env_cfg) for u in units]
            return {u.unit_id: u for u in units}

        lux_units = batch_out_of_leaf(self.units)
        n_units = self.n_units
        lux_units: Dict[str, Dict[str, Unit]] = {
            'player_0': _to_lux_units(lux_units[0], n_units[0]),
            'player_1': _to_lux_units(lux_units[1], n_units[1]),
        }

        # convert factories
        def _to_lux_factories(factories: Factory, n_factory: int) -> Dict[str, LuxFactory]:
            factories: List[Factory] = batch_out_of_leaf(factories)[:n_factory]
            factories: List[LuxFactory] = [f.to_lux(lux_teams) for f in factories]
            return {f.unit_id: f for f in factories}

        lux_factories = batch_out_of_leaf(self.factories)
        n_factories = self.n_factories
        lux_factories: Dict[str, Dict[str, LuxFactory]] = {
            'player_0': _to_lux_factories(lux_factories[0], n_factories[0]),
            'player_1': _to_lux_factories(lux_factories[1], n_factories[1]),
        }

        return LuxState(
            seed_rng=np.random.RandomState(self.seed),
            seed=self.seed,
            env_steps=int(self.env_steps),
            env_cfg=lux_env_cfg,
            board=self.board.to_lux(lux_env_cfg, lux_factories, lux_units),
            weather_schedule=np.array(self.weather_schedule),
            units=lux_units,
            factories=lux_factories,
            teams=lux_teams,
            global_id=int(self.global_id),
        )

    def __eq__(self, other: 'State') -> bool:
        if not isinstance(other, State):
            return False

        def teams_eq(teams_a: Team, teams_b: Team) -> bool:
            teams_a = batch_out_of_leaf(teams_a)
            teams_b = batch_out_of_leaf(teams_b)
            return teams_a == teams_b

        def units_eq(units_a: Unit, n_units_a: Array, units_b: Unit, n_units_b: Array) -> bool:
            if not jnp.array_equal(n_units_a, n_units_b):
                return False
            units_a_0 = jax.tree_util.tree_map(lambda x: x[0, :n_units_a[0]], units_a)
            units_b_0 = jax.tree_util.tree_map(lambda x: x[0, :n_units_b[0]], units_b)
            if not batch_out_of_leaf(units_a_0) == batch_out_of_leaf(units_b_0):
                return False
            units_a_1 = jax.tree_util.tree_map(lambda x: x[1, :n_units_a[1]], units_a)
            units_b_1 = jax.tree_util.tree_map(lambda x: x[1, :n_units_b[1]], units_b)
            if not batch_out_of_leaf(units_a_1) == batch_out_of_leaf(units_b_1):
                return False
            return True

        def factories_eq(factories_a: Factory, n_factories_a: Array, factories_b: Factory,
                         n_factories_b: Array) -> bool:
            if not jnp.array_equal(n_factories_a, n_factories_b):
                return False
            factories_a_0 = jax.tree_util.tree_map(lambda x: x[0, :n_factories_a[0]], factories_a)
            factories_b_0 = jax.tree_util.tree_map(lambda x: x[0, :n_factories_b[0]], factories_b)
            if not batch_out_of_leaf(factories_a_0) == batch_out_of_leaf(factories_b_0):
                return False
            factories_a_1 = jax.tree_util.tree_map(lambda x: x[1, :n_factories_a[1]], factories_a)
            factories_b_1 = jax.tree_util.tree_map(lambda x: x[1, :n_factories_b[1]], factories_b)
            if not batch_out_of_leaf(factories_a_1) == batch_out_of_leaf(factories_b_1):
                return False
            return True

        return (self.env_cfg == other.env_cfg and self.env_steps == other.env_steps and self.board == other.board
                and jnp.array_equal(self.weather_schedule, other.weather_schedule)
                and teams_eq(self.teams, other.teams)
                and factories_eq(self.factories, self.n_factories, other.factories, other.n_factories)
                and units_eq(self.units, self.n_units, other.units, other.n_units))
