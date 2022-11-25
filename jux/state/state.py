from typing import Dict, List, NamedTuple, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental import checkify
from luxai2022.state import State as LuxState

import jux
from jux.actions import ActionQueue, FactoryAction, UnitAction, UnitActionType
from jux.config import EnvConfig, JuxBufferConfig
from jux.factory import Factory, LuxFactory
from jux.map import Board, Weather
from jux.map.position import Direction, Position, direct2delta_xy
from jux.team import LuxTeam, Team
from jux.unit import LuxUnit, Unit, UnitType
from jux.unit_cargo import ResourceType

INT32_MAX = jnp.iinfo(jnp.int32).max


class JuxAction(NamedTuple):
    factory_action: Array  # int[2, MAX_N_FACTORIES]
    unit_action_queue: UnitAction  # UnitAction[2, MAX_N_UNITS, UNIT_ACTION_QUEUE_SIZE]
    unit_action_queue_count: Array  # int[2, MAX_N_UNITS]
    unit_action_queue_update: Array  # bool[2, MAX_N_UNITS]


class State(NamedTuple):
    env_cfg: EnvConfig

    seed: int  # the seed for reproducibility
    rng_state: jax.random.KeyArray  # current rng state

    env_steps: int
    board: Board
    weather_schedule: Array

    units: Unit  # Unit[2, MAX_N_UNITS], the unit_id and team_id of non-existent units are jnp.iinfo(jnp.int32).max
    unit_id2idx: Array  # [MAX_N_UNITS, 2], the idx of non-existent units is jnp.iinfo(jnp.int32).max
    n_units: Array  # int[2]
    '''
    The `unit_id2idx` is organized such that
    ```
    [team_id, unit_idx] = unit_id2idx[unit_id]
    assert units[team_id, unit_idx].unit_id == unit_id
    ```

    For non-existent units, its `unit_id`, `team_id`, `pos`, and `unit_idx` are all INT32_MAX.
    ```
    INT32_MAX = jnp.iinfo(np.int32).max
    assert unit.unit_id[0, n_units[0]:] == INT32_MAX
    assert unit.team_id[0, n_units[0]:] == INT32_MAX
    assert (unit.pos.pos[0, n_units[0]:] == INT32_MAX).all()

    assert unit.unit_id[1, n_units[1]:] == INT32_MAX
    assert unit.team_id[1, n_units[1]:] == INT32_MAX
    assert (unit.pos.pos[1, n_units[1]:] == INT32_MAX).all()
    ```
    '''

    factories: Factory  # Factory[2, MAX_N_FACTORIES]
    factory_id2idx: Array  # int[MAX_N_FACTORIES, 2]
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
        empty_unit = Unit.empty(env_cfg, env_cfg.UNIT_ACTION_QUEUE_SIZE)
        empty_unit = jax.tree_util.tree_map(lambda x: jnp.array(x)[None, ...], empty_unit)
        padding_units = (  # padding to length of buf_cfg.max_units
            jax.tree_util.tree_map(lambda x: x.repeat(buf_cfg.MAX_N_UNITS - n_units[0], axis=0), empty_unit),
            jax.tree_util.tree_map(lambda x: x.repeat(buf_cfg.MAX_N_UNITS - n_units[1], axis=0), empty_unit),
        )
        units: Unit = jux.tree_util.batch_into_leaf([  # batch into leaf
            jux.tree_util.concat_in_leaf([jux.tree_util.batch_into_leaf(units[0]), padding_units[0]])
            if n_units[0] > 0 else padding_units[0],
            jux.tree_util.concat_in_leaf([jux.tree_util.batch_into_leaf(units[1]), padding_units[1]])
            if n_units[1] > 0 else padding_units[1],
        ])
        n_units = jnp.array(n_units)
        unit_id2idx = State.generate_unit_id2idx(units, buf_cfg.MAX_N_UNITS)

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
        f = Factory.empty()
        factories = (  # padding to length of buf_cfg.max_factories
            factories[0] + [f] * (buf_cfg.MAX_N_FACTORIES - n_factories[0]),
            factories[1] + [f] * (buf_cfg.MAX_N_FACTORIES - n_factories[1]),
        )
        factories: Factory = jux.tree_util.batch_into_leaf([  # batch into leaf
            jux.tree_util.batch_into_leaf(factories[0]),
            jux.tree_util.batch_into_leaf(factories[1]),
        ])
        n_factories = jnp.array(n_factories)
        factory_id2idx = State.generate_factory_id2idx(factories, buf_cfg.MAX_N_FACTORIES)

        teams: List[Team] = [Team.from_lux(team, buf_cfg) for team in lux_state.teams.values()]
        teams.sort(key=lambda team: team.team_id)
        teams: Team = jux.tree_util.batch_into_leaf(teams)

        state = State(
            env_cfg=env_cfg,
            seed=lux_state.seed,
            rng_state=jax.random.PRNGKey(lux_state.seed),
            env_steps=lux_state.env_steps,
            board=Board.from_lux(lux_state.board, buf_cfg),
            weather_schedule=jnp.array(lux_state.weather_schedule),
            units=units,
            n_units=n_units,
            unit_id2idx=unit_id2idx,
            factories=factories,
            n_factories=n_factories,
            factory_id2idx=factory_id2idx,
            teams=teams,
        )
        state.check_id2idx()
        return state

    @staticmethod
    def generate_unit_id2idx(units: Unit, max_n_units: int) -> Array:
        '''
        organize unit_id2idx such that
            unit_id2idx[unit_id] == [team_id, unit_idx]
            units[team_id, unit_idx].unit_id == unit_id
        '''
        unit_id2idx = jnp.ones((max_n_units, 2), dtype=np.int32) * jnp.iinfo(np.int32).max
        unit_id2idx = unit_id2idx.at[units.unit_id[..., 0, :]].set(
            jnp.array([
                0 * jnp.ones(max_n_units, dtype=np.int32),
                jnp.arange(max_n_units),
            ]).T,
            mode='drop',
        )
        unit_id2idx = unit_id2idx.at[units.unit_id[..., 1, :]].set(
            jnp.array([
                1 * jnp.ones(max_n_units, dtype=np.int32),
                jnp.arange(max_n_units),
            ]).T,
            mode='drop',
        )
        return unit_id2idx

    @staticmethod
    def generate_factory_id2idx(factories: Factory, max_n_factories: int) -> Array:
        factory_id2idx = jnp.ones((max_n_factories, 2), dtype=np.int32) * jnp.iinfo(np.int32).max
        factory_id2idx = factory_id2idx.at[factories.unit_id[..., 0, :]].set(
            jnp.array([
                0 * jnp.ones(max_n_factories, dtype=np.int32),
                jnp.arange(max_n_factories),
            ]).T,
            mode='drop',
        )
        factory_id2idx = factory_id2idx.at[factories.unit_id[..., 1, :]].set(
            jnp.array([
                1 * jnp.ones(max_n_factories, dtype=np.int32),
                jnp.arange(max_n_factories),
            ]).T,
            mode='drop',
        )
        return factory_id2idx

    def check_id2idx(self):
        n_units = self.n_units[0]
        unit_id = self.units.unit_id[0, :n_units]
        unit_id2idx = self.unit_id2idx[unit_id]
        assert jnp.array_equiv(unit_id2idx[:, 0], 0)
        assert jnp.array_equal(unit_id2idx[:, 1], jnp.arange(n_units))

        n_units = self.n_units[1]
        unit_id = self.units.unit_id[1, :n_units]
        unit_id2idx = self.unit_id2idx[unit_id]
        assert jnp.array_equiv(unit_id2idx[:, 0], 1)
        assert jnp.array_equal(unit_id2idx[:, 1], jnp.arange(n_units))

        n_factories = self.n_factories[0]
        factory_id = self.factories.unit_id[0, :n_factories]
        factory_id2idx = self.factory_id2idx[factory_id]
        assert jnp.array_equiv(factory_id2idx[:, 0], 0)
        assert jnp.array_equal(factory_id2idx[:, 1], jnp.arange(n_factories))

        n_factories = self.n_factories[1]
        factory_id = self.factories.unit_id[1, :n_factories]
        factory_id2idx = self.factory_id2idx[factory_id]
        assert jnp.array_equiv(factory_id2idx[:, 0], 1)
        assert jnp.array_equal(factory_id2idx[:, 1], jnp.arange(n_factories))

    def to_lux(self) -> LuxState:
        lux_env_cfg = self.env_cfg.to_lux()

        # convert teams
        lux_teams: List[Team] = jux.tree_util.batch_out_of_leaf(self.teams)
        lux_teams: Dict[str, LuxTeam] = {f"player_{team.team_id}": team.to_lux() for team in lux_teams}

        # convert units
        def _to_lux_units(units: Unit, n_unit: int) -> Dict[str, LuxUnit]:
            units: List[Unit] = jux.tree_util.batch_out_of_leaf(units)[:n_unit]
            units: List[LuxUnit] = [u.to_lux(lux_teams, lux_env_cfg) for u in units]
            return {u.unit_id: u for u in units}

        lux_units = jux.tree_util.batch_out_of_leaf(self.units)
        n_units = self.n_units
        lux_units: Dict[str, Dict[str, Unit]] = {
            'player_0': _to_lux_units(lux_units[0], n_units[0]),
            'player_1': _to_lux_units(lux_units[1], n_units[1]),
        }

        # convert factories
        def _to_lux_factories(factories: Factory, n_factory: int) -> Dict[str, LuxFactory]:
            factories: List[Factory] = jux.tree_util.batch_out_of_leaf(factories)[:n_factory]
            factories: List[LuxFactory] = [f.to_lux(lux_teams) for f in factories]
            return {f.unit_id: f for f in factories}

        lux_factories = jux.tree_util.batch_out_of_leaf(self.factories)
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

        self.check_id2idx()
        other.check_id2idx()

        def teams_eq(teams_a: Team, teams_b: Team) -> bool:
            teams_a = jux.tree_util.batch_out_of_leaf(teams_a)
            teams_b = jux.tree_util.batch_out_of_leaf(teams_b)
            return teams_a == teams_b

        def units_eq(units_a: Unit, n_units_a: Array, units_b: Unit, n_units_b: Array) -> bool:
            if not jnp.array_equal(n_units_a, n_units_b):
                return False
            units_a_0 = jax.tree_util.tree_map(lambda x: x[0, :n_units_a[0]], units_a)
            units_b_0 = jax.tree_util.tree_map(lambda x: x[0, :n_units_b[0]], units_b)
            units_a_0 = jux.tree_util.batch_out_of_leaf(units_a_0)
            units_b_0 = jux.tree_util.batch_out_of_leaf(units_b_0)
            units_a_0.sort(key=lambda x: x.unit_id)
            units_b_0.sort(key=lambda x: x.unit_id)
            if not units_a_0 == units_b_0:
                return False

            units_a_1 = jax.tree_util.tree_map(lambda x: x[1, :n_units_a[1]], units_a)
            units_b_1 = jax.tree_util.tree_map(lambda x: x[1, :n_units_b[1]], units_b)
            units_a_1 = jux.tree_util.batch_out_of_leaf(units_a_1)
            units_b_1 = jux.tree_util.batch_out_of_leaf(units_b_1)
            units_a_1.sort(key=lambda x: x.unit_id)
            units_b_1.sort(key=lambda x: x.unit_id)
            if not units_a_1 == units_b_1:
                return False
            return True

        def factories_eq(factories_a: Factory, n_factories_a: Array, factories_b: Factory,
                         n_factories_b: Array) -> bool:
            if not jnp.array_equal(n_factories_a, n_factories_b):
                return False
            factories_a_0 = jax.tree_util.tree_map(lambda x: x[0, :n_factories_a[0]], factories_a)
            factories_b_0 = jax.tree_util.tree_map(lambda x: x[0, :n_factories_b[0]], factories_b)
            factories_a_0 = jux.tree_util.batch_out_of_leaf(factories_a_0)
            factories_b_0 = jux.tree_util.batch_out_of_leaf(factories_b_0)
            factories_a_0.sort(key=lambda x: x.unit_id)
            factories_b_0.sort(key=lambda x: x.unit_id)
            if not factories_a_0 == factories_b_0:
                return False

            factories_a_1 = jax.tree_util.tree_map(lambda x: x[1, :n_factories_a[1]], factories_a)
            factories_b_1 = jax.tree_util.tree_map(lambda x: x[1, :n_factories_b[1]], factories_b)
            factories_a_1 = jux.tree_util.batch_out_of_leaf(factories_a_1)
            factories_b_1 = jux.tree_util.batch_out_of_leaf(factories_b_1)
            factories_a_1.sort(key=lambda x: x.unit_id)
            factories_b_1.sort(key=lambda x: x.unit_id)
            if not factories_a_1 == factories_b_1:
                return False
            return True

        return (self.env_cfg == other.env_cfg and self.env_steps == other.env_steps and self.board == other.board
                and jnp.array_equal(self.weather_schedule, other.weather_schedule)
                and teams_eq(self.teams, other.teams)
                and factories_eq(self.factories, self.n_factories, other.factories, other.n_factories)
                and units_eq(self.units, self.n_units, other.units, other.n_units))

    @property
    def MAX_N_FACTORIES(self):
        return self.factories.unit_id.shape[-1]

    @property
    def MAX_N_UNITS(self):
        return self.units.unit_id.shape[-1]

    @property
    def UNIT_ACTION_QUEUE_SIZE(self):
        return self.units.action_queue.data.shape[-2]

    @property
    def factory_idx(self):
        factory_idx = jnp.array([
            jnp.arange(self.MAX_N_FACTORIES),
            jnp.arange(self.MAX_N_FACTORIES),
        ])
        return factory_idx

    @property
    def factory_mask(self):
        factory_mask = self.factory_idx < self.n_factories[:, None]
        chex.assert_shape(factory_mask, (2, self.MAX_N_FACTORIES))
        return factory_mask

    @property
    def unit_idx(self):
        unit_idx = jnp.array([
            jnp.arange(self.MAX_N_UNITS),
            jnp.arange(self.MAX_N_UNITS),
        ])
        return unit_idx

    @property
    def unit_mask(self):
        unit_mask = self.unit_idx < self.n_units[:, None]
        chex.assert_shape(unit_mask, (2, self.MAX_N_UNITS))
        return unit_mask

    def parse_actions_from_dict(self, actions: Dict[str, Dict[str, Union[int, Array]]]) -> Tuple[Array, Array]:
        # TODO
        factory_action = np.empty((2, self.MAX_N_FACTORIES), dtype=np.int32)
        factory_action.fill(FactoryAction.DO_NOTHING.value)

        unit_action_queue = np.empty((2, self.MAX_N_UNITS, self.env_cfg.UNIT_ACTION_QUEUE_SIZE, 5), dtype=np.int32)
        unit_action_queue_count = np.zeros((2, self.MAX_N_UNITS), dtype=np.int32)
        unit_action_queue_update = np.zeros((2, self.MAX_N_UNITS), dtype=np.bool_)

        for player_id, player_actions in actions.items():
            player_id = int(player_id.split('_')[-1])
            for unit_id, action in player_actions.items():
                if unit_id.startswith('factory_'):
                    unit_id = int(unit_id.split('_')[-1])
                    pid, idx = self.factory_id2idx[unit_id]
                    assert pid == player_id
                    assert 0 <= idx < self.n_factories[player_id]
                    factory_action[player_id, idx] = action
                elif unit_id.startswith('unit_'):
                    unit_id = int(unit_id.split('_')[-1])
                    pid, idx = self.unit_id2idx[unit_id]
                    assert pid == player_id
                    assert 0 <= idx < self.n_units[player_id]

                    queue_size = len(action)
                    unit_action_queue[player_id, idx, :queue_size, :] = action
                    unit_action_queue_count[player_id, idx] = queue_size
                    unit_action_queue_update[player_id, idx] = True
                else:
                    raise ValueError(f'Unknown unit_id: {unit_id}')

        return JuxAction(
            factory_action,
            UnitAction(unit_action_queue),
            unit_action_queue_count,
            unit_action_queue_update,
        )

    def check_actions(self, actions: JuxAction) -> None:
        chex.assert_shape(actions.factory_action, (2, self.MAX_N_FACTORIES))
        chex.assert_shape(actions.unit_action_queue, (2, self.MAX_N_UNITS, self.env_cfg.UNIT_ACTION_QUEUE_SIZE, 5))
        chex.assert_shape(actions.unit_action_queue_count, (2, self.MAX_N_UNITS))
        chex.assert_shape(actions.unit_action_queue_update, (2, self.MAX_N_UNITS))

    # def step(self, actions: JuxAction) -> 'State':
    #     # TODO
    #     checkify.check(
    #         jnp.array_equiv(self.board.factories_per_team, self.board.factories_per_team[0]),
    #         "all envs shall have the same factories_per_team",
    #     )
    #     early_game = ((self.env_steps == 0) |
    #                   (self.env_cfg.BIDDING_SYSTEM & self.env_steps <= self.board.factories_per_team + 1)).all()
    #     return jax.lax(
    #         cond=early_game,
    #         true_fun=State._step_early_game,
    #         false_fun=State._step_late_game,
    #         operand=actions,
    #     )

    def _step_early_game(self, actions: JuxAction) -> 'State':
        # TODO
        checkify.check(False, "not implemented")
        return self

    def _step_late_game(self, actions: JuxAction) -> 'State':
        # TODO
        real_env_steps = jnp.where(
            self.env_cfg.BIDDING_SYSTEM,
            self.env_steps - self.board.factories_per_team + 1 + 1,
            self.env_steps,
        )
        unit_mask = self.unit_mask
        factory_mask = self.factory_mask

        # handle weather effects
        current_weather = self.weather_schedule[real_env_steps]
        weather_cfg = jax.lax.switch(
            current_weather,
            [
                # NONE
                lambda cfg: dict(power_gain_factor=1.0, power_loss_factor=1),
                # MARS_QUAKE
                lambda cfg: dict(power_gain_factor=1.0, power_loss_factor=1),
                # COLD_SNAP
                lambda cfg: dict(power_gain_factor=1.0, power_loss_factor=cfg.COLD_SNAP.POWER_CONSUMPTION),
                # DUST_STORM
                lambda cfg: dict(power_gain_factor=cfg.DUST_STORM.POWER_GAIN.astype(float), power_loss_factor=1),
                # SOLAR_FLARE
                lambda cfg: dict(power_gain_factor=cfg.SOLAR_FLARE.POWER_GAIN.astype(float), power_loss_factor=1),
            ],
            self.env_cfg.WEATHER,
        )
        self: 'State' = jax.lax.cond(
            current_weather == Weather.MARS_QUAKE,
            lambda s: s._mars_quake(),
            lambda s: s,
            self,
        )

        # 1. Check for malformed actions
        failed_players = jnp.zeros((2, ), dtype=np.bool_)

        # check factories
        failed_factory = ((actions.factory_action < FactoryAction.DO_NOTHING) |
                          (actions.factory_action > FactoryAction.WATER))
        failed_factory = failed_factory & factory_mask
        chex.assert_shape(failed_factory, (2, self.MAX_N_FACTORIES))
        failed_factory = jnp.any(failed_factory, axis=-1)
        chex.assert_shape(failed_factory, (2, ))
        failed_players = failed_players | failed_factory

        # check units
        action_mask = jnp.arange(self.UNIT_ACTION_QUEUE_SIZE)
        action_mask = jnp.repeat(action_mask[None, :], 2 * self.MAX_N_UNITS, axis=-2).reshape((2, self.MAX_N_UNITS, -1))
        chex.assert_shape(action_mask, (2, self.MAX_N_UNITS, self.UNIT_ACTION_QUEUE_SIZE))
        action_mask = action_mask < actions.unit_action_queue_count[..., None]
        chex.assert_shape(action_mask, (2, self.MAX_N_UNITS, self.UNIT_ACTION_QUEUE_SIZE))

        failed_players = failed_players | (actions.unit_action_queue_update & ~unit_mask).any(-1)

        failed_action = ~actions.unit_action_queue.is_valid(self.env_cfg.max_transfer_amount)
        failed_action = failed_action & action_mask
        failed_action = failed_action.any(-1).any(-1)
        chex.assert_shape(failed_action, (2, ))
        failed_players = failed_players | failed_action

        # update units action queue
        action_queue_power_cost = jnp.array(self.env_cfg.UNIT_ACTION_QUEUE_POWER_COST)
        update_power_req = action_queue_power_cost[self.units.unit_type] * weather_cfg["power_loss_factor"]
        chex.assert_shape(update_power_req, (2, self.MAX_N_UNITS))
        update_queue = actions.unit_action_queue_update & unit_mask & (update_power_req <= self.units.power)
        new_power = jnp.where(update_queue, self.units.power - update_power_req, self.units.power)
        chex.assert_shape(new_power, (2, self.MAX_N_UNITS))
        new_action_queue = ActionQueue(
            data=jnp.where(update_queue[..., None, None], actions.unit_action_queue.code, self.units.action_queue.data),
            count=jnp.where(update_queue, actions.unit_action_queue_count, self.units.action_queue.count),
            front=jnp.where(update_queue, 0, self.units.action_queue.front),
            rear=jnp.where(update_queue, actions.unit_action_queue_count, self.units.action_queue.rear),
        )
        new_self = self._replace(units=self.units._replace(
            power=new_power,
            action_queue=new_action_queue,
        ))
        chex.assert_trees_all_equal_shapes(new_self, self)
        self = new_self

        # 3. validate all actions against current state, throw away impossible actions TODO
        new_units, unit_action = jax.vmap(jax.vmap(Unit.next_action))(self.units)
        new_units = jux.tree_util.tree_where(unit_mask & ~failed_players[..., None], new_units, self.units)
        unit_action = jux.tree_util.tree_where(unit_mask & ~failed_players[..., None], unit_action, UnitAction())

        factory_actions = jnp.where(factory_mask, actions.factory_action, FactoryAction.DO_NOTHING)

        self = self._handle_transfer_actions(unit_action)
        self = self._handle_pickup_actions(unit_action)
        self = self._handle_dig_actions(unit_action, weather_cfg)
        self = self._handle_self_destruct_actions(unit_action)
        self = self._handle_factory_build_actions(factory_actions, weather_cfg)
        self = self._handle_movement_actions(unit_action, weather_cfg)
        self = self._handle_recharge_actions(unit_action)
        self = self._handle_factory_water_actions(factory_actions)
        return self

    def _handle_transfer_actions(self, actions: UnitAction):
        # TODO
        # breakpoint()
        # is_transfer = (actions.action_type == UnitActionType.TRANSFER)
        # transfer_amount =
        return self

    def _handle_pickup_actions(self, actions: UnitAction):
        # TODO
        return self

    def _handle_dig_actions(self: 'State', actions: UnitAction, weather_cfg: Dict[str, np.ndarray]):
        # 1. check if dig action is valid
        unit_mask = self.unit_mask
        units = self.units
        is_dig = (actions.action_type == UnitActionType.DIG) & unit_mask

        # cannot dig if no enough power
        power_req = units.unit_cfg.DIG_COST * weather_cfg["power_loss_factor"]
        is_dig = is_dig & (power_req <= units.power)

        # cannot dig if on top of a factory
        y, x = units.pos.y, units.pos.x
        factory_id_in_pos = self.board.factory_occupancy_map[y, x]
        factory_player_id = self.factory_id2idx.at[factory_id_in_pos].get(mode='fill', fill_value=INT32_MAX)[..., 0]
        is_dig = is_dig & (factory_player_id == INT32_MAX)
        self.factory_id2idx

        # 2. execute dig action
        new_power = jnp.where(is_dig, units.power - power_req, units.power)
        units = units._replace(power=new_power)

        # rubble
        dig_rubble = is_dig & self.board.rubble[y, x] > 0
        new_rubble = self.board.rubble.at[y, x].add(-units.unit_cfg.DIG_RUBBLE_REMOVED * dig_rubble)
        new_rubble = jnp.maximum(new_rubble, 0)

        # lichen
        dig_lichen = is_dig & ~dig_rubble & (self.board.lichen[y, x] > 0)
        new_lichen = self.board.lichen.at[y, x].add(-units.unit_cfg.DIG_LICHEN_REMOVED * dig_lichen)
        new_lichen = jnp.maximum(new_lichen, 0)

        # ice
        dig_ice = is_dig & ~dig_rubble & ~dig_lichen & self.board.ice[y, x]
        add_resource = jax.vmap(jax.vmap(Unit.add_resource, in_axes=(0, None, 0)), in_axes=(0, None, 0))
        units, _ = add_resource(units, ResourceType.ice, units.unit_cfg.DIG_RESOURCE_GAIN * dig_ice)

        # ore
        dig_ore = is_dig & ~dig_rubble & ~dig_lichen & ~dig_ice & (self.board.ore[y, x] > 0)
        units, _ = add_resource(units, ResourceType.ore, units.unit_cfg.DIG_RESOURCE_GAIN * dig_ore)

        new_self = self._replace(
            units=units,
            board=self.board._replace(
                map=self.board.map._replace(rubble=new_rubble, ),
                lichen=new_lichen,
            ),
        )
        return new_self

    def _handle_self_destruct_actions(self, actions: UnitAction):
        # TODO
        return self

    def _handle_factory_build_actions(self, factory_actions: Array, weather_cfg: Dict[str, np.ndarray]):
        # TODO
        return self

    def _handle_movement_actions(self, actions: UnitAction, weather_cfg: Dict[str, np.ndarray]):
        unit_mask = self.unit_mask
        player_id = jnp.array([0, 1])[..., None]

        # 1. check if moving action is valid
        is_moving_action = ((actions.action_type == UnitActionType.MOVE) &
                            (actions.direction != Direction.CENTER)) & unit_mask

        # can't move off the map
        new_pos = Position(self.units.pos.pos + direct2delta_xy[actions.direction])
        off_map = ((new_pos.pos < jnp.array([0, 0])) | (new_pos.pos >= self.env_cfg.map_size)).any(-1)
        is_moving_action = is_moving_action & ~off_map

        # can't move into a cell occupied by opponent's factory
        factory_id_in_new_pos = self.board.factory_occupancy_map[new_pos.y, new_pos.x]
        factory_player_id = self.factory_id2idx.at[factory_id_in_new_pos].get(mode='fill', fill_value=INT32_MAX)[..., 0]
        opponent_id = player_id[::-1]
        target_is_opponent_factory = factory_player_id == opponent_id
        is_moving_action = is_moving_action & ~target_is_opponent_factory

        # can't move if power is not enough
        target_rubble = self.board.rubble[new_pos.y, new_pos.x]
        power_required = jax.vmap(jax.vmap(Unit.move_power_cost))(self.units, target_rubble)
        power_required = power_required * weather_cfg["power_loss_factor"]
        is_moving_action = is_moving_action & (power_required <= self.units.power)

        # 2. update unit position and power
        new_pos = self.units.pos.pos + direct2delta_xy[actions.direction] * is_moving_action[..., None]
        new_power = self.units.power - power_required * is_moving_action
        units = self.units._replace(
            pos=Position(new_pos),
            power=new_power,
        )

        # 3. resolve unit collision
        # classify units into groups
        light = (units.unit_type == UnitType.LIGHT) & unit_mask  # bool[2, MAX_N_UNITS]
        heavy = (units.unit_type == UnitType.HEAVY) & unit_mask  # bool[2, MAX_N_UNITS]
        moving = is_moving_action & unit_mask
        still = (~is_moving_action) & unit_mask  # bool[2, MAX_N_UNITS]
        chex.assert_shape(light, (2, self.MAX_N_UNITS))  # bool[2, MAX_N_UNITS]
        chex.assert_equal_shape([light, heavy, moving, still])

        # count the number of different types of units in each location
        x, y = units.pos.x, units.pos.y

        cnt = jnp.zeros_like(self.board.units_map)
        still_light_cnt = cnt.at[y, x].add(light & still, mode='drop')  # int[H, W]
        moving_light_cnt = cnt.at[y, x].add(light & moving, mode='drop')  # int[H, W]
        still_heavy_cnt = cnt.at[y, x].add(heavy & still, mode='drop')  # int[H, W]
        moving_heavy_cnt = cnt.at[y, x].add(heavy & moving, mode='drop')  # int[H, W]
        chex.assert_equal_shape([still_light_cnt, moving_light_cnt, still_heavy_cnt, moving_heavy_cnt, cnt])

        # map above count to agent-wise
        still_light_cnt = still_light_cnt[y, x]  # int[2, MAX_N_UNITS]
        moving_light_cnt = moving_light_cnt[y, x]  # int[2, MAX_N_UNITS]
        still_heavy_cnt = still_heavy_cnt[y, x]  # int[2, MAX_N_UNITS]
        moving_heavy_cnt = moving_heavy_cnt[y, x]  # int[2, MAX_N_UNITS]
        chex.assert_shape(still_light_cnt, (2, self.MAX_N_UNITS))  # bool[2, MAX_N_UNITS]
        chex.assert_equal_shape([still_light_cnt, moving_light_cnt, still_heavy_cnt, moving_heavy_cnt])

        # dead cases
        cases = [
            # case 1 you are light and there is a heavy:
            (light & (still_heavy_cnt + moving_heavy_cnt > 0)),
            # case 2 you are light but still, and there is a another still light:
            (light & still & (still_light_cnt > 1)),
            # case 3 you are light but still, and there is a moving light:
            (light & still & (moving_light_cnt > 0)),
            # case 4 you are moving light, and there is another moving light:
            (light & moving & (moving_light_cnt > 1)),
            # case 5 you are heavy but still, and there is another still heavy:
            (heavy & still & (still_heavy_cnt > 1)),
            # case 6 you are heavy but still, and there is a moving heavy:
            (heavy & still & (moving_heavy_cnt > 0)),
            # case 7 you are heavy but moving, and there is another moving heavy:
            (heavy & moving & (moving_heavy_cnt > 1)),
        ]
        # or them together
        dead = cases[0]
        for case in cases[1:]:
            dead = dead | case

        # remove dead units, put them into the end of the array
        is_alive = ~dead & unit_mask
        unit_idx = jnp.where(is_alive, self.unit_idx, jnp.iinfo(jnp.int32).max)
        arg = jnp.argsort(unit_idx)

        empty_unit = jax.tree_map(
            lambda x: jnp.array(x)[None, None],
            Unit.empty(self.env_cfg, self.UNIT_ACTION_QUEUE_SIZE),
        )
        units = jux.tree_util.tree_where(dead, empty_unit, units)
        units = jax.tree_map(lambda x: x[player_id, arg], units)

        # 4. update other states
        n_units = self.n_units - dead.sum(axis=1)
        unit_id2idx = State.generate_unit_id2idx(units, self.MAX_N_UNITS)

        # update board
        units_map = jnp.full_like(self.board.units_map, fill_value=INT32_MAX)
        units_map = units_map.at[new_pos[..., 0], new_pos[..., 1]].set(units.unit_id, mode='drop')
        self = self._replace(
            units=units,
            n_units=n_units,
            unit_id2idx=unit_id2idx,
            board=self.board._replace(units_map=units_map),
        )
        return self

    def _handle_recharge_actions(self, actions: UnitAction):
        # TODO
        return self

    def _handle_factory_water_actions(self, factory_actions: Array):
        # TODO
        return self

    def _mars_quake(self) -> 'State':
        x, y = self.units.pos.x, self.units.pos.y  # int[]
        chex.assert_shape(x, (2, self.MAX_N_UNITS))
        chex.assert_shape(y, (2, self.MAX_N_UNITS))
        rubble_amount = jnp.array(self.env_cfg.WEATHER.MARS_QUAKE.RUBBLE)[self.units.unit_type]
        chex.assert_shape(rubble_amount, (2, self.MAX_N_UNITS))

        rubble = self.board.rubble.at[x, y].add(rubble_amount, mode='drop')
        rubble = jnp.minimum(rubble, self.env_cfg.MAX_RUBBLE)
        chex.assert_equal_shape([rubble, self.board.rubble])

        board = self.board
        map = board.map
        map = map._replace(rubble=rubble)
        board = board._replace(map=map)
        self = self._replace(board=board)
        return self
