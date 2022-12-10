from typing import Dict, List, NamedTuple, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental import checkify
from luxai2022.state import State as LuxState

import jux
from jux.actions import FactoryAction, JuxAction, UnitAction, UnitActionType
from jux.config import EnvConfig, JuxBufferConfig
from jux.factory import Factory, LuxFactory
from jux.map import Board
from jux.map.position import Direction, Position, direct2delta_xy
from jux.team import LuxTeam, Team
from jux.unit import LuxUnit, Unit, UnitCargo, UnitType
from jux.unit_cargo import ResourceType

INT32_MAX = jnp.iinfo(jnp.int32).max


def is_day(env_cfg: EnvConfig, env_step):
    return env_step % env_cfg.CYCLE_LENGTH < env_cfg.DAY_LENGTH


@jax.jit
def sort_by_unit_id(units: Union[Unit, Factory]):
    idx = jnp.argsort(units.unit_id, axis=1)
    units = jax.tree_util.tree_map(lambda x: x[jnp.arange(2)[:, None], idx], units)
    return units


batch_into_leaf_jitted = jax.jit(jux.tree_util.batch_into_leaf, static_argnames=('axis', ))


class State(NamedTuple):
    env_cfg: EnvConfig

    seed: int  # the seed for reproducibility
    rng_state: jax.random.KeyArray  # current rng state

    env_steps: int
    board: Board
    weather_schedule: Array

    units: Unit  # Unit[2, MAX_N_UNITS], the unit_id and team_id of non-existent units are jnp.iinfo(jnp.int32).max
    unit_id2idx: Array  # [2 * MAX_N_UNITS, 2], the idx of non-existent units is jnp.iinfo(jnp.int32).max
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
    factory_id2idx: Array  # int[2 * MAX_N_FACTORIES, 2]
    n_factories: Array  # int[2]

    teams: Team  # Team[2]

    global_id: int = jnp.int32(0)

    @property
    def real_env_steps(self):
        return jnp.where(
            self.env_cfg.BIDDING_SYSTEM,
            self.env_steps - (self.board.factories_per_team * 2 + 1),
            self.env_steps,
        )

    @classmethod
    def from_lux(cls, lux_state: LuxState, buf_cfg: JuxBufferConfig) -> "State":
        with jax.default_device(jax.devices("cpu")[0]):
            env_cfg = EnvConfig.from_lux(lux_state.env_cfg)

            # convert units
            def convert_units(lux_units: LuxUnit) -> Tuple[Unit, Array]:
                units: Tuple[List[Unit], List[Unit]] = (
                    [Unit.from_lux(unit, env_cfg) for unit in lux_units['player_0'].values()],
                    [Unit.from_lux(unit, env_cfg) for unit in lux_units['player_1'].values()],
                )
                units[0].sort(key=lambda unit: unit.unit_id)  # sort
                units[1].sort(key=lambda unit: unit.unit_id)
                n_units = [
                    len(lux_units['player_0']),
                    len(lux_units['player_1']),
                ]
                assert (n_units[0] <= buf_cfg.MAX_N_UNITS) and (n_units[1] <= buf_cfg.MAX_N_UNITS)
                empty_unit = Unit.empty(env_cfg)
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
                return units, n_units

            units, n_units = convert_units(lux_state.units)
            unit_id2idx = State.generate_unit_id2idx(units, buf_cfg.MAX_N_UNITS)

            # convert factories
            def convert_factories(lux_factories: LuxFactory) -> Tuple[Factory, Array]:
                factories: Tuple[List[Factory], List[Factory]] = (
                    [Factory.from_lux(fac) for fac in lux_factories['player_0'].values()],
                    [Factory.from_lux(fac) for fac in lux_factories['player_1'].values()],
                )
                factories[0].sort(key=lambda fac: fac.unit_id)  # sort
                factories[1].sort(key=lambda fac: fac.unit_id)
                n_factories = [
                    len(lux_factories['player_0']),
                    len(lux_factories['player_1']),
                ]
                assert (n_factories[0] <= buf_cfg.MAX_N_FACTORIES) and (n_factories[1] <= buf_cfg.MAX_N_FACTORIES)
                f = Factory.empty()
                factories = (  # padding to length of buf_cfg.MAX_N_FACTORIES
                    factories[0] + [f] * (buf_cfg.MAX_N_FACTORIES - n_factories[0]),
                    factories[1] + [f] * (buf_cfg.MAX_N_FACTORIES - n_factories[1]),
                )
                factories: Factory = batch_into_leaf_jitted([  # batch into leaf
                    batch_into_leaf_jitted(factories[0]),
                    batch_into_leaf_jitted(factories[1]),
                ])
                n_factories = jnp.array(n_factories)
                return factories, n_factories

            factories, n_factories = convert_factories(lux_state.factories)
            factory_id2idx = State.generate_factory_id2idx(factories, buf_cfg.MAX_N_FACTORIES)

            teams: List[Team] = [Team.from_lux(team, buf_cfg) for team in lux_state.teams.values()]
            teams.sort(key=lambda team: team.team_id)
            teams: Team = batch_into_leaf_jitted(teams)

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
                global_id=jnp.int32(lux_state.global_id),
            )
        state = jax.device_put(state, jax.devices()[0])
        # state.check_id2idx()
        return state

    @staticmethod
    def generate_unit_id2idx(units: Unit, max_n_units: int) -> Array:
        '''
        organize unit_id2idx such that
            unit_id2idx[unit_id] == [team_id, unit_idx]
            units[team_id, unit_idx].unit_id == unit_id
        '''
        units.unit_id: Array
        unit_id2idx = jnp.ones((max_n_units * 2, 2), dtype=np.int32) * jnp.iinfo(np.int32).max
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
        factories.unit_id: Array
        factory_id2idx = jnp.ones((max_n_factories * 2, 2), dtype=np.int32) * jnp.iinfo(np.int32).max
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
            seed=int(self.seed),
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

        # self.check_id2idx()
        # other.check_id2idx()

        def teams_eq(teams_a: Team, teams_b: Team) -> bool:
            teams_a = jux.tree_util.batch_out_of_leaf(teams_a)
            teams_b = jux.tree_util.batch_out_of_leaf(teams_b)
            return (teams_a[0] == teams_b[0]) & (teams_a[1] == teams_b[1])

        def units_eq(units_a: Unit, n_units_a: Array, units_b: Unit, n_units_b: Array) -> bool:

            def when_n_eq(self, units_a, units_b):
                units_a = sort_by_unit_id(units_a)
                units_b = sort_by_unit_id(units_b)
                eq = jax.vmap(jax.vmap(Unit.__eq__))(units_a, units_b)
                unit_mask = self.unit_mask
                return (eq | ~unit_mask).all()

            return jax.lax.cond(
                jnp.array_equal(n_units_a, n_units_b),
                when_n_eq,  # when numbers are equal, we compare them.
                lambda *args: False,  # when nubmer differ, return false
                self,
                units_a,
                units_b,
            )

        def factories_eq(factories_a: Factory, n_factories_a: Array, factories_b: Factory,
                         n_factories_b: Array) -> bool:

            def when_n_eq(self, factories_a, factories_b):
                factories_a = sort_by_unit_id(factories_a)
                factories_b = sort_by_unit_id(factories_b)
                eq = jax.vmap(jax.vmap(Factory.__eq__))(factories_a, factories_b)
                factory_mask = self.factory_mask
                return (eq | ~factory_mask).all()

            return jax.lax.cond(
                jnp.array_equal(n_factories_a, n_factories_b),
                when_n_eq,  # when numbers are equal, we compare them.
                lambda *args: False,  # when number differ, return false
                self,
                factories_a,
                factories_b)

        # self = jax.device_put(self, jax.devices("cpu")[0])
        # other = jax.device_put(other, jax.devices("cpu")[0])
        return ((self.env_steps == other.env_steps) & (self.board == other.board)
                & jnp.array_equal(self.weather_schedule, other.weather_schedule)
                & teams_eq(self.teams, other.teams)
                & factories_eq(self.factories, self.n_factories, other.factories, other.n_factories)
                & units_eq(self.units, self.n_units, other.units, other.n_units))

    @property
    def MAX_N_FACTORIES(self):
        self.factories.unit_id: Array
        return self.factories.unit_id.shape[-1]

    @property
    def MAX_N_UNITS(self):
        self.units.unit_id: Array
        return self.units.unit_id.shape[-1]

    @property
    def UNIT_ACTION_QUEUE_SIZE(self):
        self.units.action_queue.data: Array
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

    def parse_actions_from_dict(self, actions: Dict[str, Dict[str, Union[int, Array]]]) -> JuxAction:
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

        factory_action = jnp.array(factory_action)
        unit_action_queue = jnp.array(unit_action_queue)
        unit_action_queue_count = jnp.array(unit_action_queue_count)
        unit_action_queue_update = jnp.array(unit_action_queue_update)

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
    #     # TODO: check input actions's shape
    #     early_game = ((self.env_steps == 0) | (self.env_cfg.BIDDING_SYSTEM &
    #                                            (self.env_steps <= self.board.factories_per_team + 1))).all()

    #     self = jax.lax.cond(
    #         early_game,
    #         State._step_early_game,
    #         State._step_late_game,
    #         self,
    #         actions,
    #     )

    #     # always set rubble under factories to 0.
    #     rubble = jnp.where(self.board.factory_occupancy_map != INT32_MAX, 0, self.board.rubble)
    #     self = self._replace(board=self.board._replace(map=self.board.map._replace(rubble=rubble)))

    #     # TODO: calculate reward, dones, observations

    #     return self

    def _step_early_game(self, actions: JuxAction) -> 'State':
        # TODO
        checkify.check(False, "not implemented")
        return self

    def _step_late_game(self, actions: JuxAction) -> 'State':
        # TODO
        real_env_steps = self.real_env_steps
        unit_mask = self.unit_mask
        factory_mask = self.factory_mask

        # handle weather effects
        current_weather = self.weather_schedule[real_env_steps]
        weather_cfg = jux.weather.get_weather_cfg(self.env_cfg.WEATHER, current_weather)
        self: 'State' = jax.lax.cond(
            current_weather == jux.weather.Weather.MARS_QUAKE,
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
        action_mask = jnp.repeat(
            action_mask[None, :],
            2 * self.MAX_N_UNITS,
            axis=-2,
        ).reshape((2, self.MAX_N_UNITS, -1))  # bool[2, U, Q]
        chex.assert_shape(action_mask, (2, self.MAX_N_UNITS, self.UNIT_ACTION_QUEUE_SIZE))
        action_mask = action_mask < actions.unit_action_queue_count[..., None]  # bool[2, U, Q]
        chex.assert_shape(action_mask, (2, self.MAX_N_UNITS, self.UNIT_ACTION_QUEUE_SIZE))

        failed_players = failed_players | (actions.unit_action_queue_update & ~unit_mask).any(-1)

        failed_action = ~actions.unit_action_queue.is_valid(self.env_cfg.max_transfer_amount)  # bool[2, U, Q]
        failed_action = failed_action & action_mask
        failed_action = failed_action.any(-1).any(-1)
        chex.assert_shape(failed_action, (2, ))
        failed_players = failed_players | failed_action

        # update units action queue
        update_power_req = self.units.unit_cfg.ACTION_QUEUE_POWER_COST * weather_cfg["power_loss_factor"]
        chex.assert_shape(update_power_req, (2, self.MAX_N_UNITS))
        update_queue = actions.unit_action_queue_update & unit_mask & (update_power_req <= self.units.power)
        new_power = jnp.where(update_queue, self.units.power - update_power_req, self.units.power)
        chex.assert_shape(new_power, (2, self.MAX_N_UNITS))
        new_action_queue = jux.actions.ActionQueue(
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

        # pop next actions
        unit_action = jax.vmap(jax.vmap(Unit.next_action))(self.units)
        unit_action = jux.tree_util.tree_where(
            unit_mask & ~failed_players[..., None],
            unit_action,
            UnitAction.do_nothing(),
        )
        factory_actions = jnp.where(
            factory_mask & ~failed_players[..., None],
            actions.factory_action,
            FactoryAction.DO_NOTHING,
        )

        # 3. execute actions.
        # validating all actions against current state is not implemented here, but implemented in each action.
        water_info = self._cache_water_info()

        success = jnp.zeros((2, self.MAX_N_UNITS), dtype=jnp.bool_)  # bool[2, U]

        self, suc = self._handle_transfer_actions(unit_action)
        success = success | suc

        self, suc = self._handle_pickup_actions(unit_action)
        success = success | suc

        self, suc = self._handle_dig_actions(unit_action, weather_cfg)
        success = success | suc

        # self_destruct changes the unit order. In such case, success and
        # unit_action must be sort accordingly, so we need to pass in and out
        # success and unit_action. So does _handle_movement_actions. Movement
        # may change the unit order because of collisions.
        self, unit_action, success = self._handle_self_destruct_actions(unit_action, success)

        self = self._handle_factory_build_actions(factory_actions, weather_cfg)

        self, unit_action, success = self._handle_movement_actions(unit_action, weather_cfg, success)

        self, suc = self._handle_recharge_actions(unit_action)
        success = success | suc

        self = self._handle_factory_water_actions(factory_actions, water_info)

        # handle action pop and repeat
        units = jax.vmap(jax.vmap(Unit.repeat_action))(self.units, success)
        self = self._replace(units=units)

        # update lichen
        new_lichen = self.board.lichen - 1
        new_lichen = new_lichen.clip(0, self.env_cfg.MAX_LICHEN_PER_TILE)
        new_lichen_strains = jnp.where(new_lichen == 0, INT32_MAX, self.board.lichen_strains)
        self = self._replace(board=self.board._replace(
            lichen=new_lichen,
            lichen_strains=new_lichen_strains,
        ))

        # resources refining
        factory = self.factories.refine_step(self.env_cfg)
        water_cost = self.env_cfg.FACTORY_WATER_CONSUMPTION * self.factory_mask
        stock = factory.cargo.stock.at[..., ResourceType.water].add(-water_cost)
        factory = factory._replace(cargo=factory.cargo._replace(stock=stock))
        factories_to_destroy = (factory.cargo.water < 0)  # noqa
        # TODO: destroy factories
        self = self._replace(factories=factory)

        # power gain
        def _gain_power(self: 'State') -> Unit:
            new_units = self.units.gain_power(weather_cfg["power_gain_factor"])
            new_units = new_units._replace(power=new_units.power * self.unit_mask)
            return new_units

        self = self._replace(units=jax.lax.cond(
            is_day(self.env_cfg, real_env_steps),
            _gain_power,
            lambda self: self.units,
            self,
        ))
        '''
        # this if statement is same as above jax.lax.cond
        if is_day(self.env_cfg, real_env_steps):
            new_units = self.units.gain_power(weather_cfg["power_gain_factor"])
            new_units = new_units._replace(power=new_units.power * self.unit_mask)
            self = self._replace(units=new_units)
        '''
        # Factories are immune to weather thanks to using nuclear reactors instead
        new_factory_power = self.factories.power + self.env_cfg.FACTORY_CHARGE * self.factory_mask
        self = self._replace(factories=self.factories._replace(power=new_factory_power))

        # update step number
        self = self._replace(env_steps=self.env_steps + 1)

        # always set rubble under factories to 0.
        rubble = jnp.where(self.board.factory_occupancy_map != INT32_MAX, 0, self.board.rubble)
        self = self._replace(board=self.board._replace(map=self.board.map._replace(rubble=rubble)))

        return self

    def _handle_transfer_actions(self, actions: UnitAction) -> Tuple['State', Array]:
        # pytype: disable=attribute-error
        # pytype: disable=unsupported-operands

        # 1. validate the action
        is_transfer = (actions.action_type == UnitActionType.TRANSFER)

        # the target must be in map
        target_pos = Position(self.units.pos.pos + direct2delta_xy[actions.direction])  # int[2, U, 2]
        within_map = ((target_pos.pos >= 0) & (target_pos.pos < self.env_cfg.map_size)).all(-1)  # bool[2, U]
        is_transfer = is_transfer & within_map
        success = is_transfer

        # 2. handle the action
        # decide target
        target_factory_id = self.board.factory_occupancy_map[target_pos.x, target_pos.y]  # int[2, U]
        target_factory_idx = self.factory_id2idx.at[target_factory_id].get('fill', fill_value=INT32_MAX)  # int[2, U, 2]
        there_is_a_factory = target_factory_idx[..., 1] < self.n_factories[:, None]  # bool[2, U]

        target_unit_id = self.board.units_map[target_pos.x, target_pos.y]  # int[2, U]
        target_unit_idx = self.unit_id2idx.at[target_unit_id].get('fill', fill_value=INT32_MAX)  # int[2, U, 2]
        there_is_an_unit = target_unit_idx[..., 1] < self.n_units[:, None]  # bool[2, U]

        transfer_to_factory = is_transfer & there_is_a_factory  # bool[2, U]
        transfer_to_unit = is_transfer & ~there_is_a_factory & there_is_an_unit  # bool[2, U]
        is_power = (actions.resource_type == ResourceType.power)

        # deduce from unit
        transfer_amount = jnp.where(is_transfer, actions.amount, 0)  # int[2, U]
        units, transfer_amount = jax.vmap(jax.vmap(Unit.sub_resource))(self.units, actions.resource_type,
                                                                       transfer_amount)
        self = self._replace(units=units)

        # transfer to factory
        transferred_resource = jnp.where(transfer_to_factory & ~is_power, transfer_amount, 0)  # int[2, U]
        transferred_power = jnp.where(transfer_to_factory & is_power, transfer_amount, 0)  # int[2, U]
        factory_stock = self.factories.cargo.stock.at[(
            jnp.arange(2)[:, None],
            target_factory_idx[..., 1],
            actions.resource_type,
        )].add(transferred_resource, mode='drop')  # int[2, F, 4]
        factory_power = self.factories.power.at[(
            jnp.arange(2)[:, None],
            target_factory_idx[..., 1],
        )].add(transferred_power, mode='drop')  # int[2, F]
        factories = self.factories._replace(
            cargo=UnitCargo(factory_stock),
            power=factory_power,
        )
        self = self._replace(factories=factories)

        # transfer to unit
        transferred_resource = jnp.where(transfer_to_unit & ~is_power, transfer_amount, 0)  # int[2, U]
        transferred_power = jnp.where(transfer_to_unit & is_power, transfer_amount, 0)  # int[2, U]
        unit_stock = self.units.cargo.stock.at[(
            jnp.arange(2)[:, None],
            target_unit_idx[..., 1],
            actions.resource_type,
        )].add(transferred_resource, mode='drop')  # int[2, U, 4]
        unit_stock = jnp.minimum(unit_stock, self.units.unit_cfg.CARGO_SPACE[..., None])  # int[2, U, 4]

        unit_power = self.units.power.at[(
            jnp.arange(2)[:, None],
            target_unit_idx[..., 1],
        )].add(transferred_power, mode='drop')  # int[2, U]
        unit_power = jnp.minimum(unit_power, self.units.unit_cfg.BATTERY_CAPACITY)  # int[2, U]
        units = self.units._replace(
            cargo=UnitCargo(unit_stock),
            power=unit_power,
        )
        self = self._replace(units=units)

        return self, success
        # pytype: enable=attribute-error
        # pytype: enable=unsupported-operands

    def _handle_pickup_actions(self, actions: UnitAction) -> Tuple['State', Array]:
        # TODO: align with v1.1.1
        there_is_a_factory = self.board.factory_occupancy_map[self.units.pos.x,
                                                              self.units.pos.y] != INT32_MAX  # bool[2, U]
        success = (UnitActionType.PICKUP == actions.action_type) & (there_is_a_factory)

        # This action is difficult to vectorize, because pickup actions are not
        # independent from each other. Two robots may pick up from the same
        # factory. Assume there two robots, both pickup 10 power from the same
        # factory. If the factory has 15 power, then the first one gets 10
        # power, and the second one gets only 5 power.
        #
        # The following implementation relies on the fact that
        #  1. there are at most 9 robots on one factory.
        #  2. opponent's robots cannot move into our factory.

        # pytype: disable=attribute-error
        # pytype: disable=unsupported-operands

        # 1. prepare some data
        # get the pos of 9 cells around the factory
        occupy_delta = jnp.array([[0, -1], [0, 0], [0, 1],])[None] + \
            jnp.array([[-1, 0], [0, 0], [1, 0]])[:, None]  # int[3, 3, 2]
        occupy_delta = occupy_delta.reshape((-1, 2))  # int[9, 2]
        occupy_pos = Position(self.factories.pos.pos[..., None, :] + occupy_delta)  # int[2, F, 9, 2]
        chex.assert_shape(occupy_pos, (2, self.MAX_N_FACTORIES, 9, 2))

        # get the unit idx on factories
        unit_id_on_factory = self.board.units_map[occupy_pos.x, occupy_pos.y]
        unit_id_on_factory = jnp.sort(unit_id_on_factory, axis=-1)  # sort by id, so small ids have higher priority
        unit_idx_on_factory = self.unit_id2idx[unit_id_on_factory]  # int[2, F, 9, 2]
        unit_team_idx, unit_idx = unit_idx_on_factory[..., 0], unit_idx_on_factory[..., 1]  # int[2, F, 9]
        chex.assert_shape(unit_idx_on_factory, (2, self.MAX_N_FACTORIES, 9, 2))

        # get action info
        action_type = actions.action_type.at[unit_team_idx, unit_idx].get(mode="fill", fill_value=0)  # int[2, F, 9]
        is_pickup = (UnitActionType.PICKUP == action_type)  # int[2, F, 9]
        resource_type = actions.resource_type.at[unit_team_idx, unit_idx].get(mode="fill", fill_value=0)  # int[2, F, 9]
        amount = actions.amount.at[unit_team_idx, unit_idx].get(mode="fill", fill_value=0)  # int[2, F, 9]
        amount = amount * is_pickup  # int[2, F, 9]
        chex.assert_shape(action_type, (2, self.MAX_N_FACTORIES, 9))
        chex.assert_equal_shape([action_type, is_pickup, resource_type, amount])

        # 2. calculate the amount of resource each robot can pickup
        # this is the core logic. The most important thing is to calculate a cumulative sum of pick up amount.
        # For a playere `p`, if its unit `u` successfully pick up resource `r` from factory `f`,
        #   then the factory will lose `cumsum[p, f, u, r]` amount of resource `r` in total in this turn,
        #   just after the unit finish picking up.
        amount_by_type = jnp.zeros((2, self.MAX_N_FACTORIES, 9, 5), dtype=jnp.int32)  # int[2, F, 9, 5]
        amount_by_type = amount_by_type.at[(
            jnp.arange(2)[:, None, None],
            jnp.arange(self.MAX_N_FACTORIES)[None, :, None],
            jnp.arange(9)[None, None, :],
            resource_type,
        )].set(amount)
        chex.assert_shape(amount_by_type, (2, self.MAX_N_FACTORIES, 9, 5))
        cumsum = jnp.cumsum(amount_by_type, axis=-2)  # int[2, F, 9, 5]
        stock = jnp.concatenate([self.factories.cargo.stock, self.factories.power[..., None]], axis=-1)  # int[2, F, 5]
        real_cumsum = jnp.minimum(cumsum, stock[:, :, None, :])  # int[2, F, 9, 5]
        real_cumsum_without_self = jnp.concatenate(
            [
                jnp.zeros((2, self.MAX_N_FACTORIES, 1, 5), dtype=jnp.int32),
                real_cumsum[:, :, :-1, :],
            ],
            axis=-2,
        )  # int[2, F, 9, 5]
        real_pickup_amount = real_cumsum - real_cumsum_without_self  # int[2, F, 9, 5]
        chex.assert_equal_shape([amount_by_type, cumsum, real_cumsum, real_cumsum_without_self, real_pickup_amount])

        # 3. apply the real_pickup_amount
        factory_lose = real_cumsum[:, :, -1, :]  # int[2, F, 5]
        fac_power = self.factories.power - factory_lose[:, :, ResourceType.power]  # int[2, F]
        fac_stock = self.factories.cargo.stock - factory_lose[:, :, :4]  # int[2, F, 4]
        new_factories = self.factories._replace(power=fac_power, cargo=UnitCargo(fac_stock))

        units_power = self.units.power.at[unit_team_idx, unit_idx].add(
            real_pickup_amount[..., ResourceType.power],
            mode='drop',
        )  # int[2, U]
        units_stock = self.units.cargo.stock.at[unit_team_idx, unit_idx].add(
            real_pickup_amount[..., :4],
            mode='drop',
        )  # int[2, U, 4]
        units_stock = jnp.minimum(units_stock, self.units.unit_cfg.CARGO_SPACE[..., None])
        new_units = self.units._replace(power=units_power, cargo=UnitCargo(units_stock))
        self = self._replace(factories=new_factories, units=new_units)

        return self, success
        # pytype: enable=attribute-error
        # pytype: enable=unsupported-operands

    def _handle_dig_actions(self: 'State', actions: UnitAction, weather_cfg: Dict[str,
                                                                                  np.ndarray]) -> Tuple['State', Array]:
        # TODO: align with v1.1.1
        # 1. check if dig action is valid
        unit_mask = self.unit_mask
        units = self.units
        is_dig = (actions.action_type == UnitActionType.DIG) & unit_mask

        # cannot dig if no enough power
        power_req = units.unit_cfg.DIG_COST * weather_cfg["power_loss_factor"]
        is_dig = is_dig & (power_req <= units.power)

        # cannot dig if on top of a factory
        x, y = units.pos.x, units.pos.y
        factory_id_in_pos = self.board.factory_occupancy_map[x, y]
        factory_player_id = self.factory_id2idx.at[factory_id_in_pos].get(mode='fill', fill_value=INT32_MAX)[..., 0]
        is_dig = is_dig & (factory_player_id == INT32_MAX)
        success = is_dig

        # 2. execute dig action
        new_power = jnp.where(is_dig, units.power - power_req, units.power)
        units = units._replace(power=new_power)

        # rubble
        dig_rubble = is_dig & (self.board.rubble[x, y] > 0)
        new_rubble = self.board.rubble.at[x, y].add(-units.unit_cfg.DIG_RUBBLE_REMOVED * dig_rubble)
        new_rubble = jnp.maximum(new_rubble, 0)

        # lichen
        dig_lichen = is_dig & ~dig_rubble & (self.board.lichen[x, y] > 0)
        new_lichen = self.board.lichen.at[x, y].add(-units.unit_cfg.DIG_LICHEN_REMOVED * dig_lichen)
        new_lichen = jnp.maximum(new_lichen, 0)

        # ice
        dig_ice = is_dig & ~dig_rubble & ~dig_lichen & self.board.ice[x, y]
        add_resource = jax.vmap(jax.vmap(Unit.add_resource, in_axes=(0, None, 0)), in_axes=(0, None, 0))
        units, _ = add_resource(units, ResourceType.ice, units.unit_cfg.DIG_RESOURCE_GAIN * dig_ice)

        # ore
        dig_ore = is_dig & ~dig_rubble & ~dig_lichen & ~dig_ice & (self.board.ore[x, y] > 0)
        units, _ = add_resource(units, ResourceType.ore, units.unit_cfg.DIG_RESOURCE_GAIN * dig_ore)

        new_self = self._replace(
            units=units,
            board=self.board._replace(
                map=self.board.map._replace(rubble=new_rubble, ),
                lichen=new_lichen,
            ),
        )
        return new_self, success

    def _handle_self_destruct_actions(self, actions: UnitAction, success: Array) -> Tuple['State', UnitAction, Array]:
        # TODO
        success = success | (actions.action_type == UnitActionType.SELF_DESTRUCT)
        return self, actions, success

    def _handle_factory_build_actions(self: 'State', factory_actions: Array, weather_cfg: Dict[str, np.ndarray]):
        factory_mask = self.factory_mask
        player_id = jnp.array([0, 1])[..., None]
        is_build_heavy = (factory_actions == FactoryAction.BUILD_HEAVY) & factory_mask
        is_build_light = (factory_actions == FactoryAction.BUILD_LIGHT) & factory_mask

        # 1. check if build action is valid
        # check if power is enough
        light_power_cost = self.env_cfg.ROBOTS[UnitType.LIGHT].POWER_COST * weather_cfg["power_loss_factor"]
        heavy_power_cost = self.env_cfg.ROBOTS[UnitType.HEAVY].POWER_COST * weather_cfg["power_loss_factor"]
        is_build_light = is_build_light & (self.factories.power >= light_power_cost)
        is_build_heavy = is_build_heavy & (self.factories.power >= heavy_power_cost)

        # check if metal is enough
        light_metal_cost = self.env_cfg.ROBOTS[UnitType.LIGHT].METAL_COST
        heavy_metal_cost = self.env_cfg.ROBOTS[UnitType.HEAVY].METAL_COST
        is_build_light = is_build_light & (self.factories.cargo.metal >= light_metal_cost)
        is_build_heavy = is_build_heavy & (self.factories.cargo.metal >= heavy_metal_cost)

        is_build = is_build_heavy | is_build_light

        # 2. deduct power and metal
        power_cost = is_build_light * light_power_cost + is_build_heavy * heavy_power_cost
        metal_cost = is_build_light * light_metal_cost + is_build_heavy * heavy_metal_cost
        factory_sub_resource = jax.vmap(jax.vmap(Factory.sub_resource, in_axes=(0, None, 0)), in_axes=(0, None, 0))
        factories = self.factories
        factories, _ = factory_sub_resource(factories, ResourceType.power, power_cost)
        factories, _ = factory_sub_resource(factories, ResourceType.metal, metal_cost)
        self = self._replace(factories=factories)

        # 3. create new units
        unit_new_vmap = jax.vmap(jax.vmap(Unit.new, in_axes=(None, 0, 0, None)), in_axes=(0, 0, 0, None))
        n_new_units = is_build.sum(axis=1)
        start_id = jnp.array([self.global_id, self.global_id + n_new_units[0]])[..., None]
        unit_id = jnp.cumsum(is_build, axis=1) - 1 + start_id

        created_units = unit_new_vmap(
            jnp.array([0, 1]),  # team_id
            is_build_heavy.astype(jnp.int32),  # unit_type
            unit_id,  # unit_id
            # replace UNIT_ACTION_QUEUE_SIZE with a concrete value to make it JIT-able
            self.env_cfg._replace(UNIT_ACTION_QUEUE_SIZE=self.UNIT_ACTION_QUEUE_SIZE),  # env_cfg
        )
        created_units = created_units._replace(pos=self.factories.pos)

        # put created units into self.units
        created_units_idx = jnp.cumsum(is_build, axis=1) - 1 + self.n_units[..., None]
        created_units_idx = jnp.where(is_build, created_units_idx, INT32_MAX)

        def set_unit_attr(units_attr, created_attr):
            return units_attr.at[player_id, created_units_idx, ...].set(created_attr, mode='drop')

        new_units = jax.tree_map(set_unit_attr, self.units, created_units)
        new_n_units = self.n_units + n_new_units

        return self._replace(
            units=new_units,
            n_units=new_n_units,
            unit_id2idx=State.generate_unit_id2idx(new_units, self.MAX_N_UNITS),
            board=self.board.update_units_map(new_units),
            global_id=self.global_id + n_new_units.sum(),
        )

    def _handle_movement_actions(self, actions: UnitAction, weather_cfg: Dict[str, np.ndarray],
                                 success: Array) -> Tuple['State', UnitAction, Array]:
        # TODO: align with v1.1.1
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
        factory_id_in_new_pos = self.board.factory_occupancy_map[new_pos.x, new_pos.y]
        factory_player_id = self.factory_id2idx.at[factory_id_in_new_pos].get(mode='fill', fill_value=INT32_MAX)[..., 0]
        opponent_id = player_id[::-1]
        target_is_opponent_factory = factory_player_id == opponent_id
        is_moving_action = is_moving_action & ~target_is_opponent_factory

        # can't move if power is not enough
        target_rubble = self.board.rubble[new_pos.x, new_pos.y]
        power_required = jax.vmap(jax.vmap(Unit.move_power_cost))(self.units, target_rubble)
        power_required = power_required * weather_cfg["power_loss_factor"]
        is_moving_action = is_moving_action & (power_required <= self.units.power)

        # moving to center is always considered as success
        success = success | is_moving_action | (((actions.action_type == UnitActionType.MOVE) &
                                                 (actions.direction == Direction.CENTER)) & unit_mask)

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
        still_light_cnt = cnt.at[x, y].add(light & still, mode='drop')  # int[H, W]
        moving_light_cnt = cnt.at[x, y].add(light & moving, mode='drop')  # int[H, W]
        still_heavy_cnt = cnt.at[x, y].add(heavy & still, mode='drop')  # int[H, W]
        moving_heavy_cnt = cnt.at[x, y].add(heavy & moving, mode='drop')  # int[H, W]
        chex.assert_equal_shape([still_light_cnt, moving_light_cnt, still_heavy_cnt, moving_heavy_cnt, cnt])

        # map above count to agent-wise
        still_light_cnt = still_light_cnt[x, y]  # int[2, MAX_N_UNITS]
        moving_light_cnt = moving_light_cnt[x, y]  # int[2, MAX_N_UNITS]
        still_heavy_cnt = still_heavy_cnt[x, y]  # int[2, MAX_N_UNITS]
        moving_heavy_cnt = moving_heavy_cnt[x, y]  # int[2, MAX_N_UNITS]
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

        self = self._replace(units=units)
        self, live_idx = self.destroy_unit(dead)

        unit_mask = self.unit_mask  # bool[2, U]

        success = success[jnp.arange(2)[:, None], live_idx]
        success = success & unit_mask

        actions = UnitAction(
            jnp.where(
                unit_mask[..., None],
                actions.code[jnp.arange(2)[:, None], live_idx],
                UnitAction.do_nothing().code,
            ))

        return self, actions, success

    def destroy_unit(self, dead: Array) -> Tuple['State', Array]:
        '''
        Destroy dead units, and put them into the end of the array.

        Args:
            dead: bool[2, U], dead indicator.

        Returns:
            new_state: State, new state.
            live_idx: int[2, U], the index of live units. Only the part with self.unit_mask == True is valid.
        '''
        unit_mask = self.unit_mask  # bool[2, U]
        units = self.units

        # add rubble to the board, and remove lichen
        rubble = self.board.rubble.at[(
            units.pos.x,
            units.pos.y,
        )].add(dead * self.units.unit_cfg.RUBBLE_AFTER_DESTRUCTION, mode='drop')
        lichen = self.board.lichen.at[(
            units.pos.x,
            units.pos.y,
        )].min(jnp.where(dead, 0, INT32_MAX), mode='drop')
        lichen_strains = self.board.lichen_strains.at[(
            units.pos.x,
            units.pos.y,
        )].max(jnp.where(dead, INT32_MAX, -1), mode='drop')

        # remove dead units, put them into the end of the array
        is_alive = ~dead & unit_mask
        unit_idx = jnp.where(is_alive, self.unit_idx, INT32_MAX)
        live_idx = jnp.argsort(unit_idx)

        empty_unit = jax.tree_map(
            lambda x: jnp.array(x)[None, None],
            # replace UNIT_ACTION_QUEUE_SIZE with a concrete value to make it JIT-able
            Unit.empty(self.env_cfg._replace(UNIT_ACTION_QUEUE_SIZE=self.UNIT_ACTION_QUEUE_SIZE)),
        )

        units = jux.tree_util.tree_where(dead, empty_unit, units)
        units = jax.tree_map(lambda x: x[jnp.arange(2)[:, None], live_idx], units)

        # update other states
        n_units = self.n_units - dead.sum(axis=1)
        unit_id2idx = State.generate_unit_id2idx(units, self.MAX_N_UNITS)

        # update board
        board = self.board.update_units_map(units)
        board = board._replace(
            map=board.map._replace(rubble=rubble),
            lichen=lichen,
            lichen_strains=lichen_strains,
        )

        self = self._replace(
            units=units,
            n_units=n_units,
            unit_id2idx=unit_id2idx,
            board=board,
        )
        return self, live_idx

    def _handle_recharge_actions(self, actions: UnitAction):
        is_recharge = (actions.action_type == UnitActionType.RECHARGE)
        success = is_recharge & (self.units.power >= actions.amount)
        return self, success

    def _handle_factory_water_actions(self, factory_actions: Array, water_info: Array) -> 'State':

        H, W = self.board.lichen_strains.shape

        color = water_info
        cmp_cnt = jux.map_generator.flood.component_sum(1, color)  # int[H, W]

        # -9 for the factory occupied cells
        grow_lichen_size = cmp_cnt[self.factories.pos.x, self.factories.pos.y] - 9  # int[2, F]
        water_cost = jnp.ceil(grow_lichen_size / self.env_cfg.LICHEN_WATERING_COST_FACTOR).astype(jnp.int32)

        # 3. perform action
        is_water = (factory_actions == FactoryAction.WATER) & (self.factories.cargo.water >= water_cost)

        # water cost
        water_cost = jnp.where(is_water, water_cost, 0)
        new_stock = self.factories.cargo.stock.at[..., ResourceType.water].add(-water_cost)

        # lichen growth
        delta_lichen = jnp.zeros((H, W), dtype=jnp.int32)  # int[H, W]
        factory_color = color.at[:, self.factories.pos.x,
                                 self.factories.pos.y].get(mode='fill', fill_value=INT32_MAX)  # int[2, 2, F]
        delta_lichen = delta_lichen.at[factory_color[0], factory_color[1]].add(is_water, mode='drop')
        delta_lichen = delta_lichen.at[color[0], color[1]].get(mode='fill', fill_value=0)
        delta_lichen = jnp.where(self.board.factory_occupancy_map == INT32_MAX, delta_lichen, 0)
        new_lichen = self.board.lichen + delta_lichen * 2

        # lichen strain
        lichen_strain = jnp.zeros((H, W), dtype=jnp.int32)  # int[H, W]
        lichen_strain = lichen_strain.at[factory_color[0], factory_color[1]].set(self.factories.unit_id, mode='drop')
        lichen_strain = lichen_strain.at[color[0], color[1]].get(mode='fill', fill_value=0)
        new_lichen_strains = jnp.where(delta_lichen > 0, lichen_strain, self.board.lichen_strains)

        # 4. update self
        self = self._replace(
            board=self.board._replace(
                lichen=new_lichen,
                lichen_strains=new_lichen_strains,
            ),
            factories=self.factories._replace(cargo=UnitCargo(new_stock)),
        )

        return self

    def _cache_water_info(self) -> 'Array':
        """
        Run flood fill algorithm to color cells. All cells to be watered by the
        same factory will have the same color.

        Returns:
            Array: int[2, H, W]. the first dimension represent the 'color'.The
            'color' is represented by the coordinate of the factory a tile
            belongs to. If a tile is not connected to any factory, its color its
            own coordinate. In such a way, different lichen strains will have
            different colors.
        """
        # The key idea here is to prepare a list of neighbors for each cell it connects to when watered.
        # neighbor_ij is a 4x2xHxW array, where the first dimension is the neighbors (4 at most), the second dimension is the 2 coordinates.
        H, W = self.board.lichen_strains.shape

        ij = jnp.mgrid[:H, :W]
        delta_ij = jnp.array([
            [-1, 0],
            [0, 1],
            [1, 0],
            [0, -1],
        ])  # int[2, H, W]
        neighbor_ij = delta_ij[..., None, None] + ij[None, ...]  # int[4, 2, H, W]

        # handle map boundary.
        neighbor_ij = neighbor_ij.at[0, 0, 0, :].set(0)
        neighbor_ij = neighbor_ij.at[1, 1, :, W - 1].set(W - 1)
        neighbor_ij = neighbor_ij.at[2, 0, H - 1, :].set(H - 1)
        neighbor_ij = neighbor_ij.at[3, 1, :, 0].set(0)

        # 1. calculate strain connections.
        color = jnp.minimum(self.board.lichen_strains, self.board.factory_occupancy_map)  # int[H, W]

        # handle a corner case where there may be rubbles on strains when movement collision happens.
        color = jnp.where(self.board.rubble == 0, color, INT32_MAX)

        neighbor_color = color.at[(
            neighbor_ij[:, 0],
            neighbor_ij[:, 1],
        )].get(mode='fill', fill_value=INT32_MAX)

        connect_cond = ((color == neighbor_color) & (color != INT32_MAX))  # bool[4, H, W]

        color = jux.map_generator.flood._flood_fill(jnp.where(connect_cond[:, None], neighbor_ij, ij))  # int[2, H, W]
        factory_color = color.at[:, self.factories.pos.x,
                                 self.factories.pos.y].get(mode='fill', fill_value=INT32_MAX)  # int[2, 2, F]
        connected_lichen = jnp.full((H, W), fill_value=INT32_MAX, dtype=jnp.int32)  # int[H, W]
        connected_lichen = connected_lichen.at[factory_color[0], factory_color[1]].set(self.factories.unit_id,
                                                                                       mode='drop')
        connected_lichen = connected_lichen.at[color[0], color[1]].get(mode='fill', fill_value=INT32_MAX)

        # 2. handle cells to expand to.
        # 2.1 cells that are allowed to expand to, only if
        #   1. it is not a lichen strain, and
        #   2. it has no rubble, and
        #   3. it is not resource.
        allow_grow = (self.board.rubble == 0) & ~(self.board.ice | self.board.ore) & \
                (self.board.lichen_strains == INT32_MAX) & (self.board.factory_occupancy_map == INT32_MAX)

        # 2.2 when a non-lichen cell connects two different strains, then it is not allowed to expand to.
        neighbor_lichen_strain = self.board.lichen_strains[neighbor_ij[:, 0], neighbor_ij[:, 1]]  # int[4, H, W]
        neighbor_is_lichen = neighbor_lichen_strain != INT32_MAX
        center_connects_two_different_strains = (self.board.lichen_strains == INT32_MAX) & ( \
            ((neighbor_lichen_strain[0] != neighbor_lichen_strain[1]) & neighbor_is_lichen[0] & neighbor_is_lichen[1]) | \
            ((neighbor_lichen_strain[0] != neighbor_lichen_strain[2]) & neighbor_is_lichen[0] & neighbor_is_lichen[2]) | \
            ((neighbor_lichen_strain[0] != neighbor_lichen_strain[3]) & neighbor_is_lichen[0] & neighbor_is_lichen[3]) | \
            ((neighbor_lichen_strain[1] != neighbor_lichen_strain[2]) & neighbor_is_lichen[1] & neighbor_is_lichen[2]) | \
            ((neighbor_lichen_strain[1] != neighbor_lichen_strain[3]) & neighbor_is_lichen[1] & neighbor_is_lichen[3]) | \
            ((neighbor_lichen_strain[2] != neighbor_lichen_strain[3]) & neighbor_is_lichen[2] & neighbor_is_lichen[3]) \
        )
        allow_grow = allow_grow & ~center_connects_two_different_strains

        # 2.3 calculate the strains id, if it is expanded to.
        expand_center = (connected_lichen != INT32_MAX) & (self.board.lichen >= self.env_cfg.MIN_LICHEN_TO_SPREAD)
        expand_center = expand_center | (self.board.factory_occupancy_map != INT32_MAX)
        expand_center = jnp.where(expand_center, connected_lichen, INT32_MAX)
        strain_id_if_expand = jnp.minimum(  # int[H, W]
            jnp.minimum(
                jnp.roll(expand_center, 1, axis=0).at[0, :].set(INT32_MAX),
                jnp.roll(expand_center, -1, axis=0).at[-1, :].set(INT32_MAX),
            ),
            jnp.minimum(
                jnp.roll(expand_center, 1, axis=1).at[:, 0].set(INT32_MAX),
                jnp.roll(expand_center, -1, axis=1).at[:, -1].set(INT32_MAX),
            ),
        )
        strain_id_if_expand = jnp.where(allow_grow, strain_id_if_expand, INT32_MAX)

        # 3. get the final color result.
        strain_id = jnp.minimum(connected_lichen, strain_id_if_expand)  # int[H, W]
        factory_idx = self.factory_id2idx[strain_id]  # int[2, H, W]
        color = self.factories.pos.pos[factory_idx[..., 0], factory_idx[..., 1]]  # int[H, W, 2]
        color = color.transpose((2, 0, 1))  # int[2, H, W]
        color = jnp.where(strain_id == INT32_MAX, ij, color)
        return color

    def _mars_quake(self) -> 'State':
        x, y = self.units.pos.x, self.units.pos.y  # int[]
        chex.assert_shape(x, (2, self.MAX_N_UNITS))
        chex.assert_shape(y, (2, self.MAX_N_UNITS))
        rubble_amount = self.units.unit_cfg.RUBBLE_AFTER_DESTRUCTION
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
