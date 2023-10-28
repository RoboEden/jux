import functools
from typing import Dict, List, NamedTuple, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from luxai_s2.state import State as LuxState

import jux.map_generator.flood
import jux.tree_util
from jux.actions import FactoryAction, JuxAction, UnitAction, UnitActionType
from jux.config import EnvConfig, JuxBufferConfig
from jux.factory import Factory, LuxFactory
from jux.map import Board
from jux.map.position import Direction, Position, direct2delta_xy
from jux.team import FactionTypes, LuxTeam, Team
from jux.unit import ActionQueue, LuxUnit, Unit, UnitCargo, UnitType
from jux.unit_cargo import ResourceType
from jux.utils import INT32_MAX, imax


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

    seed: jnp.uint32  # the seed for reproducibility
    rng_state: jax.Array  # current rng state

    env_steps: jnp.int16
    board: Board

    # the unit_id and team_id of non-existent units are jnp.iinfo(jnp.int32).max
    units: Unit  # Unit[2, U]

    # the idx of non-existent units is jnp.iinfo(jnp.int32).max
    unit_id2idx: Unit.id_dtype()  # int16[MAX_GLOBAL_ID, 2]
    n_units: Unit.id_dtype()  # int16[2]
    '''
    The `unit_id2idx` is organized such that
    ```
    [team_id, unit_idx] = unit_id2idx[unit_id]
    assert units[team_id, unit_idx].unit_id == unit_id
    ```

    For non-existent units, its `unit_id`, `team_id`, `pos`, and `unit_idx` are all INT_MAX of the corresponding dtype.
    ```
    INT16_MAX = jnp.iinfo(np.int16).max
    INT8_MAX = jnp.iinfo(np.int8).max
    assert unit.unit_id[0, n_units[0]:] == INT16_MAX
    assert unit.team_id[0, n_units[0]:] == INT8_MAX
    assert (unit.pos.pos[0, n_units[0]:] == INT8_MAX).all()

    assert unit.unit_id[1, n_units[1]:] == INT16_MAX
    assert unit.team_id[1, n_units[1]:] == INT8_MAX
    assert (unit.pos.pos[1, n_units[1]:] == INT8_MAX).all()
    ```
    '''

    factories: Factory  # Factory[2, F]
    factory_id2idx: Factory.id_dtype()  # int8[2 * F, 2]
    n_factories: Factory.id_dtype()  # int8[2]

    teams: Team  # Team[2]

    global_id: Unit.id_dtype() = Unit.id_dtype()(0)  # int16
    place_first: jnp.int8 = jnp.int8(0)  # 0/1, the player to place first

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
        return self.units.action_queue.capacity

    @property
    def MAX_GLOBAL_ID(self):
        return self.unit_id2idx.shape[-2]

    @property
    def factory_idx(self):
        factory_idx = jnp.array(
            [
                jnp.arange(self.MAX_N_FACTORIES),
                jnp.arange(self.MAX_N_FACTORIES),
            ],
            dtype=Factory.id_dtype(),
        )
        return factory_idx

    @property
    def factory_mask(self):
        factory_mask = self.factory_idx < self.n_factories[..., None]
        chex.assert_shape(factory_mask, (..., 2, self.MAX_N_FACTORIES))
        return factory_mask

    @property
    def unit_idx(self):
        unit_idx = jnp.array(
            [
                jnp.arange(self.MAX_N_UNITS),
                jnp.arange(self.MAX_N_UNITS),
            ],
            dtype=Unit.id_dtype(),
        )
        return unit_idx

    @property
    def unit_mask(self):
        unit_mask = self.unit_idx < self.n_units[..., None]
        chex.assert_shape(unit_mask, (..., 2, self.MAX_N_UNITS))
        return unit_mask

    @property
    def real_env_steps(self):
        return jnp.where(
            self.env_cfg.BIDDING_SYSTEM,
            self.env_steps - (self.board.factories_per_team * 2 + 1),
            self.env_steps,
        )

    @property
    def next_player(self):
        """The next player to place a factory.

        Returns:
            int: player_id 0 or 1.
        """
        return (self.env_steps + self.place_first + 1) % 2  # plus 1 for bid step

    @classmethod
    def new(cls, seed: int, env_cfg: EnvConfig, buf_cfg: JuxBufferConfig) -> "State":
        key = jax.random.PRNGKey(seed)
        board = Board.new(
            seed=jnp.uint32(seed),
            env_cfg=env_cfg,
            buf_cfg=buf_cfg,
        )
        key, subkey = jax.random.split(key)
        empty_unit = Unit.empty(env_cfg)
        empty_unit = jax.tree_map(lambda x: x if isinstance(x, Array) else np.array(x), empty_unit)
        units = jax.tree_map(lambda x: x[None].repeat(buf_cfg.MAX_N_UNITS, axis=0), empty_unit)
        units = jax.tree_map(lambda x: x[None].repeat(2, axis=0), units)
        unit_id2idx = State.generate_unit_id2idx(units, buf_cfg.MAX_GLOBAL_ID)
        n_units = jnp.zeros(shape=(2, ), dtype=Unit.id_dtype())

        empty_factory = Factory.empty()
        factories = jax.tree_map(lambda x: x[None].repeat(buf_cfg.MAX_N_FACTORIES, axis=0), empty_factory)
        factories = jax.tree_map(lambda x: x[None].repeat(2, axis=0), factories)
        factory_id2idx = State.generate_factory_id2idx(factories, buf_cfg.MAX_N_FACTORIES)
        n_factories = jnp.zeros(shape=(2, ), dtype=Factory.id_dtype())

        teams = jux.tree_util.batch_into_leaf([Team.new(team_id=id, buf_cfg=buf_cfg) for id in range(2)])

        state = cls(
            env_cfg=env_cfg,
            seed=jnp.uint32(seed),
            rng_state=key,
            env_steps=State.__annotations__['env_steps'](0),
            board=board,
            units=units,
            unit_id2idx=unit_id2idx,
            n_units=n_units,
            factories=factories,
            factory_id2idx=factory_id2idx,
            n_factories=n_factories,
            teams=teams,
        )
        return state

    @classmethod
    def from_lux(cls, lux_state: LuxState, buf_cfg: JuxBufferConfig = JuxBufferConfig()) -> "State":
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
                n_units = jnp.array(n_units, dtype=Unit.id_dtype())
                return units, n_units

            units, n_units = convert_units(lux_state.units)
            unit_id2idx = State.generate_unit_id2idx(units, buf_cfg.MAX_GLOBAL_ID)

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
                n_factories = jnp.array(n_factories, dtype=Factory.id_dtype())
                return factories, n_factories

            factories, n_factories = convert_factories(lux_state.factories)
            factory_id2idx = State.generate_factory_id2idx(factories, buf_cfg.MAX_N_FACTORIES)

            teams: List[Team] = [Team.from_lux(team, buf_cfg) for team in lux_state.teams.values()]
            teams.sort(key=lambda team: team.team_id)
            if len(teams) == 0:
                teams = [Team.new(team_id=0, buf_cfg=buf_cfg), Team.new(team_id=1, buf_cfg=buf_cfg)]
            teams: Team = batch_into_leaf_jitted(teams)

            if 'player_0' in lux_state.teams and lux_state.teams['player_0'].place_first:
                place_first = State.__annotations__['place_first'](0)
            else:
                place_first = State.__annotations__['place_first'](1)

            seed = lux_state.seed if lux_state.seed is not None else INT32_MAX

            state = State(
                env_cfg=env_cfg,
                seed=jnp.uint32(seed),
                rng_state=jax.random.PRNGKey(seed),
                env_steps=State.__annotations__['env_steps'](lux_state.env_steps),
                board=Board.from_lux(lux_state.board, buf_cfg),
                units=units,
                n_units=n_units,
                unit_id2idx=unit_id2idx,
                factories=factories,
                n_factories=n_factories,
                factory_id2idx=factory_id2idx,
                teams=teams,
                global_id=State.__annotations__['global_id'](lux_state.global_id),
                place_first=place_first,
            )
        state = jax.device_put(state, jax.devices()[0])
        # state.check_id2idx()
        return state

    @staticmethod
    def generate_unit_id2idx(units: Unit, max_global_id: int) -> Array:
        '''
        organize unit_id2idx such that
            unit_id2idx[unit_id] == [team_id, unit_idx]
            units[team_id, unit_idx].unit_id == unit_id
        '''
        units.unit_id: Array
        max_n_units = units.unit_id.shape[-1]
        unit_id2idx = jnp.ones((max_global_id, 2), dtype=Unit.id_dtype()) * imax(Unit.id_dtype())
        unit_id2idx = unit_id2idx.at[units.unit_id[..., 0, :]].set(
            jnp.array(
                [jnp.zeros(max_n_units), jnp.arange(max_n_units)],
                dtype=Unit.id_dtype(),
            ).T,
            mode='drop',
        )
        unit_id2idx = unit_id2idx.at[units.unit_id[..., 1, :]].set(
            jnp.array(
                [jnp.ones(max_n_units), jnp.arange(max_n_units)],
                dtype=Unit.id_dtype(),
            ).T,
            mode='drop',
        )
        return unit_id2idx

    @staticmethod
    def generate_factory_id2idx(factories: Factory, max_n_factories: int) -> Array:
        factories.unit_id: Array
        factory_id2idx = jnp.ones((max_n_factories * 2, 2), dtype=Factory.id_dtype()) * imax(Factory.id_dtype())
        factory_id2idx = factory_id2idx.at[factories.unit_id[..., 0, :]].set(
            jnp.array(
                [jnp.zeros(max_n_factories), jnp.arange(max_n_factories)],
                dtype=Factory.id_dtype(),
            ).T,
            mode='drop',
        )
        factory_id2idx = factory_id2idx.at[factories.unit_id[..., 1, :]].set(
            jnp.array(
                [jnp.ones(max_n_factories), jnp.arange(max_n_factories)],
                dtype=Factory.id_dtype(),
            ).T,
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
        lux_teams: Dict[str, LuxTeam] = {
            f"player_{team.team_id}": team.to_lux(team.team_id == self.place_first)
            for team in lux_teams
        }

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

        seed = int(self.seed) if self.seed != INT32_MAX else None
        return LuxState(
            seed_rng=np.random.RandomState(seed),
            seed=seed,
            env_steps=int(self.env_steps),
            env_cfg=lux_env_cfg,
            board=self.board.to_lux(lux_env_cfg, lux_factories, lux_units),
            units=lux_units,
            factories=lux_factories,
            teams=lux_teams,
            global_id=int(self.global_id),
        )

    def to_torch(self) -> 'State':
        # convert seed from uint32 to int32, because torch does not support uint32
        self = self._replace(
            seed=self.seed.astype(jnp.int32),
            rng_state=self.rng_state.astype(jnp.int32),
            board=self.board._replace(seed=self.board.seed.astype(jnp.int32)),
        )
        return jax.tree_map(jux.torch.to_torch, self)

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
                & teams_eq(self.teams, other.teams)
                & factories_eq(self.factories, self.n_factories, other.factories, other.n_factories)
                & units_eq(self.units, self.n_units, other.units, other.n_units))

    def parse_actions_from_dict(self, actions: Dict[str, Dict[str, Union[int, Array]]]) -> JuxAction:
        return JuxAction.from_lux(self, actions)

    def check_actions(self, actions: JuxAction) -> None:
        chex.assert_shape(actions.factory_action, (2, self.MAX_N_FACTORIES))
        chex.assert_shape(actions.unit_action_queue, (2, self.MAX_N_UNITS, self.env_cfg.UNIT_ACTION_QUEUE_SIZE, 5))
        chex.assert_shape(actions.unit_action_queue_count, (2, self.MAX_N_UNITS))
        chex.assert_shape(actions.unit_action_queue_update, (2, self.MAX_N_UNITS))

    def _step_bid(self, bid: Array, faction: Array) -> 'State':
        """The initial bidding step.

        Args:
            bid (Array): int[2], two players' bid.
            facion (Array): int[2]
        Returns:
            State: new game state
        """
        init_resource = Team.__annotations__['init_water'](self.env_cfg.INIT_WATER_METAL_PER_FACTORY)
        init_resource = init_resource * self.board.factories_per_team
        init_water = jnp.full(shape=(2, ), fill_value=init_resource, dtype=Team.__annotations__['init_water'])
        init_metal = jnp.full(shape=(2, ), fill_value=init_resource, dtype=Team.__annotations__['init_metal'])

        valid_actions = (bid <= init_resource) & (bid >= -init_resource)

        # player_0 win if
        #  1. player_1 propose invalid bid, or
        #  2. player_0 propose valid bid and bid is higher than player_1
        bid_abs = jnp.abs(bid)
        player_0_won = ~valid_actions[1] | (valid_actions[0] & (bid_abs[0] >= bid_abs[1]))
        won = jnp.array([
            player_0_won,
            ~player_0_won,
        ])

        init_water = init_water - bid_abs * won * valid_actions
        init_metal = init_metal - bid_abs * won * valid_actions

        won_player_id = jnp.where(player_0_won, 0, 1)
        place_first = jnp.where(bid[won_player_id] >= 0, won_player_id, 1 - won_player_id)
        place_first = jnp.where(~valid_actions.any(), 0, place_first)

        faction = faction.astype(jnp.int8)
        faction = jnp.clip(faction, 0, len(FactionTypes) - 1)
        self = self._replace(
            teams=self.teams._replace(
                faction=faction,
                init_water=init_water.astype(Team.__annotations__['init_water']),
                init_metal=init_metal.astype(Team.__annotations__['init_metal']),
                factories_to_place=jnp.array(
                    [self.board.factories_per_team] * 2,
                    dtype=Team.__annotations__['factories_to_place'],
                ),
                bid=bid.astype(Team.__annotations__['bid']),
            ),
            place_first=State.__annotations__['place_first'](place_first),
            env_steps=self.env_steps + 1,
        )
        return self

    def add_factory(self, team_id: int, pos: Array, water: int, metal: int):

        def _add_factory(self: 'State', team_id: int, pos: Array, water: int, metal: int):
            factory = Factory.new(
                team_id=team_id,
                unit_id=self.global_id,
                pos=Position.new(pos),
                power=self.env_cfg.INIT_POWER_PER_FACTORY,
                cargo=UnitCargo.new(0, 0, water, metal),
            )

            idx = self.n_factories[team_id]
            n_factories = self.n_factories.at[team_id].add(1)
            factories = jax.tree_map(
                lambda fs, f: fs.at[team_id, idx].set(f),
                self.factories,
                factory,
            )

            teams_factory_strains = self.teams.factory_strains.at[team_id,
                                                                  self.teams.n_factory[team_id]].set(factory.unit_id)
            teams_n_factory = self.teams.n_factory.at[team_id].add(1)

            board = self.board.update_factories_map(factories)
            occupancy = factories.occupancy
            board = board._replace(map=board.map._replace(
                rubble=board.rubble.at[occupancy.x, occupancy.y].set(0),
                ice=board.ice.at[occupancy.x, occupancy.y].set(False),
                ore=board.ore.at[occupancy.x, occupancy.y].set(False),
            ), )
            return self._replace(
                factories=factories,
                n_factories=n_factories,
                factory_id2idx=self.generate_factory_id2idx(
                    factories,
                    self.MAX_N_FACTORIES,
                ),
                global_id=self.global_id + 1,
                board=board,
                teams=self.teams._replace(
                    factory_strains=teams_factory_strains,
                    n_factory=teams_n_factory,
                ),
            )

        return jax.lax.cond(
            self.board.valid_spawns_mask[pos[0], pos[1]],
            _add_factory,
            lambda self, *_: self,
            *(self, team_id, pos, water, metal),
        )

    def _step_factory_placement(self, spawn, water, metal) -> 'State':
        """
        The early game step for factory placement. Only half of input arrays is
        used, the other half is ignored, depending on which player's turn it is.

        Args:
            actions (Array): int[2, 2], two players' factory placement.
                Only one of position[0] or position[1] is valid, depending on
                the current player.
            water (Array): int[2], the amount of water to be assigned to
                factory. Only one of water[0] or water[1] is valid, depending on
                the current player.
            metal (Array): int[2], the amount of metal to be assigned to
                factory. Only one of metal[0] or metal[1] is valid, depending on
                the current player.
        Returns:
            State: new game state.
        """
        # decide
        player = self.next_player
        water = jnp.clip(water[player], 0, self.teams.init_water[player])
        metal = jnp.clip(metal[player], 0, self.teams.init_metal[player])

        x, y = spawn[player, 0], spawn[player, 1]
        valid = self.board.valid_spawns_mask[x, y]

        # factory
        self = self.add_factory(player, spawn[player, :], water, metal)

        # teams
        factories_to_place = self.teams.factories_to_place.at[player].add(-1 * valid)
        water = self.teams.init_water.at[player].add(-water * valid)
        metal = self.teams.init_metal.at[player].add(-metal * valid)
        teams = self.teams._replace(
            factories_to_place=factories_to_place,
            init_water=water,
            init_metal=metal,
        )

        self = self._replace(
            teams=teams,
            env_steps=self.env_steps + 1,
        )
        return self

    def _step_late_game(self, actions: JuxAction) -> 'State':
        real_env_steps = self.real_env_steps
        unit_mask = self.unit_mask
        factory_mask = self.factory_mask

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
        action_mask = jnp.arange(self.UNIT_ACTION_QUEUE_SIZE, dtype=ActionQueue.__annotations__['count'])
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
        action_queue_power_cost = jnp.array(
            [unit_cfg.ACTION_QUEUE_POWER_COST for unit_cfg in self.env_cfg.ROBOTS],
            dtype=Unit.__annotations__['power'],
        )
        update_power_req = action_queue_power_cost[self.units.unit_type]
        chex.assert_shape(update_power_req, (2, self.MAX_N_UNITS))
        update_queue = actions.unit_action_queue_update & unit_mask & (update_power_req <= self.units.power)
        new_power = jnp.where(update_queue, self.units.power - update_power_req, self.units.power)
        chex.assert_shape(new_power, (2, self.MAX_N_UNITS))
        new_action_queue = jux.actions.ActionQueue(
            data=jux.tree_util.tree_where(
                update_queue[..., None],
                actions.unit_action_queue,
                self.units.action_queue.data,
            ),
            count=jnp.where(update_queue, actions.unit_action_queue_count, self.units.action_queue.count),
            front=jnp.where(update_queue, 0, self.units.action_queue.front),
            rear=jnp.where(update_queue, actions.unit_action_queue_count % self.UNIT_ACTION_QUEUE_SIZE, self.units.action_queue.rear),
        )
        new_self: State = self._replace(units=self.units._replace(
            power=new_power,
            action_queue=new_action_queue,
        ))
        chex.assert_trees_all_equal_shapes(new_self, self)
        self = new_self

        # get next actions
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

        # 3. validating all actions against current state is not implemented here, but implemented in each action.
        valid_movement, movement_power_cost = self._validate_movement_actions(unit_action)

        action_info = dict(
            valid_transfer=self._validate_transfer_actions(unit_action),  # bool[2, U]
            valid_pickup=self._validate_pickup_actions(unit_action),  # bool[2, U]
            valid_dig=self._validate_dig_actions(unit_action),  # bool[2, U]
            valid_self_destruct=self._validate_self_destruct_actions(unit_action),  # bool[2, U]
            valid_factory_build=self._validate_factory_build_actions(factory_actions),  # bool[2, F]
            movement_info=dict(
                valid=valid_movement,  # bool[2, U]
                power_cost=movement_power_cost,  # int[2, U]
            ),
            valid_recharge=self._validate_recharge_actions(unit_action),
        )

        # 4. execute actions.
        self = self._handle_dig_actions(unit_action, action_info['valid_dig'])
        self, dead = self._handle_self_destruct_actions(unit_action, action_info['valid_self_destruct'])
        self = self._handle_factory_build_actions(factory_actions, action_info['valid_factory_build'])
        self, new_dead = self._handle_movement_actions(unit_action, action_info['movement_info'], dead)
        dead = dead | new_dead
        # Not all valid recharge actions are executed successfully, there is a `suc` indicator returned.
        self, recharge_success = self._handle_recharge_actions(unit_action, action_info['valid_recharge'])

        self = self.add_rubble_for_dead_units(dead)

        color, grow_lichen_size, connected_lichen_size = self._cache_water_info(factory_actions)
        self = self._handle_factory_water_actions(factory_actions, color, grow_lichen_size)
        self = self._handle_transfer_actions(unit_action, action_info['valid_transfer'])
        self = self._handle_pickup_actions(unit_action, action_info['valid_pickup'])

        # handle action pop and repeat
        success = (
            action_info['valid_dig'] | \
            recharge_success | \
            action_info['movement_info']['valid'] | \
            action_info['valid_transfer'] | \
            action_info['valid_pickup']
        )
        units = jax.vmap(jax.vmap(Unit.repeat_action))(self.units, success)
        self = self._replace(units=units)

        # destroy dead units
        self, _ = self.destroy_unit(dead)

        # update lichen
        new_lichen = self.board.lichen - 1
        new_lichen = new_lichen.clip(0, self.env_cfg.MAX_LICHEN_PER_TILE)
        new_lichen_strains = jnp.where(new_lichen == 0, imax(self.board.lichen_strains.dtype),
                                       self.board.lichen_strains)
        self = self._replace(board=self.board._replace(
            lichen=new_lichen,
            lichen_strains=new_lichen_strains,
        ))

        # resources refining
        factories = self.factories.refine_step(self.env_cfg)
        water_cost = self.env_cfg.FACTORY_WATER_CONSUMPTION * self.factory_mask
        stock = factories.cargo.stock.at[..., ResourceType.water].add(-water_cost)
        factories = factories._replace(cargo=factories.cargo._replace(stock=stock))
        self = self._replace(factories=factories)

        # factories gain power
        delta_power = self.env_cfg.FACTORY_CHARGE + connected_lichen_size * self.env_cfg.POWER_PER_CONNECTED_LICHEN_TILE
        new_factory_power = self.factories.power + jnp.where(self.factory_mask, delta_power, 0)
        self = self._replace(factories=self.factories._replace(power=new_factory_power))

        # destroy factories without water
        factories_to_destroy = (self.factories.cargo.water < 0)  # noqa
        self = self.destroy_factories(factories_to_destroy)

        # power gain
        def _gain_power(self: 'State') -> Unit:
            new_units = self.units.gain_power(self.env_cfg.ROBOTS)
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

        # update step number
        self = self._replace(env_steps=self.env_steps + 1)

        return self

    def _validate_transfer_actions(self, actions: UnitAction):
        valid = (actions.action_type == UnitActionType.TRANSFER)

        # the target must be in map
        target_pos = Position(self.units.pos.pos + direct2delta_xy[actions.direction])  # int[2, U, 2]
        within_map = ((target_pos.pos >= 0) & (target_pos.pos < self.env_cfg.map_size)).all(-1)  # bool[2, U]
        valid = valid & within_map

        return valid

    def _handle_transfer_actions(self, actions: UnitAction, valid: Array) -> 'State':
        # pytype: disable=attribute-error
        # pytype: disable=unsupported-operands

        # decide target
        target_pos = Position(self.units.pos.pos + direct2delta_xy[actions.direction])  # int[2, U, 2]
        target_factory_id = self.board.factory_occupancy_map[target_pos.x, target_pos.y]  # int[2, U]
        target_factory_idx = self.factory_id2idx.at[target_factory_id] \
                                                .get(mode='fill', fill_value=imax(self.factory_id2idx.dtype))  # int[2, U, 2]
        there_is_a_factory = target_factory_idx[..., 1] < self.n_factories[target_factory_idx[..., 0]]  # bool[2, U]

        target_unit_id = self.board.units_map[target_pos.x, target_pos.y]  # int[2, U]
        target_unit_idx = self.unit_id2idx.at[target_unit_id] \
                                          .get(mode='fill', fill_value=imax(self.unit_id2idx.dtype))  # int[2, U, 2]
        there_is_an_unit = target_unit_idx[..., 1] < self.n_units[target_unit_idx[..., 0]]  # bool[2, U]

        transfer_to_factory = valid & there_is_a_factory  # bool[2, U]
        transfer_to_unit = valid & ~there_is_a_factory & there_is_an_unit  # bool[2, U]
        is_power = (actions.resource_type == ResourceType.power)

        # deduce from unit
        transfer_amount = jnp.where(valid, actions.amount, 0)  # int[2, U]
        units, transfer_amount = jax.vmap(jax.vmap(Unit.sub_resource))(
            self.units,
            actions.resource_type,
            transfer_amount,
        )
        self = self._replace(units=units)

        # transfer to factory
        transferred_resource = jnp.where(transfer_to_factory & ~is_power, transfer_amount, 0)  # int[2, U]
        transferred_power = jnp.where(transfer_to_factory & is_power, transfer_amount, 0)  # int[2, U]
        factory_stock = self.factories.cargo.stock.at[(
            target_factory_idx[..., 0],
            target_factory_idx[..., 1],
            actions.resource_type,
        )].add(transferred_resource, mode='drop')  # int[2, F, 4]
        factory_power = self.factories.power.at[(
            target_factory_idx[..., 0],
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
            target_unit_idx[..., 0],
            target_unit_idx[..., 1],
            actions.resource_type,
        )].add(transferred_resource, mode='drop')  # int[2, U, 4]
        cargo_space = self.units.get_cfg("CARGO_SPACE", self.env_cfg.ROBOTS)
        unit_stock = jnp.minimum(unit_stock, cargo_space[..., None])  # int[2, U, 4]

        unit_power = self.units.power.at[(
            target_unit_idx[..., 0],
            target_unit_idx[..., 1],
        )].add(transferred_power, mode='drop')  # int[2, U]
        battery_capacity = self.units.get_cfg("BATTERY_CAPACITY", self.env_cfg.ROBOTS)
        unit_power = jnp.minimum(unit_power, battery_capacity)  # int[2, U]
        units = self.units._replace(
            cargo=UnitCargo(unit_stock),
            power=unit_power,
        )
        self = self._replace(units=units)

        return self
        # pytype: enable=attribute-error
        # pytype: enable=unsupported-operands

    def _validate_pickup_actions(self, actions: UnitAction) -> Array:
        # bool[2, U]
        there_is_a_factory = self.board.factory_occupancy_map[(
            self.units.pos.x,
            self.units.pos.y,
        )] != imax(self.board.factory_occupancy_map.dtype)
        valid = (UnitActionType.PICKUP == actions.action_type) & (there_is_a_factory)

        return valid

    def _handle_pickup_actions(self, actions: UnitAction, valid: Array) -> 'State':
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
        occupy_pos = self.factories.occupancy
        chex.assert_shape(occupy_pos, (2, self.MAX_N_FACTORIES, 9, 2))

        # get the unit idx on factories
        pickup_pos = jnp.where(valid[..., None], self.units.pos.pos, imax(self.units.pos.pos.dtype))  # int[2, U, 2]
        board = self.board.update_units_map(self.units._replace(pos=Position(pickup_pos)))
        units_map = board.units_map  # int[H, W]
        unit_id_on_factory = units_map[occupy_pos.x, occupy_pos.y]
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
        amount_by_type = jnp.zeros((2, self.MAX_N_FACTORIES, 9, 5), dtype=amount.dtype)  # int[2, F, 9, 5]
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
                jnp.zeros((2, self.MAX_N_FACTORIES, 1, 5), dtype=real_cumsum.dtype),
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
        battery_capacity = self.units.get_cfg("BATTERY_CAPACITY", self.env_cfg.ROBOTS)
        units_power = jnp.minimum(units_power, battery_capacity)
        units_stock = self.units.cargo.stock.at[unit_team_idx, unit_idx].add(
            real_pickup_amount[..., :4],
            mode='drop',
        )  # int[2, U, 4]
        cargo_space = self.units.get_cfg("CARGO_SPACE", self.env_cfg.ROBOTS)
        units_stock = jnp.minimum(units_stock, cargo_space[..., None])
        new_units = self.units._replace(power=units_power, cargo=UnitCargo(units_stock))
        self = self._replace(factories=new_factories, units=new_units)

        return self
        # pytype: enable=attribute-error
        # pytype: enable=unsupported-operands

    def _validate_dig_actions(self: 'State', actions: UnitAction) -> Array:
        unit_mask = self.unit_mask
        units = self.units
        valid = (actions.action_type == UnitActionType.DIG) & unit_mask

        # cannot dig if no enough power
        power_cost = units.get_cfg("DIG_COST", self.env_cfg.ROBOTS)
        valid = valid & (power_cost <= units.power)

        # cannot dig if on top of a factory
        x, y = units.pos.x, units.pos.y
        factory_id_in_pos = self.board.factory_occupancy_map[x, y]
        factory_player_id = self.factory_id2idx.at[factory_id_in_pos] \
                                               .get(mode='fill', fill_value=imax(self.factory_id2idx.dtype))
        factory_player_id = factory_player_id[..., 0]
        valid = valid & (factory_player_id == imax(factory_player_id.dtype))

        return valid

    def _handle_dig_actions(self: 'State', actions: UnitAction, valid: Array) -> 'State':
        units = self.units
        power_cost = units.get_cfg("DIG_COST", self.env_cfg.ROBOTS)
        x, y = self.units.pos.x, self.units.pos.y

        # deduce power
        new_power = jnp.where(valid, units.power - power_cost, units.power)
        units = units._replace(power=new_power)

        # rubble
        dig_rubble = valid & (self.board.rubble[x, y] > 0)
        dig_rubble_removed = units.get_cfg("DIG_RUBBLE_REMOVED", self.env_cfg.ROBOTS).astype(self.board.rubble.dtype)
        new_rubble = self.board.rubble.at[x, y].add(-dig_rubble_removed * dig_rubble)
        new_rubble = jnp.maximum(new_rubble, 0)

        # lichen
        dig_lichen = valid & ~dig_rubble & (self.board.lichen[x, y] > 0)
        dig_lichen_removed = units.get_cfg("DIG_LICHEN_REMOVED", self.env_cfg.ROBOTS).astype(self.board.lichen.dtype)
        new_lichen = self.board.lichen.at[x, y].add(-dig_lichen_removed * dig_lichen)
        new_lichen = jnp.maximum(new_lichen, 0)
        # When digging out last lichen, then rubble equal to `DIG_RUBBLE_REMOVED` is added to the tile
        new_rubble = new_rubble.at[x, y].add(jnp.where(dig_lichen & (new_lichen[x, y] == 0), dig_rubble_removed, 0))

        # resources
        add_resource = jax.vmap(Unit.add_resource, in_axes=(0, None, 0, None))
        add_resource = jax.vmap(add_resource, in_axes=(0, None, 0, None))
        dig_resource_gain = units.get_cfg("DIG_RESOURCE_GAIN", self.env_cfg.ROBOTS).astype(UnitCargo.dtype())

        # ice
        dig_ice = valid & ~dig_rubble & ~dig_lichen & self.board.ice[x, y]
        units, _ = add_resource(units, ResourceType.ice, dig_resource_gain * dig_ice, self.env_cfg.ROBOTS)

        # ore
        dig_ore = valid & ~dig_rubble & ~dig_lichen & ~dig_ice & (self.board.ore[x, y] > 0)
        units, _ = add_resource(units, ResourceType.ore, dig_resource_gain * dig_ore, self.env_cfg.ROBOTS)

        new_self = self._replace(
            units=units,
            board=self.board._replace(
                map=self.board.map._replace(rubble=new_rubble),
                lichen=new_lichen,
            ),
        )
        return new_self

    def _validate_self_destruct_actions(self: 'State', actions: UnitAction) -> Array:
        power_cost = self.units.get_cfg("SELF_DESTRUCT_COST", self.env_cfg.ROBOTS)
        valid = (actions.action_type == UnitActionType.SELF_DESTRUCT) & (self.units.power >= power_cost)

        return valid

    def _handle_self_destruct_actions(self, actions: UnitAction, valid: Array) -> Tuple['State', Array]:

        # bool indicator of dead units
        dead = valid

        units = self.units
        pos_without_dead = jnp.where(
            dead[..., None],
            imax(units.pos.pos.dtype),
            units.pos.pos,
        )
        self = self._replace(
            units=units,
            board=self.board.update_units_map(units._replace(pos=Position(pos_without_dead))),
        )

        return self, dead

    def _validate_factory_build_actions(self: 'State', factory_actions: Array) -> Array:
        factory_mask = self.factory_mask
        is_build_heavy = (factory_actions == FactoryAction.BUILD_HEAVY) & factory_mask
        is_build_light = (factory_actions == FactoryAction.BUILD_LIGHT) & factory_mask

        # check if power is enough
        light_power_cost = self.env_cfg.ROBOTS[UnitType.LIGHT].POWER_COST
        heavy_power_cost = self.env_cfg.ROBOTS[UnitType.HEAVY].POWER_COST
        is_build_light = is_build_light & (self.factories.power >= light_power_cost)
        is_build_heavy = is_build_heavy & (self.factories.power >= heavy_power_cost)

        # check if metal is enough
        light_metal_cost = self.env_cfg.ROBOTS[UnitType.LIGHT].METAL_COST
        heavy_metal_cost = self.env_cfg.ROBOTS[UnitType.HEAVY].METAL_COST
        is_build_light = is_build_light & (self.factories.cargo.metal >= light_metal_cost)
        is_build_heavy = is_build_heavy & (self.factories.cargo.metal >= heavy_metal_cost)

        valid = is_build_heavy | is_build_light

        return valid

    def _handle_factory_build_actions(self: 'State', factory_actions: Array, valid: Array) -> 'State':

        # 1. double check if build action is valid. Because robots may pickup resources from factory
        valid = valid & self._validate_factory_build_actions(factory_actions)

        is_build_heavy = valid & (factory_actions == FactoryAction.BUILD_HEAVY)
        is_build_light = valid & (factory_actions == FactoryAction.BUILD_LIGHT)
        light_power_cost = self.env_cfg.ROBOTS[UnitType.LIGHT].POWER_COST
        heavy_power_cost = self.env_cfg.ROBOTS[UnitType.HEAVY].POWER_COST
        light_metal_cost = self.env_cfg.ROBOTS[UnitType.LIGHT].METAL_COST
        heavy_metal_cost = self.env_cfg.ROBOTS[UnitType.HEAVY].METAL_COST

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
        n_new_units = valid.sum(axis=1, dtype=Unit.id_dtype())
        start_id = jnp.array([self.global_id, self.global_id + n_new_units[0]])[..., None]
        unit_id = jnp.cumsum(valid, axis=1, dtype=Unit.id_dtype()) - 1 + start_id

        created_units = unit_new_vmap(
            jnp.array([0, 1]),  # team_id
            is_build_heavy,  # unit_type
            unit_id,  # unit_id
            # replace UNIT_ACTION_QUEUE_SIZE with a concrete value to make it JIT-able
            self.env_cfg._replace(UNIT_ACTION_QUEUE_SIZE=self.UNIT_ACTION_QUEUE_SIZE),  # env_cfg
        )
        created_units = created_units._replace(pos=self.factories.pos)

        # put created units into self.units
        created_units_idx = jnp.cumsum(valid, axis=1) - 1 + self.n_units[..., None]
        created_units_idx = jnp.where(valid, created_units_idx, imax(created_units_idx.dtype))

        def set_unit_attr(units_attr, created_attr):
            return units_attr.at[jnp.arange(2)[..., None], created_units_idx, ...].set(created_attr, mode='drop')

        new_units = jax.tree_map(set_unit_attr, self.units, created_units)
        new_n_units = self.n_units + n_new_units

        return self._replace(
            units=new_units,
            n_units=new_n_units,
            unit_id2idx=State.generate_unit_id2idx(new_units, self.MAX_GLOBAL_ID),
            board=self.board.update_units_map(new_units),
            global_id=self.global_id + n_new_units.sum(dtype=n_new_units.dtype),
        )

    def _validate_movement_actions(self, actions: UnitAction) -> Tuple[Array, Array]:
        unit_mask = self.unit_mask
        player_id = jnp.array([0, 1])[..., None].astype(Team.__annotations__['team_id'])

        is_moving = ((actions.action_type == UnitActionType.MOVE) & (actions.direction != Direction.CENTER)) & unit_mask

        # can't move off the map
        new_pos = Position(self.units.pos.pos + direct2delta_xy[actions.direction])
        off_map = ((new_pos.pos < jnp.array([0, 0])) | (new_pos.pos >= self.env_cfg.map_size)).any(-1)
        is_moving = is_moving & ~off_map

        # can't move into a cell occupied by opponent's factory
        factory_id_in_new_pos = self.board.factory_occupancy_map[new_pos.x, new_pos.y]  # int8[2, U]
        factory_player_id = self.factory_id2idx.at[factory_id_in_new_pos] \
                                               .get(mode='fill', fill_value=imax(self.factory_id2idx.dtype))
        factory_player_id = factory_player_id[..., 0]
        opponent_id = player_id[::-1]
        target_is_opponent_factory = factory_player_id == opponent_id
        is_moving = is_moving & ~target_is_opponent_factory

        # can't move if power is not enough
        target_rubble = self.board.rubble[new_pos.x, new_pos.y]
        power_cost = self.units.move_power_cost(target_rubble, self.env_cfg.ROBOTS)
        is_moving = is_moving & (power_cost <= self.units.power)

        # moving to center is always considered as success
        valid = is_moving | (((actions.action_type == UnitActionType.MOVE) &
                              (actions.direction == Direction.CENTER)) & unit_mask)
        return valid, power_cost

    def _handle_movement_actions(self, actions: UnitAction, movement_info: Dict[str, Array],
                                 already_dead: Array) -> Tuple['State', Array]:
        valid, power_cost = movement_info['valid'], movement_info['power_cost']

        # move to center is not considered as moving
        is_moving = valid & (actions.direction != Direction.CENTER)

        # 1. update unit position and power
        new_pos = self.units.pos.pos + direct2delta_xy[actions.direction] * is_moving[..., None]
        new_power = self.units.power - power_cost * is_moving
        units = self.units._replace(
            pos=Position(new_pos),
            power=new_power,
        )

        # 2. resolve unit collision
        # classify units into groups
        unit_mask = self.unit_mask
        light = (units.unit_type == UnitType.LIGHT) & unit_mask  # bool[2, U]
        heavy = (units.unit_type == UnitType.HEAVY) & unit_mask  # bool[2, U]
        moving = is_moving & unit_mask
        still = (~is_moving) & unit_mask  # bool[2, U]
        chex.assert_shape(light, (2, self.MAX_N_UNITS))  # bool[2, U]
        chex.assert_equal_shape([light, heavy, moving, still])

        # count the number of different types of units in each location
        x, y = units.pos.x, units.pos.y
        is_living = ~already_dead & unit_mask

        cnt = jnp.zeros_like(self.board.units_map, dtype=jnp.int8)
        still_light_cnt = cnt.at[x, y].add(light & still & is_living, mode='drop')  # int[H, W]
        moving_light_cnt = cnt.at[x, y].add(light & moving & is_living, mode='drop')  # int[H, W]
        still_heavy_cnt = cnt.at[x, y].add(heavy & still & is_living, mode='drop')  # int[H, W]
        moving_heavy_cnt = cnt.at[x, y].add(heavy & moving & is_living, mode='drop')  # int[H, W]
        chex.assert_equal_shape([still_light_cnt, moving_light_cnt, still_heavy_cnt, moving_heavy_cnt, cnt])

        # get the second max power of moving light units
        max_power = jnp.full_like(self.board.units_map, fill_value=-1, dtype=units.power.dtype)
        moving_light_max_power = max_power.at[x, y].max(  # int[H, W]
            jnp.where(light & moving & is_living, units.power, -1),
            mode='drop',
        )
        max_power_in_unit_pos = moving_light_max_power[x, y]  # int[2, U]
        moving_light_second_max = max_power.at[x, y].max(  # int[H, W]
            jnp.where(light & moving & is_living & (units.power != max_power_in_unit_pos), units.power, -1),
            mode='drop',
        )
        moving_light_max_cnt = cnt.at[x, y].add(  # int[H, W]
            light & moving & is_living & (units.power == max_power_in_unit_pos),
            mode='drop',
        )
        moving_light_second_max = jnp.where(moving_light_max_cnt > 1, moving_light_max_power, moving_light_second_max)

        # get the second max power of moving heavy units
        moving_heavy_max_power = max_power.at[x, y].max(  # int[H, W]
            jnp.where(heavy & moving & is_living, units.power, -1),
            mode='drop',
        )
        max_power_in_unit_pos = moving_heavy_max_power[x, y]  # int[2, U]
        moving_heavy_second_max = max_power.at[x, y].max(  # int[H, W]
            jnp.where(heavy & moving & is_living & (units.power != max_power_in_unit_pos), units.power, -1),
            mode='drop',
        )
        moving_heavy_max_cnt = cnt.at[x, y].add(  # int[H, W]
            heavy & moving & is_living & (units.power == max_power_in_unit_pos),
            mode='drop',
        )
        moving_heavy_second_max = jnp.where(moving_heavy_max_cnt > 1, moving_heavy_max_power, moving_heavy_second_max)

        # map above count to agent-wise
        still_light_cnt = still_light_cnt[x, y]  # int[2, U]
        moving_light_cnt = moving_light_cnt[x, y]  # int[2, U]
        still_heavy_cnt = still_heavy_cnt[x, y]  # int[2, U]
        moving_heavy_cnt = moving_heavy_cnt[x, y]  # int[2, U]
        moving_light_second_max = moving_light_second_max[x, y]  # int[2, U]
        moving_heavy_second_max = moving_heavy_second_max[x, y]  # int[2, U]
        chex.assert_shape(still_light_cnt, (2, self.MAX_N_UNITS))  # bool[2, U]
        chex.assert_equal_shape([still_light_cnt, moving_light_cnt, still_heavy_cnt, moving_heavy_cnt])

        # dead cases
        cases = [
            # case 1 you are light and there is a heavy:
            (light & (still_heavy_cnt + moving_heavy_cnt > 0)),
            # case 2 you are light but still, and there is a another still light:
            (light & still & (still_light_cnt > 1)),
            # case 3 you are light but still, and there is a moving light:
            (light & still & (moving_light_cnt > 0)),
            # case 4 you are moving light, and there is another moving light having more or equal power:
            (light & moving & (moving_light_cnt > 1) & (units.power <= moving_light_second_max)),
            # case 5 you are heavy but still, and there is another still heavy:
            (heavy & still & (still_heavy_cnt > 1)),
            # case 6 you are heavy but still, and there is a moving heavy:
            (heavy & still & (moving_heavy_cnt > 0)),
            # case 7 you are heavy but moving, and there is another moving heavy having more or equal power:
            (heavy & moving & (moving_heavy_cnt > 1) & (units.power <= moving_heavy_second_max)),
        ]
        # or them together
        new_dead = functools.reduce(jnp.logical_or, cases)

        power_loss = (light & moving) * jnp.ceil(moving_light_second_max * self.env_cfg.POWER_LOSS_FACTOR).astype(units.power.dtype) \
            + (heavy & moving) * jnp.ceil(moving_heavy_second_max * self.env_cfg.POWER_LOSS_FACTOR).astype(units.power.dtype)
        power_loss = jnp.maximum(power_loss, 0)
        units = units._replace(power=units.power - power_loss)

        pos_without_dead = jnp.where(
            (already_dead | new_dead)[..., None],
            imax(units.pos.pos.dtype),
            units.pos.pos,
        )
        self = self._replace(
            units=units,
            board=self.board.update_units_map(units._replace(pos=Position(pos_without_dead))),
        )
        return self, new_dead

    def add_rubble_for_dead_units(self, dead: Array) -> 'State':
        units = self.units

        # add rubble to the board, and remove lichen
        rubble_after_destruction = units.get_cfg("RUBBLE_AFTER_DESTRUCTION", self.env_cfg.ROBOTS)
        rubble_after_destruction = rubble_after_destruction.astype(self.board.rubble.dtype)
        rubble = self.board.rubble.at[(
            units.pos.x,
            units.pos.y,
        )].add(dead * rubble_after_destruction, mode='drop')
        rubble = jnp.minimum(rubble, self.env_cfg.MAX_RUBBLE)
        lichen = self.board.lichen.at[(
            units.pos.x,
            units.pos.y,
        )].min(jnp.where(dead, 0, imax(self.board.lichen.dtype)), mode='drop')
        lichen_strains = self.board.lichen_strains.at[(
            units.pos.x,
            units.pos.y,
        )].max(jnp.where(dead, imax(self.board.lichen_strains.dtype), -1), mode='drop')

        # always set rubble under factories to 0.
        occupancy = self.factories.occupancy
        rubble = rubble.at[occupancy.x, occupancy.y].set(0, mode='drop')

        board = self.board._replace(
            map=self.board.map._replace(rubble=rubble),
            lichen=lichen,
            lichen_strains=lichen_strains,
        )

        return self._replace(board=board)

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

        # remove dead units, put them into the end of the array
        is_alive = ~dead & unit_mask
        unit_idx = jnp.where(is_alive, self.unit_idx, imax(self.unit_idx.dtype))
        live_idx = jnp.argsort(unit_idx).astype(self.units.unit_id.dtype)

        empty_unit = jax.tree_map(
            lambda x: jnp.array(x)[None, None],
            # replace UNIT_ACTION_QUEUE_SIZE with a concrete value to make it JIT-able
            Unit.empty(self.env_cfg._replace(UNIT_ACTION_QUEUE_SIZE=self.UNIT_ACTION_QUEUE_SIZE)),
        )

        units = jux.tree_util.tree_where(dead, empty_unit, units)
        units = jax.tree_map(lambda x: x[jnp.arange(2)[:, None], live_idx], units)

        # update other states
        n_units = self.n_units - dead.sum(axis=1, dtype=self.n_units.dtype)
        unit_id2idx = State.generate_unit_id2idx(units, self.MAX_GLOBAL_ID)

        # update board
        board = self.board.update_units_map(units)

        self = self._replace(
            units=units,
            n_units=n_units,
            unit_id2idx=unit_id2idx,
            board=board,
        )
        return self, live_idx

    def destroy_factories(self, dead: Array) -> 'State':
        '''
        Destroy dead factories, and put them into the end of the array.

        Args:
            dead: bool[2, F], dead indicator.

        Returns:
            new_state: State.
        '''
        factory_mask = self.factory_mask  # bool[2, F]
        factories = self.factories

        # add rubble to the board, and remove lichen
        occupancy = factories.occupancy

        rubble = self.board.rubble.at[(
            occupancy.x,
            occupancy.y,
        )].add(dead[..., None] * self.env_cfg.FACTORY_RUBBLE_AFTER_DESTRUCTION, mode='drop')
        rubble = jnp.minimum(rubble, self.env_cfg.MAX_RUBBLE)

        lichen = self.board.lichen.at[(
            occupancy.x,
            occupancy.y,
        )].min(jnp.where(dead[..., None], 0, imax(self.board.lichen.dtype)), mode='drop')
        lichen_strains = self.board.lichen_strains.at[(
            occupancy.x,
            occupancy.y,
        )].max(jnp.where(dead[..., None], imax(self.board.lichen_strains.dtype), -1), mode='drop')

        # remove dead factories, put them into the end of the array
        is_alive = ~dead & factory_mask
        factory_idx = jnp.where(is_alive, self.factory_idx, imax(self.factory_idx.dtype))
        live_idx = jnp.argsort(factory_idx).astype(self.factories.unit_id.dtype)

        empty_factory = jax.tree_map(
            lambda x: jnp.array(x)[None, None],
            Factory.empty(),
        )

        factories = jux.tree_util.tree_where(dead, empty_factory, factories)
        factories = jax.tree_map(lambda x: x[jnp.arange(2)[:, None], live_idx], factories)

        # update other states
        n_factories = self.n_factories - dead.sum(axis=1, dtype=self.n_factories.dtype)
        factory_id2idx = State.generate_factory_id2idx(factories, self.MAX_N_FACTORIES)

        # update board
        board = self.board.update_factories_map(factories)
        board = board._replace(
            map=board.map._replace(rubble=rubble),
            lichen=lichen,
            lichen_strains=lichen_strains,
        )

        self = self._replace(
            factories=factories,
            n_factories=n_factories,
            factory_id2idx=factory_id2idx,
            board=board,
        )
        return self

    def _validate_recharge_actions(self, actions: UnitAction):
        valid = (actions.action_type == UnitActionType.RECHARGE)
        return valid

    def _handle_recharge_actions(self, actions: UnitAction, valid: Array):
        # valid = self._validate_recharge_actions(actions)
        success = valid & (self.units.power >= actions.amount)
        return self, success

    def _handle_factory_water_actions(self, factory_actions: Array, color: Array, grow_lichen_size: Array) -> 'State':

        H, W = self.board.lichen_strains.shape

        # check validity
        water_cost = jnp.ceil(grow_lichen_size / self.env_cfg.LICHEN_WATERING_COST_FACTOR).astype(UnitCargo.dtype())
        valid = (factory_actions == FactoryAction.WATER) & (self.factories.cargo.water >= water_cost)  # bool[2, F]
        water_cost = jnp.where(valid, water_cost, 0)  # int[2, F]

        # new factory stocks
        new_stock = self.factories.cargo.stock.at[..., ResourceType.water].add(-water_cost)

        # lichen growth
        factory_color = color.at[self.factories.pos.x, self.factories.pos.y] \
                             .get(mode='fill', fill_value=imax(color.dtype))  # int[2, F, 2]
        delta_lichen = jnp.zeros((H, W), dtype=Board.__annotations__['lichen'])  # int[H, W]
        delta_lichen = delta_lichen.at[factory_color[..., 0], factory_color[..., 1]].add(valid * 2, mode='drop')
        delta_lichen = delta_lichen.at[color[..., 0], color[..., 1]].get(mode='fill', fill_value=0)
        factory_occupancy = self.factories.occupancy
        delta_lichen = delta_lichen.at[factory_occupancy.x, factory_occupancy.y].set(0, mode='drop')
        new_lichen = self.board.lichen + delta_lichen

        # lichen strain
        lichen_strains = jnp.zeros((H, W), dtype=Board.__annotations__['lichen_strains'])  # int[H, W]
        lichen_strains = lichen_strains.at[factory_color[..., 0], factory_color[..., 1]]\
                                       .set(self.factories.unit_id, mode='drop')
        lichen_strains = lichen_strains.at[color[..., 0], color[..., 1]].get(mode='fill', fill_value=0)
        new_lichen_strains = jnp.where(delta_lichen > 0, lichen_strains, self.board.lichen_strains)

        # update self
        self = self._replace(
            board=self.board._replace(
                lichen=new_lichen,
                lichen_strains=new_lichen_strains,
            ),
            factories=self.factories._replace(cargo=UnitCargo(new_stock)),
        )

        return self

    def _cache_water_info(self, factory_actions: Array) -> Tuple[Array, Array, Array]:
        """
        Run flood fill algorithm to color cells. All cells to be watered by the
        same factory will have the same color.

        Returns:
            color: int[H, W, 2].
                The first dimension represent the 'color'.The 'color' is
                represented by the coordinate of the factory a tile belongs to.
                If a tile is not connected to any factory, its color its own
                coordinate. In such a way, different lichen strains will have
                different colors.

            grow_lichen_size: int[2, F].
                The number of positions to be watered by each factory.
        """
        # The key idea here is to prepare a list of neighbors for each cell it
        # connects to when watered. neighbor_ij is a 4x2xHxW array, where the
        # first dimension is the neighbors (4 at most), the second dimension is
        # the coordinates (x,y) of neighbors.
        H, W = self.board.lichen_strains.shape

        ij = jnp.mgrid[:H, :W].astype(Position.dtype())
        delta_ij = jnp.array([
            [-1, 0],
            [0, 1],
            [1, 0],
            [0, -1],
        ], dtype=ij.dtype)  # int[2, H, W]
        neighbor_ij = delta_ij[..., None, None] + ij[None, ...]  # int[4, 2, H, W]

        # handle map boundary.
        neighbor_ij = neighbor_ij.at[0, 0, 0, :].set(0)
        neighbor_ij = neighbor_ij.at[1, 1, :, W - 1].set(W - 1)
        neighbor_ij = neighbor_ij.at[2, 0, H - 1, :].set(H - 1)
        neighbor_ij = neighbor_ij.at[3, 1, :, 0].set(0)

        # 1. calculate strain connections.
        strains_and_factory = jnp.minimum(self.board.lichen_strains, self.board.factory_occupancy_map)  # int[H, W]

        # handle a corner case where there may be rubbles on strains when movement collision happens.
        strains_and_factory = jnp.where(self.board.rubble == 0, strains_and_factory, imax(strains_and_factory.dtype))

        neighbor_color = strains_and_factory.at[(
            neighbor_ij[:, 0],
            neighbor_ij[:, 1],
        )].get(mode='fill', fill_value=imax(strains_and_factory.dtype))

        connect_cond = (
            (strains_and_factory == neighbor_color) & (strains_and_factory != imax(strains_and_factory.dtype))
        )  # bool[4, H, W]

        color = jux.map_generator.flood._flood_fill(  # int[H, W, 2]
            jnp.concatenate(  # int[H, W, 5, 2]
                [
                    jnp.where(connect_cond[:, None], neighbor_ij, ij).transpose(2, 3, 0, 1),  # int[H, W, 4, 2]
                    ij[None].transpose(2, 3, 0, 1),  # int[H, W, 1, 2]
                ],
                axis=-2,
            ))
        factory_color = color.at[self.factories.pos.x,self.factories.pos.y] \
                             .get(mode='fill', fill_value=imax(color.dtype))  # int[2, F, 2]
        connected_lichen = jnp.full((H, W), fill_value=imax(Factory.id_dtype()))  # int[H, W]
        connected_lichen = connected_lichen.at[factory_color[..., 0], factory_color[..., 1]] \
                                           .set(self.factories.unit_id, mode='drop')
        connected_lichen = connected_lichen.at[color[..., 0], color[..., 1]]\
                                           .get(mode='fill', fill_value=imax(connected_lichen.dtype))

        # compute connected lichen size
        connected_lichen_size = jux.map_generator.flood.component_sum(UnitCargo.dtype()(1), color)  # int[H, W]
        # -9 for the factory occupied cells
        connected_lichen_size = connected_lichen_size[self.factories.pos.x, self.factories.pos.y] - 9  # int[2, F]

        # 2. handle cells to expand to.
        # 2.1 cells that are allowed to expand to, only if
        #   1. it is not a lichen strain, and
        #   2. it has no rubble, and
        #   3. it is not resource.
        allow_grow = (self.board.rubble == 0) & \
                     ~(self.board.ice | self.board.ore) & \
                     (self.board.lichen_strains == imax(self.board.lichen_strains.dtype)) & \
                     (self.board.factory_occupancy_map == imax(self.board.factory_occupancy_map.dtype))

        # 2.2 when a non-lichen cell connects two different strains, then it is not allowed to expand to.
        neighbor_lichen_strain = strains_and_factory[neighbor_ij[:, 0], neighbor_ij[:, 1]]  # int[4, H, W]
        neighbor_is_lichen = neighbor_lichen_strain != imax(neighbor_lichen_strain.dtype)
        center_connects_two_different_strains = (strains_and_factory == imax(strains_and_factory.dtype)) & ( \
            ((neighbor_lichen_strain[0] != neighbor_lichen_strain[1]) & neighbor_is_lichen[0] & neighbor_is_lichen[1]) | \
            ((neighbor_lichen_strain[0] != neighbor_lichen_strain[2]) & neighbor_is_lichen[0] & neighbor_is_lichen[2]) | \
            ((neighbor_lichen_strain[0] != neighbor_lichen_strain[3]) & neighbor_is_lichen[0] & neighbor_is_lichen[3]) | \
            ((neighbor_lichen_strain[1] != neighbor_lichen_strain[2]) & neighbor_is_lichen[1] & neighbor_is_lichen[2]) | \
            ((neighbor_lichen_strain[1] != neighbor_lichen_strain[3]) & neighbor_is_lichen[1] & neighbor_is_lichen[3]) | \
            ((neighbor_lichen_strain[2] != neighbor_lichen_strain[3]) & neighbor_is_lichen[2] & neighbor_is_lichen[3]) \
        )
        allow_grow = allow_grow & ~center_connects_two_different_strains

        # 2.3 calculate the strains id, if it is expanded to.
        expand_center = (connected_lichen != imax(connected_lichen.dtype)) & \
                        (self.board.lichen >= self.env_cfg.MIN_LICHEN_TO_SPREAD)
        factory_occupancy = self.factories.occupancy
        expand_center = expand_center.at[factory_occupancy.x, factory_occupancy.y].set(True, mode='drop')
        expand_center = jnp.where(expand_center, connected_lichen, imax(connected_lichen.dtype))
        INT_MAX = imax(expand_center.dtype)
        strain_id_if_expand = jnp.minimum(  # int[H, W]
            jnp.minimum(
                jnp.roll(expand_center, 1, axis=0).at[0, :].set(INT_MAX),
                jnp.roll(expand_center, -1, axis=0).at[-1, :].set(INT_MAX),
            ),
            jnp.minimum(
                jnp.roll(expand_center, 1, axis=1).at[:, 0].set(INT_MAX),
                jnp.roll(expand_center, -1, axis=1).at[:, -1].set(INT_MAX),
            ),
        )
        strain_id_if_expand = jnp.where(allow_grow, strain_id_if_expand, INT_MAX)

        # 3. get the final color result.
        strain_id = jnp.minimum(connected_lichen, strain_id_if_expand)  # int[H, W]
        factory_idx = self.factory_id2idx[strain_id]  # int[2, H, W]
        color = self.factories.pos.pos[factory_idx[..., 0], factory_idx[..., 1]]  # int[H, W, 2]
        color = jnp.where((strain_id == imax(strain_id.dtype))[..., None], ij.transpose(1, 2, 0), color)

        # 4. grow_lichen_size
        cmp_cnt = jux.map_generator.flood.component_sum(UnitCargo.dtype()(1), color)  # int[H, W]
        # -9 for the factory occupied cells
        grow_lichen_size = cmp_cnt[self.factories.pos.x, self.factories.pos.y] - 9  # int[2, F]

        return color, grow_lichen_size, connected_lichen_size

    def _mars_quake(self) -> 'State':
        return self.add_rubble_for_dead_units(self.unit_mask)

    def team_lichen_score(self: 'State') -> Array:
        factory_id2idx = self.generate_factory_id2idx(
            self.factories._replace(unit_id=self.teams.factory_strains),
            self.MAX_N_FACTORIES,
        )

        factory_lichen = jnp.zeros(factory_id2idx.shape[0], dtype=self.board.lichen.dtype)  # int[2 * F]
        factory_lichen = factory_lichen.at[self.board.lichen_strains].add(self.board.lichen, mode='drop')  # int[2 * F]
        self.teams.factory_strains
        lichen_score = jnp.zeros((2, self.MAX_N_FACTORIES), dtype=factory_lichen.dtype).at[(
            factory_id2idx[..., 0],
            factory_id2idx[..., 1],
        )].set(factory_lichen, mode='drop')

        return lichen_score.sum(-1)  # int[2]
