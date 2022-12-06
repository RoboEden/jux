from typing import Dict, NamedTuple, Type

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from luxai2022.env import Factory as LuxFactory
from luxai2022.env import Unit as LuxUnit
from luxai2022.map.board import Board as LuxBoard

from jux.config import EnvConfig, JuxBufferConfig, LuxEnvConfig
from jux.map_generator.generator import GameMap, MapType, SymmetryType

INT32_MAX = jnp.iinfo(jnp.int32).max


class Board(NamedTuple):
    '''
    Map-related Arrays are stored in shape (JuxBufferConfig.MAX_MAP_SIZE, JuxBufferConfig.MAX_MAP_SIZE).
    However, only the first height * width elements are valid.

    ```python
    from jux.config import JuxBufferConfig, EnvConfig

    env_cfg = EnvConfig()
    buffer_cfg = JuxBufferConfig()

    MAX_MAP_SIZE = buffer_cfg.MAX_MAP_SIZE
    assert lichen.shape == (MAX_MAP_SIZE, MAX_MAP_SIZE)

    # the valid part
    MAP_SIZE = env_cfg.MAP_SIZE
    lichen[:MAP_SIZE, :MAP_SIZE]
    ```
    '''
    height: int
    width: int
    seed: int
    factories_per_team: int

    map: GameMap

    lichen: Array  # int[height, width]

    lichen_strains: Array  # int[height, width]
    '''
    ownership of lichen by factory id, a simple mask.
    INT32_MAX = no ownership.
    type: int[height, width]
    '''

    units_map: Array  # int[height, width]
    '''
    unit_id (or may be unit_idx) in the cell.
    INT32_MAX = no unit.
    type: int[height, width]
    '''

    factory_map: Array  # int[height, width]
    '''
    factory_id (or may be factory_idx) in the cell.
    INT32_MAX = no factory.
    type: int[height, width]
    '''

    factory_occupancy_map: Array  # int[height, width]
    '''
    factory_id (or may be factory_idx) occupying current cell. Note: each factory occupies 3x3 cells.
    INT32_MAX = no factory.
    type: int[height, width]
    '''

    spawn_masks: Array  # int[height, width]
    '''
    team_id allowed to spawn factory in each cell.
    INT32_MAX = no team.
    type: int[height, width]
    '''

    # spawns is duplicated with spawn_masks, please use spawn_masks instead.

    @classmethod
    def new(cls, seed: int, env_cfg: EnvConfig, buf_cfg: JuxBufferConfig):
        height = buf_cfg.MAX_MAP_SIZE
        width = buf_cfg.MAX_MAP_SIZE
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        map_type = jax.random.choice(
            key=subkey,
            a=jnp.array([MapType.CAVE, MapType.MOUNTAIN]),
        )
        key, subkey = jax.random.split(key)
        symmetry = jax.random.choice(
            key=subkey,
            a=jnp.array([SymmetryType.HORIZONTAL, SymmetryType.VERTICAL]),
        )
        map = GameMap.random_map(
            seed=seed,
            symmetry=symmetry,
            map_type=map_type,
            width=width,
            height=height,
        )
        key, subkey = jax.random.split(key)
        factories_per_team = jax.random.randint(
            key=subkey,
            shape=(1, ),
            minval=env_cfg.MIN_FACTORIES,
            maxval=env_cfg.MAX_FACTORIES + 1,
        )[0]

        lichen = jnp.zeros(shape=(height, width), dtype=jnp.int32)
        lichen_strains = jnp.full(shape=(height, width), fill_value=INT32_MAX)
        units_map = jnp.full(shape=(height, width), fill_value=INT32_MAX)

        factory_map = jnp.full(shape=(height, width), fill_value=INT32_MAX)
        factory_occupancy_map = jnp.full(shape=(height, width), fill_value=INT32_MAX)
        spawn_masks = cls.get_valid_spawns(height, width, symmetry)

        return cls(
            height=height,
            width=width,
            seed=seed,
            factories_per_team=factories_per_team,
            map=map,
            lichen=lichen,
            lichen_strains=lichen_strains,
            units_map=units_map,
            factory_map=factory_map,
            factory_occupancy_map=factory_occupancy_map,
            spawn_masks=spawn_masks,
        )

    @classmethod
    def from_lux(cls: Type['Board'], lux_board: LuxBoard, buf_cfg: JuxBufferConfig) -> "Board":
        buf_size = (buf_cfg.MAX_MAP_SIZE, buf_cfg.MAX_MAP_SIZE)
        height, width = lux_board.height, lux_board.width

        lichen = jnp.zeros(shape=(height, width), dtype=jnp.int32)
        lichen = lichen.at[:height, :width].set(lux_board.lichen)

        lichen_strains = jnp.full(shape=(height, width), fill_value=INT32_MAX, dtype=jnp.int32)
        lichen_strains = lichen_strains.at[:height, :width].set(lux_board.lichen_strains)
        lichen_strains = lichen_strains.at[lichen_strains == -1].set(INT32_MAX)

        # put factories id to map
        lux_board.factory_map: Dict[str, 'LuxFactory']
        xs = []
        ys = []
        factory_id = []
        for k, v in lux_board.factory_map.items():
            x, y = eval(k)
            v = int(v.unit_id[len('factory_'):])  # factory_0
            xs.append(x)
            ys.append(y)
            factory_id.append(v)
        factory_map = jnp.full(buf_size, fill_value=INT32_MAX, dtype=jnp.int32)  # default value is INT32_MAX
        factory_id = jnp.array(factory_id, dtype=jnp.int32)
        factory_map = factory_map.at[ys, xs].set(factory_id)

        factory_occupancy_map = jnp.full(buf_size, fill_value=INT32_MAX, dtype=jnp.int32)
        factory_occupancy_map = factory_occupancy_map.at[:height, :width].set(lux_board.factory_occupancy_map)
        factory_occupancy_map = factory_occupancy_map.at[factory_occupancy_map == -1].set(INT32_MAX)

        # put unit_id to map
        xs = []
        ys = []
        unit_id = []
        for k, v in lux_board.units_map.items():
            x, y = eval(k)
            if len(v) == 0:
                continue
            assert len(v) == 1
            v = int(v[0].unit_id[len('unit_'):])  # factory_0
            xs.append(x)
            ys.append(y)
            unit_id.append(v)
        units_map = jnp.full(buf_size, fill_value=INT32_MAX, dtype=jnp.int32)  # default value is INT32_MAX
        unit_id = jnp.array(unit_id, dtype=jnp.int32)
        units_map = units_map.at[ys, xs].set(unit_id)

        # spawn_masks
        spawn_masks = jnp.full(buf_size, fill_value=INT32_MAX, dtype=jnp.int32)
        spawn_masks = spawn_masks.at[lux_board.spawn_masks['player_0']].set(0)
        spawn_masks = spawn_masks.at[lux_board.spawn_masks['player_1']].set(1)

        return cls(
            height=height,
            width=width,
            seed=lux_board.seed,
            factories_per_team=lux_board.factories_per_team,
            map=GameMap.from_lux(lux_board.map, buf_cfg),
            lichen=lichen,
            lichen_strains=lichen_strains,
            units_map=units_map,
            factory_map=factory_map,
            factory_occupancy_map=factory_occupancy_map,
            spawn_masks=spawn_masks,
        )

    def to_lux(
        self,
        lux_env_cfg: LuxEnvConfig,
        lux_factories: Dict[str, LuxFactory],
        lux_units: Dict[str, LuxUnit],
    ) -> LuxBoard:
        lux_board = LuxBoard.__new__(LuxBoard)
        height, width = self.height, self.width

        lux_board.env_cfg = lux_env_cfg
        lux_board.height = int(height)
        lux_board.width = int(width)
        lux_board.seed = self.seed
        lux_board.factories_per_team = int(self.factories_per_team)
        lux_board.map = self.map.to_lux()
        lux_board.lichen = np.array(self.lichen[:self.height, :self.width])

        lichen_strains = self.lichen_strains[:self.height, :self.width]
        lichen_strains = lichen_strains.at[lichen_strains == INT32_MAX].set(-1)
        lux_board.lichen_strains = np.array(lichen_strains)

        ys, xs = (self.units_map[:self.height, :self.width] != INT32_MAX).nonzero()
        unit_id = self.units_map[ys, xs]
        ys, xs, unit_id = np.array(ys), np.array(xs), np.array(unit_id)
        lux_units = {**lux_units['player_0'], **lux_units['player_1']}
        lux_board.units_map = {}
        for y, x, uid in zip(ys, xs, unit_id):
            lux_board.units_map.setdefault(f'({x}, {y})', []).append(lux_units[f"unit_{int(uid)}"])

        ys, xs = (self.factory_map[:self.height, :self.width] != INT32_MAX).nonzero()
        factory_id = self.factory_map[ys, xs]
        ys, xs, factory_id = np.array(ys), np.array(xs), np.array(factory_id)
        lux_factories = {**lux_factories['player_0'], **lux_factories['player_1']}
        lux_board.factory_map = {
            f'({x}, {y})': lux_factories[f"factory_{int(fid)}"]
            for y, x, fid in zip(ys, xs, factory_id)
        }

        lux_board.factory_occupancy_map = np.array(self.factory_occupancy_map[:self.height, :self.width])
        lux_board.factory_occupancy_map[lux_board.factory_occupancy_map == INT32_MAX] = -1
        lux_board.spawn_masks = {
            "player_0": np.array(self.spawn_masks == 0),
            "player_1": np.array(self.spawn_masks == 1),
        }
        spawns0 = np.array(lux_board.spawn_masks['player_0'].nonzero()).T
        spawns1 = np.array(lux_board.spawn_masks['player_1'].nonzero()).T
        spawns0 = spawns0[(spawns0 > 0).all(axis=1) & (spawns0 < np.array([height, width]) - 1).all(axis=1)]
        spawns1 = spawns1[(spawns1 > 0).all(axis=1) & (spawns1 < np.array([height, width]) - 1).all(axis=1)]
        lux_board.spawns = {
            "player_0": spawns0,
            "player_1": spawns1,
        }
        return lux_board

    @property
    def rubble(self) -> Array:
        return self.map.rubble

    @property
    def ice(self) -> Array:
        return self.map.ice

    @property
    def ore(self) -> Array:
        return self.map.ore

    def __eq__(self, __o: "Board") -> bool:
        if not isinstance(__o, Board):
            return False
        return ((self.height == __o.height) & (self.width == __o.width) & (self.seed == __o.seed)
                & (self.factories_per_team == __o.factories_per_team) & (self.map == __o.map)
                & jnp.array_equal(self.lichen, __o.lichen)
                & jnp.array_equal(self.lichen_strains, __o.lichen_strains)
                & jnp.array_equal(self.units_map, __o.units_map)
                & jnp.array_equal(self.factory_map, __o.factory_map)
                & jnp.array_equal(self.factory_occupancy_map, __o.factory_occupancy_map)
                & jnp.array_equal(self.spawn_masks, __o.spawn_masks))

    @staticmethod
    def get_valid_spawns(height, width, symmetry):
        xx, yy = np.mgrid[:int(width), :int(height)]

        def _horizontal0():
            spawns_mask = yy < (height - 2) / 2
            return spawns_mask

        def _vertical0():
            spawns_mask = xx < (width - 2) / 2
            return spawns_mask

        def _horizontal1():
            spawns_mask = yy >= (height + 2) / 2
            return spawns_mask

        def _vertical1():
            spawns_mask = xx >= (width + 2) / 2
            return spawns_mask

        spawns_mask0 = lax.switch(
            symmetry,
            [_horizontal0, _vertical0],
        )
        spawns_mask1 = lax.switch(
            symmetry,
            [_horizontal1, _vertical1],
        )

        spawns_masks = spawns_mask0 * 0 + spawns_mask1 * 1 + ~(spawns_mask0 | spawns_mask1) * INT32_MAX
        return spawns_masks

    def update_units_map(self, units) -> 'Board':
        units_map = jnp.full_like(self.units_map, fill_value=INT32_MAX)
        pos = units.pos
        units_map = units_map.at[pos.y, pos.x].set(units.unit_id, mode='drop')
        return self._replace(units_map=units_map)
