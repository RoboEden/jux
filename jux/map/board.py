from typing import Dict, List, NamedTuple, Type

import jax.numpy as jnp
import numpy as np
from jax import Array
from luxai2022.env import Factory as LuxFactory
from luxai2022.env import Unit as LuxUnit
from luxai2022.map.board import Board as LuxBoard

from jux.config import EnvConfig, JuxBufferConfig
from jux.map_generator.generator import GameMap


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
    -1 = no ownership.
    type: int[height, width]
    '''

    units_map: Array  # int[height, width]
    '''
    unit_id (or may be unit_idx) in the cell.
    type: int[height, width]
    '''

    factory_map: Array  # int[height, width]
    '''
    factory_id (or may be factory_idx) in the cell.
    type: int[height, width]
    '''

    factory_occupancy_map: Array  # int[height, width]
    '''
    factory_id (or may be factory_idx) occupying current cell. Note: each factory occupies 3x3 cells.
    type: int[height, width]
    '''

    spawn_masks: Array  # int[height, width]
    '''
    team_id allowed to spawn factory in each cell.
    type: int[height, width]
    '''

    # spawns is duplicated with spawn_masks, please use spawn_masks instead.

    @classmethod
    def from_lux(cls: Type['Board'], lux_board: LuxBoard, buf_cfg: JuxBufferConfig) -> "Board":
        buf_size = (buf_cfg.MAX_MAP_SIZE, buf_cfg.MAX_MAP_SIZE)
        height, width = lux_board.height, lux_board.width

        lichen = jnp.empty(buf_size, jnp.int32)
        lichen = lichen.at[:height, :width].set(lux_board.lichen)

        lichen_strains = jnp.empty(buf_size, jnp.int32)
        lichen_strains = lichen_strains.at[:height, :width].set(lux_board.lichen_strains)

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
        factory_map = -jnp.ones(buf_size, dtype=jnp.int32)  # default value is -1
        factory_id = jnp.array(factory_id, dtype=jnp.int32)
        factory_map = factory_map.at[ys, xs].set(factory_id)

        factory_occupancy_map = jnp.empty(buf_size, jnp.int32)
        factory_occupancy_map = factory_occupancy_map.at[:height, :width].set(lux_board.factory_occupancy_map)

        # put unit_id to map
        xs = []
        ys = []
        unit_id = []
        for k, v in lux_board.factory_map.items():
            x, y = eval(k)
            if len(v) == 0:
                continue
            assert len(v) == 1
            v = int(v[0].unit_id[len('unit_'):])  # factory_0
            xs.append(x)
            ys.append(y)
            unit_id.append(v)
        units_map = -jnp.ones(buf_size, dtype=jnp.int32)  # default value is -1
        unit_id = jnp.array(unit_id, dtype=jnp.int32)
        units_map = units_map.at[ys, xs].set(unit_id)

        # spawn_masks
        spawn_masks = -jnp.ones(buf_size, dtype=jnp.int32)
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

    def to_lux(self, env_cfg: EnvConfig, factories: List[LuxFactory], units: List[LuxUnit]) -> LuxBoard:
        lux_board = LuxBoard.__new__(LuxBoard)
        height, width = self.height, self.width

        lux_board.env_cfg = env_cfg.to_lux()
        lux_board.height = int(height)
        lux_board.width = int(width)
        lux_board.seed = self.seed
        lux_board.factories_per_team = int(self.factories_per_team)
        lux_board.map = self.map.to_lux()
        lux_board.lichen = np.array(self.lichen[:self.height, :self.width])
        lux_board.lichen_strains = np.array(self.lichen_strains[:self.height, :self.width])

        ys, xs = (self.units_map[:self.height, :self.width] != -1).nonzero()
        unit_id = self.units_map[ys, xs]
        ys, xs, unit_id = np.array(ys), np.array(xs), np.array(unit_id)
        lux_board.units_map = {f'({x}, {y})': units[uid] for y, x, uid in zip(ys, xs, unit_id)}

        ys, xs = (self.factory_map[:self.height, :self.width] != -1).nonzero()
        factory_id = self.factory_map[ys, xs]
        ys, xs, factory_id = np.array(ys), np.array(xs), np.array(factory_id)
        lux_board.factory_map = {f'({x}, {y})': factories[fid] for y, x, fid in zip(ys, xs, factory_id)}

        lux_board.factory_occupancy_map = np.array(self.factory_occupancy_map[:self.height, :self.width])
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

    def __eq__(self, __o: object) -> bool:
        return (self.height == __o.height and self.width == __o.width and self.seed == __o.seed
                and self.factories_per_team == __o.factories_per_team and self.map == __o.map
                and (self.lichen == __o.lichen).all() and (self.lichen_strains == __o.lichen_strains).all()
                and (self.units_map == __o.units_map).all() and (self.factory_map == __o.factory_map).all()
                and (self.factory_occupancy_map == __o.factory_occupancy_map).all()
                and (self.spawn_masks == __o.spawn_masks).all())
