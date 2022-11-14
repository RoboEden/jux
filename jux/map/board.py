from typing import NamedTuple

from jax import Array
from luxai2022.map.board import Board as LuxBoard

from jux.map_generator.generator import GameMap


class Board(NamedTuple):
    '''
    Arrays whose shapes should be (height, width) are stored in shape (JuxBufferConfig.MAX_MAP_SIZE, JuxBufferConfig.MAX_MAP_SIZE).
    However, only the first height * width elements are valid.

    ```python
    from jux.config import JuxBufferConfig, EnvConfig

    env_cfg = EnvConfig()
    buffer_cfg = JuxBufferConfig()

    MAX_MAP_SIZE = buffer_cfg.MAX_MAP_SIZE
    assert lichen.shape == (MAX_MAP_SIZE, MAX_MAP_SIZE)

    MAP_SIZE = env_cfg.MAP_SIZE
    lichen[:MAP_SIZE, :MAP_SIZE] # the valid part
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
    allowed team_id (or may be team_idx) to spawn factory in each cell.
    type: int[height, width]
    '''

    # spawns is duplicated with spawn_masks, please use spawn_masks instead.

    @classmethod
    def from_lux(cls, lux_board: LuxBoard) -> "Board":
        # TODO
        pass

    def to_lux(self) -> LuxBoard:
        # TODO
        pass

    @property
    def rubble(self) -> Array:
        return self.map.rubble

    @property
    def ice(self) -> Array:
        return self.map.ice

    @property
    def ore(self) -> Array:
        return self.map.ore
