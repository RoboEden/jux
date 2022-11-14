from enum import IntEnum
from typing import NamedTuple, Union

import chex
import jax.numpy as jnp
from jax import Array
from luxai2022.unit import Unit as LuxUnit
from luxai2022.unit import UnitCargo as LuxUnitCargo

from jux.config import EnvConfig, UnitConfig
from jux.map.position import Position


class UnitType(IntEnum):
    LIGHT = 0
    HEAVY = 1


class ResourceType(IntEnum):
    ice = 0
    ore = 1
    water = 2
    metal = 3
    power = 4


class UnitCargo(NamedTuple):
    ice: int = 0
    ore: int = 0
    water: int = 0
    metal: int = 0

    @classmethod
    def from_lux(cls, lux_cargo: LuxUnitCargo) -> "UnitCargo":
        # TODO
        pass

    def to_lux(self) -> LuxUnitCargo:
        # TODO
        pass

    def add_resource(self,
                     resource_id: ResourceType,
                     amount: int,
                     cargo_space: int = jnp.iinfo(jnp.int32).max // 2) -> Union[int, Array]:
        '''JUX implementation for luxai2022.unit.Unit.add_resource and luxai2022.factory.Factory.add_resource.

        For Unit, cargo_space should be unit.cargo_space.
        For Factory, cargo_space should be inf (here is int32.max//2).

        Returns:
            int: the amount of resource that really transferred.
        '''
        # TODO

    def sub_resource(self, resource_id: ResourceType, amount: int) -> Union[int, Array]:
        '''JUX implementation for luxai2022.unit.Unit.sub_resource and luxai2022.factory.Factory.sub_resource.

        Returns:
            int: the amount of resource that really transferred.
        '''
        # TODO


class Unit(NamedTuple):
    unit_type: UnitType
    team_id: int
    # team # no need team object, team_id is enough
    unit_id: int
    pos: Position

    cargo: UnitCargo
    action_queue: Array  # int[UNIT_ACTION_QUEUE_SIZE, 5]
    unit_cfg: UnitConfig
    power: int

    @classmethod
    def new(cls, team_id: int, unit_type: UnitType, unit_id: int, env_cfg: EnvConfig):
        return cls(
            unit_type=unit_type,
            team_id=team_id,
            unit_id=unit_id,
            pos=Position.new(),
            cargo=UnitCargo(),
            action_queue=chex.zeros((env_cfg.UNIT_ACTION_QUEUE_SIZE, 5), dtype=jnp.int32),
            unit_cfg=env_cfg.unit_cfg,
            power=env_cfg.unit_cfg.INIT_POWER,
        )

    @property
    def cargo_space(self):
        return self.unit_cfg.CARGO_SPACE

    @property
    def battery_capacity(self):
        return self.unit_cfg.BATTERY_CAPACITY

    @classmethod
    def from_lux(cls, lux_cargo: LuxUnit) -> "Unit":
        # TODO
        pass

    def to_lux(self) -> LuxUnit:
        # TODO
        pass

    def is_heavy(self) -> Union[bool, Array]:
        return self.unit_type == UnitType.HEAVY

    def move_power_cost(self, rubble_at_target: int):
        return self.unit_cfg.MOVE_COST + self.unit_cfg.RUBBLE_MOVEMENT_COST * rubble_at_target

    def add_resource(self, resource: ResourceType, amount: int) -> Union[int, Array]:
        # TODO
        # If resource != ResourceType.power, call UnitCargo.add_resource.
        # else, call Unit.add_power.
        pass

    def sub_resource(self, resource: ResourceType, amount: int) -> Union[int, Array]:
        # TODO
        # If resource != ResourceType.power, call UnitCargo.add_resource.
        # else, call Unit.sub_resource.
        pass
