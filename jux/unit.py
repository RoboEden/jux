from enum import IntEnum
from typing import NamedTuple, Tuple, Union

import jax.numpy as jnp
from jax import Array, lax
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
    stock: Array = jnp.zeros(4, jnp.int32)  # int[4]

    @property
    def ice(self):
        return self.stock[ResourceType.ice]

    @property
    def ore(self):
        return self.stock[ResourceType.ore]

    @property
    def water(self):
        return self.stock[ResourceType.water]

    @property
    def metal(self):
        return self.stock[ResourceType.metal]

    @classmethod
    def from_lux(cls, lux_cargo: LuxUnitCargo) -> "UnitCargo":
        # TODO
        pass

    def to_lux(self) -> LuxUnitCargo:
        # TODO
        pass

    def add_resource(self,
                     resource: ResourceType,
                     amount: int,
                     cargo_space: int = jnp.iinfo(jnp.int32).max // 2) -> Union[int, Array]:
        '''JUX implementation for luxai2022.unit.Unit.add_resource and luxai2022.factory.Factory.add_resource.

        For Unit, cargo_space should be unit.cargo_space.
        For Factory, cargo_space should be inf (here is int32.max//2).

        Returns:
            int: the amount of resource that really transferred.
        '''
        amount = jnp.maximum(amount, 0)
        stock = self.stock[resource]
        transfer_amount = jnp.minimum(cargo_space - stock, amount)
        new_stock = self.stock.at[resource].add(transfer_amount)
        new_cargo = self._replace(stock=new_stock)
        return new_cargo, transfer_amount

    def sub_resource(self, resource: ResourceType, amount: int) -> Union[int, Array]:
        '''JUX implementation for luxai2022.unit.Unit.sub_resource and luxai2022.factory.Factory.sub_resource.

        Returns:
            int: the amount of resource that really transferred.
        '''
        amount = jnp.maximum(amount, 0)
        stock = self.stock[resource]
        transfer_amount = jnp.minimum(stock, amount)
        new_stock = self.stock.at[resource].add(-transfer_amount)
        new_cargo = self._replace(stock=new_stock)
        return new_cargo, transfer_amount


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
            action_queue=jnp.zeros((env_cfg.UNIT_ACTION_QUEUE_SIZE, 5), dtype=jnp.int32),
            unit_cfg=env_cfg.ROBOTS[unit_type],
            power=env_cfg.ROBOTS[unit_type].INIT_POWER,
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

    def add_resource(self, resource: ResourceType, amount: int) -> Tuple['Unit', Union[int, Array]]:
        # If resource != ResourceType.power, call UnitCargo.add_resource.
        # else, call Unit.add_power.
        amount = jnp.maximum(amount, 0)

        def add_power(self, resource: ResourceType, amount: int):
            transfer_amount = jnp.minimum(self.battery_capacity - self.power, amount)
            new_unit = self._replace(power=self.power + transfer_amount)
            return new_unit, transfer_amount

        def add_others(self: Unit, resource: ResourceType, amount: int):
            new_cargo, transfer_amount = self.cargo.add_resource(
                resource=resource,
                amount=amount,
                cargo_space=self.cargo_space,
            )
            new_unit = self._replace(cargo=new_cargo)
            return new_unit, transfer_amount

        new_unit, transfer_amount = lax.cond(
            resource == ResourceType.power,
            add_power,
            add_others,
            *(self, resource, amount),
        )
        return new_unit, transfer_amount

    def sub_resource(self, resource: ResourceType, amount: int) -> Union[int, Array]:
        # If resource != ResourceType.power, call UnitCargo.add_resource.
        # else, call Unit.sub_resource.
        def sub_power(self, resource: ResourceType, amount: int):
            transfer_amount = jnp.minimum(self.power, amount)
            new_unit = self._replace(power=self.power - transfer_amount)
            return new_unit, transfer_amount

        def sub_others(self: Unit, resource: ResourceType, amount: int):
            new_cargo, transfer_amount = self.cargo.sub_resource(
                resource=resource,
                amount=amount,
            )
            new_unit = self._replace(cargo=new_cargo)
            return new_unit, transfer_amount

        new_unit, transfer_amount = lax.cond(
            resource == ResourceType.power,
            sub_power,
            sub_others,
            *(self, resource, amount),
        )
        return new_unit, transfer_amount
