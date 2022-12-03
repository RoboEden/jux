from enum import IntEnum
from typing import NamedTuple, Union

import jax.numpy as jnp
from jax import Array
from luxai2022.unit import UnitCargo as LuxUnitCargo


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
        return self.stock[..., ResourceType.ice]

    @property
    def ore(self):
        return self.stock[..., ResourceType.ore]

    @property
    def water(self):
        return self.stock[..., ResourceType.water]

    @property
    def metal(self):
        return self.stock[..., ResourceType.metal]

    @classmethod
    def from_lux(cls, lux_cargo: LuxUnitCargo) -> "UnitCargo":
        return UnitCargo(jnp.array([lux_cargo.ice, lux_cargo.ore, lux_cargo.water, lux_cargo.metal], dtype=jnp.int32))

    def to_lux(self) -> LuxUnitCargo:
        return LuxUnitCargo(int(self.ice), int(self.ore), int(self.water), int(self.metal))

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, UnitCargo) and jnp.array_equal(self.stock, __o.stock)

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
