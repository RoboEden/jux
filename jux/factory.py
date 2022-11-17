from typing import NamedTuple, Tuple

from jax import Array, lax
from jax import numpy as jnp
from luxai2022.factory import Factory as LuxFactory
from luxai2022.team import Team as LuxTeam

from jux.map.position import Position
from jux.unit import ResourceType, Unit, UnitCargo


class Factory(NamedTuple):
    team_id: int
    # team # no need team object, team_id is enough
    unit_id: int
    pos: Array  # int16[2]
    power: int
    cargo: UnitCargo  # int[4]
    num_id: int

    # action_queue # Do we need action queue for factories?

    def add_resource(self, resource: ResourceType, transfer_amount: int) -> Tuple['Factory', int]:
        # If resource != ResourceType.power, call UnitCargo.add_resource.
        # else, call Unit.add_power.
        transfer_amount = jnp.maximum(transfer_amount, 0)

        def add_power(self: Factory, transfer_amount: int):
            new_factory = self._replace(power=self.power + transfer_amount)
            return new_factory, transfer_amount

        def add_others(self: Factory, transfer_amount: int):
            new_cargo, transfer_amount = self.cargo.add_resource(
                resource=resource,
                amount=transfer_amount,
                cargo_space=jnp.iinfo(jnp.int32).max // 2,
            )
            new_factory = self._replace(cargo=new_cargo)
            return new_factory, transfer_amount

        new_factory, transfer_amount = lax.cond(
            resource == ResourceType.power,
            add_power,
            add_others,
            *(self, transfer_amount),
        )
        return new_factory, transfer_amount

    def sub_resource(self, resource: ResourceType, amount: int) -> Tuple['Factory', int]:
        # If resource != ResourceType.power, call UnitCargo.add_resource.
        # else, call Unit.sub_resource.
        def sub_power(self, resource: ResourceType, amount: int):
            transfer_amount = jnp.minimum(self.power, amount)
            new_factory = self._replace(power=self.power - transfer_amount)
            return new_factory, transfer_amount

        def sub_others(self: Unit, resource: ResourceType, amount: int):
            new_cargo, transfer_amount = self.cargo.sub_resource(
                resource=resource,
                amount=amount,
            )
            new_factory = self._replace(cargo=new_cargo)
            return new_factory, transfer_amount

        new_factory, transfer_amount = lax.cond(
            resource == ResourceType.power,
            sub_power,
            sub_others,
            *(self, resource, amount),
        )
        return new_factory, transfer_amount

    @classmethod
    def from_lux(cls, lux_factory: LuxFactory) -> "Factory":
        stock = jnp.array([
            lux_factory.cargo.ice,
            lux_factory.cargo.ore,
            lux_factory.cargo.water,
            lux_factory.cargo.metal,
        ])
        return cls(
            team_id=lux_factory.team_id,
            unit_id=lux_factory.unit_id,
            pos=Position(jnp.array(lux_factory.pos)),
            power=lux_factory.power,
            cargo=UnitCargo(stock=stock),
            num_id=lux_factory.num_id,
        )

    def to_lux(self, team: LuxTeam) -> LuxFactory:
        return LuxFactory(
            team=team,
            unit_id=self.unit_id,
            num_id=self.num_id,
        )
