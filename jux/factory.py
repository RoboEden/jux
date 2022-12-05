from typing import Dict, NamedTuple, Tuple

import jax
from jax import numpy as jnp
from luxai2022.factory import Factory as LuxFactory
from luxai2022.team import Team as LuxTeam

from jux.config import EnvConfig
from jux.map.position import Position
from jux.unit import ResourceType, Unit, UnitCargo

INT32_MAX = jnp.iinfo(jnp.int32).max


class Factory(NamedTuple):
    team_id: int = INT32_MAX
    # team # no need team object, team_id is enough
    unit_id: int = INT32_MAX
    pos: Position = Position()  # int16[2]
    power: int = jnp.int32(0)
    cargo: UnitCargo = UnitCargo()  # int[4]
    num_id: int = INT32_MAX

    # action_queue # Do we need action queue for factories?
    @classmethod
    def empty(cls):
        return cls()

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

        new_factory, transfer_amount = jax.lax.cond(
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

        new_factory, transfer_amount = jax.lax.cond(
            resource == ResourceType.power,
            sub_power,
            sub_others,
            *(self, resource, amount),
        )
        return new_factory, transfer_amount

    @classmethod
    def from_lux(cls, lux_factory: LuxFactory) -> "Factory":
        return cls(
            team_id=lux_factory.team_id,
            unit_id=int(lux_factory.unit_id[len('factory_'):]),
            pos=Position.from_lux(lux_factory.pos),
            power=lux_factory.power,
            cargo=UnitCargo.from_lux(lux_factory.cargo),
            num_id=lux_factory.num_id,
        )

    def to_lux(self, teams: Dict[str, LuxTeam]) -> LuxFactory:
        lux_factory = LuxFactory(
            team=teams[f"player_{self.team_id}"],
            unit_id=f"factory_{self.unit_id}",
            num_id=int(self.num_id),
        )
        lux_factory.pos = self.pos.to_lux()
        lux_factory.power = int(self.power)
        lux_factory.cargo = self.cargo.to_lux()
        return lux_factory

    def refine_step(self, env_cfg: EnvConfig) -> "Factory":
        max_consumed_ice = jnp.minimum(self.cargo.ice, env_cfg.FACTORY_PROCESSING_RATE_WATER)
        max_consumed_ore = jnp.minimum(self.cargo.ore, env_cfg.FACTORY_PROCESSING_RATE_METAL)
        # permit refinement of blocks of resources, no floats.
        produced_water = max_consumed_ice // env_cfg.ICE_WATER_RATIO
        produced_metal = max_consumed_ore // env_cfg.ORE_METAL_RATIO

        stock_change = jnp.stack(
            [
                -produced_water * env_cfg.ICE_WATER_RATIO,
                -produced_metal * env_cfg.ORE_METAL_RATIO,
                produced_water,
                produced_metal,
            ],
            axis=-1,
        )  # int[..., 4]

        new_stock = self.cargo.stock + stock_change
        return self._replace(cargo=UnitCargo(new_stock))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Factory):
            return False
        eq = True
        eq = eq & (self.team_id == other.team_id)
        eq = eq & (self.unit_id == other.unit_id)
        eq = eq & (self.pos == other.pos)
        eq = eq & (self.power == other.power)
        eq = eq & (self.cargo == other.cargo)
        eq = eq & (self.num_id == other.num_id)
        return eq
