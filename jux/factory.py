from typing import Dict, NamedTuple, Tuple

import jax
from jax import numpy as jnp
from luxai_s2.factory import Factory as LuxFactory
from luxai_s2.team import Team as LuxTeam

from jux.config import EnvConfig
from jux.map.position import Position, direct2delta_xy
from jux.unit import ResourceType, Unit, UnitCargo
from jux.utils import INT32_MAX, imax


class Factory(NamedTuple):
    team_id: jnp.int8 = imax(jnp.int8)
    unit_id: jnp.int8 = imax(jnp.int8)
    pos: Position = Position()  # int16[2]
    power: jnp.int32 = jnp.int32(0)
    cargo: UnitCargo = UnitCargo()  # int[4]

    @property
    def num_id(self):
        return self.unit_id

    @property
    def occupancy(self) -> Position:
        edge_pos = self.pos.pos[..., None, :] + direct2delta_xy[1:]  # [4, 2]
        corner_pos = edge_pos + direct2delta_xy[[2, 3, 4, 1], :]  # [4, 2]
        occupy = jnp.concatenate([self.pos.pos[..., None, :], edge_pos, corner_pos], axis=-2)  # [9, 2]
        occupy = Position(occupy)
        return occupy

    @staticmethod
    def id_dtype():
        return Factory.__annotations__['unit_id']

    @classmethod
    def empty(cls):
        return cls()

    @classmethod
    def new(cls, team_id: int, unit_id: int, pos: Position, power: int, cargo: UnitCargo):
        return cls(
            team_id=Factory.__annotations__['team_id'](team_id),
            unit_id=Factory.__annotations__['unit_id'](unit_id),
            pos=pos,
            power=Factory.__annotations__['power'](power),
            cargo=UnitCargo.from_lux(cargo),
        )

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
        return cls.new(
            team_id=lux_factory.team_id,
            unit_id=lux_factory.unit_id[len('factory_'):],
            pos=Position.from_lux(lux_factory.pos),
            power=lux_factory.power,
            cargo=UnitCargo.from_lux(lux_factory.cargo),
        )

    def to_lux(self, teams: Dict[str, LuxTeam]) -> LuxFactory:
        lux_factory = LuxFactory(
            team=teams[f"player_{int(self.team_id)}"],
            unit_id=f"factory_{int(self.unit_id)}",
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
        return eq
