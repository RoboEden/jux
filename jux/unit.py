from enum import IntEnum
from typing import List, NamedTuple, Tuple, Union

import jax.numpy as jnp
from jax import Array, lax
from luxai2022.config import EnvConfig as LuxEnvConfig
from luxai2022.team import Team as LuxTeam
from luxai2022.unit import Unit as LuxUnit
from luxai2022.unit import UnitType as LuxUnitType

from jux.actions import ActionQueue
from jux.config import EnvConfig, UnitConfig
from jux.map.position import Position
from jux.unit_cargo import ResourceType, UnitCargo


class UnitType(IntEnum):
    LIGHT = 0
    HEAVY = 1

    @classmethod
    def from_lux(cls, lux_unit_type: LuxUnitType) -> "UnitType":
        if lux_unit_type == LuxUnitType.LIGHT:
            return cls.LIGHT
        elif lux_unit_type == LuxUnitType.HEAVY:
            return cls.HEAVY
        else:
            raise ValueError(f"Unknown unit type {lux_unit_type}")

    def to_lux(self) -> LuxUnitType:
        if self == self.LIGHT:
            return LuxUnitType.LIGHT
        elif self == self.HEAVY:
            return LuxUnitType.HEAVY
        else:
            raise ValueError(f"Unknown unit type {self}")


class Unit(NamedTuple):
    unit_type: UnitType
    team_id: int
    # team # no need team object, team_id is enough
    unit_id: int
    pos: Position

    cargo: UnitCargo
    action_queue: ActionQueue  # int[UNIT_ACTION_QUEUE_SIZE, 5]
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
    def from_lux(cls, lux_unit: LuxUnit, env_cfg: EnvConfig) -> "Unit":
        unit_id = int(lux_unit.unit_id[len('unit_'):])
        return Unit(
            unit_type=UnitType.from_lux(lux_unit.unit_type),
            team_id=lux_unit.team_id,
            unit_id=unit_id,
            pos=Position.from_lux(lux_unit.pos),
            cargo=UnitCargo.from_lux(lux_unit.cargo),
            action_queue=ActionQueue.from_lux(lux_unit.action_queue, env_cfg.UNIT_ACTION_QUEUE_SIZE),
            unit_cfg=UnitConfig.from_lux(lux_unit.unit_cfg),
            power=lux_unit.power,
        )

    def to_lux(self, lux_teams: List[LuxTeam], lux_env_cfg: LuxEnvConfig) -> LuxUnit:
        lux_unit = LuxUnit(
            team=lux_teams[self.team_id],
            unit_type=self.unit_type.to_lux(),
            unit_id=f"unit_{self.unit_id}",
            env_cfg=lux_env_cfg,
        )
        lux_unit.pos = self.pos.to_lux()
        lux_unit.cargo = self.cargo.to_lux()
        lux_unit.power = int(self.power)
        lux_unit.action_queue = self.action_queue.to_lux()
        return lux_unit

    # def __eq__(self, __o: object) -> bool:
    #     return isinstance(__o, Unit) and self.unit_id == __o.unit_id and self.team_id == __o.team_id and self.unit_type == __o.unit_type

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
