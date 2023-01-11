from enum import IntEnum
from typing import Dict, NamedTuple, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array, lax
from luxai_s2.config import EnvConfig as LuxEnvConfig
from luxai_s2.team import Team as LuxTeam
from luxai_s2.unit import Unit as LuxUnit
from luxai_s2.unit import UnitType as LuxUnitType

from jux.actions import ActionQueue, UnitAction
from jux.config import EnvConfig, UnitConfig
from jux.map.position import Position
from jux.unit_cargo import ResourceType, UnitCargo
from jux.utils import INT32_MAX, imax


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
    unit_type: UnitType  # int8
    action_queue: ActionQueue  # ActionQueue[UNIT_ACTION_QUEUE_SIZE, 5]
    team_id: jnp.int8 = imax(jnp.int8)
    unit_id: jnp.int16 = imax(jnp.int16)
    pos: Position = Position()

    cargo: UnitCargo = UnitCargo()
    power: jnp.int32 = jnp.int32(0)

    @staticmethod
    def id_dtype():
        return Unit.__annotations__['unit_id']

    def get_cfg(self, attr: str, unit_cfgs: Tuple[UnitConfig, UnitConfig]):
        attr_values = jnp.array([
            getattr(unit_cfgs[0], attr),
            getattr(unit_cfgs[1], attr),
        ])
        return attr_values[self.unit_type]

    @staticmethod
    def new(team_id: int, unit_type: Union[UnitType, int], unit_id: int, env_cfg: EnvConfig):
        unit = Unit(
            unit_type=jnp.int8(unit_type),
            team_id=Unit.__annotations__['team_id'](team_id),
            unit_id=Unit.__annotations__['unit_id'](unit_id),
            pos=Position(),
            cargo=UnitCargo(),
            action_queue=ActionQueue.empty(env_cfg.UNIT_ACTION_QUEUE_SIZE),
            power=None,
        )
        init_power = unit.get_cfg('INIT_POWER', env_cfg.ROBOTS)
        return unit._replace(power=Unit.__annotations__['power'](init_power))

    @classmethod
    def empty(cls, env_cfg: EnvConfig):
        return cls(
            unit_type=jnp.int8(UnitType.LIGHT),
            action_queue=ActionQueue.empty(env_cfg.UNIT_ACTION_QUEUE_SIZE),
        )

    @classmethod
    def from_lux(cls, lux_unit: LuxUnit, env_cfg: EnvConfig) -> "Unit":
        unit_id = int(lux_unit.unit_id[len('unit_'):])
        return Unit(
            unit_type=jnp.int8(UnitType.from_lux(lux_unit.unit_type)),
            team_id=Unit.__annotations__['team_id'](lux_unit.team_id),
            unit_id=Unit.__annotations__['unit_id'](unit_id),
            pos=Position.from_lux(lux_unit.pos),
            cargo=UnitCargo.from_lux(lux_unit.cargo),
            action_queue=ActionQueue.from_lux(lux_unit.action_queue, env_cfg.UNIT_ACTION_QUEUE_SIZE),
            power=Unit.__annotations__['power'](lux_unit.power),
        )

    def to_lux(self, lux_teams: Dict[str, LuxTeam], lux_env_cfg: LuxEnvConfig) -> LuxUnit:
        lux_unit = LuxUnit(
            team=lux_teams[f'player_{int(self.team_id)}'],
            unit_type=UnitType(self.unit_type).to_lux(),
            unit_id=f"unit_{int(self.unit_id)}",
            env_cfg=lux_env_cfg,
        )
        lux_unit.pos = self.pos.to_lux()
        lux_unit.cargo = self.cargo.to_lux()
        lux_unit.power = int(self.power)
        lux_unit.action_queue = self.action_queue.to_lux()
        return lux_unit

    def next_action(self) -> UnitAction:
        act = self.action_queue.peek()
        act = jax.lax.cond(
            self.action_queue.is_empty(),
            lambda: UnitAction.do_nothing(),  # empty action
            lambda: act,
        )
        return act

    def repeat_action(self, success: bool) -> 'Unit':
        '''
        Currently, invalid actions in luxai2021 are not executed and also not
        removed from the action queue. So, wee need an indicator 'success' to
        indicate whether the action is executed successfully. Only when the
        action is executed successfully, we can pop/repeat it.

        Args:
            success (bool[2, U]): whether the action is executed successfully
        Returns:
            Unit: the unit with updated action queue
        '''

        act = self.action_queue.peek()
        not_empty = ~self.action_queue.is_empty()
        n_minus_one = (act.n > 1) & success & not_empty
        pop_only = (act.n <= 1) & ~act.repeat & success & not_empty
        pop_and_push_back = (act.n <= 1) & act.repeat & success & not_empty

        data = jax.tree_map(
            lambda queue, a: queue.at[self.action_queue.rear].set(a),
            self.action_queue.data,
            act._replace(n=jnp.where(pop_and_push_back, 1, act.n)),
        )
        n = data.n.at[self.action_queue.front].add(-n_minus_one.astype(data.n.dtype))
        data = data._replace(n=n)

        front = jnp.where(
            pop_only | pop_and_push_back,
            (self.action_queue.front + 1) % self.action_queue.capacity,
            self.action_queue.front,
        )

        rear = jnp.where(
            pop_and_push_back,
            (self.action_queue.rear + 1) % self.action_queue.capacity,
            self.action_queue.rear,
        )

        count = jnp.where(
            pop_only,
            self.action_queue.count - 1,
            self.action_queue.count,
        )

        action_queue = self.action_queue._replace(
            data=data,
            front=front,
            rear=rear,
            count=count,
        )
        return self._replace(action_queue=action_queue)

    def is_heavy(self) -> Union[bool, Array]:
        return self.unit_type == UnitType.HEAVY

    def move_power_cost(self, rubble_at_target: int, unit_cfgs: Tuple[UnitConfig, UnitConfig]):
        move_cost = self.get_cfg("MOVE_COST", unit_cfgs)
        rubble_movement_cost = self.get_cfg("RUBBLE_MOVEMENT_COST", unit_cfgs)
        return move_cost + jnp.floor(rubble_movement_cost * rubble_at_target).astype(move_cost.dtype)

    def add_resource(
        self,
        resource: ResourceType,
        amount: int,
        unit_cfgs: Tuple[UnitConfig, UnitConfig],
    ) -> Tuple['Unit', Union[int, Array]]:
        # If resource != ResourceType.power, call UnitCargo.add_resource.
        # else, call Unit.add_power.
        amount = jnp.maximum(amount, 0)
        cargo_space = self.get_cfg("CARGO_SPACE", unit_cfgs)
        battery_capacity = self.get_cfg("BATTERY_CAPACITY", unit_cfgs)

        def add_power(self, resource: ResourceType, amount: int):
            transfer_amount = jnp.minimum(battery_capacity - self.power, amount)
            new_unit = self._replace(power=self.power + transfer_amount)
            return new_unit, transfer_amount

        def add_others(self: Unit, resource: ResourceType, amount: int):
            new_cargo, transfer_amount = self.cargo.add_resource(
                resource=resource,
                amount=amount,
                cargo_space=cargo_space,
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

    def sub_resource(self, resource: ResourceType, amount: int) -> Tuple['Unit', Union[int, Array]]:
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

    def gain_power(self, unit_cfgs: Tuple[UnitConfig, UnitConfig]):
        charge = self.get_cfg("CHARGE", unit_cfgs)
        battery_capacity = self.get_cfg("BATTERY_CAPACITY", unit_cfgs)
        new_power = self.power + charge
        new_power = jnp.minimum(new_power, battery_capacity)
        return self._replace(power=new_power)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Unit):
            return False
        eq = True
        eq = eq & (self.unit_id == other.unit_id)
        eq = eq & (self.unit_type == other.unit_type)
        eq = eq & (self.action_queue == other.action_queue)
        eq = eq & (self.team_id == other.team_id)
        eq = eq & (self.unit_id == other.unit_id)
        eq = eq & (self.pos == other.pos)
        eq = eq & (self.cargo == other.cargo)
        eq = eq & (self.power == other.power)
        return eq
