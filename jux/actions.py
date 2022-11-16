from enum import IntEnum
from typing import NamedTuple

import jax.numpy as jnp
import luxai2022.actions as lux_actions
import numpy as np
from jax import Array
from luxai2022.actions import Action as LuxAction
from luxai2022.unit import UnitType as LuxUnitType

from jux.map.position import Direction
from jux.unit import ResourceType


class FactoryAction(IntEnum):
    BUILD_LIGHT = 0
    BUILD_HEAVY = 1
    WATER = 2

    @classmethod
    def from_lux(cls, lux_factory_action: LuxAction) -> "FactoryAction":
        return cls(lux_factory_action.state_dict())

    def to_lux(self) -> LuxAction:
        if self == self.BUILD_LIGHT:
            return lux_actions.FactoryBuildAction(unit_type=LuxUnitType.LIGHT)
        elif self == self.BUILD_HEAVY:
            return lux_actions.FactoryBuildAction(unit_type=LuxUnitType.HEAVY)
        elif self == self.WATER:
            return lux_actions.FactoryWaterAction()
        else:
            raise ValueError(f"Unknown factory action {self}")


class UnitActionType(IntEnum):
    MOVE = 0
    TRANSFER = 1
    PICKUP = 2
    DIG = 3
    SELF_DESTRUCT = 4
    RECHARGE = 5


class UnitAction(NamedTuple):
    code: Array  # int[5]

    @property
    def action_type(self):
        return self.code[..., 0]

    @property
    def direction(self):
        return self.code[..., 1]

    @property
    def resource(self):
        return self.code[..., 2]

    @property
    def amount(self):
        return self.code[..., 3]

    @property
    def repeat(self):
        return jnp.array(self.code[..., 4], jnp.bool_)

    @property
    def dist(self):
        return self.code[..., 2]

    @classmethod
    def move(cls, direction: Direction, repeat: bool = False) -> "UnitAction":
        return cls(jnp.array([UnitActionType.MOVE, direction, 1, 0, repeat], jnp.int32))

    @classmethod
    def transfer(cls, direction: Direction, resource: ResourceType, amount: int, repeat: bool = False) -> "UnitAction":
        return cls(jnp.array([UnitActionType.TRANSFER, direction, resource, amount, repeat], jnp.int32))

    @classmethod
    def pickup(cls, resource: ResourceType, amount: int, repeat: bool = False) -> "UnitAction":
        return cls(jnp.array([UnitActionType.PICKUP, 0, resource, amount, repeat], jnp.int32))

    @classmethod
    def dig(cls, repeat: bool = False) -> "UnitAction":
        return cls(jnp.array([UnitActionType.DIG, 0, 0, 0, repeat], jnp.int32))

    @classmethod
    def self_destruct(cls, repeat: bool = False) -> "UnitAction":
        return cls(jnp.array([UnitActionType.SELF_DESTRUCT, 0, 0, 0, repeat], jnp.int32))

    @classmethod
    def recharge(cls, amount, repeat: bool = False) -> "UnitAction":
        """Recharge the unit's battery until the given amount.

        Args:
            amount (int): the goal amount of battery.
            repeat (bool, optional): By rule, the recharge repeat automatically, so this arg does not matter. Defaults to False.

        Returns:
            UnitAction: a recharge action
        """
        return cls(jnp.array([UnitActionType.RECHARGE, 0, 0, amount, repeat], jnp.int32))

    def __repr__(self) -> str:
        if self.action_type == UnitActionType.MOVE:
            return f"move({Direction(int(self.direction))}, repeat={bool(self.repeat)})"
        elif self.action_type == UnitActionType.TRANSFER:
            return f"transfer({Direction(int(self.direction))}, {ResourceType(int(self.resource))}, {int(self.amount)}, repeat={bool(self.repeat)})"
        elif self.action_type == UnitActionType.PICKUP:
            return f"pick_up({ResourceType(int(self.resource))}, {int(self.amount)}, repeat={bool(self.repeat)})"
        elif self.action_type == UnitActionType.DIG:
            return f"dig(repeat={bool(self.repeat)})"
        elif self.action_type == UnitActionType.SELF_DESTRUCT:
            return f"self_destruct(repeat={bool(self.repeat)})"
        elif self.action_type == UnitActionType.RECHARGE:
            return f"recharge({int(self.amount)}, repeat={bool(self.repeat)})"
        else:
            raise ValueError(f"Unknown action type: {self.action_type}")

    @classmethod
    def from_lux(cls, lux_action: LuxAction) -> "UnitAction":
        code: np.ndarray = lux_action.state_dict()
        assert code.shape == (5, ), f"Invalid UnitAction action code: {code}"
        return cls(jnp.array(code, jnp.int32))

    def to_lux(self) -> LuxAction:
        return lux_actions.format_action_vec(np.array(self.code))

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, UnitAction) and jnp.array_equal(self.code, __o.code)
