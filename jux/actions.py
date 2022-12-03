from enum import IntEnum
from typing import List, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import luxai2022.actions as lux_actions
import numpy as np
from jax import Array
from luxai2022.actions import Action as LuxAction
from luxai2022.unit import UnitType as LuxUnitType

from jux.config import JuxBufferConfig
from jux.map.position import Direction
from jux.unit_cargo import ResourceType


class FactoryAction(IntEnum):
    DO_NOTHING = -1
    BUILD_LIGHT = 0
    BUILD_HEAVY = 1
    WATER = 2

    @classmethod
    def from_lux(cls, lux_factory_action: LuxAction) -> "FactoryAction":
        return cls(lux_factory_action.state_dict())

    def to_lux(self) -> LuxAction:
        if self == self.DO_NOTHING:
            raise ValueError(f"Cannot convert {self} to LuxAction")
        elif self == self.BUILD_LIGHT:
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
    code: Array = jnp.zeros(5, jnp.int32)  # int[5]

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
        if self.code.shape == (5, ):
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
        else:
            return f"UnitAction({self.code})"

    @classmethod
    def from_lux(cls, lux_action: LuxAction) -> "UnitAction":
        code: np.ndarray = lux_action.state_dict()
        assert code.shape == (5, ), f"Invalid UnitAction action code: {code}"
        return cls(jnp.array(code, jnp.int32))

    def to_lux(self) -> LuxAction:
        return lux_actions.format_action_vec(np.array(self.code))

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, UnitAction) and jnp.array_equal(self.code, __o.code)

    def is_valid(self, max_transfer_amount) -> bool:
        min = jnp.array([0, 0, 0, 0, False])
        max = jnp.array([
            len(UnitActionType) - 1,
            len(Direction) - 1,
            len(ResourceType) - 1,
            max_transfer_amount,
            True,
        ])
        return ((self.code >= min) & (self.code <= max)).all(-1)


class ActionQueue(NamedTuple):
    data: Array  # int[UNIT_ACTION_QUEUE_SIZE, 5]
    front: int = 0
    rear: int = 0
    count: int = 0

    @staticmethod
    def empty(capacity: int) -> "ActionQueue":
        return ActionQueue(jnp.zeros((capacity, 5), jnp.int32))

    @classmethod
    def from_lux(cls, actions: List[LuxAction], max_queue_size: int) -> "ActionQueue":
        n_actions = len(actions)
        assert n_actions <= max_queue_size, f"{n_actions} actions is too much for ActionQueue size {max_queue_size}"
        data = jnp.array([UnitAction.from_lux(act).code for act in actions], jnp.int32).reshape(-1, 5)
        pad_size = max_queue_size - len(data)
        padding = jnp.zeros((pad_size, 5), np.int32)
        data = jnp.concatenate([data, padding], axis=-2)
        return cls(data, 0, n_actions, n_actions)

    def to_lux(self) -> List[LuxAction]:
        data = np.array(self._get_sorted_data())
        return [UnitAction(code).to_lux() for code in data]

    def _get_sorted_data(self):
        '''Return data in the order of the queue. The first element is the front of the queue.'''
        if self.rear > self.front or self.is_empty():
            data = self.data[self.front:self.rear, :]
        else:
            data = jnp.concatenate([self.data[self.front:, :], self.data[:self.rear, :]], axis=-2)
        return data

    @property
    def capacity(self):
        return self.data.shape[-2]

    def push(self, action: UnitAction) -> "ActionQueue":
        """Push an action into the queue. There is no way to thrown an error in jitted function. It is user's responsibility to check if the queue is full.

        Args:
            action (UnitAction): action to push into.

        Returns:
            ActionQueue: Updated queue.
        """

        # if self.is_full():
        #     raise ValueError("ActionQueue is full")

        # There is no way to thrown an error in jitted function.
        # It is user's responsibility to check if the queue is full.
        # TODO: design an error code?
        def _push(action: UnitAction):
            data = self.data.at[..., self.rear, :].set(action.code)
            rear = (self.rear + 1) % self.capacity
            count = self.count + 1
            return ActionQueue(data, self.front, rear, count)

        return jax.lax.cond(
            self.is_full(),
            lambda _: self,
            _push,
            action,
        )

    def pop(self) -> Tuple[UnitAction, "ActionQueue"]:
        return jax.lax.cond(
            self.is_empty(),
            # if empty, return empty action and self.
            lambda self: (UnitAction(), self),
            # else, return the front action and updated queue.
            lambda self: (
                self.peek(),
                ActionQueue(
                    data=self.data,
                    front=(self.front + 1) % self.capacity,
                    rear=self.rear,
                    count=self.count - 1,
                ),
            ),
            self,
        )

    def peek(self) -> UnitAction:
        '''Return the front of the queue. There is no way to thrown an error in jitted function. It is user's responsibility to check if the queue is empty.

        Returns:
            ActionQueue: Updated queue.
        '''
        # if self.is_empty():
        #     raise ValueError("ActionQueue is empty")

        # There is no way to thrown an error in jitted function.
        # It is user's responsibility to check if the queue is empty.
        # TODO: design an error code?

        return UnitAction(self.data[..., self.front, :])

    def clear(self) -> "ActionQueue":
        return ActionQueue(self.data, 0, 0, 0)

    def is_full(self) -> bool:
        return self.count == self.capacity

    def is_empty(self) -> bool:
        return self.count == 0

    def __eq__(self, __o: 'ActionQueue') -> bool:
        if not isinstance(__o, ActionQueue):
            return False
        return (self.count == __o.count) and jnp.array_equal(self._get_sorted_data(), __o._get_sorted_data())
