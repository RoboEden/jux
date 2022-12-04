from enum import IntEnum
from typing import List, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import luxai2022.actions as lux_actions
import numpy as np
from jax import Array
from luxai2022.actions import Action as LuxAction
from luxai2022.unit import UnitType as LuxUnitType

from jux.config import EnvConfig, JuxBufferConfig
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
    def resource_type(self):
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
        n_actions = jnp.int32(len(actions))
        assert n_actions <= max_queue_size, f"{n_actions} actions is too much for ActionQueue size {max_queue_size}"
        data = jnp.array([UnitAction.from_lux(act).code for act in actions], jnp.int32).reshape(-1, 5)
        pad_size = max_queue_size - len(data)
        padding = jnp.zeros((pad_size, 5), np.int32)
        data = jnp.concatenate([data, padding], axis=-2)
        return cls(data, 0, n_actions, n_actions)

    def to_lux(self) -> List[LuxAction]:
        data = np.array(self._get_sorted_data())
        data = data[:self.count, :]
        return [UnitAction(code).to_lux() for code in data]

    def _get_sorted_data(self):
        '''Return data in the order of the queue. The first element is the front of the queue.'''
        idx = (jnp.arange(self.capacity) + self.front) % self.capacity
        data = self.data[idx, :]
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

    def __eq__(self, other: 'ActionQueue') -> bool:
        if not isinstance(other, ActionQueue):
            return False
        mask = jnp.arange(self.capacity) < self.count
        self_data = self._get_sorted_data()
        other_data = other._get_sorted_data()
        data_eq = (self_data == other_data).all(-1)  # bool[capacity]
        data_eq = (data_eq | ~mask).all(-1)
        return (self.count == other.count) & data_eq


class JuxAction(NamedTuple):
    factory_action: Array  # int[2, MAX_N_FACTORIES]
    unit_action_queue: UnitAction  # UnitAction[2, MAX_N_UNITS, UNIT_ACTION_QUEUE_SIZE]
    unit_action_queue_count: Array  # int[2, MAX_N_UNITS]
    unit_action_queue_update: Array  # bool[2, MAX_N_UNITS]

    @classmethod
    def empty(cls, env_cfg: EnvConfig, buf_cfg: JuxBufferConfig):
        return cls(
            factory_action=jnp.full((2, buf_cfg.MAX_N_FACTORIES), fill_value=FactoryAction.DO_NOTHING),
            unit_action_queue=UnitAction(code=jnp.zeros(
                (
                    2,
                    buf_cfg.MAX_N_UNITS,
                    env_cfg.UNIT_ACTION_QUEUE_SIZE,
                    5,
                ),
                dtype=jnp.int32,
            )),
            unit_action_queue_count=jnp.zeros((2, buf_cfg.MAX_N_UNITS), dtype=jnp.int32),
            unit_action_queue_update=jnp.zeros((2, buf_cfg.MAX_N_UNITS), dtype=jnp.bool_),
        )
