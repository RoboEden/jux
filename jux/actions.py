from enum import IntEnum
from functools import reduce
from typing import Any, Dict, List, NamedTuple, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import luxai_s2.actions as lux_actions
import numpy as np
from jax import Array
from luxai_s2.actions import Action as LuxAction
from luxai_s2.actions import format_action_vec
from luxai_s2.unit import UnitType as LuxUnitType

import jux.torch
import jux.tree_util
from jux.config import EnvConfig, JuxBufferConfig
from jux.map.position import Direction, Position
from jux.team import FactionTypes
from jux.unit_cargo import ResourceType, UnitCargo
from jux.utils import INT16_MAX

try:
    import torch
except ImportError:
    pass


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
    DO_NOTHING = -1
    MOVE = 0
    TRANSFER = 1
    PICKUP = 2
    DIG = 3
    SELF_DESTRUCT = 4
    RECHARGE = 5


class UnitAction(NamedTuple):
    action_type: jnp.int8 = jnp.int8(UnitActionType.DO_NOTHING)
    direction: jnp.int8 = jnp.int8(0)
    resource_type: jnp.int8 = jnp.int8(0)
    amount: jnp.int16 = jnp.int16(0)
    repeat: jnp.int16 = jnp.int16(0)
    n: jnp.int16 = jnp.int16(0)

    @classmethod
    def new(
        cls,
        action_type: UnitActionType,
        direction: Union[Direction, int] = 0,
        resource_type: Union[ResourceType, int] = 0,
        amount: jnp.int16 = 0,
        repeat: jnp.int16 = 0,
        n: jnp.int16 = 0,
    ) -> "UnitAction":
        return UnitAction(
            jnp.int8(action_type),
            jnp.int8(direction),
            jnp.int8(resource_type),
            jnp.int16(amount),
            jnp.int16(repeat),
            jnp.int16(n),
        )

    @classmethod
    def move(cls, direction: Direction, repeat: int = 0, n: int = 1) -> "UnitAction":
        return cls.new(UnitActionType.MOVE, direction=direction, repeat=repeat, n=n)

    @classmethod
    def transfer(cls,
                 direction: Direction,
                 resource_type: ResourceType,
                 amount: int,
                 repeat: int = 0,
                 n: int = 1) -> "UnitAction":
        return cls.new(UnitActionType.TRANSFER,
                       direction,
                       resource_type=resource_type,
                       amount=amount,
                       repeat=repeat,
                       n=n)

    @classmethod
    def pickup(cls, resource_type: ResourceType, amount: int, repeat: int = 0, n: int = 1) -> "UnitAction":
        return cls.new(UnitActionType.PICKUP, resource_type=resource_type, amount=amount, repeat=repeat, n=n)

    @classmethod
    def dig(cls, repeat: int = 0, n: int = 1) -> "UnitAction":
        return cls.new(UnitActionType.DIG, repeat=repeat, n=n)

    @classmethod
    def self_destruct(cls, repeat: int = 0, n: int = 1) -> "UnitAction":
        return cls.new(UnitActionType.SELF_DESTRUCT, repeat=repeat, n=n)

    @classmethod
    def recharge(cls, amount: int, repeat: int = 0, n: int = 1) -> "UnitAction":
        """Recharge the unit's battery until the given amount.

        Args:
            amount (int): the goal amount of battery.
            repeat (bool, optional): By rule, the recharge repeat automatically, so this arg does not matter. Defaults to False.

        Returns:
            UnitAction: a recharge action
        """
        return cls.new(UnitActionType.RECHARGE, amount=amount, repeat=repeat, n=n)

    @classmethod
    def do_nothing(cls) -> "UnitAction":
        return cls.new(UnitActionType.DO_NOTHING)

    @classmethod
    def from_lux(cls, lux_action: LuxAction) -> "UnitAction":
        code: np.ndarray = lux_action.state_dict()
        assert code.shape == (6, ), f"Invalid UnitAction action code: {code}"
        return cls.new(*code)

    def to_lux(self) -> LuxAction:
        return lux_actions.format_action_vec(
            np.array([
                int(self.action_type),
                int(self.direction),
                int(self.resource_type),
                int(self.amount),
                int(self.repeat),
                int(self.n),
            ]))

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, UnitAction) and \
            (self.action_type == __o.action_type) and \
            (self.direction == __o.direction) and \
            (self.resource_type == __o.resource_type) and \
            (self.amount == __o.amount) and \
            (self.repeat == __o.repeat) and \
            (self.n == __o.n)

    def is_valid(self, max_transfer_amount) -> bool:
        return (
            (0 <= self.action_type) & (self.action_type <= len(UnitActionType) - 1) & \
            (0 <= self.direction) & (self.direction <= len(Direction) - 1) & \
            (0 <= self.resource_type) & (self.resource_type <= len(ResourceType) - 1) & \
            (0 <= self.amount) & (self.amount <= max_transfer_amount) & \
            (0 <= self.n) & (self.n <= INT16_MAX) & \
            (0 <= self.repeat) & (self.repeat <= INT16_MAX) \
        )


class ActionQueue(NamedTuple):
    data: UnitAction  # UnitAction[UNIT_ACTION_QUEUE_SIZE]
    front: jnp.int8 = jnp.int8(0)
    rear: jnp.int8 = jnp.int8(0)
    count: jnp.int8 = jnp.int8(0)

    @staticmethod
    def empty(capacity: int) -> "ActionQueue":
        data = jax.tree_map(lambda x: x[None].repeat(capacity), UnitAction.do_nothing())
        return ActionQueue(data)

    @classmethod
    def from_lux(cls, actions: List[LuxAction], max_queue_size: int) -> "ActionQueue":
        n_actions = len(actions)
        assert n_actions <= max_queue_size, f"{n_actions} actions is too much for ActionQueue size {max_queue_size}"
        n_actions = jnp.int8(n_actions)
        if n_actions == 0:
            return cls.empty(max_queue_size)
        data = jux.tree_util.batch_into_leaf([UnitAction.from_lux(act) for act in actions])
        pad_size = max_queue_size - n_actions
        if pad_size > 0:
            padding = jax.tree_map(lambda x: x[None].repeat(pad_size), UnitAction.do_nothing())
            data = jux.tree_util.concat_in_leaf([data, padding], axis=-1)
        chex.assert_shape(data[0], (max_queue_size, ))
        return cls(data, jnp.int8(0), n_actions, n_actions)

    def to_lux(self) -> List[LuxAction]:
        data = np.array(self._get_sorted_data())
        data = data[:, :self.count].T
        return [format_action_vec(code) for code in data]

    def _get_sorted_data(self) -> UnitAction:
        '''Return data in the order of the queue. The first element is the front of the queue.'''
        idx = (jnp.arange(self.capacity) + self.front) % self.capacity
        data = jax.tree_map(lambda x: x[idx], self.data)
        return data

    @property
    def capacity(self) -> int:
        return self.data[0].shape[-1]

    def push_back(self, action: UnitAction) -> "ActionQueue":
        """Push an action into the back of the queue. There is no way to thrown an error in jitted function. It is user's responsibility to check if the queue is full.

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
            data = jax.tree_map(lambda data, v: data.at[..., self.rear].set(v), self.data, action)
            rear = (self.rear + 1) % self.capacity
            count = self.count + 1
            return ActionQueue(data, self.front, rear, count)

        return jax.lax.cond(
            self.is_full(),
            lambda _: self,
            _push,
            action,
        )

    def push_front(self, action: UnitAction) -> "ActionQueue":
        """Push an action into the front of the queue. There is no way to thrown an error in jitted function. It is user's responsibility to check if the queue is full.

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
            front = (self.front - 1 + self.capacity) % self.capacity
            data = jax.tree_map(lambda data, v: data.at[..., front].set(v), self.data, action)
            count = self.count + 1
            return ActionQueue(data, front, self.rear, count)

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
        return jax.tree_map(lambda x: x[..., self.front], self.data)

    def clear(self) -> "ActionQueue":
        return ActionQueue(self.data, jnp.int8(0), jnp.int8(0), jnp.int8(0))

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
        data_eq = [a == b for a, b in zip(self_data, other_data)]
        data_eq = reduce(lambda x, y: x & y, data_eq)  # and all attributes together
        data_eq = (data_eq | ~mask).all(-1)
        return (self.count == other.count) & data_eq


class JuxAction(NamedTuple):
    factory_action: Array  # int8[2, MAX_N_FACTORIES]
    unit_action_queue: UnitAction  # UnitAction[2, MAX_N_UNITS, UNIT_ACTION_QUEUE_SIZE]
    unit_action_queue_count: Array  # int8[2, MAX_N_UNITS]
    unit_action_queue_update: Array  # bool[2, MAX_N_UNITS]

    @classmethod
    def empty(cls, env_cfg: EnvConfig, buf_cfg: JuxBufferConfig):
        batch_shape = (
            2,
            buf_cfg.MAX_N_UNITS,
            env_cfg.UNIT_ACTION_QUEUE_SIZE,
        )
        unit_action_queue = jax.tree_map(
            lambda x: x[None].repeat(np.prod(batch_shape)).reshape(batch_shape),
            UnitAction.do_nothing(),
        )
        return cls(
            factory_action=jnp.full(
                (2, buf_cfg.MAX_N_FACTORIES),
                fill_value=jnp.int8(FactoryAction.DO_NOTHING),
            ),
            unit_action_queue=unit_action_queue,
            unit_action_queue_count=jnp.zeros(
                (2, buf_cfg.MAX_N_UNITS),
                dtype=ActionQueue.__annotations__['count'],
            ),
            unit_action_queue_update=jnp.zeros((2, buf_cfg.MAX_N_UNITS), dtype=jnp.bool_),
        )

    @staticmethod
    def from_lux(state, lux_action: Dict[str, Dict[str, Union[int, Array]]]) -> "JuxAction":
        factory_action = np.full(
            (2, state.MAX_N_FACTORIES),
            fill_value=np.int8(FactoryAction.DO_NOTHING.value),
        )
        batch_shape = (2, state.MAX_N_UNITS, state.env_cfg.UNIT_ACTION_QUEUE_SIZE)
        unit_action_queue = UnitAction(
            action_type=np.empty(batch_shape, dtype=UnitAction.__annotations__['action_type'].dtype),
            direction=np.empty(batch_shape, dtype=UnitAction.__annotations__['direction'].dtype),
            resource_type=np.empty(batch_shape, dtype=UnitAction.__annotations__['resource_type'].dtype),
            amount=np.empty(batch_shape, dtype=UnitAction.__annotations__['amount'].dtype),
            repeat=np.empty(batch_shape, dtype=UnitAction.__annotations__['repeat'].dtype),
            n=np.empty(batch_shape, dtype=UnitAction.__annotations__['n'].dtype),
        )
        unit_action_queue_count = np.zeros((2, state.MAX_N_UNITS), dtype=ActionQueue.__annotations__['count'])
        unit_action_queue_update = np.zeros((2, state.MAX_N_UNITS), dtype=np.bool_)

        for player_id, player_actions in lux_action.items():
            player_id = int(player_id.split('_')[-1])
            for unit_id, action in player_actions.items():
                if unit_id.startswith('factory_'):
                    unit_id = int(unit_id.split('_')[-1])
                    pid, idx = state.factory_id2idx[unit_id]
                    assert pid == player_id
                    assert 0 <= idx < state.n_factories[player_id]
                    factory_action[player_id, idx] = action
                elif unit_id.startswith('unit_'):
                    unit_id = int(unit_id.split('_')[-1])
                    pid, idx = state.unit_id2idx[unit_id]
                    assert pid == player_id
                    assert 0 <= idx < state.n_units[player_id]

                    queue_size = len(action)

                    if queue_size > 0:
                        action = np.array(action)
                        for i in range(len(UnitAction._fields)):
                            unit_action_queue[i][player_id, idx, :queue_size] = action[:, i]
                            
                    unit_action_queue_count[player_id, idx] = queue_size
                    unit_action_queue_update[player_id, idx] = True
                else:
                    raise ValueError(f'Unknown unit_id: {unit_id}')

        # handle a deprecated features in LuxAI2021
        # In old version, the 2rd dimension of action serves as `dist` for moving action.
        unit_action_queue.resource_type[unit_action_queue.action_type == UnitActionType.MOVE] = 0

        factory_action = jnp.array(factory_action)
        unit_action_queue = jax.tree_map(jnp.array, unit_action_queue)
        unit_action_queue_count = jnp.array(unit_action_queue_count)
        unit_action_queue_update = jnp.array(unit_action_queue_update)

        return JuxAction(
            factory_action,
            unit_action_queue,
            unit_action_queue_count,
            unit_action_queue_update,
        )

    def to_lux(self: "JuxAction", state) -> Dict[str, Dict[str, Union[int, Array]]]:
        """Convert `JuxAction` to dict format that can be passed into `LuxAI_S2` object.

        Args:
            self (JuxAction): self
            state (State): the current game state.

        Returns:
            Dict[str, Dict[str, Union[int, Array]]: A action dict like:
            ```
            {
                'player_0': {
                    'factory_0': 1,
                    'factory_2': 2,
                    'unit_7': [
                        [0, 0, 0, 0, 0],
                        [0, 3, 0, 0, 0],
                    ],
                    'unit_8': [
                        [3, 0, 0, 0, 0],
                        [5, 0, 0, 9, 0],
                    ],
                    ...
                },
                'player_1': {
                    ...
                },
            }
            ```
        """
        lux_action = {
            'player_0': {},
            'player_1': {},
        }
        self = jax.tree_map(lambda x: np.array(x), self)

        # factory action
        for p in range(2):
            n_factories = int(state.n_factories[p])
            for i in range(n_factories):
                id, act = int(state.factories.unit_id[p, i]), int(self.factory_action[p, i])
                if act != FactoryAction.DO_NOTHING:
                    lux_action[f'player_{p}'][f'factory_{id}'] = act

        # player action
        for p in range(2):
            n_units = int(state.n_units[p])
            for i in range(n_units):
                if self.unit_action_queue_update[p, i]:
                    id = int(state.units.unit_id[p, i])
                    queue = np.array([
                        self.unit_action_queue[a][p, i, :self.unit_action_queue_count[p, i]]
                        for a in range(len(UnitAction._fields))
                    ])
                    queue = queue.T.tolist()
                    lux_action[f'player_{p}'][f'unit_{id}'] = queue
        return lux_action

    @staticmethod
    def from_torch(
            factory_action: 'torch.Tensor',  # int8[2, MAX_N_FACTORIES]
            unit_action_queue: UnitAction,  # UnitAction[2, MAX_N_UNITS, UNIT_ACTION_QUEUE_SIZE]
            unit_action_queue_count: 'torch.Tensor',  # int8[2, MAX_N_UNITS]
            unit_action_queue_update: 'torch.Tensor',  # bool[2, MAX_N_UNITS]
            # env_cfg: EnvConfig,
            # buf_cfg: JuxBufferConfig,
    ):
        assert factory_action.dtype == torch.int8

        for i, attr in enumerate(UnitAction._fields):
            assert unit_action_queue[i].dtype == getattr(torch, UnitAction.__annotations__[attr].dtype.name), \
                f"unit_action_queue.{attr}.dtype must be {getattr(torch, UnitAction.__annotations__[attr].dtype.name)}, but got {unit_action_queue[i].dtype}."

        assert unit_action_queue_count.dtype == getattr(torch, ActionQueue.__annotations__['count'].dtype.name), \
            f"unit_action_queue_count.dtype must be {getattr(torch, ActionQueue.__annotations__['count'].dtype.name)}, but got {unit_action_queue_count.dtype}."
        assert unit_action_queue_update.dtype == torch.uint8

        chex.assert_equal_shape(unit_action_queue)

        queue_size = unit_action_queue[0].shape[-1]
        max_n_units = unit_action_queue_count.shape[-1]
        max_n_factories = factory_action.shape[-1]
        # assert queue_size == env_cfg.UNIT_ACTION_QUEUE_SIZE
        # assert max_n_units == buf_cfg.MAX_N_UNITS
        # assert max_n_factories == buf_cfg.MAX_N_FACTORIES

        assert factory_action.shape[-2:] == (2, max_n_factories)
        assert unit_action_queue_count.shape[-2:] == (2, max_n_units)
        assert unit_action_queue_update.shape[-2:] == (2, max_n_units)

        env_batch_shape = factory_action.shape[:-2]
        assert factory_action.shape[:-2] == env_batch_shape
        assert unit_action_queue[0].shape[:-3] == env_batch_shape
        assert unit_action_queue_count.shape[:-2] == env_batch_shape
        assert unit_action_queue_update.shape[:-2] == env_batch_shape

        assert (unit_action_queue_count[unit_action_queue_update == False] == 0).all()
        assert (unit_action_queue_count[unit_action_queue_update == True] <= queue_size).all()

        factory_action = jux.torch.from_torch(factory_action)
        unit_action_queue = jax.tree_map(jux.torch.from_torch, unit_action_queue)
        unit_action_queue_count = jux.torch.from_torch(unit_action_queue_count)
        unit_action_queue_update = jux.torch.from_torch(unit_action_queue_update)
        unit_action_queue_update = unit_action_queue_update.astype(jnp.bool_)

        return JuxAction(
            factory_action=factory_action,
            unit_action_queue=unit_action_queue,
            unit_action_queue_count=unit_action_queue_count,
            unit_action_queue_update=unit_action_queue_update,
        )

    def to_torch(self) -> 'JuxAction':
        """Convert leaves of `JuxAction` to `torch.Tensor`.

        Returns:
            JuxAction: a JuxAction object with leaves converted to `torch.Tensor`.
        """
        return jax.tree_map(jux.torch.to_torch, self)


def bid_action_from_lux(lux_bid_action: Dict[str, Dict[str, Any]]) -> Tuple[Array, Array]:
    '''
    Convert a `LuxAI_S2` bid action to a format that `JuxEnv.step_bid()` can receive.

    Args:
        lux_bid_action (Dict[str, Dict[str, int]]): a bid action from `LuxAI_S2`. In format of:
        ```
        {
            'player_0': {
                'bid': int,
                'faction': str,
            },
            'player_1': {
                'bid': int,
                'faction': str,
            },
        }

    Returns:
        bid (Array): int[2], bid amount for each player.
        faction (Array): int[2], faction for each player.
    '''
    bid = jnp.array(
        [
            lux_bid_action['player_0']['bid'],
            lux_bid_action['player_1']['bid'],
        ],
        dtype=UnitCargo.dtype(),
    )
    faction = jnp.array(
        [
            FactionTypes[lux_bid_action['player_0']['faction']],
            FactionTypes[lux_bid_action['player_1']['faction']],
        ],
        dtype=jnp.int8,
    )
    return bid, faction


def bid_action_to_lux(bid: Array, faction: Array) -> Dict[str, Dict[str, int]]:
    '''
    Convert a `JuxEnv.step_bid()` action to a format that `LuxAI_S2` can receive.

    Args:
        bid (Array): int[2], bid amount for each player.
        faction (Array): int[2], faction for each player.

    Returns:
        Dict[str, Dict[str, int]]: a `LuxAI_S2` bid action. In format of:
        ```
        {
            'player_0': {
                'bid': int,
                'faction': str,
            },
            'player_1': {
                'bid': int,
                'faction': str,
            },
        }
        ```
    '''
    return {
        'player_0': {
            'bid': int(bid[0]),
            'faction': FactionTypes(faction[0]).name,
        },
        'player_1': {
            'bid': int(bid[1]),
            'faction': FactionTypes(faction[1]).name,
        },
    }


def factory_placement_action_from_lux(lux_act: Dict[str, Dict[str, Any]]) -> Tuple[Array, Array, Array]:
    '''Convert a `LuxAI_S2` factory placement action to a format that `JuxEnv.step_factory_placement()` can receive.
    See `JuxEnv.step_factory_placement()` for more details.

    Args:
        lux_act (Dict[str, Dict[str, Any]]): a `LuxAI_S2` factory placement action. In format of:
        ```
        {
            'player_0': {
                'spawn': [int, int],
                'water': int,
                'metal': int,
            },
            'player_1': {},
        }
        ```

    Returns:
        spawn (Array): int[2, 2], the spawn position.
        water (Array): int[2], The initial water amount of the factory.
        metal (Array): int[2], The initial metal amount of the factory.
    '''
    spawn = jnp.array(
        [
            lux_act['player_0']['spawn'] if lux_act['player_0'] else [0, 0],
            lux_act['player_1']['spawn'] if lux_act['player_1'] else [0, 0],
        ],
        dtype=Position.__annotations__["pos"],
    )
    water = jnp.array(
        [
            lux_act['player_0']['water'] if lux_act['player_0'] else 0,
            lux_act['player_1']['water'] if lux_act['player_1'] else 0,
        ],
        dtype=UnitCargo.dtype(),
    )
    metal = jnp.array(
        [
            lux_act['player_0']['metal'] if lux_act['player_0'] else 0,
            lux_act['player_1']['metal'] if lux_act['player_1'] else 0,
        ],
        dtype=UnitCargo.dtype(),
    )
    return spawn, water, metal


def factory_placement_action_to_lux(spawn, water, metal) -> Dict[str, Dict[str, Any]]:
    """Convert factory placement action to LuxAI_S2's action format. For more details about
    input arguments, see `JuxEnv.step_factory_placement()`.

    Args:
        spawn (Array): int[2, 2]. The spawn location of the factory.
        water (Array): int[2]. The initial water amount of the factory.
        metal (Array): int[2]. The initial metal amount of the factory.

    Returns:
        Dict[str, Dict[str, Any]]: a dict in form of

        ```
        {
            'player_0': {
                'spawn': [int, int],
                'water': int,
                'metal': int,
            },
            'player_1': {
                'spawn': [int, int],
                'water': int,
                'metal': int,
            },
        }
        ```
    """
    lux_act = {
        'player_0': {
            'spawn': spawn[0].tolist(),
            'water': water[0].tolist(),
            'metal': metal[0].tolist(),
        },
        'player_1': {
            'spawn': spawn[1].tolist(),
            'water': water[1].tolist(),
            'metal': metal[1].tolist(),
        },
    }
    return lux_act
