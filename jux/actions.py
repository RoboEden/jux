from enum import IntEnum
from typing import Any, Dict, List, NamedTuple, Tuple, Union

import jax
import jax.numpy as jnp
import luxai2022.actions as lux_actions
import numpy as np
from jax import Array
from luxai2022.actions import Action as LuxAction
from luxai2022.unit import UnitType as LuxUnitType

import jux.torch
from jux.config import EnvConfig, JuxBufferConfig
from jux.map.position import Direction
from jux.team import FactionTypes
from jux.unit_cargo import ResourceType

try:
    import torch
except ImportError:
    pass

INT32_MAX = jnp.iinfo(jnp.int32).max


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
        return self.code[..., 4]

    @property
    def dist(self):
        return self.code[..., 2]

    @classmethod
    def move(cls, direction: Direction, repeat: int = 0) -> "UnitAction":
        return cls(jnp.array([UnitActionType.MOVE, direction, 0, 0, repeat], jnp.int32))

    @classmethod
    def transfer(cls, direction: Direction, resource: ResourceType, amount: int, repeat: int = 0) -> "UnitAction":
        return cls(jnp.array([UnitActionType.TRANSFER, direction, resource, amount, repeat], jnp.int32))

    @classmethod
    def pickup(cls, resource: ResourceType, amount: int, repeat: int = 0) -> "UnitAction":
        return cls(jnp.array([UnitActionType.PICKUP, 0, resource, amount, repeat], jnp.int32))

    @classmethod
    def dig(cls, repeat: int = 0) -> "UnitAction":
        return cls(jnp.array([UnitActionType.DIG, 0, 0, 0, repeat], jnp.int32))

    @classmethod
    def self_destruct(cls, repeat: int = 0) -> "UnitAction":
        return cls(jnp.array([UnitActionType.SELF_DESTRUCT, 0, 0, 0, repeat], jnp.int32))

    @classmethod
    def recharge(cls, amount: int, repeat: int = 0) -> "UnitAction":
        """Recharge the unit's battery until the given amount.

        Args:
            amount (int): the goal amount of battery.
            repeat (bool, optional): By rule, the recharge repeat automatically, so this arg does not matter. Defaults to False.

        Returns:
            UnitAction: a recharge action
        """
        return cls(jnp.array([UnitActionType.RECHARGE, 0, 0, amount, repeat], jnp.int32))

    @classmethod
    def do_nothing(cls) -> "UnitAction":
        return cls(jnp.array([UnitActionType.DO_NOTHING, 0, 0, 0, 0], jnp.int32))

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
        min = jnp.array([0, 0, 0, 0, -1])
        max = jnp.array([
            len(UnitActionType) - 1,
            len(Direction) - 1,
            len(ResourceType) - 1,
            max_transfer_amount,
            INT32_MAX,
        ])
        return ((self.code >= min) & (self.code <= max)).all(-1)


class ActionQueue(NamedTuple):
    data: Array  # int[UNIT_ACTION_QUEUE_SIZE, 5]
    front: int = jnp.int32(0)
    rear: int = jnp.int32(0)
    count: int = jnp.int32(0)

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
            data = self.data.at[..., front, :].set(action.code)
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

    @staticmethod
    def from_lux(state, lux_action: Dict[str, Dict[str, Union[int, Array]]]) -> "JuxAction":
        return state.parse_actions_from_dict(lux_action)

    def to_lux(self: "JuxAction", state) -> Dict[str, Dict[str, Union[int, Array]]]:
        """Convert `JuxAction` to dict format that can be passed into `LuxAI2022` object.

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
                    queue = self.unit_action_queue.code[p, i, :self.unit_action_queue_count[p, i]].tolist()
                    lux_action[f'player_{p}'][f'unit_{id}'] = queue
        return lux_action

    @staticmethod
    def from_torch(
            factory_action: 'torch.Tensor',  # int[2, MAX_N_FACTORIES]
            unit_action_queue: 'torch.Tensor',  # int[2, MAX_N_UNITS, UNIT_ACTION_QUEUE_SIZE]
            unit_action_queue_count: 'torch.Tensor',  # int[2, MAX_N_UNITS]
            unit_action_queue_update: 'torch.Tensor',  # bool[2, MAX_N_UNITS]
            # env_cfg: EnvConfig,
            # buf_cfg: JuxBufferConfig,
    ):
        assert factory_action.dtype == torch.int32
        assert unit_action_queue.dtype == torch.int32
        assert unit_action_queue_count.dtype == torch.int32
        assert unit_action_queue_update.dtype == torch.uint8

        queue_size = unit_action_queue.shape[-2]
        max_n_units = unit_action_queue_count.shape[-1]
        max_n_factories = factory_action.shape[-1]
        # assert queue_size == env_cfg.UNIT_ACTION_QUEUE_SIZE
        # assert max_n_units == buf_cfg.MAX_N_UNITS
        # assert max_n_factories == buf_cfg.MAX_N_FACTORIES

        assert factory_action.shape[-2:] == (2, max_n_factories)
        assert unit_action_queue.shape[-4:] == (2, max_n_units, queue_size, 5)
        assert unit_action_queue_count.shape[-2:] == (2, max_n_units)
        assert unit_action_queue_update.shape[-2:] == (2, max_n_units)

        batch_shape = factory_action.shape[:-2]
        assert factory_action.shape[:-2] == batch_shape
        assert unit_action_queue.shape[:-4] == batch_shape
        assert unit_action_queue_count.shape[:-2] == batch_shape
        assert unit_action_queue_update.shape[:-2] == batch_shape

        assert (unit_action_queue_count[unit_action_queue_update == False] == 0).all()
        assert (unit_action_queue_count[unit_action_queue_update == True] <= queue_size).all()

        factory_action = jux.torch.from_torch(factory_action)
        unit_action_queue = jux.torch.from_torch(unit_action_queue)
        unit_action_queue_count = jux.torch.from_torch(unit_action_queue_count)
        unit_action_queue_update = jux.torch.from_torch(unit_action_queue_update)
        unit_action_queue_update = unit_action_queue_update.astype(jnp.bool_)

        return JuxAction(
            factory_action=factory_action,
            unit_action_queue=UnitAction(unit_action_queue),
            unit_action_queue_count=unit_action_queue_count,
            unit_action_queue_update=unit_action_queue_update,
        )

    def to_torch(self) -> 'JuxAction':
        """Convert leaves of `JuxAction` to `torch.Tensor`.

        Returns:
            JuxAction: a JuxAction object with leaves converted to `torch.Tensor`.
        """
        return jax.tree_map(jux.torch.to_torch, self)


def bid_action_from_lux(lux_bid_action: Dict[str, Dict[str, int]]) -> Tuple[Array, Array]:
    '''
    Convert a `LuxAI2022` bid action to a format that `JuxEnv.step_bid()` can receive.

    Args:
        lux_bid_action (Dict[str, Dict[str, int]]): a bid action from `LuxAI2022`. In format of:
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
        dtype=jnp.int32,
    )
    faction = jnp.array(
        [
            FactionTypes[lux_bid_action['player_0']['faction']],
            FactionTypes[lux_bid_action['player_1']['faction']],
        ],
        dtype=jnp.int32,
    )
    return bid, faction


def bid_action_to_lux(bid: Array, faction: Array) -> Dict[str, Dict[str, int]]:
    '''
    Convert a `JuxEnv.step_bid()` action to a format that `LuxAI2022` can receive.

    Args:
        bid (Array): int[2], bid amount for each player.
        faction (Array): int[2], faction for each player.

    Returns:
        Dict[str, Dict[str, int]]: a `LuxAI2022` bid action. In format of:
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
    '''Convert a `LuxAI2022` factory placement action to a format that `JuxEnv.step_factory_placement()` can receive.
    See `JuxEnv.step_factory_placement()` for more details.

    Args:
        lux_act (Dict[str, Dict[str, Any]]): a `LuxAI2022` factory placement action. In format of:
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
    spawn = jnp.array([
        lux_act['player_0']['spawn'] if lux_act['player_0'] else [0, 0],
        lux_act['player_1']['spawn'] if lux_act['player_1'] else [0, 0],
    ])
    water = jnp.array([
        lux_act['player_0']['water'] if lux_act['player_0'] else 0,
        lux_act['player_1']['water'] if lux_act['player_1'] else 0,
    ])
    metal = jnp.array([
        lux_act['player_0']['metal'] if lux_act['player_0'] else 0,
        lux_act['player_1']['metal'] if lux_act['player_1'] else 0,
    ])
    return spawn, water, metal


def factory_placement_action_to_lux(spawn, water, metal) -> Dict[str, Dict[str, Any]]:
    """Convert factory placement action to LuxAI2022's action format. For more details about
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
