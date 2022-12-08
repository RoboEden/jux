from typing import List

import chex
import numpy as np
from luxai2022 import actions as lux_actions

import jux.actions
from jux.actions import ActionQueue, FactoryAction, LuxAction, UnitAction
from jux.config import EnvConfig, UnitConfig
from jux.map.position import Direction
from jux.unit_cargo import ResourceType

# fix a bug in the luxai2022.actions module
lux_actions.FactoryWaterAction.state_dict = lambda self: 2


class TestActions:

    def test_factory_actions(self):
        actions = [
            FactoryAction.BUILD_LIGHT,
            FactoryAction.BUILD_HEAVY,
            FactoryAction.WATER,
        ]
        for act in actions:
            assert act == FactoryAction.from_lux(act.to_lux())

    def test_unit_actions(self):
        actions = [
            UnitAction.move(Direction.UP, repeat=10),
            UnitAction.transfer(Direction.LEFT, ResourceType.ice, 10, repeat=0),
            UnitAction.pickup(ResourceType.metal, 10, repeat=-1),
            UnitAction.dig(3),
            UnitAction.self_destruct(4),
            UnitAction.recharge(20, -1),
        ]
        for act in actions:
            assert act == UnitAction.from_lux(act.to_lux())


def lux_queue_eq(a: List[LuxAction], b: List[LuxAction]) -> bool:
    return len(a) == len(b) and all([np.array_equal(i.state_dict(), j.state_dict()) for i, j in zip(a, b)])


class TestActionQueue(chex.TestCase):

    @chex.variants(with_jit=True, without_jit=True, with_device=True)
    def test_push_pop(self):
        env_cfg = EnvConfig()
        queue = ActionQueue.empty(env_cfg.UNIT_ACTION_QUEUE_SIZE)
        assert queue.is_empty()
        assert not queue.is_full()
        assert queue.count == 0

        lux_queue = [
            lux_actions.MoveAction(1, 1, False),
            lux_actions.DigAction(True),
            lux_actions.SelfDestructAction(False),
        ]

        push = self.variant(ActionQueue.push_back)
        pop = self.variant(ActionQueue.pop)
        jux_queue = ActionQueue.from_lux(lux_queue, env_cfg.UNIT_ACTION_QUEUE_SIZE)
        jux_action, jux_queue = pop(jux_queue)
        jux_queue = push(jux_queue, jux_action)

        lux_action = lux_queue[0]
        lux_queue = lux_queue[1:]
        lux_queue.append(lux_action)
        assert jux_queue == ActionQueue.from_lux(lux_queue, env_cfg.UNIT_ACTION_QUEUE_SIZE)
        assert lux_queue_eq(jux_queue.to_lux(), lux_queue)
