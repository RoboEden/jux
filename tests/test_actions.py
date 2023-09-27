from typing import List

import chex
import numpy as np
import pytest
from luxai_s2 import actions as lux_actions
from luxai_s2.actions import Action as LuxAction

import jux.utils
from jux.actions import ActionQueue, FactoryAction, JuxAction, UnitAction
from jux.config import EnvConfig, JuxBufferConfig
from jux.map.position import Direction
from jux.state import State
from jux.unit_cargo import ResourceType


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


class TestJuxAction():

    def test_to_from_lux_torch(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch is not installed")

        env, actions = jux.utils.load_replay("tests/replay2.0_0.json.gz")
        while env.env_steps < 30:
            act = next(actions)
            env.step(act)

        lux_act = next(actions)

        buf_cfg = JuxBufferConfig(MAX_N_UNITS=200)
        jux_state = State.from_lux(env.state, buf_cfg)
        jux_act = JuxAction.from_lux(jux_state, lux_act)

        assert jux_act.to_lux(jux_state) == lux_act

        torch_act = jux_act.to_torch()
        jux_from_torch = JuxAction.from_torch(
            torch_act.factory_action,
            torch_act.unit_action_queue,
            torch_act.unit_action_queue_count,
            torch_act.unit_action_queue_update,
        )
        chex.assert_trees_all_equal(jux_from_torch, jux_act)
