import gzip
import json
import os.path as osp
from typing import Dict, Iterable

import chex
from luxai2022 import LuxAI2022
from luxai2022.state import State as LuxState

from jux.config import EnvConfig, JuxBufferConfig
from jux.state import State


def get_actions_from_replay(replay: dict) -> Iterable[Dict[str, Dict]]:
    for step in replay['steps'][1:]:
        player_0, player_1 = step
        yield {
            'player_0': player_0['action'],
            'player_1': player_1['action'],
        }


def load_replay(replay: str = 'tests/replay.json.gz') -> LuxState:
    if osp.splitext(replay)[-1] == '.gz':
        with gzip.open(replay) as f:
            replay = json.load(f)
    else:
        with open(replay) as f:
            replay = json.load(f)
    seed = replay['configuration']['seed']
    env = LuxAI2022()
    env.reset(seed=seed)
    actions = get_actions_from_replay(replay)

    return env, actions


class TestState(chex.TestCase):

    def test_from_to_lux(self):
        buf_cfg = JuxBufferConfig(MAX_N_UNITS=100)
        env, actions = load_replay()
        for i in range(10):
            action = next(actions)
            env.step(action)

        lux_state = env.state
        jux_state = State.from_lux(lux_state, buf_cfg)
        assert jux_state == State.from_lux(jux_state.to_lux(), buf_cfg)

    @chex.variants(with_jit=True, without_jit=True, with_device=True)
    def test_step_late_game(self):
        # function to be tested
        state_step_late_game = self.variant(State._step_late_game)

        # prepare an environment
        buf_cfg = JuxBufferConfig(MAX_N_UNITS=40)
        env, actions = load_replay()
        for i in range(10):
            act = next(actions)
            env.step(act)

        jux_state = State.from_lux(env.state, buf_cfg)

        # step
        def step_both(jux_state, env, lux_act):

            jux_act = jux_state.parse_actions_from_dict(lux_act)
            jux_state = state_step_late_game(jux_state, jux_act)

            env.step(lux_act)
            lux_state = env.state

            return jux_state, lux_state

        # step 8 times
        # it contains only 'move' and 'dig' actions for robots
        # For factory, it contains only 'build' actions
        for i, act in zip(range(8), actions):
            jux_state, lux_state = step_both(jux_state, env, act)
            assert jux_state == State.from_lux(lux_state, buf_cfg)

        # # another step
        # # it contains a new action 'pickup'
        # act = next(actions)
        # jux_state, lux_state = step_both(jux_state, env, act)
        # assert jux_state == State.from_lux(lux_state, buf_cfg)
