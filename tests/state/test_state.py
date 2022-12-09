import gzip
import json
import os.path as osp
import urllib.request
from typing import Dict, Iterable

import chex
import jax
import jax.numpy as jnp
import pytest
from luxai2022 import LuxAI2022
from luxai2022.state import State as LuxState
from rich import print

from jux.config import EnvConfig, JuxBufferConfig
from jux.state import JuxAction, State

jnp.set_printoptions(linewidth=500, threshold=10000)


def get_actions_from_replay(replay: dict) -> Iterable[Dict[str, Dict]]:
    for step in replay['steps'][1:]:
        player_0, player_1 = step
        yield {
            'player_0': player_0['action'],
            'player_1': player_1['action'],
        }


def load_replay(replay: str = 'tests/replay.json.gz'):
    if osp.splitext(replay)[-1] == '.gz':
        with gzip.open(replay) as f:
            replay = json.load(f)
    elif replay.startswith('https://') or replay.startswith('http://'):
        with urllib.request.urlopen(replay) as f:
            replay = json.load(f)
    else:
        with open(replay) as f:
            replay = json.load(f)
    seed = replay['configuration']['seed']
    env = LuxAI2022()
    env.reset(seed=seed)
    actions = get_actions_from_replay(replay)

    return env, actions


def map_to_aval(pytree):
    return jax.tree_map(lambda x: x.aval, pytree)


state___eq___jitted = jax.jit(chex.assert_max_traces(n=1)(State.__eq__))


class TestState(chex.TestCase):

    def test_from_to_lux(self):
        buf_cfg = JuxBufferConfig(MAX_N_UNITS=30)
        env, actions = load_replay('tests/replay-45702251.json.gz')
        for i in range(10):
            action = next(actions)
            env.step(action)

        lux_state = env.state
        jux_state = State.from_lux(lux_state, buf_cfg)
        assert jux_state == State.from_lux(jux_state.to_lux(), buf_cfg)

    @chex.variants(with_jit=True, without_jit=True, with_device=True)
    def test_step_late_game(self):
        chex.clear_trace_counter()

        # 1. function to be tested
        state_step_late_game = self.variant(chex.assert_max_traces(n=1)(State._step_late_game))

        # 2. prepare an environment
        buf_cfg = JuxBufferConfig(MAX_N_UNITS=30)
        env, actions = load_replay('tests/replay-45702251.json.gz')
        while env.env_steps < 10:
            act = next(actions)
            env.step(act)

        jux_state = State.from_lux(env.state, buf_cfg)

        # 3. some helper functions

        # step
        def step_both(jux_state: State, env: LuxAI2022, lux_act):

            jux_act = jux_state.parse_actions_from_dict(lux_act)
            jux_state = state_step_late_game(jux_state, jux_act)
            jnp.array(0).block_until_ready()

            env.step(lux_act)
            lux_state = env.state

            return jux_state, lux_state

        def assert_state_eq(jux_state, lux_state):
            lux_state = State.from_lux(lux_state, buf_cfg)
            assert state___eq___jitted(jux_state, lux_state)

        # # warm up jit, for profile only
        # state___eq___jitted(jux_state, jux_state)
        # state_step_late_game(jux_state, JuxAction.empty(jux_state.env_cfg, buf_cfg))

        # 4. real test starts here
        # step 61 times
        # it contains only move/dig/transfer/pickup actions for robots
        # For factory, it contains only 'build' actions
        for i, act in zip(range(61), actions):

            if env.env_steps % 10 == 0:
                print(f"steps: {env.env_steps}")
                jux_state = State.from_lux(env.state, buf_cfg)
                jux_state, lux_state = step_both(jux_state, env, act)
                assert_state_eq(jux_state, lux_state)
            else:
                env.step(act)

    @chex.variants(with_jit=True, without_jit=True, with_device=True)
    def test_recharge_action(self):
        chex.clear_trace_counter()

        # 1. function to be tested
        state_step_late_game = self.variant(chex.assert_max_traces(n=1)(State._step_late_game))

        # 2. prepare an environment
        buf_cfg = JuxBufferConfig(MAX_N_UNITS=100)

        env, actions = load_replay("https://www.kaggleusercontent.com/episodes/45715004.json")

        # skip first several steps, since it contains no recharge action
        while env.env_steps < 10 + 149:
            act = next(actions)
            env.step(act)

        jux_state = State.from_lux(env.state, buf_cfg)

        # 3. some helper functions

        # step
        def step_both(jux_state: State, env: LuxAI2022, lux_act):

            jux_act = jux_state.parse_actions_from_dict(lux_act)
            jux_state = state_step_late_game(jux_state, jux_act)
            jnp.array(0).block_until_ready()

            env.step(lux_act)
            lux_state = env.state

            return jux_state, lux_state

        def assert_state_eq(jux_state, lux_state):
            lux_state = State.from_lux(lux_state, buf_cfg)
            assert state___eq___jitted(jux_state, lux_state)

        # # warm up jit, for profile only
        # state___eq___jitted(jux_state, jux_state)
        # state_step_late_game(jux_state, JuxAction.empty(jux_state.env_cfg, buf_cfg))

        # 4. real test starts here
        for i, act in zip(range(10), actions):
            print(f"steps: {env.env_steps}")
            jux_state, lux_state = step_both(jux_state, env, act)
            assert_state_eq(jux_state, lux_state)

    @chex.variants(with_jit=True, without_jit=True, with_device=True)
    def test_step_factory_water(self):
        chex.clear_trace_counter()

        # 1. function to be tested
        state_step_late_game = self.variant(chex.assert_max_traces(n=1)(State._step_late_game))

        # 2. prepare an environment
        buf_cfg = JuxBufferConfig(MAX_N_UNITS=100)
        env, actions = load_replay("https://www.kaggleusercontent.com/episodes/45715004.json")
        # The first 905 steps do not contains any FactoryAction.Water, so we skip them.
        while env.env_steps < 435:
            act = next(actions)
            env.step(act)

        jux_state = State.from_lux(env.state, buf_cfg)

        # 3. some helper functions

        # step
        def step_both(jux_state: State, env: LuxAI2022, lux_act):

            jux_act = jux_state.parse_actions_from_dict(lux_act)
            jux_state = state_step_late_game(jux_state, jux_act)
            jnp.array(0).block_until_ready()

            env.step(lux_act)
            lux_state = env.state

            return jux_state, lux_state

        # # warm up jit
        # state___eq___jitted(jux_state, jux_state)
        # state_step_late_game(jux_state, JuxAction.empty(jux_state.env_cfg, buf_cfg))

        def assert_state_eq(jux_state, lux_state):
            lux_state = State.from_lux(lux_state, buf_cfg)
            assert state___eq___jitted(jux_state, lux_state)

        # 4. real test starts here

        # step util end
        for i, act in zip(range(290), actions):
            if env.env_steps % 10 == 0:
                print(f"steps: {env.env_steps}")
                jux_state = State.from_lux(env.state, buf_cfg)
                jux_state, lux_state = step_both(jux_state, env, act)
                assert_state_eq(jux_state, lux_state)
            else:
                env.step(act)
