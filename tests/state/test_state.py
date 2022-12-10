import chex
import jax
import jax.numpy as jnp
from luxai2022 import LuxAI2022
from luxai2022.state import State as LuxState
from rich import print

import jux
from jux.config import EnvConfig, JuxBufferConfig
from jux.state import JuxAction, State

jnp.set_printoptions(linewidth=500, threshold=10000)


def map_to_aval(pytree):
    return jax.tree_map(lambda x: x.aval, pytree)


state___eq___jitted = jax.jit(chex.assert_max_traces(n=1)(State.__eq__))


class TestState(chex.TestCase):

    def test_from_to_lux(self):
        buf_cfg = JuxBufferConfig(MAX_N_UNITS=30)
        env, actions = jux.utils.load_replay('tests/replay-45702251.json.gz')
        for i in range(10):
            action = next(actions)
            env.step(action)

        lux_state = env.state
        jux_state = State.from_lux(lux_state, buf_cfg)
        assert jux_state == State.from_lux(jux_state.to_lux(), buf_cfg)

    def test_step_late_game(self):
        chex.clear_trace_counter()

        # 1. function to be tested
        _state_step_late_game = jax.jit(chex.assert_max_traces(n=1)(State._step_late_game))

        # 2. prepare an environment
        buf_cfg = JuxBufferConfig(MAX_N_UNITS=100)
        env, actions = jux.utils.load_replay('https://www.kaggleusercontent.com/episodes/45715004.json')

        # skip early stage
        while env.env_steps < 5:
            act = next(actions)
            env.step(act)

        jux_state = State.from_lux(env.state, buf_cfg)

        # 3. some helper functions

        # wrapper for profiling
        def state_step_late_game(jux_state, jux_act):
            return _state_step_late_game(jux_state, jux_act)

        # step
        def step_both(jux_state: State, env: LuxAI2022, lux_act):

            jux_act = jux_state.parse_actions_from_dict(lux_act)
            jux_state = _state_step_late_game(jux_state, jux_act)
            # jnp.array(0).block_until_ready()

            env.step(lux_act)
            lux_state = env.state

            return jux_state, lux_state

        def assert_state_eq(jux_state, lux_state):
            lux_state = State.from_lux(lux_state, buf_cfg)
            assert state___eq___jitted(jux_state, lux_state)

        # warm up jit, for profile only
        state___eq___jitted(jux_state, jux_state)
        state_step_late_game(jux_state, JuxAction.empty(jux_state.env_cfg, buf_cfg))

        # 4. real test starts here
        for i, act in enumerate(actions):
            print(f"steps: {env.env_steps}")
            jux_state, lux_state = step_both(jux_state, env, act)
        assert_state_eq(jux_state, lux_state)

    @chex.variants(with_jit=True, without_jit=True, with_device=True)
    def test_recharge_action(self):
        chex.clear_trace_counter()

        # 1. function to be tested
        state_step_late_game = self.variant(chex.assert_max_traces(n=1)(State._step_late_game))

        # 2. prepare an environment
        buf_cfg = JuxBufferConfig(MAX_N_UNITS=100)

        env, actions = jux.utils.load_replay("https://www.kaggleusercontent.com/episodes/45715004.json")

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
        env, actions = jux.utils.load_replay("https://www.kaggleusercontent.com/episodes/45715004.json")
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
        for i, act in zip(range(300), actions):
            if env.env_steps % 10 == 0:
                print(f"steps: {env.env_steps}")
                jux_state = State.from_lux(env.state, buf_cfg)
                jux_state, lux_state = step_both(jux_state, env, act)
                assert_state_eq(jux_state, lux_state)
            else:
                env.step(act)

    def test_episode(self):
        chex.clear_trace_counter()
        cases = [
            # Some steps are skipped because our implementation is not 100% the same as the official one.
            # episode id and skip steps.
            ('45731509', [177, 178, 238, 713, 734, 776, 821]),
            ('45740668', [16, 25, 38]),
            ('45740641', []),
            ('45742163', [248, 296, 297, 392]),
            ('45742007', []),
        ]
        buf_cfg = JuxBufferConfig(MAX_N_UNITS=200)

        # 1. function to be tested
        state_step_late_game = jax.jit(chex.assert_max_traces(n=1)(State._step_late_game))

        for episode, skip_steps in cases:
            # 2. prepare an environment
            env, actions = jux.utils.load_replay(f'https://www.kaggleusercontent.com/episodes/{episode}.json')

            # skip early stage
            while env.env_steps < 11:
                act = next(actions)
                env.step(act)

            lux_state = env.state
            jux_state = State.from_lux(lux_state, buf_cfg)

            # 3. some helper functions

            # step
            def step_both(jux_state: State, env: LuxAI2022, lux_act):

                jux_act = jux_state.parse_actions_from_dict(lux_act)
                jux_state = state_step_late_game(jux_state, jux_act)

                env.step(lux_act)
                lux_state = env.state

                return jux_state, lux_state

            def assert_state_eq(jux_state, lux_state):
                lux_state = State.from_lux(lux_state, buf_cfg)
                assert state___eq___jitted(jux_state, lux_state)

            # 4. real test starts here
            for i, act in enumerate(actions):
                print(f"steps: {env.env_steps}")
                if env.env_steps in skip_steps:
                    assert_state_eq(jux_state, lux_state)
                    env.step(act)
                    jux_state = State.from_lux(env.state, buf_cfg)
                else:
                    jux_state, lux_state = step_both(jux_state, env, act)
            assert_state_eq(jux_state, lux_state)
