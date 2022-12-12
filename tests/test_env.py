import chex
import jax
import numpy as np

import jux.env
import jux.utils
from jux.actions import JuxAction
from jux.config import JuxBufferConfig
from jux.state import State

state___eq___jitted = jax.jit(chex.assert_max_traces(n=1)(State.__eq__))


class TestJuxEnv:

    def test_step_late_game(self):
        chex.clear_trace_counter()
        cases = [
            # Some steps are skipped because of bugs in the official one.
            # episode id and skip steps.
            ('45740641', []),
            ('45742007', []),
            ('45750090', [])
        ]
        buf_cfg = JuxBufferConfig(MAX_N_UNITS=200)
        jux_env = jux.env.JuxEnv(buf_cfg=buf_cfg)

        for episode, skip_steps in cases:
            print(f"episode: {episode}")
            # 1. prepare an environment
            lux_env, actions = jux.utils.load_replay(f'https://www.kaggleusercontent.com/episodes/{episode}.json')

            # 2. skip early stage
            while lux_env.env_steps < 11:
                act = next(actions)
                lux_env.step(act)

            lux_state = lux_env.state
            jux_state = State.from_lux(lux_state, buf_cfg)

            # 3. real test starts here
            for i, lux_act in enumerate(actions):
                print(f"steps: {lux_env.env_steps}")

                lux_obs, lux_rewards, lux_dones, lux_infos = lux_env.step(lux_act)

                jux_act = JuxAction.from_lux(jux_state, lux_act)
                jux_state, (jux_obs, jux_rewards, jux_dones, jux_infos) = jux_env.step_late_game(jux_state, jux_act)

                # 4. check results
                if i % 100 == 0:
                    # comparing obs is too slow, so we only do it every 100 steps.
                    jux_obs_0 = jux_obs['player_0'].to_lux().get_obs()
                    lux_obs_0 = lux_obs['player_0']

                    # for late game, we don't need valid_spawns_mask
                    if lux_env.state.real_env_steps > 10:
                        del jux_obs_0['board']['valid_spawns_mask']
                        del lux_obs_0['board']['valid_spawns_mask']

                    compare_obs = jax.tree_map(
                        np.array_equal,
                        jux_obs_0,
                        lux_obs_0,
                    )
                    assert all(jax.tree_util.tree_leaves(compare_obs))

                assert jux_dones[0] == lux_dones['player_0']
                assert jux_dones[1] == lux_dones['player_1']

                assert jux_rewards[0] == lux_rewards['player_0']
                assert jux_rewards[1] == lux_rewards['player_1']

                assert jux_infos == lux_infos
