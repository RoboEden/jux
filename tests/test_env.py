import chex
import jax
import numpy as np
from luxai2022 import LuxAI2022

import jux.actions
import jux.utils
from jux.actions import JuxAction
from jux.config import JuxBufferConfig
from jux.env import JuxEnv
from jux.state import State

state___eq___jitted = jax.jit(chex.assert_max_traces(n=1)(State.__eq__))


class TestJuxEnv:

    def test_new(self):
        jux_env = JuxEnv()
        state1 = jux_env.reset(0)
        state2 = jux_env.reset(0)
        assert state___eq___jitted(state1, state2)

    def test_step_bid_and_factory_placement(self):
        lux_env, actions = jux.utils.load_replay(f'https://www.kaggleusercontent.com/episodes/{45740641}.json')

        buf_cfg = JuxBufferConfig(MAX_N_UNITS=200)
        jux_env, jux_state = JuxEnv.from_lux(lux_env, buf_cfg=buf_cfg)

        # Step 1: bid
        lux_act = next(actions)
        jux_bid, jux_faction = jux.actions.bid_action_from_lux(lux_act)

        lux_obs, lux_rewards, lux_dones, lux_infos = lux_env.step(lux_act)
        jux_state, (jux_obs, jux_rewards, jux_dones, jux_infos) = jux_env.step_bid(jux_state, jux_bid, jux_faction)

        lux_obs = lux_obs['player_0']
        jux_obs = jux_state.to_lux().get_obs()

        # Compare returned values
        compare_obs = jax.tree_map(
            np.array_equal,
            jux_obs,
            lux_obs,
        )
        assert all(jax.tree_util.tree_leaves(compare_obs))
        assert jux_dones[0] == lux_dones['player_0']
        assert jux_dones[1] == lux_dones['player_1']
        assert jux_rewards[0] == lux_rewards['player_0']
        assert jux_rewards[1] == lux_rewards['player_1']

        # Step 2: factory placement
        factories_per_team = int(jux_state.board.factories_per_team)
        for i in range(factories_per_team * 2):
            lux_act = next(actions)
            jux_act = jux.actions.factory_placement_action_from_lux(lux_act)

            lux_obs, lux_rewards, lux_dones, lux_infos = lux_env.step(lux_act)
            jux_state, (jux_obs, jux_rewards, jux_dones,
                        jux_infos) = jux_env.step_factory_placement(jux_state, *jux_act)

            # Compare returned values
            lux_obs = lux_obs['player_0']
            jux_obs = jux_state.to_lux().get_obs()
            compare_obs = jax.tree_map(
                np.array_equal,
                jux_obs,
                lux_obs,
            )
            assert all(jax.tree_util.tree_leaves(compare_obs))
            assert jux_dones[0] == lux_dones['player_0']
            assert jux_dones[1] == lux_dones['player_1']
            assert jux_rewards[0] == lux_rewards['player_0']
            assert jux_rewards[1] == lux_rewards['player_1']

    def test_step_late_game(self):
        chex.clear_trace_counter()
        episode_list = [
            '45740641',
            '45742007',
            '45750090',
        ]
        buf_cfg = JuxBufferConfig(MAX_N_UNITS=200)
        jux_env = JuxEnv(buf_cfg=buf_cfg)

        for episode in episode_list:
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
