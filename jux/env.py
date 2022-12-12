from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from luxai2022 import LuxAI2022

from jux.actions import JuxAction
from jux.config import EnvConfig, JuxBufferConfig
from jux.state import State


class JuxEnv:
    metadata = {"render.modes": ["human", "rgb_array"], "name": "jux_v0"}

    def __init__(self, env_cfg=EnvConfig(), buf_cfg=JuxBufferConfig()) -> None:
        self.env_cfg = env_cfg
        self.buf_cfg = buf_cfg
        self._dummy_env = LuxAI2022()  # for rendering

    @partial(jax.jit, static_argnums=(0, ))
    def reset(self, seed: int) -> Tuple[State, Tuple[Dict, int, bool, Dict]]:
        return State.new(seed, self.env_cfg, self.buf_cfg)

    @partial(jax.jit, static_argnums=(0, ))
    def step_bid(self, state: State, action: JuxAction) -> Tuple[State, Tuple[Dict, int, bool, Dict]]:
        # TODO
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0, ))
    def step_factory_placement(self, state: State, action: JuxAction) -> Tuple[State, Tuple[Dict, int, bool, Dict]]:
        # TODO
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0, ))
    def step_late_game(self, state: State, action: JuxAction) -> Tuple[State, Tuple[Dict, int, bool, Dict]]:
        state = state._step_late_game(action)

        # perfect info game, so observations = state
        observations = {
            'player_0': state,
            'player_1': state,
        }
        rewards = state.team_lichen_score() - (state.n_factories == 0) * 1000
        infos = {
            'player_0': {},
            'player_1': {},
        }

        dones = (state.n_factories == 0).any() | (state.real_env_steps >= self.env_cfg.max_episode_length)
        dones = jnp.array([dones, dones])

        return state, (observations, rewards, dones, infos)

    def render(self, state: State, mode='human', **kwargs):
        '''
        Render the environment.
        '''
        assert state.n_units.shape == (2, ), "Only support rendering for single environment."
        self._dummy_env.state = state.to_lux()
        return self._dummy_env.render(mode=mode, **kwargs)

    def close(self):
        return self._dummy_env.close()


class JuxAI2022:
    """A LuxAI2022 compatible wrapper for Jux.
    """
    # TODO
