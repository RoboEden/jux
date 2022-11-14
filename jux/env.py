from typing import Tuple

import jax

from jux.config import EnvConfig, JuxBufferConfig
from jux.state import JuxState


class Jux:

    def __init__(self, env_cfg: EnvConfig, buf_cfg: JuxBufferConfig) -> None:
        # TODO
        pass

    def reset(self, seed: jax.random.PRNGKey) -> Tuple[JuxState, Tuple['obs', int, bool, 'info']]:
        # TODO
        return jux_state, (obs, reward, done, info)

    def step(self, action) -> Tuple[JuxState, Tuple['obs', int, bool, 'info']]:
        # TODO
        return jux_state, (obs, reward, done, info)


class JuxAI2022:
    """A LuxAI2022 compatible wrapper for Jux.
    """
    # TODO
