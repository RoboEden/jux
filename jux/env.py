from typing import Dict, Tuple

import jax

from jux.config import EnvConfig, JuxBufferConfig
from jux.state import State


class Jux:

    def __init__(self, env_cfg: EnvConfig, buf_cfg: JuxBufferConfig) -> None:
        # TODO
        pass

    def reset(self, seed: jax.random.PRNGKey) -> Tuple[State, Tuple[Dict, int, bool, Dict]]:
        # TODO
        raise NotImplementedError

    def step(self, action) -> Tuple[State, Tuple[Dict, int, bool, Dict]]:
        # TODO
        raise NotImplementedError


class JuxAI2022:
    """A LuxAI2022 compatible wrapper for Jux.
    """
    # TODO
