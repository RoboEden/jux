import gzip
import json
import os.path as osp
from typing import Dict, Iterable

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


class TestState:

    def create_state(self, replay: str = 'tests/replay.json.gz', n_step=10) -> LuxState:
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
        for i in range(n_step):
            action = next(actions)
            env.step(action)

        return env.state

    def test_from_to_lux(self):
        buf_cfg = JuxBufferConfig(MAX_N_UNITS=100)
        lux_state = self.create_state()
        jux_state = State.from_lux(lux_state, buf_cfg)
        assert jux_state == State.from_lux(jux_state.to_lux(), buf_cfg)
