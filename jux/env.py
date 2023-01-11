from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from luxai_s2 import LuxAI_S2

from jux.actions import JuxAction
from jux.config import EnvConfig, JuxBufferConfig
from jux.state import State


class JuxEnv:
    metadata = {"render.modes": ["human", "rgb_array"], "name": "jux_v0"}

    def __init__(self, env_cfg=EnvConfig(), buf_cfg=JuxBufferConfig()) -> None:
        self.env_cfg = env_cfg
        self.buf_cfg = buf_cfg
        self._dummy_env = LuxAI_S2()  # for rendering

    def __hash__(self) -> int:
        return hash((JuxEnv, self.env_cfg, self.buf_cfg))

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, JuxEnv) and self.env_cfg == __o.env_cfg and self.buf_cfg == __o.buf_cfg

    @partial(jax.jit, static_argnums=(0, ))
    def reset(self, seed: int) -> State:
        return State.new(seed, self.env_cfg, self.buf_cfg)

    @partial(jax.jit, static_argnums=(0, ))
    def step_bid(self, state: State, bid: Array, faction: Array) -> Tuple[State, Tuple[Dict, Array, Array, Dict]]:
        """Step the first bidding step.

        Args:
            state (State): The current game state.
            bid (Array): int[2], two players' bids.
            faction (Array): int[2], two players' faction.

        Returns:
            state, (observations, rewards, dones, infos)

            state (State): new game state. observations (Dict): observations for two players.
                As this is perfect info game, so `observations['player_0'] == observations['player_1'] == state`.
            rewards (Array): int[2], two players' rewards, which is equal to the sum of lichens each player has.
            dones (Array): bool[2], done indicator. If game ends, then `dones[0] == dones[1] == True`
            infos (Dict): empty dict, because there is no extra info.
        """
        state = state._step_bid(bid, faction)

        # perfect info game, so observations = state
        observations = {
            'player_0': state,
            'player_1': state,
        }
        rewards = jnp.zeros(2)
        infos = {'player_0': {}, 'player_1': {}}

        dones = jnp.zeros(2, dtype=jnp.bool_)

        return state, (observations, rewards, dones, infos)

    @partial(jax.jit, static_argnums=(0, ))
    def step_factory_placement(self, state: State, spawn: Array, water: Array, metal: Array) \
                                                            -> Tuple[State, Tuple[Dict, Array, Array, Dict]]:
        """
        Step the factory placement steps.
        `state.place_first` indicates the player_id of the first player to place
        a factory. The next player is decided by `state.next_player`.
        `state.board.factories_per_team` indicates the number of factories each player can place.
        `state.team.init_water` and `state.team.init_metal` indicates the water and metal budget of each player.

        Args:
            state (State): The current game state.
            spawn (Array): int[2, 2], spawn location. Only `spawn[current_player_id]` is used.
            water (Array): int[2], water to be assigned to factory. Only `water[current_player_id]` is used.
            metal (Array): int[2], metal to be assigned to factory. Only `metal[current_player_id]` is used.

        Returns:
            state, (observations, rewards, dones, infos)

            state (State): new game state. observations (Dict): observations for two players.
                As this is perfect info game, so `observations['player_0'] == observations['player_1'] == state`.
            rewards (Array): int[2], two players' rewards, which is equal to the sum of lichens each player has.
            dones (Array): bool[2], done indicator. If game ends, then `dones[0] == dones[1] == True`
            infos (Dict): empty dict, because there is no extra info.
        """
        state = state._step_factory_placement(spawn, water, metal)

        # perfect info game, so observations = state
        observations = {'player_0': state, 'player_1': state}
        rewards = jnp.zeros(2)
        infos = {'player_0': {}, 'player_1': {}}

        dones = jnp.zeros(2, dtype=jnp.bool_)

        return state, (observations, rewards, dones, infos)

    @partial(jax.jit, static_argnums=(0, ))
    def step_late_game(self, state: State, actions: JuxAction) -> Tuple[State, Tuple[Dict, Array, Array, Dict]]:
        """
        Step the normal game steps.

        Args:
            actions (JuxAction): The actions of two players. Member variables of `actions` shall have following shape and dtype.

                actions.factory_action: `int[2, buf_cfg.MAX_N_FACTORIES]`
                    The factory action of each player. See `FactoryAction` for details. Depending on the number of factory each player
                    has (`state.n_factories`), only the first several elements are used. The rest of the elements is ignored.

                actions.unit_action_queue_update: bool[2, self.buf_cfg.MAX_N_UNITS]
                    Indicates whether to update a robot's action queue. Depending on the number of robots each player has (`state.n_units`),
                    only the first several elements are used. The rest of the elements is ignored.

                actions.unit_action_queue_count: int[2, self.buf_cfg.MAX_N_UNITS]
                    The number of actions in the robot's updated action queue. Only when `actions.unit_action_queue_update[player_i, robot_j]`
                    is True, then `actions.unit_action_queue_count[player_i, robot_j]` is used.

                actions.unit_action_queue: UnitAction[2, self.buf_cfg.MAX_N_UNITS, self.env_cfg.UNIT_ACTION_QUEUE_SIZE].
                    If update a robot's action queue, the actions in the updated queue you want.

                In short, when `actions.unit_action_queue_count[player_i, robot_j] == True`, then the `robot_j` of `player_i` will
                have following actions in queue, if the robot has enough power to update its action queue:
                ```python
                n_actions = actions.unit_action_queue_count[player_i, robot_j]
                actions_in_queue = jux.tree_util.batch_out_of_leaf(
                    jax.tree_map(lambda x: x[player_i, robot_j, :n_actions], actions.unit_action_queue)
                )
                ```

        Returns:
            state, (observations, rewards, dones, infos)

            state (State): new game state. observations (Dict): observations for two players.
                As this is perfect info game, so `observations['player_0'] == observations['player_1'] == state`.
            rewards (Array): int[2], two players' rewards, which is equal to the sum of lichens each player has.
            dones (Array): bool[2], done indicator. If game ends, then `dones[0] == dones[1] == True`
            infos (Dict): empty dict, because there is no extra info.
        """
        state = state._step_late_game(actions)

        # perfect info game, so observations = state
        observations = {'player_0': state, 'player_1': state}

        # rewards = lichen. There is 1000 penalty for losing all factories.
        rewards = state.team_lichen_score() - (state.n_factories == 0) * 1000

        # info is empty
        infos = {'player_0': {}, 'player_1': {}}

        # done if one player loses all factories or max_episode_length is reached
        dones = (state.n_factories == 0).any() | (state.real_env_steps >= self.env_cfg.max_episode_length)
        dones = jnp.array([dones, dones])

        return state, (observations, rewards, dones, infos)

    def render(self, state: State, mode='human', **kwargs):
        """render the environment.

        Args:
            state (State): The current game state.
            mode ('human' | 'rgb_array'): The mode to render. See `LuxAI_S2.render` for details.
        """
        assert state.n_units.shape == (2, ), "Only support rendering for single environment."
        self._dummy_env.state = state.to_lux()
        return self._dummy_env.render(mode=mode, **kwargs)

    def close(self):
        return self._dummy_env.close()

    @staticmethod
    def from_lux(lux_env: LuxAI_S2, buf_cfg=JuxBufferConfig()) -> Tuple['JuxEnv', 'State']:
        """
        Create a `JuxEnv` from a `LuxAI_S2` environment.

        Args:
            env (LuxAI_S2): The LuxAI_S2 environment.
            buf_cfg (JuxBufferConfig): The buffer configuration.

        Returns:
            jux_env (JuxEnv): The Jux environment.
            state (State): The current game state.
        """
        return JuxEnv(EnvConfig.from_lux(lux_env.env_cfg), buf_cfg), State.from_lux(lux_env.state, buf_cfg)


class JuxEnvBatch:

    @property
    def env_cfg(self) -> EnvConfig:
        return self.jux_env.env_cfg

    @property
    def buf_cfg(self) -> JuxBufferConfig:
        return self.jux_env.buf_cfg

    def __init__(self, env_cfg=EnvConfig(), buf_cfg=JuxBufferConfig()) -> None:
        self.jux_env = JuxEnv(env_cfg, buf_cfg)

    def __hash__(self) -> int:
        return hash((JuxEnvBatch, self.jux_env.env_cfg, self.jux_env.buf_cfg))

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, JuxEnvBatch) and self.env_cfg == __o.env_cfg and self.buf_cfg == __o.buf_cfg

    @partial(jax.jit, static_argnums=(0, ))
    def reset(self, seeds: Array) -> Tuple[State, Tuple[Dict, int, bool, Dict]]:
        states = jax.vmap(self.jux_env.reset)(seeds)

        # for env in the same batch, they must have same state.board.factories_per_team,
        # so that they have same number of steps to place factory
        factories_per_team = states.board.factories_per_team.at[:].set(states.board.factories_per_team[0])
        states = states._replace(board=states.board._replace(factories_per_team=factories_per_team))
        return states

    @partial(jax.jit, static_argnums=(0, ))
    def step_bid(self, states: State, bid: Array, faction: Array) -> Tuple[State, Tuple[Dict, Array, Array, Dict]]:
        states, (observations, rewards, dones, infos) = jax.vmap(self.jux_env.step_bid)(states, bid, faction)
        return states, (observations, rewards, dones, infos)

    @partial(jax.jit, static_argnums=(0, ))
    def step_factory_placement(self, states: State, spawn: Array, water: Array, metal: Array) \
                                                            -> Tuple[State, Tuple[Dict, Array, Array, Dict]]:
        states, (observations, rewards, dones, infos) = jax.vmap(self.jux_env.step_factory_placement)(
            states,
            spawn,
            water,
            metal,
        )
        return states, (observations, rewards, dones, infos)

    @partial(jax.jit, static_argnums=(0, ))
    def step_late_game(self, states: State, actions: JuxAction) -> Tuple[State, Tuple[Dict, Array, Array, Dict]]:
        states, (observations, rewards, dones, infos) = jax.vmap(self.jux_env.step_late_game)(states, actions)
        return states, (observations, rewards, dones, infos)
