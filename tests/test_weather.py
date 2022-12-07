import chex
import jax
import numpy as np
from jax import numpy as jnp
from luxai2022.config import EnvConfig as LuxEnvConfig

from jux.config import EnvConfig
from jux.weather import generate_weather_schedule


class TestWeather(chex.TestCase):

    # @chex.variants(without_jit=True)
    def test_generate_weather_schedule(self):
        seed = 42
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        lux_env_cfg = LuxEnvConfig()
        env_cfg = EnvConfig()
        lux_weather = self.lux_generate_weather_schedule(key=subkey, cfg=lux_env_cfg)
        jux_weather = generate_weather_schedule(key=subkey, env_cfg=env_cfg)
        jax.debug.print("lux_weather: {lux_weather}", lux_weather=lux_weather)
        jax.debug.print("jux_weather: {jux_weather}", jux_weather=jux_weather)
        assert jnp.allclose(lux_weather, jux_weather)

    @staticmethod
    def lux_generate_weather_schedule(key: jax.random.PRNGKey, cfg: LuxEnvConfig):
        # randomly generate 3-5 events, each lasting 20 turns
        # no event can overlap another
        key, subkey = jax.random.split(key)
        num_events = jax.random.randint(key=subkey,
                                        shape=(),
                                        minval=cfg.NUM_WEATHER_EVENTS_RANGE[0],
                                        maxval=cfg.NUM_WEATHER_EVENTS_RANGE[1] + 1)
        available_times = set(list(range(cfg.max_episode_length - 30)))
        schedule = np.zeros(cfg.max_episode_length, dtype=int)
        for i in range(num_events):
            key, subkey = jax.random.split(key)
            weather_id = jax.random.randint(key=subkey, shape=(), minval=1, maxval=len(cfg.WEATHER_ID_TO_NAME))
            weather = cfg.WEATHER_ID_TO_NAME[weather_id]
            weather_cfg = cfg.WEATHER[weather]
            weather_time_range = weather_cfg["TIME_RANGE"]
            # use rejection sampling
            while True:
                key, subkey = jax.random.split(key)
                start_time = jax.random.randint(key=subkey, shape=(), minval=0, maxval=cfg.max_episode_length - 20)
                key, subkey = jax.random.split(key)
                duration = jax.random.randint(key=subkey,
                                              shape=(),
                                              minval=weather_time_range[0],
                                              maxval=weather_time_range[1] + 1)
                requested_times = set(list(range(start_time, start_time + duration)))
                if requested_times.issubset(available_times):
                    available_times.difference_update(requested_times)
                    schedule[start_time:start_time + duration] = weather_id
                    break
        return schedule

    @chex.variants(with_jit=True, without_jit=True, with_device=True, without_device=True)
    def test_generate_weather_schedule(self):
        seed = 42
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        env_cfg = EnvConfig()
        jux_generate_weather_schedule = self.variant(generate_weather_schedule, static_argnames=["env_cfg"])
        jux_weather = jux_generate_weather_schedule(key=subkey, env_cfg=env_cfg)
