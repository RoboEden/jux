from enum import IntEnum
from typing import NamedTuple, Tuple

import jax
from jax import lax
from jax import numpy as jnp

import jux
from jux.utils import INT32_MAX


class Weather(IntEnum):
    NONE = 0
    MARS_QUAKE = 1
    COLD_SNAP = 2
    DUST_STORM = 3
    SOLAR_FLARE = 4


class MarsQuake(NamedTuple):
    TIME_RANGE: Tuple[int, int] = (1, 5)

    @classmethod
    def from_lux(cls, lux_mars_quake):
        return cls(TIME_RANGE=tuple(lux_mars_quake['TIME_RANGE']))

    def to_lux(self):
        return {
            'TIME_RANGE': list(self.TIME_RANGE),
        }


class ColdSnap(NamedTuple):
    POWER_CONSUMPTION: int = 2  # must be integer
    TIME_RANGE: Tuple[int, int] = (10, 30)

    @classmethod
    def from_lux(cls, lux_cold_snap):
        return cls(
            POWER_CONSUMPTION=int(lux_cold_snap['POWER_CONSUMPTION']),
            TIME_RANGE=tuple(lux_cold_snap['TIME_RANGE']),
        )

    def to_lux(self):
        return {
            'POWER_CONSUMPTION': int(self.POWER_CONSUMPTION),
            'TIME_RANGE': list(self.TIME_RANGE),
        }


class DustStorm(NamedTuple):
    POWER_GAIN: float = 0.5
    TIME_RANGE: Tuple[int, int] = (10, 30)

    @classmethod
    def from_lux(cls, lux_dust_storm):
        return cls(
            POWER_GAIN=float(lux_dust_storm['POWER_GAIN']),
            TIME_RANGE=tuple(lux_dust_storm['TIME_RANGE']),
        )

    def to_lux(self):
        return {
            'POWER_GAIN': float(self.POWER_GAIN),
            'TIME_RANGE': list(self.TIME_RANGE),
        }


class SolarFlare(NamedTuple):
    POWER_GAIN: float = 2.0
    TIME_RANGE: Tuple[int, int] = (10, 30)

    @classmethod
    def from_lux(cls, lux_solar_flare):
        return cls(
            POWER_GAIN=float(lux_solar_flare['POWER_GAIN']),
            TIME_RANGE=tuple(lux_solar_flare['TIME_RANGE']),
        )

    def to_lux(self):
        return {
            'POWER_GAIN': float(self.POWER_GAIN),
            'TIME_RANGE': list(self.TIME_RANGE),
        }


class WeatherConfig(NamedTuple):
    NONE: Tuple = ()
    MARS_QUAKE: MarsQuake = MarsQuake()
    COLD_SNAP: ColdSnap = ColdSnap()
    DUST_STORM: DustStorm = DustStorm()
    SOLAR_FLARE: SolarFlare = SolarFlare()

    @classmethod
    def from_lux(cls, lux_weather_config):
        return cls(
            NONE=(),
            MARS_QUAKE=MarsQuake.from_lux(lux_weather_config['MARS_QUAKE']),
            COLD_SNAP=ColdSnap.from_lux(lux_weather_config['COLD_SNAP']),
            DUST_STORM=DustStorm.from_lux(lux_weather_config['DUST_STORM']),
            SOLAR_FLARE=SolarFlare.from_lux(lux_weather_config['SOLAR_FLARE']),
        )

    def to_lux(self):
        return dict(
            MARS_QUAKE=self.MARS_QUAKE.to_lux(),
            COLD_SNAP=self.COLD_SNAP.to_lux(),
            DUST_STORM=self.DUST_STORM.to_lux(),
            SOLAR_FLARE=self.SOLAR_FLARE.to_lux(),
        )


def get_weather_cfg(weather_cfg, current_weather: Weather):
    return jax.lax.switch(
        current_weather,
        [
            # NONE
            lambda cfg: dict(power_gain_factor=1.0, power_loss_factor=1),
            # MARS_QUAKE
            lambda cfg: dict(power_gain_factor=1.0, power_loss_factor=1),
            # COLD_SNAP
            lambda cfg: dict(power_gain_factor=1.0, power_loss_factor=cfg.COLD_SNAP.POWER_CONSUMPTION),
            # DUST_STORM
            lambda cfg: dict(power_gain_factor=cfg.DUST_STORM.POWER_GAIN.astype(float), power_loss_factor=1),
            # SOLAR_FLARE
            lambda cfg: dict(power_gain_factor=cfg.SOLAR_FLARE.POWER_GAIN.astype(float), power_loss_factor=1),
        ],
        weather_cfg,
    )


def generate_weather_schedule(key: jax.random.KeyArray, env_cfg):
    # randomly generate 3-5 events, each lasting 20 turns
    # no event can overlap another
    key, subkey = jax.random.split(key)
    num_events = jax.random.randint(key=subkey,
                                    shape=(),
                                    minval=env_cfg.NUM_WEATHER_EVENTS_RANGE[0],
                                    maxval=env_cfg.NUM_WEATHER_EVENTS_RANGE[1] + 1)
    # last_event_end_step = 0
    available_times = jnp.arange(env_cfg.max_episode_length) < (env_cfg.max_episode_length - 30)
    schedule = jnp.zeros(env_cfg.max_episode_length, dtype=jnp.int32)
    time_ranges = jnp.array((
        (INT32_MAX, INT32_MAX),
        env_cfg.WEATHER.MARS_QUAKE.TIME_RANGE,
        env_cfg.WEATHER.COLD_SNAP.TIME_RANGE,
        env_cfg.WEATHER.DUST_STORM.TIME_RANGE,
        env_cfg.WEATHER.SOLAR_FLARE.TIME_RANGE,
    ))

    def fori_body_fun(i, val1):
        key, available_times, schedule = val1
        key, subkey = jax.random.split(key)
        weather_type = jax.random.randint(key=subkey, shape=(), minval=1, maxval=len(Weather))
        weather_time_range = time_ranges[weather_type]

        def _while_body_fun(val2):
            key, requested_time = val2
            key, subkey = jax.random.split(key)
            start_time = jax.random.randint(key=subkey, shape=(), minval=0, maxval=env_cfg.max_episode_length - 20)
            key, subkey = jax.random.split(key)
            duration = jax.random.randint(key=subkey,
                                          shape=(),
                                          minval=weather_time_range[0],
                                          maxval=weather_time_range[1] + 1)
            requested_time = jnp.arange(env_cfg.max_episode_length)

            requested_time = (requested_time >= start_time) & (requested_time < start_time + duration)
            return key, requested_time

        def _cond_func(val2):
            _, requested_time = val2
            return ~((requested_time & available_times) == requested_time).all()

        requested_time = jnp.ones_like(available_times)
        key, requested_time = lax.while_loop(cond_fun=_cond_func,
                                             body_fun=_while_body_fun,
                                             init_val=(key, requested_time))
        available_times = available_times & (~requested_time)
        schedule = schedule + requested_time * weather_type
        return key, available_times, schedule

    key, available_times, schedule = lax.fori_loop(lower=0,
                                                   upper=num_events,
                                                   body_fun=fori_body_fun,
                                                   init_val=(key, available_times, schedule))
    return schedule
