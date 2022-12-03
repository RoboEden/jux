from enum import IntEnum

import jax


class Weather(IntEnum):
    NONE = 0
    MARS_QUAKE = 1
    COLD_SNAP = 2
    DUST_STORM = 3
    SOLAR_FLARE = 4


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
