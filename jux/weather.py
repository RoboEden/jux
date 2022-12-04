from enum import IntEnum
from typing import NamedTuple, Tuple

import jax


class Weather(IntEnum):
    NONE = 0
    MARS_QUAKE = 1
    COLD_SNAP = 2
    DUST_STORM = 3
    SOLAR_FLARE = 4


class MarsQuake(NamedTuple):
    RUBBLE: Tuple[int, int] = (
        1,  # UnitType.LIGHT
        10,  # UnitType.HEAVY
    )
    TIME_RANGE: Tuple[int, int] = (1, 5)

    @classmethod
    def from_lux(cls, lux_mars_quake):
        return cls(
            RUBBLE=(
                lux_mars_quake['RUBBLE']['LIGHT'],
                lux_mars_quake['RUBBLE']['HEAVY'],
            ),
            TIME_RANGE=tuple(lux_mars_quake['TIME_RANGE']),
        )

    def to_lux(self):
        return {
            'RUBBLE': dict(LIGHT=self.RUBBLE[0], HEAVY=self.RUBBLE[1]),
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
