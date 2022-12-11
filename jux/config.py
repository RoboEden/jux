import dataclasses
from copy import copy
from typing import Any, Dict, NamedTuple, Tuple

import jax
from luxai2022.config import EnvConfig as LuxEnvConfig
from luxai2022.config import UnitConfig as LuxUnitConfig

import jux
from jux.weather import WeatherConfig


class UnitConfig(NamedTuple):
    METAL_COST: int = 100
    POWER_COST: int = 500
    CARGO_SPACE: int = 1000
    BATTERY_CAPACITY: int = 1500
    CHARGE: int = 1
    INIT_POWER: int = 50
    MOVE_COST: int = 1
    RUBBLE_MOVEMENT_COST: int = 1
    DIG_COST: int = 5
    DIG_RUBBLE_REMOVED: int = 1
    DIG_RESOURCE_GAIN: int = 2
    DIG_LICHEN_REMOVED: int = 10
    SELF_DESTRUCT_COST: int = 10
    RUBBLE_AFTER_DESTRUCTION: int = 1
    ACTION_QUEUE_POWER_COST: int = 1

    @classmethod
    def from_lux(cls, lux_unit_config: LuxUnitConfig) -> "UnitConfig":
        return UnitConfig(**dataclasses.asdict(lux_unit_config))

    def to_lux(self) -> LuxUnitConfig:
        return LuxUnitConfig(**self._asdict())


class EnvConfig(NamedTuple):
    ## various options that can be configured if needed

    ### Variable parameters that don't affect game logic much ###
    max_episode_length: int = 1000
    map_size: int = 48
    verbose: int = 1

    # this can be disabled to improve env FPS but assume your actions are well formatted
    # During online competition this is set to True
    validate_action_space: bool = True

    ### Constants ###
    # you can only ever transfer in/out 1000 as this is the max cargo space.
    max_transfer_amount: int = 10000
    MIN_FACTORIES: int = 2
    MAX_FACTORIES: int = 5
    CYCLE_LENGTH: int = 50
    DAY_LENGTH: int = 30
    UNIT_ACTION_QUEUE_SIZE: int = 20  # when set to 1, then no action queue is used

    MAX_RUBBLE: int = 100
    FACTORY_RUBBLE_AFTER_DESTRUCTION: int = 50
    INIT_WATER_METAL_PER_FACTORY: int = 150  # amount of water and metal units given to each factory
    INIT_POWER_PER_FACTORY: int = 1000

    #### LICHEN ####
    MIN_LICHEN_TO_SPREAD: int = 10
    LICHEN_LOST_WITHOUT_WATER: int = 1
    LICHEN_GAINED_WITH_WATER: int = 1
    MAX_LICHEN_PER_TILE: int = 100
    # cost of watering with a factory is `ceil(# of connected lichen tiles) / (this factor) + 1`
    LICHEN_WATERING_COST_FACTOR: int = 10

    #### Bidding System ####
    BIDDING_SYSTEM: bool = True

    #### Factories ####
    FACTORY_PROCESSING_RATE_WATER: int = 50
    ICE_WATER_RATIO: int = 2
    FACTORY_PROCESSING_RATE_METAL: int = 50
    ORE_METAL_RATIO: int = 5
    # game design note: Factories close to resource cluster = more resources are refined per turn
    # Then the high ice:water and ore:metal ratios encourages transfer of refined resources between
    # factories dedicated to mining particular clusters which is more possible as it is more compact
    #

    FACTORY_CHARGE: int = 50
    FACTORY_WATER_CONSUMPTION: int = 1
    # game design note: with a positive water consumption, game becomes quite hard for new competitors.
    # so we set it to 0

    #### Units ####
    ROBOTS: Tuple[UnitConfig, UnitConfig] = (
        # UnitType.LIGHT
        UnitConfig(
            METAL_COST=10,
            POWER_COST=50,
            INIT_POWER=50,
            CARGO_SPACE=100,
            BATTERY_CAPACITY=150,
            CHARGE=1,
            MOVE_COST=1,
            RUBBLE_MOVEMENT_COST=0,
            DIG_COST=5,
            SELF_DESTRUCT_COST=5,
            DIG_RUBBLE_REMOVED=2,
            DIG_RESOURCE_GAIN=2,
            DIG_LICHEN_REMOVED=10,
            RUBBLE_AFTER_DESTRUCTION=1,
            ACTION_QUEUE_POWER_COST=1,
        ),
        # UnitType.HEAVY
        UnitConfig(
            METAL_COST=100,
            POWER_COST=500,
            INIT_POWER=500,
            CARGO_SPACE=1000,
            BATTERY_CAPACITY=3000,
            CHARGE=10,
            MOVE_COST=20,
            RUBBLE_MOVEMENT_COST=1,
            DIG_COST=100,
            SELF_DESTRUCT_COST=100,
            DIG_RUBBLE_REMOVED=20,
            DIG_RESOURCE_GAIN=20,
            DIG_LICHEN_REMOVED=100,
            RUBBLE_AFTER_DESTRUCTION=10,
            ACTION_QUEUE_POWER_COST=10,
        ),
    )

    #### Weather ####
    # WEATHER_ID_TO_NAME: list = dataclasses.field(
    #     default_factory=lambda: ["NONE", "MARS_QUAKE", "COLD_SNAP", "DUST_STORM", "SOLAR_FLARE"])
    NUM_WEATHER_EVENTS_RANGE: Tuple[int, int] = (3, 5)
    WEATHER: WeatherConfig = WeatherConfig(  # pytype: disable=annotation-type-mismatch
        # Weather.NONE
        NONE=(),
        # Weather.MARS_QUAKE
        MARS_QUAKE=jux.weather.MarsQuake(TIME_RANGE=(1, 5)),
        # Weather.COLD_SNAP
        COLD_SNAP=jux.weather.ColdSnap(
            # power multiplier required per robot action. 2 -> requires 2x as much power to execute the same action
            POWER_CONSUMPTION=2,  # must be integer
            TIME_RANGE=(10, 30),
        ),
        # Weather.DUST_STORM
        DUST_STORM=jux.weather.DustStorm(
            # power gain multiplier. .5 -> gain .5x as much power per turn
            POWER_GAIN=0.5,
            TIME_RANGE=(10, 30),
        ),
        # Weather.SOLAR_FLARE
        SOLAR_FLARE=jux.weather.SolarFlare(
            # power gain multiplier. 2 -> gain 2x as much power per turn
            POWER_GAIN=2.0,
            TIME_RANGE=(10, 30),
        ),
    )

    @classmethod
    def from_lux(cls, lux_env_config: LuxEnvConfig):
        lux_env_config = copy(lux_env_config)

        lux_env_config.ROBOTS = (
            UnitConfig.from_lux(lux_env_config.ROBOTS['LIGHT']),
            UnitConfig.from_lux(lux_env_config.ROBOTS['HEAVY']),
        )

        lux_env_config.NUM_WEATHER_EVENTS_RANGE = tuple(lux_env_config.NUM_WEATHER_EVENTS_RANGE)
        lux_env_config.WEATHER = WeatherConfig.from_lux(lux_env_config.WEATHER)

        lux_env_config = dataclasses.asdict(lux_env_config)
        lux_env_config.pop('WEATHER_ID_TO_NAME')
        return EnvConfig(**lux_env_config)

    def to_lux(self) -> LuxEnvConfig:
        self = jax.tree_map(lambda x: x.item() if isinstance(x, jax.Array) else x, self)
        self: Dict[str, Any] = self._asdict()

        self['ROBOTS'] = dict(
            LIGHT=self['ROBOTS'][0].to_lux(),
            HEAVY=self['ROBOTS'][1].to_lux(),
        )
        self['WEATHER'] = self['WEATHER'].to_lux()
        self['NUM_WEATHER_EVENTS_RANGE'] = list(self['NUM_WEATHER_EVENTS_RANGE'])
        self['WEATHER_ID_TO_NAME'] = ["NONE"] + list(self['WEATHER'].keys())

        return LuxEnvConfig(**self)


default = EnvConfig()


class JuxBufferConfig(NamedTuple):
    MAX_N_UNITS: int = 1000
    MAX_GLOBAL_ID: int = 1000 * 2
    MAX_N_FACTORIES: int = default.MAX_FACTORIES + 1
    MAX_MAP_SIZE: int = default.map_size
