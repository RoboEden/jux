from enum import IntEnum
from typing import NamedTuple, Tuple

from jax import lax


class MapType(IntEnum):
    CAVE = 0
    CRATERS = 1
    ISLAND = 2
    MOUNTAIN = 3

    @classmethod
    def from_lux(cls, map_type: str) -> "MapType":
        idx = ["Cave", "Craters", "Island", "Mountain"].index(map_type)
        return cls(idx)

    def to_lux(self) -> str:
        return ["Cave", "Craters", "Island", "Mountain"][self.value]


class MapDistributionType(IntEnum):
    HIGH_ICE_HIGH_ORE = 0
    HIGH_ICE_LOW_ORE = 1
    LOW_ICE_HIGH_ORE = 2
    LOW_ICE_LOW_ORE = 3

    @classmethod
    def from_lux(cls, map_distribution_type: str) -> "MapDistributionType":
        idx = [
            "high_ice_high_ore",
            "high_ice_low_ore",
            "low_ice_high_ore",
            "low_ice_low_ore",
        ].index(map_distribution_type)
        return cls(idx)

    def to_lux(self) -> str:
        return [
            "high_ice_high_ore",
            "high_ice_low_ore",
            "low_ice_high_ore",
            "low_ice_low_ore",
        ][self.value]


class CaveConfig(NamedTuple):
    ice_high_range: Tuple[float, float] = (99., 100.)
    ice_mid_range: Tuple[float, float] = (91., 91.7)
    ore_high_range: Tuple[float, float] = (98.7, 100.)
    ore_mid_range: Tuple[float, float] = (81., 81.5)

    @staticmethod
    def new(map_distribution_type: MapDistributionType):
        cave_config = lax.switch(
            map_distribution_type,
            [
                lambda: CaveConfig(),  # high_ice_high_ore
                lambda: CaveConfig(
                    ore_high_range=(99.5, 100.),
                    ore_mid_range=(81., 81.4),
                ),  # high_ice_low_ore
                lambda: CaveConfig(
                    ice_high_range=(99.6, 100.),
                    ice_mid_range=(91., 91.5),
                ),  # low_ice_high_ore
                lambda: CaveConfig(
                    ore_high_range=(99.5, 100.),
                    ore_mid_range=(81., 81.4),
                    ice_high_range=(99.6, 100.),
                    ice_mid_range=(91., 91.5),
                )  # low_ice_low_ore
            ],
        )
        return cave_config


class MountainConfig(NamedTuple):

    # controls amount of ice spread along mountain peaks
    ice_high_range: Tuple[float, float] = (98.9, 100.)

    # around middle level
    ice_mid_range: Tuple[float, float] = (52.5, 53.)

    # around lower level
    ice_low_range: Tuple[float, float] = (0., 21.)

    # controls amount of ore spread along middle of the way to the mountain peaks
    ore_mid_range: Tuple[float, float] = (84., 85.)

    # controls amount of ore spread along lower part of the mountain peaks,
    # should be smaller than ore_mid_range
    ore_low_range: Tuple[float, float] = (61.4, 62.)

    @staticmethod
    def new(map_distribution_type: MapDistributionType):
        cave_config = lax.switch(
            map_distribution_type,
            [
                lambda: MountainConfig(),  # high_ice_high_ore
                lambda: MountainConfig(
                    ore_low_range=(61.6, 62.),
                    ore_mid_range=(84.5, 85.),
                ),  # high_ice_low_ore
                lambda: MountainConfig(
                    ice_high_range=(99.4, 100.),
                    ice_mid_range=(52.7, 53.),
                    ice_low_range=(0., 20.),
                ),  # low_ice_high_ore
                lambda: MountainConfig(
                    ice_high_range=(99.4, 100.),
                    ice_mid_range=(52.7, 53.),
                    ice_low_range=(0., 20.),
                    ore_low_range=(61.6, 62.),
                    ore_mid_range=(84.5, 85.),
                )  # low_ice_low_ore
            ],
        )
        return cave_config
