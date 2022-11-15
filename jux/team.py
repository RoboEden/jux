from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

from jax import Array
from luxai2022.team import Team as LuxTeam


class FactionTypes(IntEnum):
    AlphaStrike = 0
    MotherMars = 1
    TheBuilders = 2
    FirstMars = 3


@dataclass
class FactionInfo:
    color: str = "none"
    alt_color: str = "red"
    faction_id: int = -1


FactionTypes.AlphaStrike.color = "yellow"
FactionTypes.AlphaStrike.faction_id = int(FactionTypes.AlphaStrike)
FactionTypes.AlphaStrike.alt_color = "red"

FactionTypes.MotherMars.color = "green"
FactionTypes.MotherMars.faction_id = int(FactionTypes.MotherMars)
FactionTypes.MotherMars.alt_color = "red"

FactionTypes.TheBuilders.color = "blue"
FactionTypes.TheBuilders.faction_id = int(FactionTypes.TheBuilders)
FactionTypes.TheBuilders.alt_color = "red"

FactionTypes.FirstMars.color = "red"
FactionTypes.FirstMars.faction_id = int(FactionTypes.FirstMars)
FactionTypes.FirstMars.alt_color = "red"


class Team(NamedTuple):
    faction: FactionTypes
    team_id: int
    # agent: str # weather we need it?
    init_water: int
    init_metal: int
    factories_to_place: int

    # TODO: decide the data structure of factory_strains
    factory_strains: Array  # int[MAX_N_FACTORIES], factory_id belonging to this team
    n_factory: int  # usually MAX_FACTORIES or MAX_FACTORIES + 1

    @classmethod
    def from_lux(self, lux_team: LuxTeam) -> "Team":
        # TODO
        pass

    def to_lux(self) -> LuxTeam:
        # TODO
        pass
