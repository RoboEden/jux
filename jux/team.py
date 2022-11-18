from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from luxai2022.team import FactionTypes as LuxFactionTypes
from luxai2022.team import Team as LuxTeam

from jux.config import JuxBufferConfig


class FactionTypes(IntEnum):
    AlphaStrike = 0
    MotherMars = 1
    TheBuilders = 2
    FirstMars = 3

    @classmethod
    def from_lux(cls, lux_faction: LuxFactionTypes) -> "FactionTypes":
        return cls(lux_faction.value.faction_id)

    def to_lux(self) -> LuxFactionTypes:
        return LuxFactionTypes[self.name]


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

    factory_strains: Array  # int[MAX_N_FACTORIES], factory_id belonging to this team
    n_factory: int  # usually MAX_FACTORIES or MAX_FACTORIES + 1

    @classmethod
    def new(cls, team_id: int, faction: FactionTypes, buf_cfg: JuxBufferConfig) -> "Team":
        return cls(
            faction=faction,
            team_id=team_id,
            init_water=0,
            init_metal=0,
            factories_to_place=0,
            factory_strains=jnp.empty(buf_cfg.MAX_N_FACTORIES, dtype=jnp.int32),
            n_factory=0,
        )

    @classmethod
    def from_lux(cls, lux_team: LuxTeam, buf_cfg: JuxBufferConfig) -> "Team":
        factory_strains = jnp.empty(buf_cfg.MAX_N_FACTORIES, dtype=jnp.int32)

        n_factory = len(lux_team.factory_strains)
        factory_strains = factory_strains.at[:n_factory].set(jnp.array(lux_team.factory_strains, dtype=jnp.int32))
        return cls(
            faction=FactionTypes.from_lux(lux_team.faction),
            team_id=lux_team.team_id,
            init_water=lux_team.init_water,
            init_metal=lux_team.init_metal,
            factories_to_place=lux_team.factories_to_place,
            factory_strains=factory_strains,
            n_factory=n_factory,
        )

    def to_lux(self) -> LuxTeam:
        lux_team = LuxTeam(
            team_id=int(self.team_id),
            agent=f'player_{int(self.team_id)}',
            faction=FactionTypes(self.faction).to_lux(),
        )
        lux_team.init_water = self.init_water
        lux_team.init_metal = self.init_metal
        lux_team.factories_to_place = self.factories_to_place
        lux_team.factory_strains = self.factory_strains[:self.n_factory].tolist()
        return lux_team

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Team):
            return False
        return (self.faction == __o.faction and self.team_id == __o.team_id and self.init_water == __o.init_water
                and self.init_metal == __o.init_metal and self.factories_to_place == __o.factories_to_place
                and self.n_factory == __o.n_factory
                and jnp.array_equal(self.factory_strains[:self.n_factory], __o.factory_strains[:__o.n_factory]))
