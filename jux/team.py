from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple, Union

import jax.numpy as jnp
from jax import Array
from luxai_s2.team import FactionTypes as LuxFactionTypes
from luxai_s2.team import Team as LuxTeam

from jux.config import JuxBufferConfig
from jux.utils import INT32_MAX, imax


class FactionTypes(IntEnum):
    AlphaStrike = 0
    MotherMars = 1
    TheBuilders = 2
    FirstMars = 3

    @classmethod
    def from_lux(cls, lux_faction: Union[str, LuxFactionTypes]) -> "FactionTypes":
        if isinstance(lux_faction, str):
            return cls[lux_faction]
        elif isinstance(lux_faction, LuxFactionTypes):
            return cls(lux_faction.value.faction_id)
        else:
            raise ValueError(f"Unsupport type {type(lux_faction)}: {lux_faction}.")

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
    team_id: jnp.int8
    faction: FactionTypes
    init_water: jnp.int32
    init_metal: jnp.int32
    factories_to_place: jnp.int32

    # This is not duplicated with State.factories.unit_id and State.n_factories,
    # because factory may be destroyed, but a destroyed factory's lichen is still in the game.
    # To correctly count lichen, we need to keep track of all factories that have been placed.
    factory_strains: jnp.int8  # int[MAX_N_FACTORIES], factory_id belonging to this team
    n_factory: jnp.int8

    bid: jnp.int32

    @classmethod
    def new(
        cls,
        team_id: int,
        faction: Union[FactionTypes, int] = 0,
        init_water: int = 0,
        init_metal: int = 0,
        factories_to_place: int = 0,
        factory_strains: Union[Array, None] = None,
        n_factory: int = 0,
        bid: int = 0,
        *,
        buf_cfg: Union[JuxBufferConfig, None] = None,
    ) -> "Team":
        if buf_cfg is None:
            if factory_strains is None:
                raise ValueError("Either buf_cfg or factory_strains must be provided.")
        elif factory_strains is None:
            factory_strains = jnp.full(buf_cfg.MAX_N_FACTORIES,
                                       fill_value=imax(Team.__annotations__['factory_strains']))
        return cls(
            team_id=Team.__annotations__['team_id'](team_id),
            faction=jnp.int8(faction),
            init_water=Team.__annotations__['init_water'](init_water),
            init_metal=Team.__annotations__['init_metal'](init_metal),
            factories_to_place=Team.__annotations__['factories_to_place'](factories_to_place),
            factory_strains=Team.__annotations__['factory_strains'](factory_strains),
            n_factory=Team.__annotations__['n_factory'](n_factory),
            bid=Team.__annotations__['bid'](bid),
        )

    @classmethod
    def from_lux(cls, lux_team: LuxTeam, buf_cfg: JuxBufferConfig) -> "Team":
        strains_dtype = Team.__annotations__['n_factory']
        factory_strains = jnp.full(buf_cfg.MAX_N_FACTORIES, fill_value=imax(strains_dtype))

        n_factory = len(lux_team.factory_strains)
        factory_strains = factory_strains.at[:n_factory].set(jnp.array(lux_team.factory_strains, dtype=strains_dtype))

        return cls.new(
            team_id=lux_team.team_id,
            faction=FactionTypes.from_lux(lux_team.faction),
            init_water=lux_team.init_water,
            init_metal=lux_team.init_metal,
            factories_to_place=lux_team.factories_to_place,
            factory_strains=factory_strains,
            n_factory=n_factory,
            bid=lux_team.bid,
        )

    def to_lux(
        self,
        place_first: bool,
    ) -> LuxTeam:
        lux_team = LuxTeam(
            team_id=int(self.team_id),
            agent=f'player_{int(self.team_id)}',
            faction=FactionTypes(self.faction).to_lux(),
        )
        lux_team.init_water = int(self.init_water)
        lux_team.init_metal = int(self.init_metal)
        lux_team.factories_to_place = int(self.factories_to_place)
        lux_team.factory_strains = self.factory_strains[:self.n_factory].tolist()
        lux_team.place_first = place_first
        lux_team.bid = int(self.bid)
        return lux_team

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Team):
            return False
        return ((self.faction == __o.faction) & (self.team_id == __o.team_id) & (self.init_water == __o.init_water)
                & (self.init_metal == __o.init_metal) & (self.factories_to_place == __o.factories_to_place) &
                (self.bid == __o.bid))
