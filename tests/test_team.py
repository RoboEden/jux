import jux.team
from jux.config import JuxBufferConfig
from jux.team import FactionTypes, LuxFactionTypes, LuxTeam, Team


def lux_team_eq(a: LuxTeam, b: LuxTeam) -> bool:
    return (a.team_id == b.team_id and a.faction == b.faction and a.init_water == b.init_water
            and a.init_metal == b.init_metal and a.factories_to_place == b.factories_to_place
            and a.factory_strains == b.factory_strains)


class TestTeam:

    def test_from_to_lux(self):
        buf_cfg = JuxBufferConfig()

        lux_team = LuxTeam(0, 'player_0', LuxFactionTypes.AlphaStrike)
        assert lux_team_eq(
            lux_team,
            Team.from_lux(lux_team, buf_cfg).to_lux(),
        )

        jux_team = Team.new(0, FactionTypes.FirstMars, buf_cfg)
        assert jux_team == Team.from_lux(jux_team.to_lux(), buf_cfg)
