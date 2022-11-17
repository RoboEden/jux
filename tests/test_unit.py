import chex
from jax import numpy as jnp
from luxai2022.team import FactionTypes as LuxFactionTypes

from jux.config import EnvConfig
from jux.team import LuxTeam
from jux.unit import LuxUnit, LuxUnitType, Unit, UnitType
from jux.unit_cargo import ResourceType


class TestUnit(chex.TestCase):

    @staticmethod
    def create_unit():
        team_id: int = 0
        unit_type: UnitType = 1
        unit_id: int = 1
        env_cfg: EnvConfig = EnvConfig()
        return Unit.new(team_id=team_id, unit_type=unit_type, unit_id=unit_id, env_cfg=env_cfg)

    @chex.variants(with_jit=True, without_jit=True, with_device=True, without_device=True)
    def test_add_resourece(self):
        # key = jax.random.PRNGKey(int(datetime.now().timestamp()))
        amount = 10

        unit: Unit = self.create_unit()
        unit_add_resource = self.variant(Unit.add_resource, static_argnames=())
        unit, transfer_amount = unit_add_resource(
            unit,
            ResourceType.ice,
            amount,
        )
        chex.assert_type(transfer_amount, int)
        assert transfer_amount == 10
        assert unit.cargo.ice == 10

    @chex.variants(
        with_jit=True,
        without_jit=True,
        with_device=True,
        without_device=True,
    )
    def test_sub_resourece(self):
        # key = jax.random.PRNGKey(int(datetime.now().timestamp()))
        amount = 10
        unit: Unit = self.create_unit()

        unit = unit._replace(cargo=unit.cargo._replace(stock=jnp.array([100, 0, 0, 0, 0])))
        unit_sub_resource = self.variant(Unit.sub_resource, static_argnames=())
        unit, transfer_amount = unit_sub_resource(unit, ResourceType.ice, amount)
        chex.assert_type(transfer_amount, int)
        assert transfer_amount == 10
        assert unit.cargo.ice == 90

    def test_to_from_lux(self):
        env_cfg = EnvConfig()
        lux_env_cfg = env_cfg.to_lux()
        lux_teams = [LuxTeam(0, LuxFactionTypes.AlphaStrike)]
        lux_unit = LuxUnit(team=lux_teams[0], unit_type=LuxUnitType.HEAVY, unit_id="unit_1", env_cfg=lux_env_cfg)

        jux_unit = Unit.from_lux(lux_unit, env_cfg)
        assert jux_unit == Unit.from_lux(jux_unit.to_lux(lux_teams, lux_env_cfg), env_cfg)
