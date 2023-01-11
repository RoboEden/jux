import chex
import jax
from jax import numpy as jnp
from luxai_s2.team import FactionTypes as LuxFactionTypes

from jux.config import EnvConfig
from jux.team import LuxTeam
from jux.tree_util import batch_into_leaf
from jux.unit import LuxUnit, LuxUnitType, Unit, UnitType
from jux.unit_cargo import ResourceType, UnitCargo


class TestUnit(chex.TestCase):

    @staticmethod
    def create_unit(env_cfg):
        team_id: int = 0
        unit_type: UnitType = UnitType.LIGHT
        unit_id: int = 1
        return Unit.new(team_id=team_id, unit_type=unit_type, unit_id=unit_id, env_cfg=env_cfg)

    @chex.variants(with_jit=True, without_jit=True, with_device=True, without_device=True)
    def test_add_resourece(self):
        # key = jax.random.PRNGKey(int(datetime.now().timestamp()))
        env_cfg = EnvConfig()
        amount = 10

        unit: Unit = self.create_unit(env_cfg)

        unit_add_resource = self.variant(Unit.add_resource, static_argnames=())
        unit, transfer_amount = unit_add_resource(
            unit,
            ResourceType.ice,
            amount,
            unit_cfgs=env_cfg.ROBOTS,
        )
        chex.assert_type(transfer_amount, int)
        assert transfer_amount == 10
        assert unit.cargo.ice == 10

        unit: Unit = self.create_unit(env_cfg)
        cargo = UnitCargo(stock=jnp.array([80, 70, 60, 50]))
        units: Unit = batch_into_leaf([
            unit._replace(cargo=cargo),
            unit._replace(cargo=cargo),
            unit._replace(cargo=cargo),
            unit._replace(power=60),
            unit._replace(power=60),
        ])
        res_type = jnp.array([1, 2, 3, 4, 4])
        amount = jnp.array([400, -10, 50, 20, 100])
        new_units, transfer_amount = self.variant(jax.vmap(Unit.add_resource, in_axes=(0, 0, 0, None)))(
            units,
            res_type,
            amount,
            env_cfg.ROBOTS,
        )

        assert (transfer_amount == jnp.array([30, 0, 50, 20, 90])).all()
        assert (new_units.cargo.stock == jnp.array([
            [80, 100, 60, 50],
            [80, 70, 60, 50],
            [80, 70, 60, 100],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])).all()
        assert (new_units.power == jnp.array([50, 50, 50, 80, 150])).all()

    @chex.variants(
        with_jit=True,
        without_jit=True,
        with_device=True,
        without_device=True,
    )
    def test_sub_resourece(self):
        # key = jax.random.PRNGKey(int(datetime.now().timestamp()))
        env_cfg = EnvConfig()
        amount = 10
        unit: Unit = self.create_unit(env_cfg)

        unit = unit._replace(cargo=unit.cargo._replace(stock=jnp.array([100, 0, 0, 0, 0])))
        unit_sub_resource = self.variant(Unit.sub_resource, static_argnames=())
        unit, transfer_amount = unit_sub_resource(unit, ResourceType.ice, amount)
        chex.assert_type(transfer_amount, int)
        assert transfer_amount == 10
        assert unit.cargo.ice == 90

    def test_to_from_lux(self):
        env_cfg = EnvConfig()
        lux_env_cfg = env_cfg.to_lux()
        lux_teams = {'player_0': LuxTeam(0, LuxFactionTypes.AlphaStrike)}
        lux_unit = LuxUnit(team=lux_teams['player_0'],
                           unit_type=LuxUnitType.HEAVY,
                           unit_id="unit_1",
                           env_cfg=lux_env_cfg)

        jux_unit = Unit.from_lux(lux_unit, env_cfg)
        assert jux_unit == Unit.from_lux(jux_unit.to_lux(lux_teams, lux_env_cfg), env_cfg)
