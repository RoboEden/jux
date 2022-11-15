import chex
from jax import numpy as jnp

from jux.unit import EnvConfig, ResourceType, Unit, UnitType


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
