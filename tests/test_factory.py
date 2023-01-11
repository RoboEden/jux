import chex
from jax import Array
from jax import numpy as jnp
from luxai_s2.team import FactionTypes as LuxFactionTypes
from luxai_s2.team import Team as LuxTeam

from jux.factory import Factory, LuxFactory, LuxTeam
from jux.map.position import Position
from jux.unit import ResourceType, Unit, UnitCargo


class TestFactory(chex.TestCase):

    @staticmethod
    def create_factory():
        team_id: int = 0
        unit_id: int = 1
        pos: Array = Position()
        power: int = 10
        cargo: UnitCargo = UnitCargo()
        return Factory(
            team_id=team_id,
            unit_id=unit_id,
            pos=pos,
            power=power,
            cargo=cargo,
        )

    @chex.variants(with_jit=True, without_jit=True, with_device=True, without_device=True)
    def test_add_resource(self):
        factory: Factory = self.create_factory()

        transfer_amount = 10
        factory_add_source = self.variant(Factory.add_resource, static_argnames=())
        factory, transfer_amount = factory_add_source(
            factory,
            ResourceType.ice,
            transfer_amount,
        )
        chex.assert_type(transfer_amount, int)
        assert transfer_amount == 10
        assert factory.cargo.ice == 10

    @chex.variants(with_jit=True, without_jit=True, with_device=True, without_device=True)
    def test_sub_resourece(self):
        # key = jax.random.PRNGKey(int(datetime.now().timestamp()))
        amount = 10
        factory: Factory = self.create_factory()

        factory = factory._replace(cargo=factory.cargo._replace(stock=jnp.array([100, 0, 0, 0, 0])))
        factory_sub_resource = self.variant(Unit.sub_resource, static_argnames=())
        factory, transfer_amount = factory_sub_resource(factory, ResourceType.ice, amount)
        chex.assert_type(transfer_amount, int)
        assert transfer_amount == 10
        assert factory.cargo.ice == 90

    def test_from_to_lux(self):
        factory: Factory = self.create_factory()
        lux_factory: LuxFactory = factory.to_lux({'player_0': LuxTeam(
            0,
            'player_0',
            LuxFactionTypes.MotherMars,
        )})
        factory_from_lux: Factory = Factory.from_lux(lux_factory)
        assert factory == factory_from_lux
