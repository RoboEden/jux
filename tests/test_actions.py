from luxai2022 import actions as lux_actions

import jux.actions
from jux.actions import Direction, FactoryAction, ResourceType, UnitAction

# fix a bug in the luxai2022.actions module
lux_actions.FactoryWaterAction.state_dict = lambda self: 2


class TestActions:

    def test_factory_actions(self):
        actions = [
            FactoryAction.BUILD_LIGHT,
            FactoryAction.BUILD_HEAVY,
            FactoryAction.WATER,
        ]
        for act in actions:
            assert act == FactoryAction.from_lux(act.to_lux())

    def test_unit_actions(self):
        actions = [
            UnitAction.move(Direction.UP, repeat=True),
            UnitAction.transfer(Direction.LEFT, ResourceType.ice, 10, repeat=True),
            UnitAction.pickup(ResourceType.metal, 10, repeat=False),
            UnitAction.dig(True),
            UnitAction.self_destruct(False),
            UnitAction.recharge(20, False),
        ]
        for act in actions:
            assert act == UnitAction.from_lux(act.to_lux())
