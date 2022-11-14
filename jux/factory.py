from typing import NamedTuple, Union

from jax import Array
from luxai2022.factory import Factory as LuxFactory

from jux.unit import ResourceType, UnitCargo


class Factory(NamedTuple):
    team_id: int
    # team # no need team object, team_id is enough
    unit_id: int
    pos: Array  # int16[2]
    power: int
    cargo: UnitCargo  # int[4]
    num_id: int

    # action_queue # Do we need action queue for factories?

    def add_resource(self, resource: ResourceType, amount: int) -> Union[int, Array]:
        # TODO
        # If resource != ResourceType.power, call UnitCargo.add_resource.
        # else, call Unit.add_power.
        pass

    def sub_resource(self, resource: ResourceType, amount: int) -> Union[int, Array]:
        # TODO
        # If resource != ResourceType.power, call UnitCargo.add_resource.
        # else, call Unit.sub_resource.
        pass

    @classmethod
    def from_lux(cls, lux_factory: LuxFactory) -> "Factory":
        # TODO
        pass

    def to_lux(self) -> LuxFactory:
        # TODO
        pass
