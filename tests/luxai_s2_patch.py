from dataclasses import dataclass
from typing import Callable

import numpy as np
from luxai_s2.factory import Factory, Position, Team, UnitCargo
from luxai_s2.unit import Unit


def Factory__init__(self, team: Team, unit_id: str, num_id: int) -> None:
    self.team_id = team.team_id
    self.team = team
    self.unit_id = unit_id
    self.pos = Position(np.zeros(2, dtype=int))
    self.power = 0
    self.cargo = UnitCargo()
    self.num_id = num_id
    self.action_queue = []
    self.grow_lichen_positions = []


def Unit_repeat_action(self, action):
    action.n -= 1
    if action.n <= 0:
        # remove from front of queue
        self.action_queue.pop(0)
        # endless repeat puts action back at end of queue
        if action.repeat:
            action.n = 1
            self.action_queue.append(action)


@dataclass
class OriginalAPI:
    Factory__init__: Callable = Factory.__init__
    Unit_repeat_action: Callable = Unit.repeat_action


def install_patches():
    Factory.__init__ = Factory__init__
    Unit.repeat_action = Unit_repeat_action


def uninstall_patches():
    Factory.__init__ = OriginalAPI.Factory__init__
    Unit.repeat_action = OriginalAPI.Unit_repeat_action
