from collections import deque
from typing import Dict, List, Tuple

import numpy as np
from luxai2022 import LuxAI2022
from luxai2022.actions import Action, TransferAction, move_deltas
from luxai2022.config import EnvConfig
from luxai2022.factory import Factory
from luxai2022.map.board import Board
from luxai2022.map.position import Position
from luxai2022.unit import Unit, UnitType

ActionsByType = Dict[str, List[Tuple[Unit, Action]]]


def install_patches():
    LuxAI2022._handle_factory_placement_step = _handle_factory_placement_step
    LuxAI2022._handle_transfer_actions = _handle_transfer_actions
    Factory.cache_water_info = cache_water_info


def _handle_factory_placement_step(self, actions):
    # factory placement rounds, which are sequential

    player_to_place_factory: str
    if self.state.teams["player_0"].place_first:
        if self.state.env_steps % 2 == 1:
            player_to_place_factory = "player_0"
        else:
            player_to_place_factory = "player_1"
    else:
        if self.state.env_steps % 2 == 1:
            player_to_place_factory = "player_1"
        else:
            player_to_place_factory = "player_0"

    failed_agents = {agent: False for agent in self.agents}
    for k, a in actions.items():
        if a is None:
            failed_agents[k] = True
            continue
        if k not in self.agents:
            raise ValueError(f"Invalid player {k}")
        if "spawn" in a and "metal" in a and "water" in a:
            if k != player_to_place_factory:
                self._log(f"{k} tried to perform an action in the early phase when it is not its turn right now.")
                continue
            if self.state.teams[k].factories_to_place <= 0:
                self._log(f"{k} cannot place additional factories. Cancelled placement of factory")
                continue
            if a["water"] < 0 or a["metal"] < 0:
                self._log(f"{k} tried to place negative water/metal in factory. Cancelled placement of factory")
                continue
            if a["water"] > self.state.teams[k].init_water:
                a["water"] = self.state.teams[k].init_water
                self._log(f" Warning - {k} does not have enough water. Using {a['water']}")
            if a["metal"] > self.state.teams[k].init_metal:
                a["metal"] = self.state.teams[k].init_metal
                self._log(f" Warning - {k} does not have enough metal. Using {a['metal']}")
            factory = self.add_factory(self.state.teams[k], a["spawn"])
            if factory is None: continue
            factory.cargo.water = a["water"]
            factory.cargo.metal = a["metal"]
            factory.power = self.env_cfg.INIT_POWER_PER_FACTORY
            self.state.teams[k].factories_to_place -= 1
            self.state.teams[k].init_metal -= a["metal"]
            self.state.teams[k].init_water -= a["water"]
        else:
            # pass, turn is skipped.
            pass
    return failed_agents


def compute_water_info(init: np.ndarray, MIN_LICHEN_TO_SPREAD: int, lichen: np.ndarray, lichen_strains: np.ndarray,
                       factory_occupancy_map: np.ndarray, strain_id: int, forbidden: np.ndarray):
    # TODO - improve the performance here with cached solution
    frontier = deque(init)
    seen = set(map(tuple, init))
    grow_lichen_positions = set()
    H, W = lichen.shape
    ct = 0
    while len(frontier) > 0:
        ct += 1
        if ct > 1_000_000:
            print("Error! Lichen Growth calculation took too long")
            break
        pos = frontier.popleft()
        if pos[0] < 0 or pos[1] < 0 or pos[0] >= forbidden.shape[0] or pos[1] >= forbidden.shape[1]:
            continue

        if forbidden[pos[0], pos[1]]:
            continue
        pos_lichen = lichen[pos[0], pos[1]]
        pos_strain = lichen_strains[pos[0], pos[1]]
        # check for surrounding tiles with lichen and no incompatible lichen strains, grow on those
        can_grow = True
        for move_delta in move_deltas[1:]:
            check_pos = pos + move_delta
            # check surrounding tiles on the map
            if check_pos[0] < 0 or check_pos[1] < 0 or check_pos[0] >= H or check_pos[1] >= W: continue

            # If any neighbor 1. has a different strain, or 2. is a different factory,
            # then the current pos cannot grow
            adj_strain = lichen_strains[check_pos[0], check_pos[1]]
            adj_factory = factory_occupancy_map[check_pos[0], check_pos[1]]
            if (adj_strain != -1 and adj_strain != strain_id) \
                or (adj_factory != -1 and adj_factory != strain_id):
                can_grow = False

            # if seen, skip
            if (check_pos[0], check_pos[1]) in seen:
                continue

            # we add it to the frontier only in two cases:
            #  1. it is an empty tile, and current pos has enough lichen to expand.
            #  2. both current tile and check_pos are of our strain.
            if (adj_strain == -1 and pos_lichen >= MIN_LICHEN_TO_SPREAD) \
                or (adj_strain == strain_id and pos_strain == strain_id):
                seen.add(tuple(check_pos))
                frontier.append(check_pos)

        if can_grow or (lichen_strains[pos[0], pos[1]] == strain_id):
            grow_lichen_positions.add((pos[0], pos[1]))
    return grow_lichen_positions


def cache_water_info(self, board: Board, env_cfg: EnvConfig):
    # TODO this can easily be a fairly slow function, can we make it much faster?
    # Caches information about which tiles lichen can grow on for this factory

    # perform a BFS from the factory position and look for non rubble, non factory tiles.
    # find the current frontier from 12 starting positions x marked below
    """
        x x x
        _ _ _
    x |     | x
    x |     | x
    x |_ _ _| x
        x x x
    """
    forbidden = (board.rubble > 0) | (board.factory_occupancy_map != -1) | (board.ice > 0) | (board.ore > 0)
    deltas = [
        np.array([0, -2]),
        np.array([-1, -2]),
        np.array([1, -2]),
        np.array([0, 2]),
        np.array([-1, 2]),
        np.array([1, 2]),
        np.array([2, 0]),
        np.array([2, -1]),
        np.array([2, 1]),
        np.array([-2, 0]),
        np.array([-2, -1]),
        np.array([-2, 1])
    ]
    init_arr = np.stack(deltas) + self.pos.pos
    self.grow_lichen_positions = compute_water_info(
        init_arr,
        env_cfg.MIN_LICHEN_TO_SPREAD,
        board.lichen,
        board.lichen_strains,
        board.factory_occupancy_map,
        self.num_id,
        forbidden,
    )


def _handle_transfer_actions(self, actions_by_type: ActionsByType):
    # It is important to first sub resource from all units, and then add
    # resource to targets. Only When splitted into two loops, the transfer
    # action is irrelevant to unit id.

    # sub from unit cargo
    amount_list = []
    for unit, transfer_action in actions_by_type["transfer"]:
        transfer_action: TransferAction
        transfer_amount = unit.sub_resource(transfer_action.resource, transfer_action.transfer_amount)
        amount_list.append(transfer_amount)

    # add to target cargo
    for (unit, transfer_action), transfer_amount in zip(actions_by_type["transfer"], amount_list):
        transfer_action: TransferAction
        transfer_pos: Position = unit.pos + move_deltas[transfer_action.transfer_dir]
        units_there = self.state.board.get_units_at(transfer_pos)

        # if there is a factory, we prefer transferring to that entity
        factory_id = f"factory_{self.state.board.factory_occupancy_map[transfer_pos.x, transfer_pos.y]}"
        if factory_id in self.state.factories[unit.team.agent]:
            factory = self.state.factories[unit.team.agent][factory_id]
            factory.add_resource(transfer_action.resource, transfer_amount)
        elif units_there is not None:
            assert len(units_there) == 1, "Fatal error here, this is a bug"
            target_unit = units_there[0]
            # add resources to target. This will waste (transfer_amount - actually_transferred) resources
            target_unit.add_resource(transfer_action.resource, transfer_amount)
        unit.repeat_action(transfer_action)
