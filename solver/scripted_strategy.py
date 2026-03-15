"""Scripted strategy: plays a pre-computed action plan in the simulator.

The plan is a dict: round_num -> {bot_id: action_str}
Actions: "move_up", "move_down", "move_left", "move_right", "pick_up", "drop_off", "wait"

For pick_up, the strategy resolves item_id from game state (nearest matching item).
"""

from __future__ import annotations
from collections import deque


class ScriptedStrategy:
    """Replays a pre-computed plan in the simulator."""

    def __init__(self, plan: dict):
        """plan: {round_num: {bot_id: action_string}} — keys can be str or int"""
        # Normalize keys to int
        self._plan = {}
        for rk, actions in plan.items():
            r = int(rk)
            self._plan[r] = {int(bk): v for bk, v in actions.items()}

    def __call__(self, state: dict) -> dict:
        round_num = state.get("round", 0)
        bots = state.get("bots", [])
        items = state.get("items", [])

        round_plan = self._plan.get(round_num, {})

        actions = []
        for bot in bots:
            bid = bot["id"]
            action = round_plan.get(bid, "wait")

            entry = {"bot": bid, "action": action}

            # Resolve item_id for pick_up
            if action == "pick_up":
                bot_pos = tuple(bot["position"])
                item_id = self._find_adjacent_item(bot_pos, items)
                if item_id:
                    entry["item_id"] = item_id
                else:
                    entry["action"] = "wait"  # no item to pick

            actions.append(entry)

        return {"actions": actions}

    @staticmethod
    def _find_adjacent_item(bot_pos: tuple, items: list) -> str | None:
        """Find any item adjacent to bot position."""
        bx, by = bot_pos
        for item in items:
            ix, iy = item["position"]
            if abs(bx - ix) + abs(by - iy) == 1:
                return item["id"]
        return None


def plan_from_bot_paths(
    bot_paths: dict[int, list[tuple[int, int]]],
    pickup_rounds: dict[int, list[int]],
    dropoff_rounds: dict[int, list[int]],
) -> dict[int, dict[int, str]]:
    """Convert bot paths + pickup/dropoff schedules into a round-action plan.

    bot_paths: {bot_id: [(x0,y0), (x1,y1), ...]} — position per round
    pickup_rounds: {bot_id: [round_nums where bot picks up]}
    dropoff_rounds: {bot_id: [round_nums where bot drops off]}
    """
    plan: dict[int, dict[int, str]] = {}

    for bot_id, path in bot_paths.items():
        pickups = set(pickup_rounds.get(bot_id, []))
        dropoffs = set(dropoff_rounds.get(bot_id, []))

        for r in range(len(path)):
            if r not in plan:
                plan[r] = {}

            if r in pickups:
                plan[r][bot_id] = "pick_up"
            elif r in dropoffs:
                plan[r][bot_id] = "drop_off"
            elif r > 0:
                prev = path[r - 1]
                curr = path[r]
                dx = curr[0] - prev[0]
                dy = curr[1] - prev[1]
                if dx == 1:
                    plan[r][bot_id] = "move_right"
                elif dx == -1:
                    plan[r][bot_id] = "move_left"
                elif dy == 1:
                    plan[r][bot_id] = "move_down"
                elif dy == -1:
                    plan[r][bot_id] = "move_up"
                else:
                    plan[r][bot_id] = "wait"
            else:
                plan[r][bot_id] = "wait"

    return plan
