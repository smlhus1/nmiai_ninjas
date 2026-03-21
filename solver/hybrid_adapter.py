"""HybridAdapter: mixed MAPF (frozen) + reactive (free) bot execution.

For each round:
1. Frozen bots follow pre-computed MAPF plan actions
2. Free bots run through BotAdapter (full reactive pipeline)
3. Frozen bots' next positions injected as temporary walls for free bots' grid

Used by LNS: freeze most bots, re-plan a few, splice improvements back.
"""
from __future__ import annotations

import copy
import logging
from typing import Any

from mapf_planner import MAPFAction, DELTAS

logger = logging.getLogger(__name__)


def _apply_action(pos: tuple[int, int], action: str) -> tuple[int, int]:
    """Compute next position given current position and action string."""
    delta = DELTAS.get(action)
    if delta:
        return (pos[0] + delta[0], pos[1] + delta[1])
    return pos


class HybridAdapter:
    """Run frozen bots from MAPF plan + free bots from BotAdapter reactively.

    Args:
        plan_actions: dict[bot_id, list[MAPFAction]] — full plan for all bots
        free_bots: set of bot IDs that should run reactively (not from plan)
        config: CoordinatorConfig for the reactive BotAdapter
    """

    def __init__(
        self,
        plan_actions: dict[int, list["MAPFAction"]],
        free_bots: set[int],
        config=None,
    ):
        self._plan_actions = plan_actions
        self._free_bots = free_bots
        self._frozen_bots = set(plan_actions.keys()) - free_bots
        self._config = config
        self._round = 0
        self._adapter = None  # Lazy init

    def __call__(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Strategy interface for Simulator.run()."""
        from Simulering.offline.bot_adapter import BotAdapter

        if self._adapter is None:
            self._adapter = BotAdapter(suppress_logs=True, config=self._config)

        round_idx = self._round
        self._round += 1

        # 1. Get frozen bots' actions from plan
        frozen_actions: dict[int, dict] = {}
        frozen_next_positions: set[tuple[int, int]] = set()

        for bid in self._frozen_bots:
            actions = self._plan_actions.get(bid, [])
            if round_idx < len(actions):
                act = actions[round_idx]
                pos = tuple(act.position)
                action = act.action
                item_type = act.item_type

                next_pos = _apply_action(pos, action)
                frozen_next_positions.add(next_pos)

                frozen_actions[bid] = {
                    "bot": bid,
                    "action": action,
                }
                if action == "pick_up" and item_type:
                    # Find item at this position matching type
                    for item in state_dict.get("items", []):
                        if item["type"] == item_type and tuple(item["position"]) == pos:
                            frozen_actions[bid]["item_id"] = item["id"]
                            break
            else:
                # Plan exhausted — wait
                frozen_actions[bid] = {"bot": bid, "action": "wait"}

        # 2. Build modified state for free bots (inject frozen positions as walls)
        if self._free_bots:
            free_state = copy.deepcopy(state_dict)

            # Add frozen bots' next positions as temporary walls
            # This prevents free bots from moving into frozen bots' paths
            existing_walls = set(tuple(w) for w in free_state["grid"]["walls"])
            for pos in frozen_next_positions:
                existing_walls.add(pos)
            free_state["grid"]["walls"] = [list(w) for w in existing_walls]

            # Remove frozen bots from the state so adapter doesn't plan for them
            free_state["bots"] = [
                b for b in free_state["bots"] if b["id"] in self._free_bots
            ]

            # Get reactive actions for free bots
            reactive_response = self._adapter(free_state)
            reactive_actions = {
                a["bot"]: a for a in reactive_response.get("actions", [])
            }
        else:
            reactive_actions = {}

        # 3. Merge: frozen plan + reactive free
        all_actions = []
        for bot_data in state_dict.get("bots", []):
            bid = bot_data["id"]
            if bid in frozen_actions:
                all_actions.append(frozen_actions[bid])
            elif bid in reactive_actions:
                all_actions.append(reactive_actions[bid])
            else:
                all_actions.append({"bot": bid, "action": "wait"})

        return {"actions": all_actions}

    def reset(self):
        """Reset for new game."""
        self._round = 0
        if self._adapter:
            self._adapter.reset()
        self._adapter = None
