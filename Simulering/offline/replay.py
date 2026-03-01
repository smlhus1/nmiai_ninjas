"""
Replay executor for Grocery Bot.
Takes a pre-computed action sequence and executes it during a live game.

Key responsibility: match planned item_type+shelf_pos to runtime item_ids,
since IDs change between runs even though positions are deterministic.

Usage:
    replayer = ReplayExecutor(planned_actions)
    response = replayer.execute(game_state_dict)
"""
from __future__ import annotations
from typing import Optional


class ReplayExecutor:
    """
    Executes a pre-computed action plan against live game state.
    
    Handles:
    - Matching planned shelf positions to runtime item IDs
    - Detecting state divergence
    - Falling back to reactive bot on divergence
    """

    def __init__(self, planned_actions: list[dict],
                 fallback_fn=None):
        """
        planned_actions: list of per-round actions from OfflinePlanner.
        fallback_fn: callable(game_state) -> response dict. Used when
                     plan diverges from reality.
        """
        self.plan = planned_actions
        self.plan_index = 0
        self.fallback = fallback_fn
        self.diverged = False
        self.stats = {
            "planned_rounds": 0,
            "fallback_rounds": 0,
            "item_matches": 0,
            "item_misses": 0,
        }

    def execute(self, state: dict) -> dict:
        """
        Given a game_state, return the next action(s).
        Returns {"actions": [...]} ready for WebSocket send.
        """
        if self.diverged and self.fallback:
            self.stats["fallback_rounds"] += 1
            return self.fallback(state)

        if self.plan_index >= len(self.plan):
            # Plan exhausted — endgame or ran out of orders
            if self.fallback:
                self.stats["fallback_rounds"] += 1
                return self.fallback(state)
            return {"actions": [{"bot": 0, "action": "wait"}]}

        planned = self.plan[self.plan_index]
        action = dict(planned)  # copy

        # If it's a pick_up, resolve item_id from shelf position
        if action["action"] == "pick_up":
            item_id = self._resolve_item_id(
                state,
                action.get("item_type"),
                action.get("shelf_pos"),
                action["bot"],
            )
            if item_id:
                action["item_id"] = item_id
                self.stats["item_matches"] += 1
            else:
                # Can't find the planned item — might be picked already
                # or state diverged
                self.stats["item_misses"] += 1
                if self._should_fallback(state, action):
                    self.diverged = True
                    if self.fallback:
                        self.stats["fallback_rounds"] += 1
                        return self.fallback(state)
                    action["action"] = "wait"

        # Clean up planning metadata before sending
        clean_action = {
            "bot": action["bot"],
            "action": action["action"],
        }
        if "item_id" in action:
            clean_action["item_id"] = action["item_id"]

        self.plan_index += 1
        self.stats["planned_rounds"] += 1

        return {"actions": [clean_action]}

    def execute_multi_bot(self, state: dict,
                          all_plans: dict[int, list[dict]],
                          plan_indices: dict[int, int]
                          ) -> dict:
        """
        Execute plans for multiple bots simultaneously.
        
        all_plans: {bot_id: [actions]}
        plan_indices: {bot_id: current_index} (mutable, updated in place)
        """
        actions = []
        num_bots = len(state["bots"])

        for bot_id in range(num_bots):
            plan = all_plans.get(bot_id, [])
            idx = plan_indices.get(bot_id, 0)

            if idx >= len(plan):
                actions.append({"bot": bot_id, "action": "wait"})
                continue

            planned = dict(plan[idx])

            if planned["action"] == "pick_up":
                item_id = self._resolve_item_id(
                    state,
                    planned.get("item_type"),
                    planned.get("shelf_pos"),
                    bot_id,
                )
                if item_id:
                    planned["item_id"] = item_id
                else:
                    planned["action"] = "wait"

            clean = {"bot": bot_id, "action": planned["action"]}
            if "item_id" in planned:
                clean["item_id"] = planned["item_id"]
            actions.append(clean)
            plan_indices[bot_id] = idx + 1

        return {"actions": actions}

    def _resolve_item_id(self, state: dict, item_type: Optional[str],
                         shelf_pos: Optional[list], bot_id: int
                         ) -> Optional[str]:
        """
        Find the actual item_id for a planned pickup.
        Match by type + position (deterministic map).
        """
        if not item_type or not shelf_pos:
            return None

        target_pos = tuple(shelf_pos) if isinstance(shelf_pos, list) else shelf_pos

        # Find items of matching type at matching shelf position
        candidates = []
        for item in state["items"]:
            if item["type"] == item_type:
                ipos = tuple(item["position"])
                if ipos == target_pos:
                    candidates.append(item)

        if candidates:
            # If multiple (respawned), return first
            return candidates[0]["id"]

        # Fallback: match by type only (position might differ slightly)
        for item in state["items"]:
            if item["type"] == item_type:
                # Check if bot is adjacent to this item
                bot = next(b for b in state["bots"] if b["id"] == bot_id)
                bpos = tuple(bot["position"])
                ipos = tuple(item["position"])
                manhattan = abs(bpos[0] - ipos[0]) + abs(bpos[1] - ipos[1])
                if manhattan == 1:
                    return item["id"]

        return None

    def _should_fallback(self, state: dict, failed_action: dict) -> bool:
        """
        Decide if a single failed pickup warrants full fallback.
        Conservative: only fallback after 2+ consecutive misses.
        """
        # If we've missed several items in a row, state has diverged
        if self.stats["item_misses"] > 2:
            return True
        return False

    def print_stats(self):
        """Print replay execution statistics."""
        total = self.stats["planned_rounds"] + self.stats["fallback_rounds"]
        print(f"\n=== REPLAY STATS ===")
        print(f"  Planned rounds: {self.stats['planned_rounds']}")
        print(f"  Fallback rounds: {self.stats['fallback_rounds']}")
        print(f"  Item matches: {self.stats['item_matches']}")
        print(f"  Item misses: {self.stats['item_misses']}")
        if total > 0:
            pct = self.stats['planned_rounds'] / total * 100
            print(f"  Plan adherence: {pct:.0f}%")
        print()


# ------------------------------------------------------------------
# Save/load plans
# ------------------------------------------------------------------

import json
import os
from datetime import date


def save_plan(actions: list[dict], fingerprint: str,
              metadata: dict = None) -> str:
    """Save action plan to JSON."""
    today = date.today().isoformat()
    filename = f"{fingerprint}_{today}_plan.json"
    path = os.path.join("logs", filename)
    os.makedirs("logs", exist_ok=True)

    data = {
        "fingerprint": fingerprint,
        "date": today,
        "num_actions": len(actions),
        "actions": actions,
        "metadata": metadata or {},
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Plan saved: {path} ({len(actions)} actions)")
    return path


def load_plan(fingerprint: str, target_date: str = None) -> Optional[list[dict]]:
    """Load action plan for a given map fingerprint and date."""
    target = target_date or date.today().isoformat()
    filename = f"{fingerprint}_{target}_plan.json"
    path = os.path.join("logs", filename)

    if not os.path.exists(path):
        return None

    with open(path) as f:
        data = json.load(f)

    print(f"Plan loaded: {path} ({data['num_actions']} actions)")
    return data["actions"]
