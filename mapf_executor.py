"""
MAPF Plan Executor for NM i AI Grocery Bot.

Thin layer that reads a pre-computed MAPF plan and returns actions per round.
Handles:
- Item-ID resolution (plan uses shelf_pos + item_type, runtime uses actual IDs)
- Collision-delay tolerance: retries blocked moves, cursor only advances on success
- Integration with Simulator for offline verification

Usage:
    # Offline verification via simulator
    py mapf_executor.py --plan mapf_plan.json --recon logs/74001e7f_2026-03-12_recon.json

    # Live game (wired through main.py --mapf)
    py main.py --url "wss://..." --mapf mapf_plan.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from mapf_planner import MAPFPlan, MAPFAction, plan_from_dict

logger = logging.getLogger(__name__)


class MAPFExecutor:
    """Executes a pre-computed MAPF plan with collision-delay tolerance.

    Per-bot cursor tracks the next plan index to confirm. The cursor only
    advances when the bot's actual position matches plan[cursor].position.
    If a move is blocked by collision, the cursor stays put and the move
    is retried next round.
    """

    def __init__(self, plan: MAPFPlan) -> None:
        self._plan = plan
        self._fallback_active = False
        self._round = 0
        # Per-bot: next plan index to confirm (bot should be at plan[cursor].position)
        self._bot_cursor: dict[int, int] = {}

    @property
    def fallback_active(self) -> bool:
        return self._fallback_active

    @property
    def plan_exhausted(self) -> bool:
        """True when all bots have finished their planned actions."""
        if not self._plan.actions:
            return True
        for bot_id, actions in self._plan.actions.items():
            cursor = self._bot_cursor.get(bot_id, 0)
            if cursor < len(actions):
                return False
        return True

    def on_game_state(self, raw: dict) -> dict:
        round_num = raw.get("round", self._round)
        self._round = round_num
        bots = raw.get("bots", [])
        items = raw.get("items", [])

        actions = []
        for bot_data in bots:
            bot_id = bot_data["id"]
            bot_pos = tuple(bot_data["position"])

            bot_actions = self._plan.actions.get(bot_id)
            if bot_actions is None:
                actions.append({"bot": bot_id, "action": "wait"})
                continue

            cursor = self._bot_cursor.get(bot_id, 0)

            if cursor >= len(bot_actions):
                # Plan exhausted — wait
                actions.append({"bot": bot_id, "action": "wait"})
                continue

            planned = bot_actions[cursor]

            if bot_pos == planned.position:
                # Bot is where the plan expects — execute the planned action
                action_dict = self._build_action(planned, bot_id, bot_pos, items)
                self._bot_cursor[bot_id] = cursor + 1
                actions.append(action_dict)

            elif cursor > 0 and bot_pos == bot_actions[cursor - 1].position:
                # Bot is at previous plan position — the last move was blocked
                # Retry the previous action (which should move bot to cursor position)
                retry = bot_actions[cursor - 1]
                action_dict = self._build_action(retry, bot_id, bot_pos, items)
                # DON'T advance cursor — we're retrying
                actions.append(action_dict)

            else:
                # Bot is somewhere unexpected — try to find it in the plan
                found = False
                # Look forward a few steps (bot might have jumped ahead somehow)
                for look in range(cursor + 1, min(cursor + 5, len(bot_actions))):
                    if bot_actions[look].position == bot_pos:
                        planned = bot_actions[look]
                        action_dict = self._build_action(planned, bot_id, bot_pos, items)
                        self._bot_cursor[bot_id] = look + 1
                        actions.append(action_dict)
                        found = True
                        break

                if not found:
                    # Truly lost — wait
                    logger.warning(
                        "Round %d bot %d: lost at %s (plan expects %s at cursor %d)",
                        round_num, bot_id, bot_pos, planned.position, cursor,
                    )
                    actions.append({"bot": bot_id, "action": "wait"})

        return {"actions": actions}

    def _build_action(
        self, planned: MAPFAction, bot_id: int, bot_pos: tuple, items: list[dict]
    ) -> dict:
        """Build action dict from planned action, resolving item IDs."""
        action_dict: dict = {"bot": bot_id, "action": planned.action}

        if planned.action == "pick_up" and planned.item_type:
            item_id = self._resolve_item_id(items, bot_pos, planned.item_type)
            if item_id:
                action_dict["item_id"] = item_id
            else:
                logger.warning(
                    "Bot %d: can't find %s at %s, waiting",
                    bot_id, planned.item_type, bot_pos,
                )
                action_dict["action"] = "wait"

        return action_dict

    def _resolve_item_id(
        self,
        items: list[dict],
        bot_pos: tuple,
        item_type: str,
    ) -> str | None:
        """Find the actual item ID for a planned pickup."""
        for item in items:
            if item.get("type") != item_type:
                continue
            item_pos = tuple(item["position"])
            dx = abs(item_pos[0] - bot_pos[0])
            dy = abs(item_pos[1] - bot_pos[1])
            if dx + dy <= 1:
                return item["id"]
        return None


# ---------------------------------------------------------------------------
# Simulator integration for offline verification
# ---------------------------------------------------------------------------

def run_with_simulator(plan_path: str, recon_path: str) -> None:
    """Run MAPF plan through the offline simulator to verify score."""
    from Simulering.offline.simulator import Simulator

    plan_data = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    plan = plan_from_dict(plan_data)

    sim = Simulator.from_recon_file(recon_path)
    executor = MAPFExecutor(plan)

    def mapf_strategy(state: dict) -> dict:
        return executor.on_game_state(state)

    result = sim.run(mapf_strategy)

    print(f"\nSimulator result:")
    print(f"  Score: {result.get('score', '?')}")
    print(f"  Rounds: {result.get('rounds', '?')}")
    print(f"  Expected: {plan.expected_score}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if "--plan" not in sys.argv or "--recon" not in sys.argv:
        print("Usage: py mapf_executor.py --plan <plan_file> --recon <recon_file>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    plan_path = sys.argv[sys.argv.index("--plan") + 1]
    recon_path = sys.argv[sys.argv.index("--recon") + 1]

    run_with_simulator(plan_path, recon_path)


if __name__ == "__main__":
    main()
