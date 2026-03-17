"""Auto-select best config for today's recon.

Runs V2 with both one-way ON and OFF in sim, picks the winner,
captures MAPF plan from the best config.

Usage:
    py auto_best_plan.py <recon_file> [output_plan]
"""
import json
import sys
from pathlib import Path

from Simulering.offline.bot_adapter import BotAdapter
from Simulering.offline.simulator import Simulator
from bot.config import CoordinatorConfig


def run_config(recon: dict, disable_ow: bool) -> int:
    sim = Simulator.from_recon_data(recon)
    cfg = CoordinatorConfig.nightmare()
    cfg.disable_one_way = disable_ow
    adapter = BotAdapter(suppress_logs=True, config=cfg)
    result = sim.run(adapter)
    adapter.reset()
    return result["score"]


def capture_plan(recon_path: str, disable_ow: bool, output: str) -> int:
    """Capture MAPF plan from best config."""
    from mapf_planner import MAPFPlan, MAPFAction, plan_to_dict

    sim = Simulator.from_recon_file(recon_path)
    cfg = CoordinatorConfig.nightmare()
    cfg.disable_one_way = disable_ow
    adapter = BotAdapter(suppress_logs=True, config=cfg)

    state = sim.reset()
    bot_actions = {}

    for round_t in range(sim.max_rounds):
        state_dict = state.to_dict()
        bots = state_dict["bots"]
        items = state_dict["items"]
        response = adapter(state_dict)
        actions = response.get("actions", [])

        action_map = {a["bot"]: a for a in actions}
        for bot_data in bots:
            bid = bot_data["id"]
            act = action_map.get(bid, {"action": "wait"})
            action = act.get("action", "wait")
            bot_actions.setdefault(bid, [])
            item_type = ""
            if action == "pick_up":
                item_id = act.get("item_id", "")
                for item in items:
                    if item["id"] == item_id:
                        item_type = item["type"]
                        break
            bot_actions[bid].append(MAPFAction(
                action=action, position=tuple(bot_data["position"]), item_type=item_type,
            ))

        state, game_over = sim.step(actions)
        if game_over:
            break

    score = sim._score
    plan = MAPFPlan(
        actions=bot_actions, total_rounds=sim._round, expected_score=score,
        order_activations={}, pickup_schedule=[], dropoff_schedule=[],
    )
    Path(output).write_text(json.dumps(plan_to_dict(plan), indent=2), encoding="utf-8")
    return score


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py auto_best_plan.py <recon_file> [output_plan]")
        sys.exit(1)

    recon_path = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else "mapf_plan_best.json"
    recon = json.loads(Path(recon_path).read_text())

    print("Testing configs...", flush=True)
    score_ow = run_config(recon, disable_ow=False)
    score_no_ow = run_config(recon, disable_ow=True)

    print(f"  One-way ON:  {score_ow}")
    print(f"  One-way OFF: {score_no_ow}")

    best_ow = score_no_ow > score_ow
    best_score = max(score_ow, score_no_ow)
    label = "OFF" if best_ow else "ON"
    print(f"  Winner: one-way {label} ({best_score})")

    print(f"\nCapturing MAPF plan (one-way {label})...", flush=True)
    final = capture_plan(recon_path, disable_ow=best_ow, output=output)
    print(f"Plan saved: {output} (score={final})")
