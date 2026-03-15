"""Capture reactive bot's actions as a MAPF plan for live replay.

Usage:
    py _capture_reactive.py <recon_file> [output_file]
    py _capture_reactive.py logs/74001e7f_2026-03-13_recon.json
    py _capture_reactive.py logs/74001e7f_2026-03-13_recon.json mapf_plan.json
"""
import json
import sys
from pathlib import Path
from Simulering.offline.simulator import Simulator
from Simulering.offline.bot_adapter import BotAdapter
from mapf_planner import MAPFPlan, MAPFAction, plan_to_dict

if len(sys.argv) < 2:
    print("Usage: py _capture_reactive.py <recon_file> [output_file]")
    sys.exit(1)

recon_file = sys.argv[1]
output_file = sys.argv[2] if len(sys.argv) > 2 else "mapf_plan_reactive.json"

config = None
if len(sys.argv) > 3:
    from bot.config import CoordinatorConfig
    config = CoordinatorConfig.from_dict(json.loads(Path(sys.argv[3]).read_text()))

sim = Simulator.from_recon_file(recon_file)
adapter = BotAdapter(suppress_logs=True, config=config)

state = sim.reset()
bot_actions: dict[int, list[MAPFAction]] = {}

for round_t in range(sim.max_rounds):
    state_dict = state.to_dict()
    bots = state_dict["bots"]
    items = state_dict["items"]

    # Get reactive bot's response
    response = adapter(state_dict)
    actions = response.get("actions", [])

    # Record actions as MAPFAction
    action_map = {a["bot"]: a for a in actions}
    for bot_data in bots:
        bid = bot_data["id"]
        bot_pos = tuple(bot_data["position"])
        act = action_map.get(bid, {"action": "wait"})
        action = act.get("action", "wait")

        if bid not in bot_actions:
            bot_actions[bid] = []

        item_type = ""
        if action == "pick_up":
            # Resolve item type from item_id
            item_id = act.get("item_id", "")
            for item in items:
                if item["id"] == item_id:
                    item_type = item["type"]
                    break

        bot_actions[bid].append(MAPFAction(
            action=action,
            position=bot_pos,
            item_type=item_type,
        ))

    state, game_over = sim.step(actions)
    if game_over:
        break

    if round_t % 50 == 0:
        print(f"Round {round_t}: score={state.to_dict()['score']}")

final = state.to_dict()
score = final["score"]
print(f"\nFinal: score={score}, rounds={sim._round}")

# Create plan
plan = MAPFPlan(
    actions=bot_actions,
    total_rounds=sim._round,
    expected_score=score,
    order_activations={},
    pickup_schedule=[],
    dropoff_schedule=[],
)

plan_dict = plan_to_dict(plan)
Path(output_file).write_text(
    json.dumps(plan_dict, indent=2), encoding="utf-8"
)
print(f"Plan saved to {output_file} ({score} expected score)")
