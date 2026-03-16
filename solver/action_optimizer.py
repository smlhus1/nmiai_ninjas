"""Per-round action optimizer: takes BotAdapter output and tries alternatives.

For each round where bots wait (collision/stuck), tries permuting actions
to reduce waits. Uses sim as oracle — only accepts changes that increase score.

This squeezes extra throughput from BotAdapter's existing planning.
"""

from __future__ import annotations

import json
import sys
import os
import time
import copy
import random
import logging
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
logging.basicConfig(level=logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).parent.parent))

from Simulering.offline.bot_adapter import BotAdapter
from Simulering.offline.simulator import Simulator
from bot.config import CoordinatorConfig


def capture_baseline(recon_path: str, config: CoordinatorConfig = None) -> tuple[list[list[dict]], int]:
    """Run BotAdapter and capture all actions per round. Returns (actions_per_round, score)."""
    adapter = BotAdapter(suppress_logs=True, config=config)
    sim = Simulator.from_recon_file(recon_path)
    state = sim.reset()

    all_actions = []
    for r in range(sim.max_rounds):
        sd = state.to_dict()
        response = adapter(sd)
        actions = response.get("actions", [])
        all_actions.append(actions)
        state, done = sim.step(actions)
        if done:
            break

    return all_actions, sim._score


def replay_with_modifications(recon_path: str, actions_per_round: list[list[dict]],
                              modifications: dict[int, list[dict]]) -> int:
    """Replay actions with specific round modifications. Returns score."""
    sim = Simulator.from_recon_file(recon_path)
    state = sim.reset()

    for r in range(min(len(actions_per_round), sim.max_rounds)):
        if r in modifications:
            actions = modifications[r]
        else:
            actions = actions_per_round[r]
        state, done = sim.step(actions)
        if done:
            break

    return sim._score


def optimize_actions(recon_path: str, n_iterations: int = 100,
                    config: CoordinatorConfig = None) -> tuple[int, dict]:
    """Optimize per-round actions via random perturbation."""
    print("Capturing baseline...", flush=True)
    baseline_actions, baseline_score = capture_baseline(recon_path, config)
    print(f"Baseline: score={baseline_score}, rounds={len(baseline_actions)}", flush=True)

    # Find rounds with many waits (optimization opportunities)
    wait_rounds = []
    for r, actions in enumerate(baseline_actions):
        n_waits = sum(1 for a in actions if a["action"] == "wait")
        if n_waits >= 3:
            wait_rounds.append((r, n_waits))

    print(f"Rounds with 3+ waits: {len(wait_rounds)}", flush=True)

    best_score = baseline_score
    best_mods = {}

    for iteration in range(n_iterations):
        # Pick a random round with waits and try modifications
        if not wait_rounds:
            break

        r, n_waits = random.choice(wait_rounds)
        original = baseline_actions[r]

        # Try: swap a waiting bot's action with a random move
        modified = copy.deepcopy(original)
        waiters = [i for i, a in enumerate(modified) if a["action"] == "wait"]
        if not waiters:
            continue

        idx = random.choice(waiters)
        new_action = random.choice(["move_up", "move_down", "move_left", "move_right"])
        modified[idx]["action"] = new_action

        mods = dict(best_mods)
        mods[r] = modified

        score = replay_with_modifications(recon_path, baseline_actions, mods)

        if score > best_score:
            best_score = score
            best_mods = mods
            print(f"  Iter {iteration}: score={best_score} (R{r} B{modified[idx]['bot']} {new_action}) ***", flush=True)

    print(f"\nBest: {best_score} (baseline {baseline_score}, +{best_score - baseline_score})", flush=True)
    return best_score, best_mods


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", required=True)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    pref_path = Path("logs/best_shelf_pref.json")
    config = CoordinatorConfig.for_difficulty(20)
    if pref_path.exists():
        config.shelf_preference = json.loads(pref_path.read_text())

    t0 = time.time()
    score, mods = optimize_actions(args.recon, n_iterations=args.iterations, config=config)
    print(f"Time: {time.time()-t0:.1f}s")
