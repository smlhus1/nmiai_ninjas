"""LNS Nightmare Optimizer — config + noise variation search.

Runs BotAdapter with many different config/noise combinations in parallel.
Each run captures a full MAPF plan. Best plan wins.

Key insight: distance_noise + noise_seed creates different routing decisions,
which break different congestion patterns. With enough trials, some avoid the
specific bottlenecks that limit baseline score.

Modes:
1. Config variation: mutate guidance/stuck/gate params
2. Noise search: vary distance_noise + noise_seed for deterministic diversity
3. Hybrid: both config and noise variation

Usage:
    py -m solver.lns_nightmare --recon logs/74001e7f_2026-03-17_recon.json --iterations 200
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent))


def _capture_with_config(args) -> tuple[int, int, dict | None, dict]:
    """Worker: run BotAdapter with given config overrides, capture MAPF plan.

    Returns (idx, score, plan_actions_or_None, config_overrides).
    """
    idx, recon_path, overrides = args
    try:
        from Simulering.offline.simulator import Simulator
        from Simulering.offline.bot_adapter import BotAdapter
        from bot.config import CoordinatorConfig
        from mapf_planner import MAPFAction

        config = CoordinatorConfig.nightmare()
        if overrides:
            for k, v in overrides.items():
                if v is not None and hasattr(config, k):
                    setattr(config, k, v)

        adapter = BotAdapter(suppress_logs=True, config=config)
        sim = Simulator.from_recon_file(recon_path)
        state = sim.reset()

        bot_actions: dict[int, list[MAPFAction]] = {}

        for round_t in range(sim.max_rounds):
            state_dict = state.to_dict()
            bots = state_dict["bots"]
            items = state_dict["items"]

            response = adapter(state_dict)
            actions = response.get("actions", [])

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

        return idx, sim._score, bot_actions, overrides

    except Exception as e:
        logging.error("Worker %d failed: %s", idx, e)
        return idx, 0, None, overrides


def _generate_configs(n: int) -> list[dict]:
    """Generate diverse config override sets for nightmare optimization."""
    configs = []

    # First entry: baseline (no overrides)
    configs.append({})

    for i in range(1, n):
        overrides = {}

        # Noise variation (primary diversity driver)
        overrides["distance_noise"] = random.choice([0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5])
        overrides["noise_seed"] = random.randint(0, 99999)

        # Shelf randomness (different routing per type)
        overrides["shelf_randomness"] = random.choice([0.0, 0.1, 0.2, 0.3])

        # Guidance variation
        if random.random() < 0.4:
            overrides["guidance_alpha"] = random.choice([0.5, 1.0, 1.5, 2.0, 3.0])
            overrides["guidance_beta"] = random.choice([1.0, 2.0, 3.0, 5.0])
            overrides["guidance_decay"] = random.choice([0.5, 0.7, 0.8, 0.9])
            overrides["guidance_update_interval"] = random.choice([2, 3, 5])

        # Stuck threshold variation
        if random.random() < 0.3:
            overrides["stuck_transit_rounds"] = random.randint(3, 8)
            overrides["stuck_pick_rounds"] = random.randint(2, 6)

        # Gate variation
        if random.random() < 0.2:
            overrides["gate_max_delay"] = random.choice([0, 2, 3, 5])
            overrides["pre_pick_rush_remaining"] = random.choice([2, 3, 4])

        # Pre-pick inventory
        if random.random() < 0.2:
            overrides["pre_pick_max_inventory"] = random.choice([2, 3])

        configs.append(overrides)

    return configs


def lns_optimize(
    recon_path: str,
    iterations: int = 100,
    n_workers: int | None = None,
    save_interval: int = 50,
    output: str = "mapf_plan_lns.json",
):
    """Run config-variation LNS optimization.

    Args:
        recon_path: Path to nightmare recon JSON
        iterations: Number of config variants to try
        n_workers: Parallel workers (default: CPU count, max 12)
        save_interval: Save best plan every N iterations
        output: Output file path for best plan
    """
    if n_workers is None:
        n_workers = min(cpu_count(), 12)

    print(f"=== LNS Nightmare Optimizer (Config Variation) ===", flush=True)
    print(f"Recon: {recon_path}", flush=True)
    print(f"Iterations: {iterations}, workers: {n_workers}", flush=True)

    configs = _generate_configs(iterations)

    t0 = time.time()
    best_score = 0
    best_plan = None
    best_config = {}
    improvements = 0
    score_history = []

    # Process in batches for memory efficiency
    batch_size = n_workers * 2
    for batch_start in range(0, len(configs), batch_size):
        batch_end = min(batch_start + batch_size, len(configs))
        batch_configs = configs[batch_start:batch_end]

        tasks = [
            (batch_start + i, recon_path, cfg)
            for i, cfg in enumerate(batch_configs)
        ]

        t_batch = time.time()
        with Pool(n_workers) as pool:
            results = pool.map(_capture_with_config, tasks)

        for idx, score, plan_actions, overrides in results:
            score_history.append(score)
            if score > best_score and plan_actions is not None:
                old_best = best_score
                best_score = score
                best_plan = plan_actions
                best_config = overrides
                improvements += 1
                print(f"  *** NEW BEST: {best_score} (iter {idx}, +{score - old_best})", flush=True)
                if overrides:
                    # Show only non-default overrides
                    key_overrides = {k: v for k, v in overrides.items()
                                     if k in ('distance_noise', 'noise_seed', 'guidance_alpha')}
                    print(f"      config: {key_overrides}", flush=True)

        elapsed = time.time() - t_batch
        print(f"  Batch {batch_end}/{len(configs)}: best={best_score}, "
              f"batch_time={elapsed:.1f}s", flush=True)

        # Save periodically
        if best_plan and batch_end % save_interval < batch_size:
            _save_plan(best_plan, best_score, output)

    # Final save
    if best_plan:
        _save_plan(best_plan, best_score, output)

    # Summary
    total_time = time.time() - t0
    score_history.sort(reverse=True)
    print(f"\n=== RESULTS ===", flush=True)
    print(f"Best: {best_score}", flush=True)
    print(f"Top 5: {score_history[:5]}", flush=True)
    print(f"Median: {score_history[len(score_history)//2]}", flush=True)
    print(f"Improvements: {improvements}/{iterations}", flush=True)
    print(f"Best config: {best_config}", flush=True)
    print(f"Total time: {total_time:.0f}s ({total_time/iterations:.1f}s/iter)", flush=True)
    print(f"Plan saved to {output}", flush=True)

    return best_plan, best_score


def _save_plan(plan_actions: dict, score: int, output_path: str):
    """Save plan as MAPF JSON."""
    from mapf_planner import MAPFPlan, plan_to_dict

    total_rounds = max(len(acts) for acts in plan_actions.values())
    plan = MAPFPlan(
        actions=plan_actions,
        total_rounds=total_rounds,
        expected_score=score,
        order_activations={},
        pickup_schedule=[],
        dropoff_schedule=[],
    )
    Path(output_path).write_text(
        json.dumps(plan_to_dict(plan), indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LNS nightmare optimizer (config variation)")
    parser.add_argument("--recon", required=True, help="Path to recon JSON")
    parser.add_argument("--iterations", type=int, default=200,
                        help="Number of config variants to try")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers (default: CPU count)")
    parser.add_argument("--output", default="mapf_plan_lns.json",
                        help="Output plan file")
    args = parser.parse_args()

    lns_optimize(
        args.recon,
        iterations=args.iterations,
        n_workers=args.workers,
        output=args.output,
    )
