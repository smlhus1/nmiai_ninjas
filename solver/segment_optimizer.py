"""Segment optimizer: checkpoint sim at order boundaries, branch on slow orders.

Core idea: deepcopy(sim) + deepcopy(adapter) at each order completion.
When an order takes >15 rounds, restore checkpoint and try N variants
with different PIBT tiebreaking (via guidance_alpha/noise tweaks).
Keep the variant where the order completes fastest, continue from there.

This accumulates improvements across the game — each optimized segment
provides a better starting state for subsequent segments.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
import time
import random
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
logging.basicConfig(level=logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).parent.parent))


def segment_optimize(
    recon_path: str,
    n_variants: int = 20,
    slow_threshold: int = 15,
    base_config: dict = None,
):
    """Optimize by branching at slow order checkpoints."""
    from Simulering.offline.bot_adapter import BotAdapter
    from Simulering.offline.simulator import Simulator
    from bot.config import CoordinatorConfig

    def make_adapter(overrides=None):
        config = CoordinatorConfig.for_difficulty(20)
        if base_config:
            for k, v in base_config.items():
                if hasattr(config, k) and v is not None:
                    setattr(config, k, v)
        if overrides:
            for k, v in overrides.items():
                if hasattr(config, k):
                    setattr(config, k, v)
        return BotAdapter(suppress_logs=True, config=config)

    def tweak_overrides():
        """Small random tweaks to create different PIBT behavior."""
        return {
            "guidance_alpha": random.uniform(0.5, 3.0),
            "guidance_beta": random.uniform(1.0, 5.0),
            "guidance_decay": random.uniform(0.4, 0.95),
            "guidance_update_interval": random.choice([2, 3, 5]),
        }

    # Initial state
    sim = Simulator.from_recon_file(recon_path)
    adapter = make_adapter()
    state = sim.reset()

    total_saved = 0
    segments_optimized = 0
    prev_order_round = 0
    prev_orders = 0

    # Checkpoint = state just after previous order completes
    checkpoint_sim = copy.deepcopy(sim)
    checkpoint_adapter = copy.deepcopy(adapter)
    checkpoint_round = 0

    print(f"Segment optimizer: {n_variants} variants per slow segment, threshold={slow_threshold}r", flush=True)

    r = 0
    while r < 500:
        sd = state.to_dict()
        resp = adapter(sd)
        state, game_over = sim.step(resp.get("actions", []))
        r += 1

        if sim._orders_completed > prev_orders:
            delta = r - prev_order_round
            order_num = sim._orders_completed
            score = sim._score

            if delta > slow_threshold and order_num > 1:
                # SLOW ORDER DETECTED — branch from checkpoint
                print(f"  Order {order_num:2d} SLOW ({delta:2d}r) at R{r}. Branching from R{checkpoint_round}...", flush=True)

                best_delta = delta
                best_sim = None
                best_adapter = None
                best_state = None
                best_round = r

                for vi in range(n_variants):
                    # Restore checkpoint
                    sim_v = copy.deepcopy(checkpoint_sim)
                    adapter_v = copy.deepcopy(checkpoint_adapter)

                    # Apply random tweaks to create different bot behavior
                    tweaks = tweak_overrides()
                    coord = adapter_v._coordinator
                    # Tweak guidance (affects PIBT tiebreaking)
                    if hasattr(coord, '_resolver') and hasattr(coord._resolver, '_guidance'):
                        g = coord._resolver._guidance
                        if g:
                            g._alpha = tweaks["guidance_alpha"]
                            g._beta = tweaks["guidance_beta"]
                            g._decay = tweaks["guidance_decay"]
                    # Tweak shelf preferences (affects WHICH items bots target)
                    if hasattr(coord, '_config') and coord._config:
                        import json as _json
                        recon_data = _json.load(open(recon_path))
                        shelf_map = recon_data.get("shelf_map", {})
                        new_pref = {}
                        for item_type, shelves in shelf_map.items():
                            if shelves and random.random() < 0.3:
                                new_pref[item_type] = random.randint(0, len(shelves) - 1)
                        if new_pref:
                            existing = coord._config.shelf_preference or {}
                            existing.update(new_pref)
                            coord._config.shelf_preference = existing
                    # Tweak stuck thresholds (affects when bots re-route)
                    if hasattr(coord, '_config') and coord._config:
                        coord._config.stuck_pick_rounds = random.randint(2, 6)
                        coord._config.stuck_transit_rounds = random.randint(3, 8)
                        coord._config.gate_max_delay = random.choice([0, 2, 4, 6, 8])

                    # Run from checkpoint until this order completes
                    state_v = sim_v._last_state if hasattr(sim_v, '_last_state') else None
                    # We need to re-derive state from sim
                    # Actually, step returns state. Let's track it differently.
                    # The sim's internal state IS the state. We can generate state from it.
                    orders_at_checkpoint = checkpoint_sim._orders_completed
                    variant_round = checkpoint_round

                    # Run until order completes or timeout
                    max_rounds = checkpoint_round + delta + 30  # allow some slack
                    completed = False
                    variant_delta = 9999

                    # Need to drive sim forward — get state from step
                    # First step: get current state
                    state_v = sim_v._get_state()

                    while variant_round < min(max_rounds, 500):
                        sd_v = state_v.to_dict()
                        resp_v = adapter_v(sd_v)
                        state_v, go = sim_v.step(resp_v.get("actions", []))
                        variant_round += 1

                        if sim_v._orders_completed > orders_at_checkpoint:
                            # Order completed!
                            variant_delta = variant_round - checkpoint_round
                            completed = True
                            break
                        if go:
                            break

                    if completed and variant_delta < best_delta:
                        saved = best_delta - variant_delta if best_delta < delta + 1 else delta - variant_delta
                        best_delta = variant_delta
                        best_sim = sim_v
                        best_adapter = adapter_v
                        best_state = state_v
                        best_round = variant_round

                if best_sim and best_delta < delta:
                    saved = delta - best_delta
                    total_saved += saved
                    segments_optimized += 1
                    print(f"    IMPROVED: {delta}r -> {best_delta}r (saved {saved}r)", flush=True)

                    # Replace current state with best variant
                    sim = best_sim
                    adapter = best_adapter
                    state = best_state
                    r = best_round
                else:
                    print(f"    No improvement found (best variant also {best_delta}r)", flush=True)
            else:
                if delta > slow_threshold:
                    print(f"  Order {order_num:2d}: {delta:2d}r at R{r} (first order, skip)", flush=True)
                else:
                    pass  # Fast order, no action

            # Save checkpoint for next segment
            checkpoint_sim = copy.deepcopy(sim)
            checkpoint_adapter = copy.deepcopy(adapter)
            checkpoint_round = r
            prev_order_round = r
            prev_orders = sim._orders_completed

        if game_over:
            break

    # Final results
    s180_est = sim._score  # Can't easily get s180 with branching, use total
    print(f"\n=== RESULT ===", flush=True)
    print(f"Total: {sim._score}, Orders: {sim._orders_completed}, Rounds: {r}", flush=True)
    print(f"Segments optimized: {segments_optimized}, Rounds saved: {total_saved}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", required=True)
    parser.add_argument("--variants", type=int, default=20)
    parser.add_argument("--threshold", type=int, default=15)
    args = parser.parse_args()

    base = {}
    bp = Path("logs/best_individual.json")
    if bp.exists():
        ind = json.loads(bp.read_text())
        base = ind.get("config", {})

    t0 = time.time()
    segment_optimize(args.recon, n_variants=args.variants,
                     slow_threshold=args.threshold, base_config=base)
    print(f"Time: {time.time() - t0:.0f}s", flush=True)
