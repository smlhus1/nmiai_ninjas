"""
Collect training data by running BotAdapter in sim.

Mode A (--mode imitation, default): Imitation learning.
  Logs V2TaskPlanner's actual (bot->item) decisions as labels.
  chosen_pair=1.0, not_chosen=0.0. Clear binary signal.

Mode B (--mode reward): Reward-based (original).
  reward_5 = score_delta over 5 rounds. Sparse signal.

Mode C (--mode shaped): Potential-based reward shaping.
  reward = gamma * phi(s') - phi(s) + score_delta
  where phi(s) = -sum(BFS_dist(bot, target)) / normalization

Usage:
    py -m ml.collect_training_data \
      --recon logs/74001e7f_2026-03-16_score274_recon.json \
      --n-games 50 --mode imitation \
      --output data/imitation_74001e7f_2026-03-16.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Simulering.offline.bot_adapter import BotAdapter
from Simulering.offline.simulator import Simulator
from bot.engine.pathfinding import PathEngine
from bot.models import GameState, Grid
from bot.strategy.task import TaskType
from ml.candidate_generator import DELIVER, IDLE
from ml.feature_extractor import FeatureContext, FeatureExtractor


def _build_shared_pe(recon: dict):
    """Build PathEngine once — grid is static across all rounds/games."""
    sim_tmp = Simulator.from_recon_data(recon)
    tmp_state = sim_tmp.reset()
    tmp_dict = tmp_state.to_dict()
    tmp_gs = GameState.from_dict(tmp_dict)
    shelves = frozenset(sim_tmp.shelves)
    merged_walls = tmp_gs.grid.walls | shelves
    merged_grid = Grid(tmp_gs.grid.width, tmp_gs.grid.height, merged_walls)
    pe = PathEngine()
    pe.set_grid(merged_grid, tmp_gs.drop_off)
    # Warm BFS cache for drop-off zones
    for z in tmp_gs.drop_off_zones:
        pe.distance((0, 0), z)
    return pe, tmp_gs.drop_off_zones


def _get_v2_decisions(adapter: BotAdapter) -> dict[int, str]:
    """Extract V2TaskPlanner's actual decisions from coordinator state.

    Returns dict[bot_id -> item_id | "DELIVER" | "IDLE"].
    """
    coord = adapter._coordinator
    if coord is None:
        return {}
    decisions = {}
    for bot_id, assignment in coord._assignments.items():
        if assignment.task is None or assignment.task.task_type == TaskType.IDLE:
            decisions[bot_id] = IDLE
        elif assignment.task.task_type == TaskType.DELIVER:
            decisions[bot_id] = DELIVER
        elif assignment.task.item_id:
            decisions[bot_id] = assignment.task.item_id
        else:
            decisions[bot_id] = IDLE
    return decisions


def _compute_potential(gs: GameState, pe: PathEngine, adapter: BotAdapter) -> float:
    """phi(s) = -sum(BFS_dist(bot, target)) for all bots with tasks."""
    coord = adapter._coordinator
    if coord is None:
        return 0.0
    total = 0.0
    for bot in gs.bots:
        a = coord._assignments.get(bot.id)
        if a and a.effective_target:
            d = pe.distance(bot.position, a.effective_target)
            if d < 9999:
                total -= d
    return total


def collect(
    recon_path: Path,
    n_games: int,
    output_path: Path,
    mode: str = "imitation",
    k_candidates: int = 5,
) -> None:
    recon = json.loads(recon_path.read_text(encoding="utf-8"))
    all_types = sorted(recon.get("shelf_map", {}).keys())
    type_index = {t: i for i, t in enumerate(all_types)}

    shared_pe, drop_off_zones = _build_shared_pe(recon)

    all_data: list[tuple[torch.Tensor, float]] = []
    t0 = time.time()
    stats = {"chosen": 0, "not_chosen": 0, "deliver": 0, "idle": 0}

    for game_idx in range(n_games):
        sim = Simulator.from_recon_data(recon)
        adapter = BotAdapter(suppress_logs=True)

        state = sim.reset()
        round_scores: list[int] = [sim._score]
        # For shaped mode: store per-round features with potentials
        round_data: list[list[tuple[torch.Tensor, float]]] = []
        prev_potential = 0.0

        for _ in range(sim.max_rounds):
            state_dict = state.to_dict()

            # Step adapter FIRST to get V2's decisions
            response = adapter(state_dict)
            actions = response.get("actions", [])

            # Parse state for feature extraction
            gs = GameState.from_dict(state_dict)
            active = gs.active_orders[0] if gs.active_orders else None
            preview = gs.preview_orders[0] if gs.preview_orders else None

            # Get V2's actual decisions
            v2_decisions = _get_v2_decisions(adapter)

            # Build claimed set from V2 decisions
            claimed = set()
            for d in v2_decisions.values():
                if d not in (DELIVER, IDLE):
                    claimed.add(d)

            ctx = FeatureContext(
                assignments=adapter._coordinator._assignments if adapter._coordinator else {},
                claimed_items=claimed,
                active_order=active,
                preview_order=preview,
                bot_positions=[b.position for b in gs.bots],
                n_bots=len(gs.bots),
                max_dist=60,
                item_type_index=type_index,
                drop_off_zones=drop_off_zones,
                score_history=list(round_scores),
            )

            # For each bot: encode candidates + V2's actual choice
            round_pairs: list[tuple[torch.Tensor, float]] = []

            for bot in gs.bots:
                v2_choice = v2_decisions.get(bot.id, IDLE)

                # Get top-K nearest items as candidates
                items_with_dist = []
                for item in gs.items:
                    d = shared_pe.distance(bot.position, item.position)
                    if d >= 9999:
                        d = min(
                            (shared_pe.distance(bot.position, (item.position[0] + dx, item.position[1] + dy))
                             for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0))),
                            default=9999,
                        )
                    items_with_dist.append((d, item))

                items_with_dist.sort(key=lambda x: x[0])
                candidate_items = items_with_dist[:k_candidates]

                # Ensure V2's chosen item is in candidates (if it's an item)
                chosen_ids = {item.id for _, item in candidate_items}
                if v2_choice not in (DELIVER, IDLE) and v2_choice not in chosen_ids:
                    # Find the chosen item and add it
                    for item in gs.items:
                        if item.id == v2_choice:
                            d = shared_pe.distance(bot.position, item.position)
                            if d >= 9999:
                                d = min(
                                    (shared_pe.distance(bot.position, (item.position[0] + dx, item.position[1] + dy))
                                     for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0))),
                                    default=9999,
                                )
                            candidate_items.append((d, item))
                            break

                # Encode item candidates
                for _, item in candidate_items:
                    feat = FeatureExtractor.encode_pair(bot, item, gs, shared_pe, ctx)
                    if mode == "imitation":
                        label = 1.0 if item.id == v2_choice else 0.0
                        if label == 1.0:
                            stats["chosen"] += 1
                        else:
                            stats["not_chosen"] += 1
                    else:
                        label = 0.0  # filled in post-processing
                    round_pairs.append((feat, label))

                # Encode DELIVER action
                from bot.models import Item as _Item
                if len(bot.inventory) > 0:
                    nearest_do = gs.drop_off
                    if drop_off_zones:
                        nearest_do = min(drop_off_zones,
                                        key=lambda z: shared_pe.distance(bot.position, z))
                    dummy = _Item(id="__deliver__", type="__deliver__", position=nearest_do)
                    feat = FeatureExtractor.encode_pair(bot, dummy, gs, shared_pe, ctx)
                    if mode == "imitation":
                        label = 1.0 if v2_choice == DELIVER else 0.0
                        if label == 1.0:
                            stats["deliver"] += 1
                    else:
                        label = 0.0
                    round_pairs.append((feat, label))

                # Encode IDLE action
                dummy_idle = _Item(id="__idle__", type="__idle__", position=bot.position)
                feat = FeatureExtractor.encode_pair(bot, dummy_idle, gs, shared_pe, ctx)
                if mode == "imitation":
                    label = 1.0 if v2_choice == IDLE else 0.0
                    if label == 1.0:
                        stats["idle"] += 1
                else:
                    label = 0.0
                round_pairs.append((feat, label))

            if mode == "imitation":
                all_data.extend(round_pairs)
            else:
                round_data.append(round_pairs)

            # Step sim
            state, game_over = sim.step(actions)
            round_scores.append(sim._score)

            # Compute potential for shaped mode
            if mode == "shaped":
                curr_potential = _compute_potential(gs, shared_pe, adapter)
                gamma = 0.99
                score_delta = round_scores[-1] - round_scores[-2]
                shaped_reward = gamma * curr_potential - prev_potential + score_delta
                shaped_reward = max(0.0, min(shaped_reward / 20.0 + 0.5, 1.0))
                # Apply shaped reward to this round's pairs
                for feat, _ in round_data[-1]:
                    all_data.append((feat, shaped_reward))
                prev_potential = curr_potential

            if game_over:
                break

        adapter.reset()

        # For reward mode: compute reward_5 post-hoc
        if mode == "reward":
            total_rounds = len(round_scores) - 1
            for r_idx, pairs in enumerate(round_data):
                future_r = min(r_idx + 5, total_rounds)
                delta = round_scores[future_r] - round_scores[r_idx]
                reward = max(0.0, min(delta / 10.0, 1.0))
                for feat, _ in pairs:
                    all_data.append((feat, reward))

        elapsed = time.time() - t0
        print(f"Game {game_idx + 1}/{n_games} done, score={round_scores[-1]}, "
              f"total_samples={len(all_data)}, elapsed={elapsed:.0f}s")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved {len(all_data)} samples to {output_path}")
    print(f"Total time: {time.time() - t0:.0f}s")
    if mode == "imitation":
        total = stats["chosen"] + stats["not_chosen"] + stats["deliver"] + stats["idle"]
        print(f"Labels: chosen={stats['chosen']} not_chosen={stats['not_chosen']} "
              f"deliver={stats['deliver']} idle={stats['idle']} "
              f"(positive rate: {(stats['chosen']+stats['deliver'])/(total+1)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Collect training data for ScorerMLP")
    parser.add_argument("--recon", required=True, help="Path to recon JSON")
    parser.add_argument("--n-games", type=int, default=50, help="Number of sim games")
    parser.add_argument("--output", help="Output pickle path")
    parser.add_argument("--mode", choices=["imitation", "reward", "shaped"],
                        default="imitation", help="Data collection mode")
    parser.add_argument("--k", type=int, default=5, help="Candidates per bot")
    args = parser.parse_args()

    recon_path = Path(args.recon)
    if not recon_path.exists():
        print(f"Recon file not found: {recon_path}")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        stem = recon_path.stem.replace("_recon", "")
        output_path = Path("data") / f"{args.mode}_{stem}.pkl"

    print(f"Collecting {args.n_games} games from {recon_path.name} (mode={args.mode})")
    print(f"Output: {output_path}\n")
    collect(recon_path, args.n_games, output_path, mode=args.mode, k_candidates=args.k)


if __name__ == "__main__":
    main()
