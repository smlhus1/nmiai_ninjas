"""
Massive PPO-style training: V2TaskPlanner as environment + scorer optimization.

V2 runs the game (scores ~354). Each round, we extract features and V2's
decisions. We score V2's chosen items and backprop to maximize score-delta.
This is reward-weighted behavioral cloning at scale.

The scorer learns which (bot, item) pairs lead to score increases.
After training, MLPlanner uses the scorer for beam search assignments.

6+ hours on RTX A3000: ~2500 episodes × 500 rounds × 20 bots = 25M updates.

Usage:
    py -m ml.ppo_trainer \
      --recon logs/74001e7f_2026-03-16_score274_recon.json \
      --checkpoint models/scorer_imitation_v1.pt \
      --episodes 2500 --output models/scorer_ppo_v1.pt
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Simulering.offline.bot_adapter import BotAdapter
from Simulering.offline.simulator import Simulator
from bot.engine.pathfinding import PathEngine
from bot.models import GameState, Grid, Item
from bot.strategy.task import TaskType
from ml.candidate_generator import DELIVER, IDLE
from ml.feature_extractor import FeatureContext, FeatureExtractor
from ml.scorer import ScorerMLP


def _build_shared_pe(recon: dict):
    sim_tmp = Simulator.from_recon_data(recon)
    tmp_state = sim_tmp.reset()
    tmp_gs = GameState.from_dict(tmp_state.to_dict())
    shelves = frozenset(sim_tmp.shelves)
    merged = Grid(tmp_gs.grid.width, tmp_gs.grid.height, tmp_gs.grid.walls | shelves)
    pe = PathEngine()
    pe.set_grid(merged, tmp_gs.drop_off)
    for z in tmp_gs.drop_off_zones:
        pe.distance((0, 0), z)
    return pe, tmp_gs.drop_off_zones


def collect_episode_with_rewards(
    recon: dict,
    pe: PathEngine,
    drop_off_zones: tuple,
    type_index: dict,
) -> list[tuple[torch.Tensor, float]]:
    """Run V2TaskPlanner, collect (features, reward-weighted label) per decision.

    For each round:
    - V2 makes assignments
    - We encode all (bot, item) candidates
    - V2's chosen items get label = shaped_reward
    - Non-chosen items get label = 0.0
    - shaped_reward = score_delta + potential_improvement (dense signal)
    """
    sim = Simulator.from_recon_data(recon)
    adapter = BotAdapter(suppress_logs=True)
    state = sim.reset()

    data: list[tuple[torch.Tensor, float]] = []
    prev_score = 0

    for round_idx in range(sim.max_rounds):
        state_dict = state.to_dict()

        # V2 makes decisions
        response = adapter(state_dict)
        actions = response.get("actions", [])

        # Parse state
        gs = GameState.from_dict(state_dict)
        active = gs.active_orders[0] if gs.active_orders else None
        preview = gs.preview_orders[0] if gs.preview_orders else None

        # Get V2's assignments
        coord = adapter._coordinator
        if not coord:
            state, done = sim.step(actions)
            continue

        v2_decisions: dict[int, str] = {}
        claimed = set()
        for bid, a in coord._assignments.items():
            if a.task is None or a.task.task_type == TaskType.IDLE:
                v2_decisions[bid] = IDLE
            elif a.task.task_type == TaskType.DELIVER:
                v2_decisions[bid] = DELIVER
            elif a.task.item_id:
                v2_decisions[bid] = a.task.item_id
                claimed.add(a.task.item_id)
            else:
                v2_decisions[bid] = IDLE

        ctx = FeatureContext(
            assignments=coord._assignments,
            claimed_items=claimed,
            active_order=active,
            preview_order=preview,
            bot_positions=[b.position for b in gs.bots],
            n_bots=len(gs.bots),
            max_dist=60,
            item_type_index=type_index,
            drop_off_zones=drop_off_zones,
        )

        # Step sim to get reward
        state, done = sim.step(actions)
        score_delta = sim._score - prev_score
        prev_score = sim._score

        # Compute shaped reward: score_delta normalized + small potential bonus
        # Score delta of 1 (item delivered) → reward ~0.5
        # Score delta of 6 (order completed) → reward ~1.0
        shaped_reward = min(score_delta / 10.0, 1.0) if score_delta > 0 else 0.0

        # Only generate training data on rounds with score changes (sparse but high quality)
        # Also sample 10% of other rounds for negative examples
        if score_delta == 0 and round_idx % 10 != 0:
            if done:
                break
            continue

        # Encode candidates for each bot
        for bot in gs.bots:
            v2_choice = v2_decisions.get(bot.id, IDLE)

            # Top-10 nearest items
            items_with_dist = []
            for item in gs.items:
                d = pe.distance(bot.position, item.position)
                if d >= 9999:
                    d = min(
                        (pe.distance(bot.position, (item.position[0]+dx, item.position[1]+dy))
                         for dx, dy in ((0,-1),(0,1),(-1,0),(1,0))),
                        default=9999,
                    )
                if d < 9999:
                    items_with_dist.append((d, item))

            items_with_dist.sort(key=lambda x: x[0])
            candidates = items_with_dist[:10]

            # Ensure V2's choice is in candidates
            if v2_choice not in (DELIVER, IDLE):
                chosen_ids = {item.id for _, item in candidates}
                if v2_choice not in chosen_ids:
                    for item in gs.items:
                        if item.id == v2_choice:
                            d = pe.distance(bot.position, item.position)
                            if d >= 9999:
                                d = min(
                                    (pe.distance(bot.position, (item.position[0]+dx, item.position[1]+dy))
                                     for dx, dy in ((0,-1),(0,1),(-1,0),(1,0))),
                                    default=9999,
                                )
                            candidates.append((d, item))
                            break

            for _, item in candidates:
                feat = FeatureExtractor.encode_pair(bot, item, gs, pe, ctx)
                label = shaped_reward if item.id == v2_choice else 0.0
                data.append((feat, label))

            # DELIVER / IDLE encoding
            if len(bot.inventory) > 0:
                dummy = Item(id="__deliver__", type="__deliver__",
                           position=min(drop_off_zones, key=lambda z: pe.distance(bot.position, z)))
                feat = FeatureExtractor.encode_pair(bot, dummy, gs, pe, ctx)
                label = shaped_reward if v2_choice == DELIVER else 0.0
                data.append((feat, label))

            dummy_idle = Item(id="__idle__", type="__idle__", position=bot.position)
            feat = FeatureExtractor.encode_pair(bot, dummy_idle, gs, pe, ctx)
            label = shaped_reward if v2_choice == IDLE else 0.0
            data.append((feat, label))

        if done:
            break

    adapter.reset()
    return data, sim._score


def train_ppo(
    recon_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    episodes: int = 2500,
    lr: float = 5e-4,
    batch_size: int = 512,
    update_every: int = 5,
    save_interval: int = 100,
) -> None:
    recon = json.loads(recon_path.read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scorer = ScorerMLP()
    if checkpoint_path.exists():
        scorer.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("Starting from scratch")

    scorer = scorer.to(device).train()
    print(f"Device: {device}")
    print(f"Episodes: {episodes}, update every {update_every}")

    pe, drop_off_zones = _build_shared_pe(recon)
    all_types = sorted(recon.get("shelf_map", {}).keys())
    type_index = {t: i for i, t in enumerate(all_types)}

    optimizer = torch.optim.AdamW(scorer.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=episodes // update_every)

    interrupted = False
    def handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\nInterrupted — saving...")
    signal.signal(signal.SIGINT, handler)

    # Training buffer
    buffer: list[tuple[torch.Tensor, float]] = []
    best_val_loss = float("inf")
    scores_window: list[int] = []
    t0 = time.time()

    print(f"\n{'Ep':>5} {'Score':>6} {'Avg50':>6} {'Samples':>8} "
          f"{'Loss':>8} {'PosRate':>7} {'LR':>10} {'Time':>7}")
    print("-" * 65)

    total_updates = 0
    for ep in range(1, episodes + 1):
        if interrupted:
            break

        # Collect episode
        ep_data, ep_score = collect_episode_with_rewards(
            recon, pe, drop_off_zones, type_index,
        )
        buffer.extend(ep_data)
        scores_window.append(ep_score)
        if len(scores_window) > 50:
            scores_window.pop(0)

        # Train on buffer every N episodes
        if ep % update_every == 0 and buffer:
            # Build tensors
            features = torch.stack([f for f, _ in buffer]).to(device)
            labels = torch.tensor([l for _, l in buffer], dtype=torch.float32).unsqueeze(1).to(device)

            # Compute positive rate
            pos_count = (labels > 0).sum().item()
            pos_rate = pos_count / max(len(labels), 1)

            # Weighted BCE: upweight positive examples
            pos_weight = max((1 - pos_rate) / max(pos_rate, 0.001), 1.0)
            pos_weight = min(pos_weight, 50.0)  # cap

            # Train for a few passes
            dataset = torch.utils.data.TensorDataset(features, labels)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

            epoch_loss = 0.0
            n_batches = 0
            scorer.train()
            for xb, yb in loader:
                pred = scorer(xb)
                raw_bce = F.binary_cross_entropy(pred, yb, reduction="none")
                weight = torch.where(yb > 0.01, pos_weight, 1.0)
                loss = (raw_bce * weight).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(scorer.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            total_updates += 1

            # Save best
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                torch.save(scorer.state_dict(), output_path)

            # Periodic checkpoint
            if ep % save_interval == 0:
                periodic = output_path.parent / f"{output_path.stem}_ep{ep}.pt"
                torch.save(scorer.state_dict(), periodic)

            elapsed = time.time() - t0
            avg50 = sum(scores_window) / len(scores_window)
            current_lr = scheduler.get_last_lr()[0]

            print(f"{ep:>5} {ep_score:>6} {avg50:>6.1f} {len(buffer):>8} "
                  f"{avg_loss:>8.5f} {pos_rate:>6.1%} {current_lr:>10.6f} {elapsed:>6.0f}s")

            # Clear buffer
            buffer.clear()

        elif ep <= 5 or ep % 50 == 0:
            elapsed = time.time() - t0
            avg50 = sum(scores_window) / len(scores_window) if scores_window else 0
            print(f"{ep:>5} {ep_score:>6} {avg50:>6.1f} {len(buffer):>8} "
                  f"{'---':>8} {'---':>7} {'---':>10} {elapsed:>6.0f}s")

    # Final save
    final_path = output_path.parent / f"{output_path.stem}_final.pt"
    torch.save(scorer.state_dict(), final_path)

    elapsed = time.time() - t0
    avg = sum(scores_window) / len(scores_window) if scores_window else 0
    print(f"\n{'='*65}")
    print(f"Done. Episodes: {ep}  Avg50: {avg:.1f}  Updates: {total_updates}")
    print(f"Total time: {elapsed/3600:.1f} hours ({elapsed:.0f}s)")
    print(f"Best checkpoint: {output_path}")
    print(f"Final checkpoint: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="PPO-style training for ScorerMLP")
    parser.add_argument("--recon", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="models/scorer_ppo_v1.pt")
    parser.add_argument("--episodes", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--update-every", type=int, default=5)
    parser.add_argument("--save-interval", type=int, default=100)
    args = parser.parse_args()

    train_ppo(
        Path(args.recon), Path(args.checkpoint), Path(args.output),
        episodes=args.episodes, lr=args.lr, batch_size=args.batch_size,
        update_every=args.update_every, save_interval=args.save_interval,
    )


if __name__ == "__main__":
    main()
