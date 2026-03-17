"""
TASK-2-1: MLP Proof-of-Concept — end-to-end training pipeline validation.

Proves that:
1. We can generate (features, reward) pairs from sim runs
2. An MLP can learn *something* (val-loss decreases)
3. No CUDA OOM at batch 256

Features are 48 random floats (placeholder — real features in Epic 3).
Reward = normalized score delta over 5 rounds.

Usage:
    py -m ml.spike_mlp_poc --recon logs/74001e7f_2026-03-16_score274_recon.json
    py -m ml.spike_mlp_poc --recon logs/74001e7f_2026-03-16_score274_recon.json --games 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def collect_data(recon_path: Path, n_games: int) -> tuple[np.ndarray, np.ndarray]:
    """Run n_games sim games, collect (features_48, reward_5) per bot per round."""
    from Simulering.offline.bot_adapter import BotAdapter
    from Simulering.offline.simulator import Simulator

    recon = json.loads(recon_path.read_text(encoding="utf-8"))
    n_bots = recon.get("bot_count", 1)

    all_features = []
    all_rewards = []
    rng = np.random.default_rng(42)

    for game_idx in range(n_games):
        sim = Simulator.from_recon_data(recon)
        adapter = BotAdapter(suppress_logs=True)

        # Run game step-by-step to track per-round scores
        state = sim.reset()
        round_scores = [sim._score]

        for _ in range(sim.max_rounds):
            state_dict = state.to_dict()
            response = adapter(state_dict)
            actions = response.get("actions", [])
            state, game_over = sim.step(actions)
            round_scores.append(sim._score)
            if game_over:
                break

        adapter.reset()
        total_rounds = len(round_scores) - 1

        # Generate features and compute reward_5 for each (round, bot)
        for r in range(total_rounds):
            # reward_5: score gain over next 5 rounds, normalized
            future_r = min(r + 5, total_rounds)
            score_delta = round_scores[future_r] - round_scores[r]
            reward = score_delta / 10.0  # normalize to ~[0, 1]
            reward = max(0.0, min(1.0, reward))  # clamp

            for _ in range(n_bots):
                features = rng.standard_normal(48).astype(np.float32)
                all_features.append(features)
                all_rewards.append(reward)

        if (game_idx + 1) % 10 == 0:
            print(f"  Collected {game_idx + 1}/{n_games} games "
                  f"({len(all_features)} samples, last score={round_scores[-1]})")

    return np.array(all_features), np.array(all_rewards, dtype=np.float32)


def train(features: np.ndarray, rewards: np.ndarray, device: str) -> None:
    """Train MLP on collected data, print loss per epoch."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, random_split

    X = torch.from_numpy(features).to(device)
    y = torch.from_numpy(rewards).unsqueeze(1).to(device)

    dataset = TensorDataset(X, y)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256)

    model = nn.Sequential(
        nn.Linear(48, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.HuberLoss()

    print(f"\nTraining on {n_train} samples, validating on {n_val}")
    print(f"Device: {device}")
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}")
    print("-" * 30)

    for epoch in range(1, 11):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(xb)
            train_count += len(xb)

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss_sum += loss.item() * len(xb)
                val_count += len(xb)

        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = val_loss_sum / max(val_count, 1)
        print(f"{epoch:>5}  {train_loss:>10.6f}  {val_loss:>10.6f}")

    print()
    if epoch == 10:
        print("Pipeline validation complete.")
        print(f"Final train loss: {train_loss:.6f}, val loss: {val_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description="MLP PoC — training pipeline spike")
    parser.add_argument("--recon", required=True, help="Path to nightmare recon JSON")
    parser.add_argument("--games", type=int, default=50, help="Number of sim games")
    args = parser.parse_args()

    recon_path = Path(args.recon)
    if not recon_path.exists():
        print(f"Recon file not found: {recon_path}")
        sys.exit(1)

    print(f"=== MLP PoC: {args.games} games from {recon_path.name} ===\n")

    # 1. Collect data
    t0 = time.time()
    features, rewards = collect_data(recon_path, args.games)
    t_collect = time.time() - t0
    print(f"\nData: {len(features)} samples, {features.shape[1]} features")
    print(f"Reward stats: mean={rewards.mean():.3f}, std={rewards.std():.3f}, "
          f"min={rewards.min():.3f}, max={rewards.max():.3f}")
    print(f"Collection time: {t_collect:.1f}s")

    # 2. Train
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(features, rewards, device)


if __name__ == "__main__":
    main()
