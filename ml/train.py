"""
Training loop for ScorerMLP.

Loads TrainingDataset, trains with AdamW + CosineAnnealingLR,
saves checkpoint. Handles Ctrl+C gracefully.

Usage:
    py -m ml.train \
      --data data/training_74001e7f_2026-03-16.pkl \
      --output models/ \
      --epochs 20 \
      --batch-size 256 \
      --lr 1e-3
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.dataset import TrainingDataset
from ml.scorer import ScorerMLP


def train(
    data_path: Path,
    output_dir: Path,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    checkpoint_name: str | None = None,
) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    dataset = TrainingDataset(data_path)
    n_total = len(dataset)
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=0)

    print(f"Data: {n_total} samples ({n_train} train, {n_val} val)")

    # Model
    model = ScorerMLP().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Detect imitation data: check if labels are binary (0.0 or 1.0)
    sample_labels = [dataset[i][1].item() for i in range(min(1000, len(dataset)))]
    is_binary = all(l in (0.0, 1.0) for l in sample_labels)
    pos_rate = sum(1 for l in sample_labels if l > 0.5) / len(sample_labels)

    if is_binary and pos_rate < 0.3:
        # Imitation mode: weighted BCE to handle class imbalance
        pos_weight = (1.0 - pos_rate) / max(pos_rate, 0.01)
        loss_fn = None  # Use manual weighted BCE
        print(f"Loss: Weighted BCE (pos_rate={pos_rate:.1%}, pos_weight={pos_weight:.1f})")
    else:
        loss_fn = nn.HuberLoss(delta=0.1)
        pos_weight = 0.0
        print(f"Loss: HuberLoss")

    # Graceful Ctrl+C
    interrupted = False

    def signal_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\nInterrupted — saving checkpoint from last completed epoch...")

    signal.signal(signal.SIGINT, signal_handler)

    # Training loop
    output_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_name is None:
        stem = data_path.stem.replace("training_", "scorer_")
        checkpoint_name = f"{stem}.pt"
    ckpt_path = output_dir / checkpoint_name

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}  {'LR':>10}  {'Time':>6}")
    print("-" * 50)

    best_val = float("inf")
    last_saved_epoch = 0

    for epoch in range(1, epochs + 1):
        if interrupted:
            break

        t0 = time.time()

        # Train
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            if loss_fn is not None:
                loss = loss_fn(pred, yb)
            else:
                # Weighted BCE for imitation learning
                raw = nn.functional.binary_cross_entropy(pred, yb, reduction="none")
                weight = torch.where(yb > 0.5, pos_weight, 1.0)
                loss = (raw * weight).mean()
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
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                if loss_fn is not None:
                    loss = loss_fn(pred, yb)
                else:
                    raw = nn.functional.binary_cross_entropy(pred, yb, reduction="none")
                    weight = torch.where(yb > 0.5, pos_weight, 1.0)
                    loss = (raw * weight).mean()
                val_loss_sum += loss.item() * len(xb)
                val_count += len(xb)

        scheduler.step()

        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = val_loss_sum / max(val_count, 1)
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(f"{epoch:>5}  {train_loss:>10.6f}  {val_loss:>10.6f}  {current_lr:>10.6f}  {elapsed:>5.1f}s")

        # Save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
            last_saved_epoch = epoch

    print(f"\nBest val loss: {best_val:.6f} (epoch {last_saved_epoch})")
    print(f"Checkpoint saved: {ckpt_path}")

    # Save final checkpoint if different from best
    if last_saved_epoch != epoch and not interrupted:
        final_path = output_dir / checkpoint_name.replace(".pt", "_final.pt")
        torch.save(model.state_dict(), final_path)
        print(f"Final checkpoint: {final_path}")

    return ckpt_path


def main():
    parser = argparse.ArgumentParser(description="Train ScorerMLP")
    parser.add_argument("--data", required=True, help="Training data pickle")
    parser.add_argument("--output", default="models/", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--name", help="Checkpoint filename (default: derived from data)")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        sys.exit(1)

    train(data_path, Path(args.output), args.epochs, args.batch_size, args.lr, args.name)


if __name__ == "__main__":
    main()
