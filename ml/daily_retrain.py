"""
Daily retraining workflow — one command to collect data, train, and validate.

Idempotent: skips data collection if pickle already exists for today's date.

Usage:
    py -m ml.daily_retrain \
      --recon logs/74001e7f_2026-03-16_score274_recon.json \
      --n-games 50 \
      --epochs 20
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Daily retrain pipeline")
    parser.add_argument("--recon", required=True, help="Recon JSON file")
    parser.add_argument("--n-games", type=int, default=50, help="Sim games for data collection")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-validate", type=int, default=5, help="Validation runs")
    args = parser.parse_args()

    recon_path = Path(args.recon)
    if not recon_path.exists():
        print(f"Recon not found: {recon_path}")
        sys.exit(1)

    today = date.today().isoformat()
    stem = recon_path.stem.split("_")[0]  # fingerprint

    data_path = Path("data") / f"training_{stem}_{today}.pkl"
    ckpt_name = f"scorer_{stem}_{today}.pt"
    model_dir = Path("models")

    total_t0 = time.time()

    # --- Step 1: Collect training data ---
    print(f"{'='*55}")
    print(f"STEP 1: Collect training data ({args.n_games} games)")
    print(f"{'='*55}")

    if data_path.exists():
        print(f"  SKIP — {data_path} already exists (idempotent)")
    else:
        try:
            from ml.collect_training_data import collect
            collect(recon_path, args.n_games, data_path)
        except Exception as e:
            print(f"\nERROR in Step 1 (data collection): {e}")
            sys.exit(1)

    # --- Step 2: Train model ---
    print(f"\n{'='*55}")
    print(f"STEP 2: Train ScorerMLP ({args.epochs} epochs)")
    print(f"{'='*55}")

    try:
        from ml.train import train
        ckpt_path = train(
            data_path, model_dir, args.epochs, args.batch_size, args.lr, ckpt_name
        )
    except Exception as e:
        print(f"\nERROR in Step 2 (training): {e}")
        sys.exit(1)

    # --- Step 3: Validate ---
    print(f"\n{'='*55}")
    print(f"STEP 3: Validate ({args.n_validate} runs)")
    print(f"{'='*55}")

    try:
        from ml.validate import validate
        validate(recon_path, ckpt_path, args.n_validate)
    except Exception as e:
        print(f"\nERROR in Step 3 (validation): {e}")
        sys.exit(1)

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*55}")
    print(f"DONE — Total time: {total_elapsed / 60:.1f} minutes")
    print(f"Checkpoint: {model_dir / ckpt_name}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
