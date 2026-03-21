"""
Train YOLOv8 1-class product detector.

Uses aggressive augmentation for dense shelf detection.
Exports best.pt as detect_model.pt for submission.
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 product detector")
    parser.add_argument("--data", type=str, default="dataset/dataset.yaml",
                        help="Path to dataset.yaml")
    parser.add_argument("--model", type=str, default="yolov8m.pt",
                        help="Pretrained model (yolov8n/s/m/l/x.pt)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (adjust for GPU memory)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Training image size (640 or 1280)")
    parser.add_argument("--name", type=str, default="product_detector",
                        help="Experiment name")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--output", type=str, default="detect_model.pt",
                        help="Output path for best weights")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of dataloader workers")
    return parser.parse_args()


def main():
    args = parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

    print(f"Model: {args.model}")
    print(f"Dataset: {data_yaml}")
    print(f"Image size: {args.imgsz}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")

    # Load pretrained model
    model = YOLO(args.model)

    # Train with aggressive augmentation for dense shelf detection
    results = model.train(
        data=str(data_yaml.resolve()),
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        resume=args.resume,

        # Augmentation — aggressive for dense product shelves
        mosaic=1.0,          # Full mosaic augmentation
        mixup=0.3,           # Mix two images
        copy_paste=0.0,      # Disabled — we do custom copy-paste
        erasing=0.4,         # Random erasing
        degrees=5.0,         # Small rotation
        translate=0.1,       # Translation
        scale=0.5,           # Scale augmentation
        shear=2.0,           # Small shear
        perspective=0.0001,  # Perspective
        flipud=0.0,          # No vertical flip (products have orientation)
        fliplr=0.5,          # Horizontal flip OK
        hsv_h=0.015,         # Hue shift
        hsv_s=0.4,           # Saturation shift (store lighting varies)
        hsv_v=0.4,           # Value shift

        # Training params
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,            # Final LR factor
        weight_decay=0.0005,
        warmup_epochs=5,
        close_mosaic=10,     # Disable mosaic last 10 epochs

        # Output
        save=True,
        save_period=-1,      # Save only best + last
        plots=True,
        verbose=True,
    )

    # --- Export best weights ---
    # Find best.pt from training run
    runs_dir = Path("runs/detect") / args.name
    best_pt = runs_dir / "weights" / "best.pt"

    if not best_pt.exists():
        # Try with incremented name
        for p in sorted(Path("runs/detect").glob(f"{args.name}*")):
            candidate = p / "weights" / "best.pt"
            if candidate.exists():
                best_pt = candidate
                break

    output_path = Path(args.output)
    if best_pt.exists():
        shutil.copy2(best_pt, output_path)
        print(f"\nBest weights copied to {output_path}")
        print(f"  Source: {best_pt}")
    else:
        print(f"\nWARNING: Could not find best.pt at {best_pt}")
        print("  Check runs/detect/ for training output")

    # Print summary
    if results:
        print(f"\nTraining complete!")
        print(f"  mAP@0.5: {getattr(results, 'maps', 'N/A')}")


if __name__ == "__main__":
    main()
