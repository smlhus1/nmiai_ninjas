"""
Train YOLOv8 detector. Works for both 1-class and multi-class datasets.
Handles PyTorch 2.6 + ultralytics 8.1.0 compatibility.

Usage:
  py train.py --data dataset_multiclass/dataset.yaml --model yolov8s.pt --epochs 30 --batch 4
  py train.py --data dataset/dataset.yaml --model yolov8m.pt --epochs 100 --batch 8 --imgsz 1280
"""

import argparse

import torch

# Fix PyTorch 2.6 weights_only=True breaking ultralytics 8.1.0
_orig_load = torch.load
torch.load = lambda *args, **kwargs: _orig_load(*args, **{**kwargs, 'weights_only': False})

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset.yaml")
    parser.add_argument("--model", default="yolov8s.pt", help="Base model (yolov8n/s/m/l/x.pt)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", default=None, help="Run name")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    model = YOLO(args.model)

    run_name = args.name or f"{args.model.replace('.pt','')}_ep{args.epochs}_bs{args.batch}_img{args.imgsz}"

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        amp=True,
        project="runs",
        name=run_name,
        exist_ok=True,
        verbose=True,
        patience=15,
        # Augmentation
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.0,
        erasing=0.3,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        scale=0.5,
        translate=0.1,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        resume=args.resume,
    )

    print(f"\nTraining complete. Best model: runs/{run_name}/weights/best.pt")
    print(f"Results: {results.results_dict}")


if __name__ == "__main__":
    main()
