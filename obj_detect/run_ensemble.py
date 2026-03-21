"""
Sandbox inference — Ensemble: YOLOv8l + YOLOv8x + YOLOv8l-1280 with WBF.
Falls back gracefully if any model is missing.

Usage: python run.py --input /data/images --output /output/predictions.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion


def load_models():
    """Load all available models."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    weights_list = []

    model_files = [
        ("detect_l.pt", 1.0),        # YOLOv8l 640
        ("detect_x.pt", 1.5),        # YOLOv8x 640 (best single, higher weight)
        ("detect_l1280.pt", 1.5),    # YOLOv8l 1280 (higher weight)
    ]

    for fname, weight in model_files:
        p = Path(__file__).parent / fname
        if p.exists():
            m = YOLO(str(p))
            models.append((m, fname))
            weights_list.append(weight)

    if not models:
        # Fallback: single model
        p = Path(__file__).parent / "detect_model.pt"
        if p.exists():
            models.append((YOLO(str(p)), "detect_model.pt"))
            weights_list.append(1.0)

    return models, weights_list


def predict_single(model, img_path, device, imgsz=640):
    """Run single model prediction, return normalized boxes + scores + labels."""
    results = model(
        str(img_path),
        device=device,
        verbose=False,
        conf=0.001,
        iou=0.7,
        max_det=500,
        imgsz=imgsz,
    )

    boxes, scores, labels = [], [], []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        img_h, img_w = r.orig_shape
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
            # Normalize to [0, 1] for WBF
            boxes.append([x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h])
            scores.append(float(r.boxes.conf[i].item()))
            labels.append(int(r.boxes.cls[i].item()))

    return boxes, scores, labels, (img_w, img_h)


def ensemble_predict(models, weights_list, img_path, device):
    """Run all models and fuse with WBF."""
    all_boxes, all_scores, all_labels = [], [], []
    img_size = None

    for model, fname in models:
        imgsz = 1280 if "1280" in fname else 640
        boxes, scores, labels, size = predict_single(model, img_path, device, imgsz=imgsz)
        all_boxes.append(boxes if boxes else [[0, 0, 0, 0]])
        all_scores.append(scores if scores else [0])
        all_labels.append(labels if labels else [0])
        img_size = size

    if len(models) == 1:
        # No ensemble needed
        fused_boxes = all_boxes[0]
        fused_scores = all_scores[0]
        fused_labels = all_labels[0]
    else:
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            weights=weights_list,
            iou_thr=0.7,
            skip_box_thr=0.001,
        )

    return fused_boxes, fused_scores, fused_labels, img_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models, weights_list = load_models()
    print(f"Loaded {len(models)} models: {[name for _, name in models]}")

    predictions = []
    input_dir = Path(args.input)
    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    for img_path in image_files:
        image_id = int(img_path.stem.split("_")[-1])
        fused_boxes, fused_scores, fused_labels, (img_w, img_h) = ensemble_predict(
            models, weights_list, img_path, device
        )

        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            if isinstance(box, np.ndarray):
                x1, y1, x2, y2 = box.tolist()
            else:
                x1, y1, x2, y2 = box
            # Denormalize from [0,1] to pixels
            x1_px = x1 * img_w
            y1_px = y1 * img_h
            w_px = (x2 - x1) * img_w
            h_px = (y2 - y1) * img_h

            if w_px < 1 or h_px < 1:
                continue

            predictions.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [round(x1_px, 1), round(y1_px, 1), round(w_px, 1), round(h_px, 1)],
                "score": round(float(score), 4),
            })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

    print(f"Wrote {len(predictions)} predictions for {len(image_files)} images")


if __name__ == "__main__":
    main()
