"""
Local validation script for detection + classification pipeline.

Runs run.py logic on the val split, then evaluates:
  - Detection mAP@0.5 (1-class)
  - Classification mAP@0.5 (356 classes)
  - Combined score: 0.7 * det_mAP + 0.3 * cls_mAP

Also validates COCO output format correctness.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Validate detection + classification pipeline")
    parser.add_argument("--input", type=str, default="dataset/val/images",
                        help="Directory with validation images")
    parser.add_argument("--gt", type=str, default="dataset/annotations.json",
                        help="Path to full COCO annotations")
    parser.add_argument("--split-info", type=str, default="dataset/split_info.json",
                        help="Path to split info (for val image IDs)")
    parser.add_argument("--predictions", type=str, default=None,
                        help="Path to predictions.json (if already generated)")
    parser.add_argument("--detect-model", type=str, default="detect_model.pt",
                        help="Detection model path")
    parser.add_argument("--embeddings", type=str, default="weights/reference_embeddings.pt",
                        help="Reference embeddings path")
    parser.add_argument("--classify-model", type=str, default=None,
                        help="Optional classifier model path")
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def compute_iou(box1, box2):
    """Compute IoU between two COCO-format boxes [x, y, w, h]."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter = max(0, xb - xa) * max(0, yb - ya)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def compute_ap(recalls, precisions):
    """Compute AP using 101-point interpolation (COCO style)."""
    recalls = np.array(recalls)
    precisions = np.array(precisions)

    # Sort by recall
    sorted_indices = np.argsort(recalls)
    recalls = recalls[sorted_indices]
    precisions = precisions[sorted_indices]

    # 101-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        mask = recalls >= t
        if mask.any():
            ap += precisions[mask].max()
    ap /= 101.0
    return ap


def evaluate_detection(predictions, gt_annotations, gt_images, iou_threshold=0.5):
    """
    Evaluate 1-class detection mAP@0.5.
    All predictions and GT are treated as class 0 (product).
    """
    # Group GT by image
    gt_by_image = {}
    for ann in gt_annotations:
        img_id = ann["image_id"]
        if img_id not in gt_by_image:
            gt_by_image[img_id] = []
        gt_by_image[img_id].append(ann)

    # Sort predictions by score (descending)
    preds_sorted = sorted(predictions, key=lambda p: p["score"], reverse=True)

    tp = []
    fp = []
    total_gt = len(gt_annotations)

    # Track matched GT per image
    matched = {}

    for pred in preds_sorted:
        img_id = pred["image_id"]
        gt_boxes = gt_by_image.get(img_id, [])

        best_iou = 0.0
        best_gt_idx = -1

        for i, gt in enumerate(gt_boxes):
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_threshold:
            gt_key = (img_id, best_gt_idx)
            if gt_key not in matched:
                tp.append(1)
                fp.append(0)
                matched[gt_key] = True
            else:
                tp.append(0)
                fp.append(1)
        else:
            tp.append(0)
            fp.append(1)

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / total_gt if total_gt > 0 else tp_cumsum
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    if len(recalls) == 0:
        return 0.0

    return compute_ap(recalls, precisions)


def evaluate_classification(predictions, gt_annotations, iou_threshold=0.5):
    """
    Evaluate per-category classification mAP@0.5.
    Only matched detections (IoU >= threshold) contribute.
    """
    # Group GT by image
    gt_by_image = {}
    for ann in gt_annotations:
        img_id = ann["image_id"]
        if img_id not in gt_by_image:
            gt_by_image[img_id] = []
        gt_by_image[img_id].append(ann)

    # Collect per-category results
    from collections import defaultdict
    cat_preds = defaultdict(list)  # cat_id -> [(score, is_tp)]
    cat_gt_counts = defaultdict(int)

    # Count GT per category
    for ann in gt_annotations:
        cat_gt_counts[ann["category_id"]] += 1

    # Match predictions to GT (category-aware)
    preds_sorted = sorted(predictions, key=lambda p: p["score"], reverse=True)
    matched = {}

    for pred in preds_sorted:
        img_id = pred["image_id"]
        pred_cat = pred["category_id"]
        gt_boxes = gt_by_image.get(img_id, [])

        best_iou = 0.0
        best_gt_idx = -1

        # Match only against same-category GT
        for i, gt in enumerate(gt_boxes):
            if gt["category_id"] != pred_cat:
                continue
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_threshold:
            gt_key = (img_id, best_gt_idx)
            if gt_key not in matched:
                cat_preds[pred_cat].append((pred["score"], True))
                matched[gt_key] = True
            else:
                cat_preds[pred_cat].append((pred["score"], False))
        else:
            cat_preds[pred_cat].append((pred["score"], False))

    # Compute AP per category
    aps = []
    for cat_id in sorted(cat_gt_counts.keys()):
        n_gt = cat_gt_counts[cat_id]
        if n_gt == 0:
            continue

        preds = cat_preds.get(cat_id, [])
        if not preds:
            aps.append(0.0)
            continue

        # Sort by score descending
        preds.sort(key=lambda x: x[0], reverse=True)

        tp = [1 if is_tp else 0 for _, is_tp in preds]
        fp = [0 if is_tp else 1 for _, is_tp in preds]

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / n_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        ap = compute_ap(recalls, precisions)
        aps.append(ap)

    return np.mean(aps) if aps else 0.0


def validate_format(predictions):
    """Check predictions.json format correctness."""
    errors = []

    if not isinstance(predictions, list):
        errors.append("Predictions must be a list")
        return errors

    for i, pred in enumerate(predictions):
        # Required fields
        for field in ["image_id", "category_id", "bbox", "score"]:
            if field not in pred:
                errors.append(f"Prediction {i}: missing '{field}'")

        # Category ID range
        if "category_id" in pred:
            cat_id = pred["category_id"]
            if not isinstance(cat_id, int) or cat_id < 0 or cat_id > 355:
                errors.append(f"Prediction {i}: invalid category_id {cat_id} (must be 0-355)")

        # BBox format: [x, y, w, h] in pixels
        if "bbox" in pred:
            bbox = pred["bbox"]
            if not isinstance(bbox, list) or len(bbox) != 4:
                errors.append(f"Prediction {i}: bbox must be [x, y, w, h], got {bbox}")
            elif any(v < 0 for v in bbox):
                errors.append(f"Prediction {i}: negative bbox values: {bbox}")
            elif bbox[2] <= 0 or bbox[3] <= 0:
                errors.append(f"Prediction {i}: zero/negative width or height: {bbox}")
            # Check if bbox looks normalized (all values 0-1) — it shouldn't be!
            elif all(0 <= v <= 1.0 for v in bbox):
                errors.append(
                    f"Prediction {i}: bbox {bbox} looks NORMALIZED — "
                    f"COCO format requires PIXEL coordinates!"
                )

        # Score range
        if "score" in pred:
            score = pred["score"]
            if not isinstance(score, (int, float)) or score < 0 or score > 1:
                errors.append(f"Prediction {i}: score {score} out of [0, 1] range")

    return errors


def main():
    args = parse_args()

    # --- Generate predictions if needed ---
    if args.predictions and Path(args.predictions).exists():
        print(f"Loading existing predictions: {args.predictions}")
        with open(args.predictions, "r") as f:
            predictions = json.load(f)
    else:
        print("Running inference pipeline...")
        import subprocess
        import sys

        pred_dir = Path("val_predictions")
        pred_dir.mkdir(exist_ok=True)

        cmd = [
            sys.executable, "run.py",
            "--input", args.input,
            "--output", str(pred_dir),
            "--detect-model", args.detect_model,
            "--embeddings", args.embeddings,
            "--conf-threshold", str(args.conf_threshold),
            "--imgsz", str(args.imgsz),
        ]
        if args.classify_model:
            cmd.extend(["--classify-model", args.classify_model])

        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            return

        pred_path = pred_dir / "predictions.json"
        with open(pred_path, "r") as f:
            predictions = json.load(f)

    # --- Load GT ---
    print(f"Loading ground truth: {args.gt}")
    with open(args.gt, "r") as f:
        gt_data = json.load(f)

    # Filter GT to val images only
    if Path(args.split_info).exists():
        with open(args.split_info, "r") as f:
            split_info = json.load(f)
        val_image_ids = set(split_info["val_image_ids"])
    else:
        # Use all images if no split info
        val_image_ids = {img["id"] for img in gt_data["images"]}

    gt_images = {img["id"]: img for img in gt_data["images"] if img["id"] in val_image_ids}
    gt_annotations = [ann for ann in gt_data["annotations"] if ann["image_id"] in val_image_ids]

    # Filter predictions to val images
    predictions = [p for p in predictions if p["image_id"] in val_image_ids]

    print(f"Val images: {len(gt_images)}")
    print(f"GT annotations: {len(gt_annotations)}")
    print(f"Predictions: {len(predictions)}")

    # --- Format validation ---
    print("\n=== Format Validation ===")
    errors = validate_format(predictions)
    if errors:
        print(f"  ERRORS ({len(errors)}):")
        for err in errors[:20]:
            print(f"    - {err}")
        if len(errors) > 20:
            print(f"    ... and {len(errors) - 20} more")
    else:
        print("  All format checks PASSED")

    # --- Detection mAP ---
    print("\n=== Detection mAP@0.5 (1-class) ===")
    det_map = evaluate_detection(predictions, gt_annotations, gt_images)
    print(f"  Detection mAP@0.5: {det_map:.4f}")

    # --- Classification mAP ---
    print("\n=== Classification mAP@0.5 ===")
    cls_map = evaluate_classification(predictions, gt_annotations)
    print(f"  Classification mAP@0.5: {cls_map:.4f}")

    # --- Combined score ---
    combined = 0.7 * det_map + 0.3 * cls_map
    print(f"\n=== Combined Score ===")
    print(f"  0.7 * {det_map:.4f} + 0.3 * {cls_map:.4f} = {combined:.4f}")

    # --- Stats ---
    print(f"\n=== Statistics ===")
    print(f"  Avg predictions per image: {len(predictions) / max(1, len(gt_images)):.1f}")
    print(f"  Avg GT per image: {len(gt_annotations) / max(1, len(gt_images)):.1f}")

    if predictions:
        scores = [p["score"] for p in predictions]
        print(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}]")
        print(f"  Score mean: {np.mean(scores):.4f}")

        cat_ids = set(p["category_id"] for p in predictions)
        print(f"  Unique categories predicted: {len(cat_ids)} of 356")


if __name__ == "__main__":
    main()
