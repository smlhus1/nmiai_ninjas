"""
Inference pipeline for NM i AI grocery product detection + classification.

Sandbox-compatible: no import os, subprocess, socket.
Uses pathlib for all file operations.

Pipeline per image:
  1. YOLOv8 detect (1-class product) -> bounding boxes
  2. Crop detections with padding
  3. DINOv2 embed crops (batched)
  4. Cosine similarity vs reference prototypes -> category_id
  5. Output COCO format predictions.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run detection + classification inference")
    parser.add_argument("--input", type=str, required=True,
                        help="Directory containing test images")
    parser.add_argument("--output", type=str, required=True,
                        help="Directory to write predictions.json")
    parser.add_argument("--detect-model", type=str, default="detect_model.pt",
                        help="Path to YOLOv8 detection weights")
    parser.add_argument("--embeddings", type=str, default="reference_embeddings.pt",
                        help="Path to reference embeddings")
    parser.add_argument("--classify-model", type=str, default=None,
                        help="Path to fine-tuned DINOv2 classifier (optional)")
    parser.add_argument("--conf-threshold", type=float, default=0.25,
                        help="Detection confidence threshold")
    parser.add_argument("--crop-padding", type=float, default=0.1,
                        help="Fractional padding around crop (0.1 = 10%%)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for DINOv2 embedding")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="YOLO input image size")
    return parser.parse_args()


def load_dinov2_classifier(model_path: str, device: str, n_classes: int = 356):
    """Load fine-tuned DINOv2 with linear classification head."""
    model = timm.create_model("vit_base_patch14_dinov2", pretrained=False,
                              num_classes=n_classes)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model


def load_dinov2_embedder(device: str):
    """Load DINOv2-base as feature extractor (no classification head)."""
    model = timm.create_model("vit_base_patch14_dinov2", pretrained=True, num_classes=0)
    model = model.to(device).eval()

    data_config = resolve_data_config(model.pretrained_cfg, model=model)
    transform = create_transform(**data_config, is_training=False)

    return model, transform


def crop_detection(image: Image.Image, bbox_xyxy: list, padding: float = 0.1) -> Image.Image:
    """Crop detection from image with padding. bbox is [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox_xyxy
    w = x2 - x1
    h = y2 - y1
    pad_x = w * padding
    pad_y = h * padding

    img_w, img_h = image.size
    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(img_w, int(x2 + pad_x))
    y2 = min(img_h, int(y2 + pad_y))

    return image.crop((x1, y1, x2, y2))


def xyxy_to_coco_xywh(bbox_xyxy: list) -> list:
    """Convert [x1, y1, x2, y2] to COCO [x, y, w, h] in pixels."""
    x1, y1, x2, y2 = bbox_xyxy
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def classify_crops_embedding(
    crops: list[Image.Image],
    model: torch.nn.Module,
    transform,
    ref_embeddings: dict,
    device: str,
    batch_size: int = 64,
) -> list[tuple[int, float]]:
    """
    Classify crops via cosine similarity against reference prototypes.
    Returns list of (category_id, confidence_score).
    """
    if not crops:
        return []

    # Prepare reference matrix
    cat_ids = sorted(ref_embeddings.keys())
    ref_matrix = torch.stack([ref_embeddings[cid] for cid in cat_ids])  # (N_cats, 768)
    ref_matrix = ref_matrix.to(device)

    results = []

    for i in range(0, len(crops), batch_size):
        batch_crops = crops[i:i + batch_size]
        tensors = []
        for crop in batch_crops:
            try:
                t = transform(crop.convert("RGB"))
                tensors.append(t)
            except Exception:
                # Fallback: tiny crop, use zero tensor
                tensors.append(torch.zeros(3, 224, 224))

        batch = torch.stack(tensors).to(device)

        with torch.no_grad():
            feats = model(batch)  # (B, 768)
            feats = feats / feats.norm(dim=1, keepdim=True)  # L2 normalize

            # Cosine similarity: (B, N_cats)
            sims = torch.mm(feats, ref_matrix.t())

            # Best match per crop
            max_sims, max_indices = sims.max(dim=1)

            for j in range(len(batch_crops)):
                cat_id = cat_ids[max_indices[j].item()]
                score = max_sims[j].item()
                results.append((cat_id, score))

    return results


def classify_crops_linear(
    crops: list[Image.Image],
    model: torch.nn.Module,
    transform,
    device: str,
    batch_size: int = 64,
) -> list[tuple[int, float]]:
    """Classify crops using fine-tuned DINOv2 with linear head."""
    if not crops:
        return []

    data_config = resolve_data_config(model.pretrained_cfg if hasattr(model, 'pretrained_cfg') else {}, model=model)
    xform = create_transform(**data_config, is_training=False)

    results = []

    for i in range(0, len(crops), batch_size):
        batch_crops = crops[i:i + batch_size]
        tensors = [xform(c.convert("RGB")) for c in batch_crops]
        batch = torch.stack(tensors).to(device)

        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            max_probs, max_indices = probs.max(dim=1)

            for j in range(len(batch_crops)):
                results.append((max_indices[j].item(), max_probs[j].item()))

    return results


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load detection model ---
    detect_model_path = Path(args.detect_model)
    print(f"Loading detector: {detect_model_path}")
    detector = YOLO(str(detect_model_path))

    # --- Load classifier ---
    use_linear = args.classify_model is not None and Path(args.classify_model).exists()

    if use_linear:
        print(f"Loading linear classifier: {args.classify_model}")
        classifier = load_dinov2_classifier(args.classify_model, device)
        embedder = None
        embed_transform = None
        ref_embeddings = None
        # Get transform from classifier
        cls_config = resolve_data_config(
            classifier.pretrained_cfg if hasattr(classifier, 'pretrained_cfg') else {},
            model=classifier
        )
        cls_transform = create_transform(**cls_config, is_training=False)
    else:
        print("Loading DINOv2 embedder for k-NN classification...")
        embedder, embed_transform = load_dinov2_embedder(device)
        classifier = None

        emb_path = Path(args.embeddings)
        if not emb_path.exists():
            raise FileNotFoundError(f"Reference embeddings not found: {emb_path}")
        ref_embeddings = torch.load(emb_path, map_location=device, weights_only=True)
        print(f"Loaded {len(ref_embeddings)} reference prototypes")

    # --- Find images ---
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_paths = sorted([
        p for p in input_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ])
    print(f"Found {len(image_paths)} images in {input_dir}")

    # --- Run inference ---
    predictions = []
    pred_id = 1

    for img_idx, img_path in enumerate(image_paths):
        if (img_idx + 1) % 10 == 0:
            print(f"  Processing {img_idx + 1}/{len(image_paths)}: {img_path.name}")

        # Image ID from filename (e.g., "img_00001.jpg" -> 1)
        stem = img_path.stem
        try:
            image_id = int(stem.replace("img_", "").lstrip("0") or "0")
        except ValueError:
            image_id = img_idx + 1

        # Detect
        results = detector(str(img_path), conf=args.conf_threshold, imgsz=args.imgsz,
                           verbose=False)

        if not results or len(results[0].boxes) == 0:
            continue

        boxes = results[0].boxes
        bboxes_xyxy = boxes.xyxy.cpu().numpy()  # (N, 4) — pixel coords
        det_confs = boxes.conf.cpu().numpy()  # (N,)

        # Load image for cropping
        pil_image = Image.open(img_path).convert("RGB")

        # Crop detections
        crops = []
        valid_indices = []
        for i, bbox in enumerate(bboxes_xyxy):
            crop = crop_detection(pil_image, bbox.tolist(), padding=args.crop_padding)
            if crop.size[0] > 2 and crop.size[1] > 2:  # Skip degenerate crops
                crops.append(crop)
                valid_indices.append(i)

        # Classify crops
        if use_linear:
            cls_results = classify_crops_linear(
                crops, classifier, cls_transform, device, args.batch_size
            )
        else:
            cls_results = classify_crops_embedding(
                crops, embedder, embed_transform, ref_embeddings, device, args.batch_size
            )

        # Build predictions
        for idx, (cat_id, cls_score) in zip(valid_indices, cls_results):
            bbox_xyxy = bboxes_xyxy[idx].tolist()
            det_conf = float(det_confs[idx])

            # CRITICAL: Convert xyxy to COCO xywh (pixels, NOT normalized!)
            bbox_coco = xyxy_to_coco_xywh(bbox_xyxy)

            # Combined score: detection confidence * classification confidence
            combined_score = det_conf * cls_score

            predictions.append({
                "id": pred_id,
                "image_id": image_id,
                "category_id": int(cat_id),
                "bbox": bbox_coco,  # [x, y, w, h] in PIXELS
                "score": round(combined_score, 4),
            })
            pred_id += 1

    # --- Write predictions ---
    output_path = output_dir / "predictions.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    print(f"\nWrote {len(predictions)} predictions to {output_path}")
    print(f"Images processed: {len(image_paths)}")


if __name__ == "__main__":
    main()
