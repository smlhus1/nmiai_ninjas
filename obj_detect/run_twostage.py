"""
Sandbox inference — Two-stage pipeline:
  Stage 1: Class-agnostic YOLOv8m detector (1 class) + manual SAHI
  Stage 2: DINOv2-base embeddings → linear probe classifier (356 classes)

Usage: python run.py --input /data/images --output /output/predictions.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def load_detector(weights_path):
    """Load class-agnostic YOLOv8 detector."""
    from ultralytics import YOLO
    return YOLO(str(weights_path))


def load_classifier(dinov2_path, probe_path, nc=356):
    """Load DINOv2 feature extractor + linear probe.

    Probe weights bundled as .json to avoid exceeding 3 weight file limit.
    """
    import timm

    # Load DINOv2-base
    model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False, num_classes=0)
    state_dict = torch.load(str(dinov2_path), map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model = model.eval()

    # Load linear probe from .json (avoids counting as weight file)
    embed_dim = 768
    probe = nn.Linear(embed_dim, nc)
    probe_data = json.loads(Path(probe_path).read_text())
    probe.weight.data = torch.tensor(probe_data['weight'])
    probe.bias.data = torch.tensor(probe_data['bias'])
    probe = probe.eval()

    # Get transforms
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    return model, probe, transform


def detect_with_sahi(detector, img_path, device, imgsz=1280, conf=0.15,
                     slice_size=640, overlap=0.2):
    """Run detection with manual SAHI (tiled inference) + full image."""
    img = Image.open(img_path).convert('RGB')
    img_w, img_h = img.size

    all_boxes = []  # normalized [x1,y1,x2,y2]
    all_scores = []
    all_source = []  # track which predictions came from where

    # Full image detection
    results = detector(
        str(img_path), device=device, verbose=False,
        conf=conf, iou=0.7, max_det=500, imgsz=imgsz,
    )
    for r in results:
        if r.boxes is None:
            continue
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
            all_boxes.append([x1, y1, x2, y2])
            all_scores.append(float(r.boxes.conf[i].item()))
            all_source.append('full')

    # Tiled detection (SAHI)
    stride = int(slice_size * (1 - overlap))
    np_img = np.array(img)

    for y_start in range(0, img_h, stride):
        for x_start in range(0, img_w, stride):
            y_end = min(y_start + slice_size, img_h)
            x_end = min(x_start + slice_size, img_w)

            # Skip tiny patches
            if (x_end - x_start) < 100 or (y_end - y_start) < 100:
                continue

            patch = np_img[y_start:y_end, x_start:x_end]
            results = detector(
                patch, device=device, verbose=False,
                conf=conf, iou=0.7, max_det=200, imgsz=imgsz,
            )
            for r in results:
                if r.boxes is None:
                    continue
                for i in range(len(r.boxes)):
                    x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                    # Offset to full image coordinates
                    all_boxes.append([x1 + x_start, y1 + y_start,
                                     x2 + x_start, y2 + y_start])
                    all_scores.append(float(r.boxes.conf[i].item()))
                    all_source.append('tile')

    # NMS to merge overlapping detections from full + tiles
    if all_boxes:
        boxes_tensor = torch.tensor(all_boxes)
        scores_tensor = torch.tensor(all_scores)
        from torchvision.ops import nms
        keep = nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
        all_boxes = [all_boxes[i] for i in keep.tolist()]
        all_scores = [all_scores[i] for i in keep.tolist()]

    return all_boxes, all_scores, (img_w, img_h)


def classify_crops(img_path, boxes, dinov2_model, probe,
                   transform, device, padding=5):
    """Crop detections and classify with DINOv2 + linear probe."""
    img = Image.open(img_path).convert('RGB')
    img_w, img_h = img.size

    if not boxes:
        return []

    # Crop and transform all detections
    crops = []
    valid_indices = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        # Add padding
        px1 = max(0, int(x1) - padding)
        py1 = max(0, int(y1) - padding)
        px2 = min(img_w, int(x2) + padding)
        py2 = min(img_h, int(y2) + padding)

        if px2 - px1 < 5 or py2 - py1 < 5:
            continue

        crop = img.crop((px1, py1, px2, py2))
        crop_tensor = transform(crop)
        crops.append(crop_tensor)
        valid_indices.append(i)

    if not crops:
        return []

    # Batch inference
    batch = torch.stack(crops).to(device)
    with torch.no_grad():
        embeddings = dinov2_model(batch)  # [N, 768]
        logits = probe(embeddings)  # [N, 356]
        category_ids = logits.argmax(dim=1).cpu().tolist()
        # Get confidence from softmax
        probs = torch.softmax(logits, dim=1)
        cls_confidences = probs.max(dim=1).values.cpu().tolist()

    return list(zip(valid_indices, category_ids, cls_confidences))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    script_dir = Path(__file__).parent

    # Load models
    detector = load_detector(script_dir / "detect_model.pt")
    dinov2_model, probe, transform = load_classifier(
        script_dir / "classify_model.pt",
        script_dir / "linear_probe.json",
    )
    dinov2_model = dinov2_model.to(device)
    probe = probe.to(device)

    predictions = []
    input_dir = Path(args.input)
    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    for img_path in image_files:
        image_id = int(img_path.stem.split("_")[-1])

        # Stage 1: Detect all products (class-agnostic)
        boxes, det_scores, (img_w, img_h) = detect_with_sahi(
            detector, img_path, device,
            imgsz=1280, conf=0.15,
            slice_size=640, overlap=0.2,
        )

        # Stage 2: Classify each detection
        classifications = classify_crops(
            img_path, boxes, dinov2_model, probe,
            transform, device,
        )

        for idx, category_id, cls_conf in classifications:
            x1, y1, x2, y2 = boxes[idx]
            det_score = det_scores[idx]
            # Combined score: detection confidence × classification confidence
            combined_score = det_score * cls_conf

            predictions.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [
                    round(x1, 1),
                    round(y1, 1),
                    round(x2 - x1, 1),
                    round(y2 - y1, 1),
                ],
                "score": round(combined_score, 4),
            })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

    print(f"Wrote {len(predictions)} predictions for {len(image_files)} images")


if __name__ == "__main__":
    main()
