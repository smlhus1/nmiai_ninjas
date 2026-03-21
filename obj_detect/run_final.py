"""
Final submission: YOLO multi-class + tiled inference + WBF + EfficientNet-B0 re-classifier.

Pipeline:
  1. Full-image YOLO detection (imgsz=1280)
  2. Tiled YOLO detection (640x640 tiles, 25% overlap)
  3. WBF fusion of full + tiled detections
  4. Crop detections → EfficientNet-B0 → override category_id
  5. Output COCO JSON

Usage: python run.py --input /data/images --output /output/predictions.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from ensemble_boxes import weighted_boxes_fusion


def load_detector(weights_path):
    from ultralytics import YOLO
    return YOLO(str(weights_path))


def load_classifier(weights_path, embeddings_path, nc=356):
    """Load EfficientNet-B0 classifier + reference embeddings."""
    import timm
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
    state = torch.load(str(weights_path), map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model = model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    # Load reference embeddings + labels
    emb_data = np.load(str(embeddings_path), allow_pickle=True).item()
    ref_embeds = torch.tensor(emb_data['embeddings'])  # [N_ref, 1280]
    ref_labels = emb_data['labels']  # [N_ref] category IDs

    # Normalize for cosine similarity
    ref_embeds = torch.nn.functional.normalize(ref_embeds, dim=1)

    return model, transform, ref_embeds, ref_labels


def detect_full(detector, img_path, device, imgsz=1280, conf=0.01):
    """Full-image detection."""
    results = detector(str(img_path), device=device, verbose=False,
                       conf=conf, iou=0.7, max_det=500, imgsz=imgsz)
    boxes, scores, labels = [], [], []
    img_w, img_h = None, None
    for r in results:
        if r.boxes is None:
            continue
        img_h, img_w = r.orig_shape
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
            boxes.append([x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h])
            scores.append(float(r.boxes.conf[i].item()))
            labels.append(int(r.boxes.cls[i].item()))
    return boxes, scores, labels, (img_w, img_h)


def detect_tiled(detector, img_path, device, imgsz=1280, conf=0.01,
                 tile_size=640, overlap=0.25):
    """Tiled detection for small/dense products."""
    img = Image.open(img_path).convert('RGB')
    img_w, img_h = img.size
    np_img = np.array(img)
    stride = int(tile_size * (1 - overlap))

    boxes, scores, labels = [], [], []
    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            ye = min(y + tile_size, img_h)
            xe = min(x + tile_size, img_w)
            if xe - x < 64 or ye - y < 64:
                continue
            patch = np_img[y:ye, x:xe]
            results = detector(patch, device=device, verbose=False,
                               conf=conf, iou=0.7, max_det=300, imgsz=imgsz)
            for r in results:
                if r.boxes is None:
                    continue
                for i in range(len(r.boxes)):
                    bx1, by1, bx2, by2 = r.boxes.xyxy[i].tolist()
                    # Offset to full image and normalize
                    boxes.append([(bx1 + x) / img_w, (by1 + y) / img_h,
                                  (bx2 + x) / img_w, (by2 + y) / img_h])
                    scores.append(float(r.boxes.conf[i].item()))
                    labels.append(int(r.boxes.cls[i].item()))

    return boxes, scores, labels, (img_w, img_h)


def fuse_detections(all_boxes_list, all_scores_list, all_labels_list,
                    weights=None, iou_thr=0.6):
    """Fuse detections from multiple sources with WBF."""
    if not any(b for b in all_boxes_list):
        return [], [], []

    # WBF needs at least one non-empty list
    clean_b, clean_s, clean_l, clean_w = [], [], [], []
    for i, (b, s, l) in enumerate(zip(all_boxes_list, all_scores_list, all_labels_list)):
        if b:
            clean_b.append(b)
            clean_s.append(s)
            clean_l.append(l)
            clean_w.append(weights[i] if weights else 1.0)

    if not clean_b:
        return [], [], []

    fb, fs, fl = weighted_boxes_fusion(
        clean_b, clean_s, clean_l,
        weights=clean_w, iou_thr=iou_thr, skip_box_thr=0.001,
    )
    return fb.tolist(), fs.tolist(), [int(x) for x in fl.tolist()]


def classify_crops(img_path, boxes_norm, yolo_labels, classifier, transform,
                   ref_embeds, ref_labels, device, img_size):
    """Re-classify detections with EfficientNet-B0 embeddings."""
    img = Image.open(img_path).convert('RGB')
    img_w, img_h = img_size

    if not boxes_norm:
        return []

    crops = []
    valid = []
    for i, (x1n, y1n, x2n, y2n) in enumerate(boxes_norm):
        px1 = max(0, int(x1n * img_w) - 3)
        py1 = max(0, int(y1n * img_h) - 3)
        px2 = min(img_w, int(x2n * img_w) + 3)
        py2 = min(img_h, int(y2n * img_h) + 3)
        if px2 - px1 < 5 or py2 - py1 < 5:
            valid.append((i, yolo_labels[i]))  # Keep YOLO label
            continue
        crop = img.crop((px1, py1, px2, py2))
        crops.append(transform(crop))
        valid.append((i, None))  # Will be classified

    if not any(c is None for _, c in valid):
        return [(i, cat) for i, cat in valid]

    # Batch classify
    crop_tensors = [c for c in crops if c is not None]
    if crop_tensors:
        batch = torch.stack(crop_tensors).to(device)
        with torch.no_grad():
            embeds = classifier(batch)
            embeds = torch.nn.functional.normalize(embeds, dim=1)
            # Cosine similarity against reference
            sim = embeds @ ref_embeds.to(device).T  # [N_crops, N_ref]
            best_ref_idx = sim.argmax(dim=1).cpu().tolist()

        crop_idx = 0
        result = []
        for i, cat in valid:
            if cat is not None:
                result.append((i, cat))
            else:
                ref_cat = ref_labels[best_ref_idx[crop_idx]]
                crop_idx += 1
                result.append((i, int(ref_cat)))
        return result
    else:
        return [(i, cat) for i, cat in valid]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    script_dir = Path(__file__).parent

    # Load models
    detector = load_detector(script_dir / "detect_model.pt")

    # Try to load classifier, fall back to YOLO-only if not present
    classifier_path = script_dir / "classifier.pt"
    embeddings_path = script_dir / "embeddings.npy"
    use_classifier = classifier_path.exists() and embeddings_path.exists()

    if use_classifier:
        classifier, cls_transform, ref_embeds, ref_labels = load_classifier(
            classifier_path, embeddings_path)
        classifier = classifier.to(device)

    predictions = []
    input_dir = Path(args.input)
    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    for img_path in image_files:
        image_id = int(img_path.stem.split("_")[-1])

        # Stage 1: Full-image detection
        boxes_full, scores_full, labels_full, img_size = detect_full(
            detector, img_path, device, imgsz=1280, conf=0.01)

        # Stage 2: Tiled detection
        boxes_tiled, scores_tiled, labels_tiled, _ = detect_tiled(
            detector, img_path, device, imgsz=1280, conf=0.01,
            tile_size=640, overlap=0.25)

        # Stage 3: WBF fusion
        fused_boxes, fused_scores, fused_labels = fuse_detections(
            [boxes_full, boxes_tiled],
            [scores_full, scores_tiled],
            [labels_full, labels_tiled],
            weights=[1.0, 1.5],
            iou_thr=0.6,
        )

        # Stage 4: Re-classify (if classifier available)
        if use_classifier and fused_boxes:
            classifications = classify_crops(
                img_path, fused_boxes, fused_labels,
                classifier, cls_transform, ref_embeds, ref_labels,
                device, img_size)
            final_labels = [cat for _, cat in classifications]
        else:
            final_labels = fused_labels

        # Stage 5: Output
        img_w, img_h = img_size
        for box, score, cat in zip(fused_boxes, fused_scores, final_labels):
            x1n, y1n, x2n, y2n = box
            x_px = x1n * img_w
            y_px = y1n * img_h
            w_px = (x2n - x1n) * img_w
            h_px = (y2n - y1n) * img_h
            if w_px < 1 or h_px < 1:
                continue
            predictions.append({
                "image_id": image_id,
                "category_id": int(cat),
                "bbox": [round(x_px, 1), round(y_px, 1),
                         round(w_px, 1), round(h_px, 1)],
                "score": round(float(score), 4),
            })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

    print(f"Wrote {len(predictions)} predictions for {len(image_files)} images")


if __name__ == "__main__":
    main()
