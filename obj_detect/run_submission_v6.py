"""
NorgesGruppen Object Detection — 3-model ensemble + tiled inference + WBF
Submission run.py for sandbox (torch 2.6.0 + ultralytics 8.1.0)

Requires torch.load monkey-patch in sandbox because ultralytics 8.1.0
uses pickle-based model saves and PyTorch 2.6 defaults to weights_only=True.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ensemble_boxes import weighted_boxes_fusion

# Note: ultralytics 8.1.0 has built-in patch in utils/patches.py
# that sets weights_only=False for PyTorch 2.6. No monkey-patch needed.


def load_models():
    from ultralytics import YOLO
    script_dir = Path(__file__).parent
    models = []
    for fname, imgsz, weight in [
        ("detect_l.pt", 1280, 1.0),
        ("detect_x.pt", 640, 1.0),
        ("detect_l2.pt", 640, 1.0),
    ]:
        p = script_dir / fname
        if p.exists():
            models.append((YOLO(str(p)), imgsz, weight, fname))
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = load_models()
    print(f"Loaded {len(models)} models: {[f for _,_,_,f in models]}")

    predictions = []
    input_dir = Path(args.input)
    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    for img_path in image_files:
        image_id = int(img_path.stem.split("_")[-1])
        all_boxes, all_scores, all_labels = [], [], []
        img_size = None

        for model, imgsz, weight, fname in models:
            # Full image
            results = model(str(img_path), device=device, verbose=False,
                           conf=0.001, iou=0.7, max_det=500, imgsz=imgsz)
            boxes, scores, labels = [], [], []
            img_w, img_h = None, None
            for r in results:
                if r.boxes is None: continue
                img_h, img_w = r.orig_shape
                for i in range(len(r.boxes)):
                    x1,y1,x2,y2 = r.boxes.xyxy[i].tolist()
                    boxes.append([x1/img_w, y1/img_h, x2/img_w, y2/img_h])
                    scores.append(float(r.boxes.conf[i].item()))
                    labels.append(int(r.boxes.cls[i].item()))
            all_boxes.append(boxes if boxes else [[0,0,0,0]])
            all_scores.append(scores if scores else [0])
            all_labels.append(labels if labels else [0])
            img_size = (img_w, img_h) if img_w else img_size

            # Tiled (only for 1280 model)
            if imgsz == 1280:
                img = Image.open(img_path).convert('RGB')
                iw, ih = img.size
                np_img = np.array(img)
                tile, overlap = 640, 0.25
                stride = int(tile * (1 - overlap))
                bt, st, lt = [], [], []
                for y in range(0, ih, stride):
                    for x in range(0, iw, stride):
                        ye, xe = min(y+tile, ih), min(x+tile, iw)
                        if xe-x < 64 or ye-y < 64: continue
                        res = model(np_img[y:ye, x:xe], device=device, verbose=False,
                                   conf=0.001, iou=0.7, max_det=300, imgsz=imgsz)
                        for r in res:
                            if r.boxes is None: continue
                            for i in range(len(r.boxes)):
                                bx1,by1,bx2,by2 = r.boxes.xyxy[i].tolist()
                                bt.append([(bx1+x)/iw, (by1+y)/ih, (bx2+x)/iw, (by2+y)/ih])
                                st.append(float(r.boxes.conf[i].item()))
                                lt.append(int(r.boxes.cls[i].item()))
                all_boxes.append(bt if bt else [[0,0,0,0]])
                all_scores.append(st if st else [0])
                all_labels.append(lt if lt else [0])
                if not img_size: img_size = (iw, ih)

        # WBF fusion
        fb, fs, fl = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            iou_thr=0.5, skip_box_thr=0.1)

        iw, ih = img_size
        for box, score, label in zip(fb, fs, fl):
            x1n,y1n,x2n,y2n = box
            w_px = (x2n-x1n)*iw
            h_px = (y2n-y1n)*ih
            if w_px < 1 or h_px < 1: continue
            predictions.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [round(x1n*iw,1), round(y1n*ih,1),
                         round(w_px,1), round(h_px,1)],
                "score": round(float(score), 3),
            })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)
    print(f"Wrote {len(predictions)} predictions for {len(image_files)} images")


if __name__ == "__main__":
    main()
