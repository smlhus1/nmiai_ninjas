# Research: Closing the 0.574 -> 0.92+ mAP Gap in NM i AI Grocery Detection

> Researched: 2026-03-21 | Sources consulted: 42 | Confidence: High

## TL;DR

The 0.574 -> 0.92+ gap is likely caused by a combination of: (1) ONNX export degradation vs .pt inference, (2) suboptimal confidence/NMS thresholds at inference time, (3) insufficient augmentation diversity for 248 images, and (4) missing TTA (test-time augmentation). The single biggest quick win is probably **switching from ONNX to .pt inference** (if allowed) or fixing ONNX export settings, combined with **lowering conf threshold to 0.001 for evaluation** and **using TTA with augment=True**. For training improvements, pseudo-labeling, corrected-annotation fine-tuning, and optimized WBF ensemble parameters can each add 2-5% mAP.

## Key Findings

### 1. ONNX vs .pt — The Silent Killer

This is potentially the LARGEST single source of score loss. Multiple documented issues confirm significant accuracy degradation:

**Root causes:**
- **Image size mismatch**: ONNX export must use the EXACT same imgsz as training (accounting for 32-pixel stride alignment). If you trained at imgsz=1280, you MUST export with `imgsz=1280`. Mismatch causes confidence scores to drop dramatically. One documented case showed 70+ confidence in .pt vs 40+ in ONNX from size mismatch alone.
- **NMS differences**: PyTorch .pt uses ultralytics' built-in NMS; ONNX models require manual post-processing. Even slight NMS implementation differences change which boxes survive.
- **Precision drift**: PyTorch defaults to float32; ONNX may use float16, causing numerical drift.
- **Letterboxing inconsistency**: Some ONNX inference implementations resize directly to target size instead of letterboxing (maintaining aspect ratio with padding). This changes how objects are scaled.

**Critical fix for your pipeline:**
```python
# When using ultralytics to load ONNX, it handles preprocessing internally.
# But if you're doing manual ONNX inference:

# WRONG: Direct resize
img = cv2.resize(img, (1280, 1280))

# RIGHT: Letterbox resize (maintain aspect ratio, pad to 32-multiple)
from ultralytics.data.augment import LetterBox
letterbox = LetterBox(new_shape=(1280, 1280))
img = letterbox(image=img)
```

**Recommendation**: If sandbox allows .pt inference (via `ultralytics.YOLO("model.pt").predict()`), use it instead of ONNX. The ultralytics predict pipeline handles all preprocessing correctly. If ONNX is mandatory, ensure:
1. Export with exact training imgsz: `model.export(format='onnx', imgsz=1280)`
2. Use ultralytics' ONNX inference path: `YOLO("model.onnx").predict()` — this handles letterboxing/NMS identically to .pt

Sources:
- [GitHub #4791: ONNX vs .pt results differ](https://github.com/ultralytics/ultralytics/issues/4791)
- [GitHub #1247: Confidence pt vs ONNX](https://github.com/ultralytics/ultralytics/issues/1247)
- [GitHub #10289: ONNX accuracy decrease](https://github.com/ultralytics/ultralytics/issues/10289)
- [GitHub #5016: 46% accuracy drop in ONNX classification](https://github.com/ultralytics/ultralytics/issues/5016)

---

### 2. Confidence Threshold and mAP Evaluation

**Critical insight**: YOLOv8 validation uses `conf=0.001` and `iou=0.7` internally for mAP calculation. This is NOT optional — mAP is computed across ALL confidence thresholds. Using conf=0.15 or conf=0.25 at inference time WILL lower your mAP score because you're discarding valid low-confidence detections that contribute to recall.

**Your current code uses conf=0.15 in run_submission_v6.py and conf=0.01 in run_final.py.** Even conf=0.01 may be too high.

**For mAP@0.5 evaluation, you want:**
- `conf=0.001` (or even lower) — let the evaluator threshold
- `iou=0.7` for NMS during inference
- `max_det=500` or higher for dense shelf images (300 may miss products)

**Why this matters for the scoring gap**: If top teams use conf=0.001 and you use conf=0.15, you could be discarding 20-40% of true positive detections that happen to have lower confidence. This directly kills recall, which mAP@0.5 rewards heavily.

```python
# WRONG for mAP evaluation
results = model(img, conf=0.15, iou=0.6, max_det=300)

# RIGHT for mAP evaluation
results = model(img, conf=0.001, iou=0.7, max_det=1000)
```

Sources:
- [Ultralytics Performance Metrics Docs](https://docs.ultralytics.com/guides/yolo-performance-metrics/)
- [GitHub #5315: Confidence/NMS queries](https://github.com/ultralytics/ultralytics/issues/5315)
- [GitHub #8985: Confidence in validation](https://github.com/ultralytics/ultralytics/issues/8985)

---

### 3. Test-Time Augmentation (TTA)

TTA applies left-right flip + 3 different scales during inference, merging results before NMS. YOLOv8 supports TTA via `augment=True`:

```python
# Prediction with TTA
results = model.predict(img, augment=True, conf=0.001, iou=0.7, max_det=1000)
```

**Expected improvement**: +1-2% mAP (documented: 0.504 -> 0.516 on COCO, +2.4% relative)
**Cost**: 2-3x inference time

**Important**: TTA works with .pt models via `augment=True`. For ONNX models, TTA must be implemented manually (run inference 3x at different scales + flip, then merge with WBF/NMS).

TTA is especially valuable for:
- Small objects (products on shelves)
- Scale variation (near vs far shelves)
- Dense scenes (grocery shelves)

**Practical TTA for sandbox** (if timeout allows):
```python
from ultralytics import YOLO
model = YOLO("best.pt")

# TTA inference — 2-3x slower but higher mAP
results = model.predict(
    source=img_path,
    augment=True,       # Enable TTA
    conf=0.001,
    iou=0.7,
    max_det=1000,
    imgsz=1280
)
```

Sources:
- [YOLOv5 TTA Docs](https://docs.ultralytics.com/yolov5/tutorials/test_time_augmentation/)
- [GitHub #1469: TTA in YOLOv8](https://github.com/ultralytics/ultralytics/issues/1469)
- [GitHub #3154: augment=True issue](https://github.com/ultralytics/ultralytics/issues/3154)

---

### 4. Optimal YOLOv8 Training for 248 Images

#### Model Size
- **Use YOLOv8l, NOT YOLOv8x** for 248 images. YOLOv8x (68.2M params) will overfit harder than YOLOv8l (43.7M). Your current approach of using both l and x in ensemble is fine, but primary model should be l.
- Consider YOLOv8m (25.9M) as a third ensemble member — lower capacity = less overfitting = better generalization.

#### Freeze Strategy
- **Freeze 10-15 backbone layers** (you already use `freeze=10`, good).
- Research shows: "Deeper fine-tuning (unfreezing down to layer 10) yields +10% absolute mAP50 on fine-grained tasks compared to only training the head." Your freeze=10 is optimal.
- Do NOT freeze=22 (entire backbone) — this prevents domain adaptation.

#### Learning Rate
- `lr0=0.001` with AdamW is correct for fine-tuning.
- `cos_lr=True` gives smoother convergence — already in your config.
- `lrf=0.01` (final LR = 0.001 * 0.01 = 0.00001) is appropriate.

#### Epochs and Patience
- **Your patience=0 is WRONG**. This trains ALL epochs regardless of overfitting.
- With 248 images, overfitting typically starts at epoch 80-120. Training 200 epochs with patience=0 means 80-120 wasted epochs of overfitting.
- **Recommended**: `patience=30, epochs=300` — early stopping prevents overfitting while allowing enough training.
- Alternatively: `patience=50, epochs=500` if augmentation is aggressive enough.

#### Batch Size
- With 248 training images and 20% val split, you have ~198 training images.
- `batch=4` at imgsz=1280 is fine for GPU memory.
- `batch=8-16` at imgsz=640 would be better for batch norm statistics.
- **Key**: Use largest batch that fits GPU memory.

#### close_mosaic
- Default `close_mosaic=10` means mosaic disabled for last 10 epochs.
- For 200 epochs, mosaic is off from epoch 190.
- **For 248 images**: Increase to `close_mosaic=30` — let the model "settle" on real data longer before stopping.

#### Multi-Scale Training
- `multi_scale=True` dynamically varies input size between 0.5x and 1.5x of imgsz during training.
- For imgsz=1280, this means 640-1920 range.
- **Useful for grocery shelves** where product sizes vary greatly.
- Costs: ~50% more training time, ~50% more VRAM.

#### Augmentation Settings (optimized for 248 shelf images)
```python
model.train(
    # Architecture
    data=data, imgsz=1280, batch=4, device=0, amp=True,
    freeze=10,  # Keep backbone features

    # Training schedule
    epochs=300, patience=30,
    optimizer='AdamW', lr0=0.001, lrf=0.01,
    cos_lr=True, warmup_epochs=5,

    # Augmentation — aggressive for small dataset
    mosaic=1.0,          # 4-image mosaic (critical)
    mixup=0.15,          # Blend images (helps generalization)
    copy_paste=0.0,      # DOES NOT WORK for detection (needs seg masks)
    erasing=0.3,         # Random erasing
    scale=0.5,           # Scale jitter +-50%
    fliplr=0.5,          # Left-right flip
    hsv_h=0.015,         # Hue jitter
    hsv_s=0.7,           # Saturation jitter
    hsv_v=0.4,           # Value jitter
    translate=0.1,       # Translation
    degrees=0.0,         # No rotation (shelf images are always upright)
    perspective=0.0,     # No perspective (adds noise for shelves)

    # Regularization
    label_smoothing=0.1,  # Helps with noisy labels (good for 248 images)
    close_mosaic=30,      # Disable mosaic earlier for fine-tuning

    # Multi-scale
    multi_scale=True,     # Vary input size during training (+~2% mAP)
)
```

Sources:
- [GitHub #6201: Small dataset best practices](https://github.com/ultralytics/ultralytics/issues/6201)
- [Ultralytics Discussion #2799: Training tips](https://github.com/orgs/ultralytics/discussions/2799)
- [Ultralytics Training Docs](https://docs.ultralytics.com/modes/train/)
- [YOLOv8 Layer Freezing Discussion](https://github.com/orgs/ultralytics/discussions/3862)

---

### 5. Copy-Paste Augmentation — What You Need to Know

**Critical fact**: YOLOv8's built-in `copy_paste` parameter requires segmentation masks. For detection-only tasks with bounding boxes, it does NOT work (confirmed by ultralytics team).

**Your manual copy-paste approach in `colab_train_v6.py` is correct** — pasting reference product images onto training shelf images. However, the implementation can be improved:

**Current issues with your copy-paste:**
1. Products are pasted at random positions without regard to shelf structure
2. No realistic occlusion handling
3. Scale (0.15-0.6) may not match actual product sizes
4. 3 augmented copies per image * 3-10 pasted objects = limited diversity

**Improvements:**
```python
# Better copy-paste strategy:
# 1. Paste products ALONG shelf lines (not random positions)
# 2. Use realistic scale based on existing annotations in the image
# 3. Apply slight rotation, brightness jitter to pasted products
# 4. Generate MORE copies (10-20 per original, not just 3)
# 5. Vary number of pasted objects more (1-15 instead of 3-10)

# Most impactful: match the SCALE of pasted products to real ones
existing_widths = [ann['bbox'][2] for ann in img_anns[img_id]]
median_w = np.median(existing_widths)
# Scale reference images to match median product width
scale = median_w / ref.width * random.uniform(0.7, 1.3)
```

**Expected improvement**: +2-4% mAP from better copy-paste (+3.65% mAP50 documented in fish detection study).

Sources:
- [GitHub #6590: copy_paste for detection](https://github.com/ultralytics/ultralytics/issues/6590)
- [CVPR 2021: Simple Copy-Paste Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.pdf)
- [GitHub #18073: copy-paste for detection](https://github.com/ultralytics/ultralytics/issues/18073)

---

### 6. Pseudo-Labeling for Detection

Step-by-step practical guide for self-training with YOLOv8:

**Step 1: Train base model**
```python
# Train on original 248 images
model = YOLO('yolov8l-oiv7.pt')
model.train(data='dataset.yaml', epochs=200, ...)
```

**Step 2: Generate pseudo-labels on augmented/unlabeled data**
```python
# Create augmented versions of your training images (heavy augmentation)
# OR use the test images if you have access to unlabeled shelf images
base_model = YOLO('runs/best.pt')
results = base_model.predict(
    source='unlabeled_images/',
    conf=0.5,       # HIGH threshold for pseudo-labels (precision > recall)
    iou=0.5,
    save_txt=True,   # Save YOLO format labels
    save_conf=True
)
```

**Step 3: Filter pseudo-labels**
- Use conf >= 0.5 for first iteration (high precision pseudo-labels)
- After iteration 2, can lower to 0.3
- Remove any pseudo-labeled image where prediction count is < 50% of expected (likely bad)

**Step 4: Combine and retrain**
```python
# Merge original labels + pseudo-labels into combined dataset
# Weight: original images appear 2-3x more than pseudo-labeled
model2 = YOLO('yolov8l-oiv7.pt')  # Start fresh from pretrained
model2.train(data='combined_dataset.yaml', epochs=200, ...)
```

**Step 5: Iterate (2-3 rounds max)**
- Round 1: conf=0.5, expect +2-3% mAP
- Round 2: conf=0.3 using round 1 model, expect +1-2% mAP
- Round 3: diminishing returns, stop if mAP doesn't improve

**Key insight**: You can pseudo-label your OWN augmented training data. Create heavy augmentations (large color jitter, blur, noise) of training images, predict on them with the base model, and use those predictions as training data. This teaches the model to be robust to image degradation.

**Expected improvement**: +2-5% mAP total across 2-3 iterations.

**Warning**: If your base model has systematic errors (always misses category X), pseudo-labeling will REINFORCE those errors. Only use pseudo-labels from a well-performing base model.

Sources:
- [Self-Training Survey (arxiv 2202.12040)](https://arxiv.org/pdf/2202.12040)
- [Pseudo-Label Review (arxiv 2408.07221)](https://arxiv.org/pdf/2408.07221)
- [Taming Self-Training for Open-Vocab Detection](https://arxiv.org/html/2308.06412)

---

### 7. The Corrected Annotations Trick

Your dataset has a "corrected" field on annotations. Research shows annotation quality has MASSIVE impact:

**Key finding**: Eliminating 5% noisy annotations can improve mAP@50 by up to 0.085 (8.5 percentage points) — equivalent to a 20% relative improvement. (CVPR 2022 Workshop)

**Types of annotation noise and their impact:**
1. **Missing annotations** (forgot to label an object): MOST destructive — model learns to ignore real objects
2. **Wrong class labels**: Moderate impact — confuses classifier
3. **Imprecise bounding boxes**: Least impact — model is somewhat robust to bbox noise

**Two-stage training approach (RECOMMENDED):**
```python
# Stage 1: Train on ALL data (corrected + uncorrected) — maximizes data volume
model = YOLO('yolov8l-oiv7.pt')
model.train(data='all_data.yaml', epochs=150, patience=30, ...)

# Stage 2: Fine-tune on CORRECTED-ONLY data — clean up noise
model2 = YOLO('runs/stage1/weights/best.pt')
model2.train(
    data='corrected_only.yaml',
    epochs=50,
    patience=15,
    lr0=0.0001,        # Lower LR for fine-tuning
    freeze=15,          # Freeze more layers
    mosaic=0.0,         # No mosaic (want clean data)
    mixup=0.0,          # No mixup
    close_mosaic=0,
    label_smoothing=0.0,  # No smoothing — trust corrected labels
)
```

**Expected improvement**: +3-8% mAP depending on annotation noise level.

**Alternative**: Weight corrected annotations higher during training. Unfortunately, YOLOv8 doesn't natively support per-annotation weights. The two-stage approach is the practical workaround.

Sources:
- [CVPR 2022: Effect of Improving Annotation Quality](https://openaccess.thecvf.com/content/CVPR2022W/VDU/papers/Ma_The_Effect_of_Improving_Annotation_Quality_on_Object_Detection_Datasets_CVPRW_2022_paper.pdf)
- [Combating Noisy Labels in Object Detection](https://arxiv.org/html/2211.13993v3)
- [Universal Noise Annotation Impact](https://arxiv.org/html/2312.13822v1)

---

### 8. supervision 0.18.0 InferenceSlicer

**Availability**: InferenceSlicer IS available in supervision 0.18.0 (introduced in 0.14.0).

**Threading**: YES, it uses `ThreadPoolExecutor` with configurable `thread_workers` parameter (default=1). If the sandbox blocks threading, set `thread_workers=1` explicitly — this runs slices sequentially (no threading).

**Known issues:**
- `thread_workers > 1` causes segfault in some versions (documented bug in 0.24.0+)
- Performance can be SLOWER than SAHI with threading due to GIL + overhead
- For sandbox: `thread_workers=1` is safest

**Usage with ultralytics:**
```python
import supervision as sv
from ultralytics import YOLO

model = YOLO("best.pt")

def callback(tile: np.ndarray) -> sv.Detections:
    results = model(tile, conf=0.001, iou=0.7, max_det=500, verbose=False)[0]
    return sv.Detections.from_ultralytics(results)

slicer = sv.InferenceSlicer(
    callback=callback,
    slice_wh=(640, 640),        # Tile size
    overlap_wh=(160, 160),      # 25% overlap
    overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
    iou_threshold=0.5,
    thread_workers=1            # Disable threading for sandbox safety
)

image = cv2.imread("shelf_image.jpg")
detections = slicer(image)
```

**Performance vs manual tiling**: InferenceSlicer adds NMS-based overlap filtering automatically. Your manual tiling in run_submission_v6.py already does WBF fusion, which is arguably BETTER than NMS for overlapping tiles. **Recommendation: Stick with your manual tiling + WBF approach** — it's more flexible and WBF typically outperforms NMS for tile merging.

Sources:
- [supervision InferenceSlicer Docs](https://supervision.roboflow.com/develop/detection/tools/inference_slicer/)
- [GitHub #1695: InferenceSlicer threading slower than SAHI](https://github.com/roboflow/supervision/issues/1695)
- [GitHub #1632: Segfault with thread_workers > 1](https://github.com/roboflow/supervision/issues/1632)

---

### 9. WBF Ensemble Optimization

Your current WBF setup can be significantly improved:

**Your current settings (from run_submission_v6.py):**
```python
# Implicit defaults — likely suboptimal
weighted_boxes_fusion(boxes, scores, labels, iou_thr=0.5, skip_box_thr=0.0001)
```

**Optimal WBF parameters for dense grocery detection:**
```python
from ensemble_boxes import weighted_boxes_fusion

boxes_fused, scores_fused, labels_fused = weighted_boxes_fusion(
    boxes_list,
    scores_list,
    labels_list,
    weights=[1.0, 1.0, 1.0],  # Equal weights (or tune on val)
    iou_thr=0.43,              # Lower than 0.5 — grocery products are dense
    skip_box_thr=0.001,        # Keep low for mAP evaluation
    conf_type='avg',           # Average confidence (best for competition)
    allows_overflow=False
)
```

**Key WBF insights:**
- `iou_thr=0.43` was optimal in documented benchmarks (vs default 0.55)
- `skip_box_thr=0.001` (NOT 0.15) — let evaluator threshold
- `conf_type='avg'` is usually best; `'box_and_model_avg'` is alternative
- Model weights should be proportional to individual model mAP on validation
- WBF with 3+ models typically gives +1-3% mAP over single best model
- WBF is WORSE than NMS for single-model output — only use for actual ensembles

**Tune WBF on your validation set:**
```python
# Grid search optimal WBF parameters
best_map = 0
for iou_thr in [0.3, 0.35, 0.4, 0.43, 0.45, 0.5, 0.55, 0.6]:
    for skip_thr in [0.0001, 0.001, 0.01]:
        fused = weighted_boxes_fusion(..., iou_thr=iou_thr, skip_box_thr=skip_thr)
        map_score = evaluate(fused, ground_truth)
        if map_score > best_map:
            best_map = map_score
            best_params = (iou_thr, skip_thr)
```

Sources:
- [ZFTurbo WBF GitHub](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
- [WBF Paper](https://arxiv.org/abs/1910.13302)
- [WBF LearnOpenCV Tutorial](https://learnopencv.com/weighted-boxes-fusion/)

---

### 10. Bonus: Pose-Model Trick for Small Object Detection

A clever technique uses YOLOv8 pose models to improve small object detection by 12%+ with NO inference overhead:

1. Convert bounding box labels to keypoint format (box center as keypoint)
2. Train a pose model with reduced box loss weight
3. The keypoint head acts as auxiliary task improving feature learning
4. After training, remove the pose head — get a pure detection model with better features

**Documented improvement**: mAP50 44.8% -> 50.6% (+12% relative) on small traffic signs.

This is worth exploring if your per-product mAP is suffering on small products.

Source: [Y-T-G: Increase YOLOv8 Accuracy on Small Objects](https://y-t-g.github.io/tutorials/yolov8-increase-accuracy/)

---

## The 0.574 -> 0.92+ Gap Analysis

Breaking down the likely contributors to the gap:

| Factor | Estimated Impact | Status | Priority |
|--------|-----------------|--------|----------|
| ONNX vs .pt inference | -5 to -15% mAP | **Using ONNX** | **P0 — Fix first** |
| Confidence threshold too high | -3 to -8% mAP | conf=0.15 in v6 | **P0 — Trivial fix** |
| max_det too low | -1 to -3% mAP | max_det=300 | **P0 — Trivial fix** |
| Missing TTA | -1 to -3% mAP | Not used | P1 — Easy |
| WBF params suboptimal | -1 to -2% mAP | Default params | P1 — Easy |
| Training: patience=0 (overfitting) | -2 to -5% mAP | patience=0 | P1 — Retrain |
| Training: no multi_scale | -1 to -2% mAP | Not used | P1 — Retrain |
| No corrected-annotation fine-tune | -3 to -8% mAP | Not used | P1 — Retrain |
| Pseudo-labeling | -2 to -5% mAP | Not used | P2 — Time-consuming |
| Copy-paste quality | -1 to -3% mAP | Basic impl | P2 — Moderate |
| close_mosaic too late | -0.5 to -1% mAP | Default 10 | P2 — Retrain |

**Total addressable gap**: ~20-55% mAP (cumulative, not additive — some overlap)

**Your current**: 0.574
**With P0 fixes only** (ONNX->pt, conf=0.001, max_det=1000): potentially 0.65-0.75
**With P0+P1** (add TTA, WBF tuning, retrained models): potentially 0.75-0.85
**With all fixes**: potentially 0.80-0.92

---

## Immediate Action Plan (Priority Order)

### P0: Zero-retraining fixes (30 minutes)
1. **Switch to .pt inference** instead of ONNX (or fix ONNX imgsz)
2. **Set conf=0.001** in all inference calls
3. **Set max_det=1000** (or higher)
4. **Set iou=0.7** for NMS
5. **Enable TTA**: `model.predict(augment=True, ...)`

### P1: Quick retraining wins (2-4 hours)
6. **Tune WBF parameters** on validation set (grid search iou_thr, skip_box_thr)
7. **Retrain with patience=30** instead of patience=0
8. **Add multi_scale=True** to training
9. **Increase close_mosaic=30**
10. **Two-stage corrected annotation training**

### P2: Advanced improvements (4-8 hours)
11. **Pseudo-labeling** 2-3 iterations
12. **Improve copy-paste** realism (match shelf structure, scale)
13. **Add YOLOv8m** as additional ensemble member (diversity)
14. **Pose-model trick** for small products

---

## Corrected Submission Script Template

```python
"""
Optimized submission — fixes major mAP-killing issues.
"""
import torch
_orig_load = torch.load
torch.load = lambda *args, **kwargs: _orig_load(*args, **{**kwargs, 'weights_only': False})

from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
import numpy as np
from pathlib import Path
import json, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load .pt models (NOT ONNX)
    models = [
        (YOLO("detect_l.pt"), 1280, 1.0),
        (YOLO("detect_x.pt"), 640, 1.0),
        (YOLO("detect_l2.pt"), 640, 1.0),
    ]

    predictions = []
    input_dir = Path(args.input)
    image_files = sorted(p for p in input_dir.iterdir()
                         if p.suffix.lower() in (".jpg", ".jpeg", ".png"))

    for img_path in image_files:
        image_id = int(img_path.stem.split("_")[-1])
        all_boxes, all_scores, all_labels = [], [], []
        img_size = None

        for model, imgsz, weight in models:
            # KEY FIX 1: Use .pt predict with TTA + low confidence
            results = model.predict(
                str(img_path),
                device=device,
                verbose=False,
                augment=True,       # TTA: +1-2% mAP
                conf=0.001,         # KEY FIX 2: Very low conf for mAP eval
                iou=0.7,            # Standard NMS threshold
                max_det=1000,       # KEY FIX 3: High max_det for dense shelves
                imgsz=imgsz
            )

            boxes, scores, labels = [], [], []
            for r in results:
                if r.boxes is None:
                    continue
                img_h, img_w = r.orig_shape
                img_size = (img_w, img_h)
                for i in range(len(r.boxes)):
                    x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                    boxes.append([x1/img_w, y1/img_h, x2/img_w, y2/img_h])
                    scores.append(float(r.boxes.conf[i].item()))
                    labels.append(int(r.boxes.cls[i].item()))

            all_boxes.append(boxes if boxes else [[0,0,0,0]])
            all_scores.append(scores if scores else [0.0])
            all_labels.append(labels if labels else [0])

        # KEY FIX 4: Optimized WBF parameters
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            weights=[1.0, 1.0, 1.0],
            iou_thr=0.43,           # Optimized for dense objects
            skip_box_thr=0.001,     # Keep everything for mAP
            conf_type='avg'
        )

        if img_size is None:
            continue
        img_w, img_h = img_size

        for i in range(len(fused_boxes)):
            x1, y1, x2, y2 = fused_boxes[i]
            x1, y1, x2, y2 = x1*img_w, y1*img_h, x2*img_w, y2*img_h
            predictions.append({
                "image_id": image_id,
                "category_id": int(fused_labels[i]),
                "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                "score": float(fused_scores[i])
            })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(predictions))
    print(f"Saved {len(predictions)} predictions")

if __name__ == "__main__":
    main()
```

---

## Gotchas and Considerations

- **Copy-paste augmentation in ultralytics REQUIRES segmentation masks** — your bounding-box-only dataset cannot use the built-in `copy_paste` parameter. Continue with manual copy-paste.
- **TTA takes 2-3x longer** — verify it fits within 300s sandbox timeout. With 3 models * TTA, you might timeout. Consider TTA on only the best single model if time is tight.
- **conf=0.001 generates MANY predictions** — can be 5000+ per image. Ensure your COCO JSON output handles this (no truncation).
- **WBF with very low skip_box_thr** generates many merged boxes — this is intentional for mAP but may slow evaluation.
- **Patience=0 in your training config** means NO early stopping — the model likely overtrained significantly.
- **supervision 0.18.0 InferenceSlicer** uses threading — if sandbox blocks threading, use `thread_workers=1` or stick with manual tiling.
- **OIV7 pretrained models** have 600 classes vs COCO's 80 — more relevant features for grocery products. Good choice to continue using.
- **ONNX export with NMS embedded** (using `nms=True` flag) has known batch processing bugs — avoid.

## Sources

1. [GitHub #4791: ONNX vs .pt inference differences](https://github.com/ultralytics/ultralytics/issues/4791) — Root causes for ONNX quality loss
2. [GitHub #1247: Confidence .pt vs ONNX](https://github.com/ultralytics/ultralytics/issues/1247) — Image size mismatch fix
3. [GitHub #10289: ONNX accuracy decrease](https://github.com/ultralytics/ultralytics/issues/10289) — Community troubleshooting
4. [GitHub #5016: 46% accuracy drop ONNX](https://github.com/ultralytics/ultralytics/issues/5016) — Severe classification drop
5. [GitHub #6201: Small dataset best practices](https://github.com/ultralytics/ultralytics/issues/6201) — Training recommendations
6. [Ultralytics Discussion #2799: Training tips](https://github.com/orgs/ultralytics/discussions/2799) — Official best practices
7. [GitHub #6590: copy_paste for detection](https://github.com/ultralytics/ultralytics/issues/6590) — Confirmed: needs seg masks
8. [GitHub #1469: TTA in YOLOv8](https://github.com/ultralytics/ultralytics/issues/1469) — TTA support status
9. [YOLOv5 TTA Docs](https://docs.ultralytics.com/yolov5/tutorials/test_time_augmentation/) — TTA implementation details
10. [CVPR 2022: Annotation Quality Effect](https://openaccess.thecvf.com/content/CVPR2022W/VDU/papers/Ma_The_Effect_of_Improving_Annotation_Quality_on_Object_Detection_Datasets_CVPRW_2022_paper.pdf) — Corrected labels impact
11. [supervision InferenceSlicer](https://supervision.roboflow.com/develop/detection/tools/inference_slicer/) — API docs
12. [ZFTurbo WBF GitHub](https://github.com/ZFTurbo/Weighted-Boxes-Fusion) — WBF implementation + params
13. [Ultralytics Augmentation Docs](https://docs.ultralytics.com/guides/yolo-data-augmentation/) — Full augmentation reference
14. [Ultralytics Config Docs](https://docs.ultralytics.com/usage/cfg/) — All training parameters
15. [Y-T-G: Small Object Trick](https://y-t-g.github.io/tutorials/yolov8-increase-accuracy/) — Pose-model auxiliary training
16. [Ultralytics Export Docs](https://docs.ultralytics.com/modes/export/) — ONNX export reference
17. [GitHub #23397: ONNX NMS-free export bug](https://github.com/ultralytics/ultralytics/issues/23397) — ONNX export breaks NMS behavior
18. [Noisy Labels in Detection (arxiv)](https://arxiv.org/html/2211.13993v3) — Handling noisy annotations
19. [Self-Training Survey (arxiv)](https://arxiv.org/pdf/2202.12040) — Pseudo-labeling overview
20. [WBF Paper (arxiv)](https://arxiv.org/abs/1910.13302) — Weighted Boxes Fusion method
21. [Layer Freezing Analysis (MDPI)](https://www.mdpi.com/2227-7390/13/15/2539) — Optimal freeze strategies
22. [Fine-Tuning Without Forgetting (arxiv)](https://arxiv.org/html/2505.01016v1) — Freeze=10 yields +10% mAP
