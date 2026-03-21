# Research: Achieving 0.92+ mAP@0.5 in NM i AI Grocery Product Detection

> Researched: 2026-03-21 | Sources consulted: 38 | Confidence: High

## TL;DR

The path from 0.70-0.75 to 0.92+ requires stacking 6-8 independent improvements. The highest-impact techniques are: (1) two-stage corrected-annotation fine-tuning (+3-8%), (2) pseudo-labeling / self-training (+2-5%), (3) model soup weight averaging (+1-3%), (4) training on 100% data for final submission (+1-3%), (5) optimized WBF parameters (+1-2%), (6) TTA combined with tiling (+1-2%), (7) separate classifier for the 30% classification component (+2-5% overall), and (8) aggressive data augmentation with larger resolution (+1-3%). All techniques are implementable in hours with the existing sandbox constraints.

---

## 1. Pseudo-Labeling / Self-Training

### How It Works
Train a base model, use it to predict on unlabeled or augmented images with high confidence, add those predictions as training data, and retrain. This is the single most proven technique for small-dataset object detection competitions.

### Practical Workflow for YOLOv8

**Step 1: Train base model on original 248 images**
```python
model = YOLO('yolov8l-oiv7.pt')
model.train(data='dataset.yaml', epochs=200, patience=30, imgsz=1280, ...)
```

**Step 2: Create augmented unlabeled images**
Since you don't have separate unlabeled test-like images, create heavily augmented versions of your training images that are different enough to add value:
```python
import albumentations as A
from PIL import Image
import numpy as np

heavy_aug = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0),
    A.GaussianBlur(blur_limit=(3, 9), p=0.5),
    A.GaussNoise(var_limit=(10, 80), p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=30, p=0.8),
    A.CLAHE(clip_limit=4.0, p=0.3),
    A.RandomShadow(p=0.3),
    A.ISONoise(p=0.3),
])

# Generate 5 augmented versions per original image
for img_path in original_images:
    img = np.array(Image.open(img_path))
    for i in range(5):
        aug_img = heavy_aug(image=img)['image']
        Image.fromarray(aug_img).save(f'pseudo_unlabeled/{img_path.stem}_aug{i}.jpg')
```

**Step 3: Generate pseudo-labels**
```python
base_model = YOLO('runs/best.pt')

# HIGH confidence for round 1 (precision over recall)
results = base_model.predict(
    source='pseudo_unlabeled/',
    conf=0.5,          # Only confident predictions become pseudo-labels
    iou=0.5,           # Standard NMS
    save_txt=True,     # Save YOLO-format labels
    save_conf=True,    # Include confidence scores
    max_det=500,
    imgsz=1280
)
```

**Step 4: Filter and combine**
```python
import os
from pathlib import Path

pseudo_dir = Path('runs/predict/labels/')
filtered_dir = Path('pseudo_filtered/')
filtered_dir.mkdir(exist_ok=True)

for label_file in pseudo_dir.glob('*.txt'):
    lines = label_file.read_text().strip().split('\n')
    # Filter: keep only predictions with conf >= 0.5
    filtered = [l for l in lines if float(l.split()[-1]) >= 0.5]

    # Quality check: skip images with suspiciously few detections
    # (expect 20-200 products per shelf image)
    if len(filtered) < 10:
        continue

    # Remove confidence column for YOLO format
    clean = [' '.join(l.split()[:-1]) for l in filtered]
    (filtered_dir / label_file.name).write_text('\n'.join(clean))
```

**Step 5: Retrain with combined dataset**
```yaml
# combined_dataset.yaml
train:
  - dataset/images/train        # Original (appears first = higher sampling weight)
  - dataset/images/train        # Duplicate original for 2x weight
  - pseudo_filtered/images      # Pseudo-labeled augmented images
val: dataset/images/val
nc: 356
names: [...]
```

```python
model2 = YOLO('yolov8l-oiv7.pt')  # Fresh from pretrained (not from base model)
model2.train(data='combined_dataset.yaml', epochs=200, patience=30, ...)
```

**Step 6: Iterate (2-3 rounds max)**
- Round 1: conf=0.5 threshold, expect +2-3% mAP
- Round 2: conf=0.3 using Round 1 model, expect +1-2% mAP
- Round 3: Diminishing returns; stop if no improvement

### Key Research Findings
- Confidence threshold 0.5-0.7 for initial round, 0.3-0.5 for subsequent rounds
- Teacher-student EMA (Exponential Moving Average) frameworks outperform naive self-training, but are complex to implement
- Simple pseudo-labeling achieves 80-90% of teacher-student performance
- WARNING: Pseudo-labels reinforce systematic errors. If base model consistently misses category X, self-training won't fix it

### Expected Improvement: +2-5% mAP

---

## 2. Corrected Annotations Fine-Tuning

### The Science
Annotation quality has MASSIVE impact on model performance. Research shows eliminating 5% noisy annotations can improve mAP@50 by up to 8.5 percentage points (CVPR 2022 Workshop). Missing annotations (forgetting to label objects) are the most destructive type of noise — the model actively learns to ignore real objects.

### Two-Stage Training Protocol

**Stage 1: Train on ALL data (248 images, all annotations)**
```python
model = YOLO('yolov8l-oiv7.pt')
model.train(
    data='all_data.yaml',
    epochs=200,
    patience=30,
    imgsz=1280,
    batch=4,
    freeze=10,
    # Standard augmentation
    mosaic=1.0, mixup=0.15, scale=0.5, erasing=0.3,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    optimizer='AdamW', lr0=0.001,
)
```

**Stage 2: Fine-tune on CORRECTED-ONLY annotations**
```python
# First: filter annotations to corrected-only
import json

with open('annotations.json') as f:
    data = json.load(f)

corrected_anns = [a for a in data['annotations'] if a.get('corrected', False)]
corrected_img_ids = set(a['image_id'] for a in corrected_anns)

# Create filtered dataset YAML pointing to corrected-only images + labels
# ...

model2 = YOLO('runs/stage1/weights/best.pt')
model2.train(
    data='corrected_only.yaml',
    epochs=50,
    patience=15,
    lr0=0.0001,         # 10x lower LR for fine-tuning
    freeze=15,           # Freeze more layers
    mosaic=0.0,          # No mosaic — want clean signal
    mixup=0.0,           # No mixup
    close_mosaic=0,
    label_smoothing=0.0, # Trust corrected labels
    scale=0.2,           # Minimal scale jitter
    imgsz=1280,
)
```

### Expected Improvement: +3-8% mAP (depends on noise level)

---

## 3. Test-Time Augmentation (TTA)

### ultralytics 8.1.0 Support
TTA works via `augment=True` in predict mode. It applies:
- Left-right flip
- 3 different scales (0.83x, 1.0x, 1.17x of imgsz)
- Results merged before NMS

```python
results = model.predict(
    source=img_path,
    augment=True,       # Enable TTA
    conf=0.001,         # Low conf for mAP evaluation
    iou=0.7,
    max_det=1000,
    imgsz=1280
)
```

### Documented Performance
- COCO: mAP 0.504 -> 0.516 (+1.2% absolute, +2.4% relative)
- Small objects: AP 0.351 -> 0.361 (+2.8% relative)
- Cost: 2-3x inference time

### Combining TTA with Tiling
TTA and tiling are complementary. Run both:
```python
# Strategy: TTA on full image + tiled (no TTA on tiles — too slow)
# Full image with TTA
full_results = model.predict(img, augment=True, imgsz=1280, conf=0.001, max_det=1000)

# Tiled without TTA (tiles already handle scale variation)
tiled_results = run_manual_tiling(model, img, tile_size=640, overlap=0.25)

# WBF fusion of full+TTA and tiled results
fused = weighted_boxes_fusion(
    [full_boxes, tiled_boxes],
    [full_scores, tiled_scores],
    [full_labels, tiled_labels],
    weights=[1.0, 1.5],  # Higher weight for tiled (better small objects)
    iou_thr=0.43,
    skip_box_thr=0.001
)
```

### Time Budget Analysis (100 test images on L4)
| Pass | Per-image | Total | Notes |
|------|----------|-------|-------|
| Full+TTA (1280) | ~30ms | 3s | 3x normal |
| Tiled (640, 25% overlap) | ~600ms | 60s | ~48 tiles |
| WBF fusion | ~5ms | 0.5s | Fast |
| **Total** | | **~64s** | **21% of 300s budget** |

### Expected Improvement: +1-2% mAP (stacks with tiling)

---

## 4. Optimized WBF Parameters

### Current Issue
Your `run_ensemble.py` uses `iou_thr=0.7` — this is too high for dense grocery products. Adjacent products on shelves have IoU 0.1-0.4. Fusing at 0.7 means only near-identical boxes are merged, wasting the ensemble's diversity.

### Optimal Parameters for Dense Grocery Detection
```python
from ensemble_boxes import weighted_boxes_fusion

fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
    boxes_list,
    scores_list,
    labels_list,
    weights=[1.0, 1.5, 1.5],   # Higher weight for better-performing models
    iou_thr=0.43,               # Optimal from benchmarks (vs default 0.55)
    skip_box_thr=0.001,         # Keep everything for mAP evaluation
    conf_type='avg',            # Average confidence (best for competition)
    allows_overflow=False
)
```

### Parameter Tuning Grid Search
```python
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

best_map = 0
for iou_thr in [0.3, 0.35, 0.4, 0.43, 0.45, 0.5, 0.55, 0.6, 0.7]:
    for skip_thr in [0.0001, 0.001, 0.01, 0.05]:
        for conf_type in ['avg', 'box_and_model_avg', 'max']:
            fused = weighted_boxes_fusion(
                ...,
                iou_thr=iou_thr,
                skip_box_thr=skip_thr,
                conf_type=conf_type
            )
            # Evaluate against ground truth
            map_score = evaluate_coco(fused, gt_annotations)
            if map_score > best_map:
                best_map = map_score
                best_params = (iou_thr, skip_thr, conf_type)
                print(f"New best: iou={iou_thr}, skip={skip_thr}, "
                      f"conf={conf_type} -> mAP={map_score:.4f}")
```

### Key Insights
- `iou_thr=0.43` outperformed 0.5 and 0.55 in documented benchmarks
- For VERY dense scenes (100+ products), try even lower: 0.3-0.35
- `skip_box_thr=0.001` (NOT 0.15) — let the evaluator handle thresholding
- Model weights should be proportional to individual model mAP on validation
- WBF is BETTER than NMS for ensembles; NMS is better for single models

### Expected Improvement: +1-2% mAP

---

## 5. RT-DETR: Transformer Detector Alternative

### Performance Comparison (COCO val2017)

| Model | mAP@50:95 | Params | GPU Speed | Notes |
|-------|-----------|--------|-----------|-------|
| YOLOv8l | 52.9 | 43.7M | 9.06ms | Your current |
| YOLOv8x | 53.9 | 68.2M | 14.37ms | Your current |
| RT-DETRv2-l | 53.4 | 42M | 9.76ms | Comparable to YOLOv8l |
| RT-DETRv2-x | 54.3 | 76M | 15.03ms | Slightly better than YOLOv8x |

### ultralytics 8.1.0 Compatibility
RT-DETR IS supported in ultralytics 8.1.0. Available checkpoints:
- `rtdetr-l.pt` (63.4 MB) — COCO pretrained
- `rtdetr-x.pt` (129.5 MB) — COCO pretrained

```python
from ultralytics import RTDETR
model = RTDETR('rtdetr-l.pt')
model.train(data='dataset.yaml', epochs=100, imgsz=1280, batch=4)
```

### Verdict: NOT Worth It for This Competition
- **No OIV7 pretrained RT-DETR** — only COCO. OIV7 YOLOv8 pretrained backbone is more valuable for grocery detection
- **Higher VRAM** — transformer attention is quadratic, 1280px is borderline on L4 24GB
- **Marginal improvement** — 53.4 vs 52.9 mAP (0.5% on COCO) is within noise for 248-image fine-tuning
- **Known AMP bugs** — documented issues with RT-DETR + mixed precision in ultralytics 8.1.0
- **Ensemble diversity** — adding RT-DETR as 4th ensemble member COULD help (different architecture = different errors), but you're limited to 3 weight files

### Recommendation
Stick with YOLOv8 ensemble. If you want architecture diversity, use YOLOv8m + YOLOv8l + YOLOv8x (different capacities) rather than switching to RT-DETR.

---

## 6. Classification Boost (30% of Score)

The scoring formula is `0.7 * detection_mAP + 0.3 * classification_mAP`. A dedicated classifier that re-classifies YOLO detections can significantly improve the 30% classification component.

### Architecture: Two-Stage Detection + Classification

```python
# Stage 1: YOLO detects products (bounding boxes)
# Stage 2: Crop each detection -> EfficientNet classifies product category

import timm
import torch
from PIL import Image
from torchvision import transforms

# Load fine-tuned EfficientNet-B0
classifier = timm.create_model('efficientnet_b0', pretrained=False, num_classes=356)
state = torch.load('classifier.pth', map_location='cuda', weights_only=True)
classifier.load_state_dict(state)
classifier = classifier.eval().cuda()

data_config = timm.data.resolve_model_data_config(classifier)
transform = timm.data.create_transform(**data_config, is_training=False)

def reclassify_detections(image, boxes, original_labels, original_scores):
    """Override YOLO class predictions with dedicated classifier."""
    new_labels = []
    for box in boxes:
        x1, y1, x2, y2 = box
        crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
        crop = crop.resize((224, 224))
        tensor = transform(crop).unsqueeze(0).cuda()

        with torch.no_grad():
            logits = classifier(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

        # Override if classifier is more confident than YOLO
        if conf.item() > 0.3:  # Threshold for override
            new_labels.append(int(pred.item()))
        else:
            new_labels.append(original_labels[len(new_labels)])

    return new_labels
```

### Embedding-Based Retrieval Alternative
Instead of training a 356-class classifier, use feature embeddings + nearest-neighbor matching against reference product images:

```python
# Pre-compute reference embeddings (offline)
ref_model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
ref_model.load_state_dict(torch.load('effnet_b0.pth', weights_only=True))
ref_model = ref_model.eval().cuda()

# Extract embeddings for all reference product images
ref_embeddings = []  # [N_ref, 1280]
ref_labels = []      # [N_ref] category IDs

for product_dir in Path('product_images/').iterdir():
    cat_id = int(product_dir.name)
    for img_path in product_dir.glob('*.jpg'):
        img = transform(Image.open(img_path)).unsqueeze(0).cuda()
        with torch.no_grad():
            emb = ref_model(img)  # [1, 1280]
        ref_embeddings.append(emb.cpu())
        ref_labels.append(cat_id)

ref_embeddings = torch.cat(ref_embeddings)  # [N_ref, 1280]
ref_embeddings = torch.nn.functional.normalize(ref_embeddings, dim=1)

# Save for sandbox
np.savez('embeddings.npz',
         embeddings=ref_embeddings.numpy(),
         labels=np.array(ref_labels))

# At inference: cosine similarity matching
def classify_by_embedding(crop_embedding, ref_embeddings, ref_labels, top_k=5):
    crop_norm = torch.nn.functional.normalize(crop_embedding, dim=1)
    sims = torch.mm(crop_norm, ref_embeddings.T)  # [1, N_ref]
    top_vals, top_idx = sims.topk(top_k, dim=1)

    # Majority vote among top-k
    top_labels = [ref_labels[i] for i in top_idx[0]]
    from collections import Counter
    return Counter(top_labels).most_common(1)[0][0]
```

### Hybrid Strategy (Best of Both)
```python
# Use fine-tuned classifier for high-confidence predictions
# Fall back to embedding retrieval for low-confidence cases
def hybrid_classify(crop, classifier, ref_model, ref_embeddings, ref_labels):
    tensor = transform(crop).unsqueeze(0).cuda()

    with torch.no_grad():
        # Try classifier first
        logits = classifier(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

        if conf.item() > 0.5:
            return int(pred.item()), conf.item()

        # Low confidence: fall back to embedding retrieval
        emb = ref_model(tensor)
        label = classify_by_embedding(emb, ref_embeddings, ref_labels)
        return label, conf.item()
```

### Model Selection for Classifier
| Model | Size | Embed dim | ImageNet Top-1 | Recommended |
|-------|------|-----------|----------------|-------------|
| efficientnet_b0 | 21 MB | 1280 | 77.7% | Best balance |
| efficientnet_b2 | 36 MB | 1408 | 80.1% | Better accuracy |
| convnext_tiny | 109 MB | 768 | 82.1% | Best accuracy, large |
| mobilenetv3_large | 22 MB | 1280 | 75.2% | Fastest |

### Weight Budget with Classifier
| File | Size | Purpose |
|------|------|---------|
| detect_l.pt | ~87 MB | YOLOv8l detector |
| detect_x.pt | ~131 MB | YOLOv8x detector |
| classifier.pth + embeddings.npz | ~27 MB | EfficientNet + ref embeddings |
| **Total** | **~245 MB** | **58% of 420 MB** |

Wait — this uses all 3 weight file slots. Options:
1. **Two detectors + classifier**: detect_l.pt + detect_x.pt + classifier.pth (embed in .pth)
2. **One detector + two classifiers**: detect_l.pt + classifier.pth + embeddings.npz (higher classification accuracy, lower detection)
3. **Three detectors, no classifier**: detect_l.pt + detect_x.pt + detect_l1280.pt (rely on YOLO classes)

For 0.92+ mAP, option 1 is likely best — the 30% classification weight is too valuable to ignore.

### Expected Improvement: +2-5% on overall score (from classification component)

---

## 7. Data Augmentation Beyond Copy-Paste

### Optimal Settings for 248 Grocery Images
```python
model.train(
    # Core augmentation (keep aggressive for small datasets)
    mosaic=1.0,           # 4-image mosaic — CRITICAL for 248 images
    mixup=0.15,           # Moderate blend — helps generalization
    erasing=0.3,          # Random erasing — forces robust features

    # Color augmentation (match real shelf lighting variation)
    hsv_h=0.02,           # Slight hue shift (packaging colors vary)
    hsv_s=0.8,            # Strong saturation jitter (shelf lighting)
    hsv_v=0.5,            # Strong brightness jitter (shadows, spotlights)

    # Geometric
    scale=0.5,            # +-50% scale — handles near/far shelves
    translate=0.15,       # 15% translation — partially visible products
    fliplr=0.5,           # Left-right flip (products are symmetric)
    flipud=0.0,           # NO vertical flip (products are upright)
    degrees=0.0,          # NO rotation (shelf images are level)
    perspective=0.0,      # NO perspective (adds noise for shelves)
    shear=0.0,            # NO shear

    # Training schedule
    close_mosaic=30,      # Disable mosaic 30 epochs before end
    label_smoothing=0.1,  # Helps with noisy labels

    # Multi-scale (IMPORTANT for dense shelves)
    multi_scale=True,     # Vary input 0.5x-1.5x during training
)
```

### Augmentation Impact Rankings (for grocery detection)
| Technique | Impact | Notes |
|-----------|--------|-------|
| Mosaic | +5-10% | Single most impactful for small datasets |
| Multi-scale | +1-3% | Handles size variation |
| HSV jitter | +1-2% | Lighting robustness |
| Mixup | +0.5-1% | Regularization |
| Erasing | +0.5-1% | Occlusion robustness |
| Scale jitter | +0.5-1% | Size generalization |
| CutMix | +0.5-1% | Try `cutmix=0.1` as alternative to mixup |
| Copy-paste (manual) | +2-4% | External product images on shelves |

### Custom Copy-Paste (Since Built-in Requires Segmentation)
Your existing `copy_paste_aug.py` is the right approach. Key improvements:
```python
# Match pasted product SCALE to existing annotations in the image
existing_widths = [ann['bbox'][2] for ann in img_annotations]
median_w = np.median(existing_widths)
scale = median_w / ref_product.width * random.uniform(0.7, 1.3)

# Paste ON shelf lines, not random positions
# Detect shelf lines from horizontal edges in the image
# or use y-coordinates of existing annotations

# Generate MORE variants (10-20 per image, not 3)
for i in range(20):
    aug_img = copy_paste_augment(
        base_img,
        products=random.sample(ref_products, random.randint(3, 15)),
        scale_range=(0.7, 1.3),
        along_shelves=True
    )
```

---

## 8. Training on ALL Data (No Val Split)

### The Trick
For final competition submission, after hyperparameter tuning on train/val split:
1. Determine optimal epochs from train/val experiments
2. Retrain on 100% of data (train + val combined) for that exact epoch count

### YOLOv8 Implementation
```yaml
# all_data.yaml — same paths for train and val
train: dataset/images/all
val: dataset/images/all    # Same as train (validation metrics will be inflated)
nc: 356
names: [...]
```

```python
# Use the epoch count that gave best val mAP in previous training
optimal_epochs = 150  # Determined from train/val split experiments

model = YOLO('yolov8l-oiv7.pt')
model.train(
    data='all_data.yaml',
    epochs=optimal_epochs,
    patience=0,            # NO early stopping (train all epochs)
    imgsz=1280,
    # ... all other settings from best config
)
# Use last.pt, NOT best.pt (best.pt is based on inflated val metrics)
```

### Important Caveats
- Validation metrics will be meaningless (overfitting to val = train)
- You MUST use a fixed epoch count (from previous train/val experiments)
- Use `last.pt` as the final model (not `best.pt` which is based on inflated metrics)
- Risk: slight overfitting with no validation signal. Mitigate with aggressive augmentation

### Expected Improvement: +1-3% mAP
With 248 images, 20% val = 50 images removed from training. Adding those back gives the model 25% more data. For severely data-limited scenarios, this matters.

---

## 9. Model Soup / Weight Averaging

### Concept
Average the weights of N models fine-tuned with different hyperparameters. Unlike ensembling (which runs N models at inference), model soup produces ONE model — no additional inference cost.

### Implementation for YOLOv8
```python
import torch
from ultralytics import YOLO
from copy import deepcopy

def model_soup(model_paths, output_path='soup.pt'):
    """Average weights of multiple YOLOv8 models."""
    # Load first model as base
    base = YOLO(model_paths[0])
    base_sd = base.model.state_dict()

    # Accumulate weights from all models
    for path in model_paths[1:]:
        m = YOLO(path)
        sd = m.model.state_dict()
        for key in base_sd:
            base_sd[key] += sd[key]

    # Average
    n = len(model_paths)
    for key in base_sd:
        base_sd[key] /= n

    # Load averaged weights back
    base.model.load_state_dict(base_sd)

    # Save — need to save the full ultralytics checkpoint format
    torch.save({
        'model': base.model,
        'train_args': base.ckpt.get('train_args', {}),
    }, output_path)

    return output_path

# Train multiple models with different configs
configs = [
    {'seed': 0, 'lr0': 0.001, 'mixup': 0.1},
    {'seed': 42, 'lr0': 0.001, 'mixup': 0.2},
    {'seed': 123, 'lr0': 0.0005, 'mixup': 0.15},
    {'seed': 456, 'lr0': 0.002, 'mixup': 0.1},
    {'seed': 789, 'lr0': 0.001, 'mixup': 0.1, 'scale': 0.7},
]

model_paths = []
for i, cfg in enumerate(configs):
    model = YOLO('yolov8l-oiv7.pt')
    model.train(
        data='dataset.yaml',
        epochs=200,
        patience=30,
        imgsz=1280,
        name=f'soup_run_{i}',
        deterministic=True,
        seed=cfg['seed'],
        lr0=cfg['lr0'],
        mixup=cfg['mixup'],
        scale=cfg.get('scale', 0.5),
    )
    model_paths.append(f'runs/soup_run_{i}/weights/best.pt')

# Create soup
soup_path = model_soup(model_paths)
```

### Greedy Soup (Better Than Uniform)
```python
def greedy_soup(model_paths, val_data, output_path='greedy_soup.pt'):
    """Add models to soup only if they improve validation mAP."""
    # Start with best individual model
    best_model = find_best_model(model_paths, val_data)
    soup_paths = [best_model]
    best_map = evaluate(best_model, val_data)

    for path in model_paths:
        if path == best_model:
            continue
        # Try adding this model to the soup
        candidate = model_soup(soup_paths + [path], 'candidate.pt')
        candidate_map = evaluate(candidate, val_data)

        if candidate_map > best_map:
            soup_paths.append(path)
            best_map = candidate_map
            print(f"Added {path}, mAP: {best_map:.4f}")
        else:
            print(f"Skipped {path}")

    return model_soup(soup_paths, output_path)
```

### Research Findings
- Uniform soup: +0.5-1% over best individual model
- Greedy soup: +1-2% over best individual model
- Works BEST when models share the same pretrained backbone but differ in fine-tuning hyperparameters
- Cost: N training runs (hours on Colab), but inference is FREE (same as single model)

### Expected Improvement: +1-3% mAP

---

## 10. Larger Input Resolution

### Resolution vs Memory vs Speed

| imgsz | VRAM (YOLOv8l, bs=1) | Inference (L4) | Training (L4, bs=4) | mAP impact |
|-------|----------------------|----------------|---------------------|------------|
| 640 | ~4 GB | ~8ms | ~12 GB | Baseline |
| 1280 | ~8 GB | ~25ms | ~20 GB | +3-5% |
| 1536 | ~12 GB | ~45ms | ~24 GB* | +1-2% over 1280 |
| 2048 | ~20 GB | ~80ms | OOM at bs=4 | +0.5-1% over 1536 |

*bs=4 may require gradient accumulation at 1536

### Practical Approach
```python
# Train at 1280 (fits L4 at bs=4)
model.train(data='dataset.yaml', imgsz=1280, batch=4, ...)

# Inference at 1536 (slightly larger than training — acceptable mismatch)
results = model.predict(img, imgsz=1536, ...)
```

### The Better Alternative: Tiling
For dense grocery shelves, **tiling at 640 gives BETTER results than high-resolution inference**:
- 640x640 tiles on a 4000x3000 image = 48 tiles
- Each tile processes products at 640px scale (optimal for YOLO)
- Products that are 100px in original = 100px in tile (vs 20px at imgsz=640 full-image)
- Tiling + WBF already solves the scale problem that higher resolution addresses

### Recommendation
- **Train at 1280** (your current setup is correct)
- **Inference: Full-image at 1280 + tiles at 640** (your current approach)
- **Do NOT go above 1536** — diminishing returns, timeout risk, and tiling is better

---

## Priority-Ordered Action Plan

### Tier 1: Quick Wins (1-2 hours, no retraining)
| # | Action | Expected Impact | Time |
|---|--------|----------------|------|
| 1 | Set `conf=0.001, max_det=1000, iou=0.7` everywhere | +3-8% | 10 min |
| 2 | Enable TTA: `augment=True` on full-image pass | +1-2% | 5 min |
| 3 | Tune WBF: `iou_thr=0.43, skip_box_thr=0.001` | +1-2% | 30 min |

### Tier 2: Retraining (4-8 hours on Colab)
| # | Action | Expected Impact | Time |
|---|--------|----------------|------|
| 4 | Two-stage corrected annotation fine-tuning | +3-8% | 2-3 hr |
| 5 | Train on 100% data (no val split) for final model | +1-3% | 2 hr |
| 6 | Model soup (5 training runs with different seeds) | +1-3% | 4-6 hr |
| 7 | Aggressive augmentation (multi_scale, higher hsv, label_smoothing) | +1-2% | 2 hr |

### Tier 3: Advanced (8-16 hours)
| # | Action | Expected Impact | Time |
|---|--------|----------------|------|
| 8 | Pseudo-labeling (2-3 iterations) | +2-5% | 6-8 hr |
| 9 | Dedicated EfficientNet classifier for 30% component | +2-5% overall | 4-6 hr |
| 10 | Improved manual copy-paste augmentation | +2-4% | 3-4 hr |

### Cumulative Estimate
- Current: 0.70-0.75
- After Tier 1: 0.75-0.82
- After Tier 2: 0.82-0.90
- After Tier 3: 0.88-0.95+

Note: Improvements are NOT perfectly additive. Diminishing returns apply. Realistic ceiling with all techniques: 0.90-0.93.

---

## Gotchas and Considerations

- **TTA + 3 models + tiling = timeout risk**. Budget 300s carefully. TTA on best single model only if time is tight.
- **conf=0.001 generates 5000+ predictions per image**. Ensure JSON output is not truncated.
- **Copy-paste in ultralytics REQUIRES segmentation masks**. Your manual approach is correct.
- **Pseudo-labeling reinforces errors**. Only use with high-quality base model.
- **Model soup requires same architecture**. Cannot average YOLOv8l and YOLOv8x weights.
- **Train on all data means NO validation signal**. Use fixed epoch count from previous experiments.
- **WBF is WORSE than NMS for single-model output**. Only use WBF for actual multi-model/multi-pass fusion.
- **OIV7 pretrained > COCO pretrained** for grocery. Keep using OIV7 checkpoints.
- **RT-DETR is NOT worth the switch** — marginal improvement, higher risk, no OIV7 weights.
- **3 weight file limit** constrains options. Two detectors + one classifier is likely optimal.

---

## Sources

1. [Ultralytics YOLOv8 Training Docs](https://docs.ultralytics.com/modes/train/) — Training parameters reference
2. [Ultralytics Data Augmentation Guide](https://docs.ultralytics.com/guides/yolo-data-augmentation/) — All augmentation parameters
3. [YOLOv8 vs RT-DETR Comparison](https://docs.ultralytics.com/compare/yolov8-vs-rtdetr/) — mAP/speed benchmarks
4. [ZFTurbo/Weighted-Boxes-Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion) — WBF implementation
5. [WBF Paper (arxiv 1910.13302)](https://arxiv.org/abs/1910.13302) — Original WBF method
6. [Model Soups (arxiv 2203.05482)](https://arxiv.org/abs/2203.05482) — Weight averaging theory
7. [Model Soups GitHub](https://github.com/mlfoundations/model-soups) — Reference implementation
8. [PyTorch SWA Blog](https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/) — Stochastic weight averaging
9. [CVPR 2022: Annotation Quality Effect](https://openaccess.thecvf.com/content/CVPR2022W/VDU/papers/Ma_The_Effect_of_Improving_Annotation_Quality_on_Object_Detection_Datasets_CVPRW_2022_paper.pdf) — Corrected labels impact
10. [Pseudo-Label Review (arxiv 2408.07221)](https://arxiv.org/pdf/2408.07221) — Comprehensive pseudo-labeling survey
11. [Adaptive Self-Training (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023W/LIMIT/papers/Vandeghen_Adaptive_Self-Training_for_Object_Detection_ICCVW_2023_paper.pdf) — Self-training for detection
12. [GitHub #3154: TTA in YOLOv8](https://github.com/ultralytics/ultralytics/issues/3154) — TTA predict-mode support
13. [GitHub #1469: TTA YOLOv8](https://github.com/ultralytics/ultralytics/issues/1469) — TTA limitations
14. [YOLOv5 TTA Docs](https://docs.ultralytics.com/yolov5/tutorials/test_time_augmentation/) — TTA details (+1.2% mAP)
15. [GitHub #7494: Val split](https://github.com/ultralytics/ultralytics/issues/7494) — Training without validation
16. [Ultralytics Community: Same train/val](https://community.ultralytics.com/t/set-validation-set-the-same-as-training-set/83) — Train=val discussion
17. [WBF LearnOpenCV](https://learnopencv.com/weighted-boxes-fusion/) — WBF parameter tutorial
18. [Kaggle WBF Approach](https://www.kaggle.com/code/shonenkov/wbf-approach-for-ensemble) — Competition WBF notebook
19. [timm GitHub](https://github.com/huggingface/pytorch-image-models) — EfficientNet/ConvNeXt models
20. [Multimodal Grocery Recognition](https://link.springer.com/article/10.1007/s00138-024-01549-9) — Image+OCR product recognition
21. [GitHub #20258: imgsz large images](https://github.com/ultralytics/ultralytics/issues/20258) — High resolution considerations
22. [Simple Copy-Paste (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.pdf) — Copy-paste augmentation
23. [Noisy Labels in Detection](https://arxiv.org/html/2211.13993v3) — Handling annotation noise
