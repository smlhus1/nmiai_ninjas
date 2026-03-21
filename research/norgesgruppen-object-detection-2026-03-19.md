# Research: NorgesGruppen Object Detection — NM i AI 2026

> Researched: 2026-03-19 | Sources consulted: 28 | Confidence: High

## TL;DR

With 254 images and 357 classes, this is an extremely data-scarce fine-grained detection problem. The winning strategy is a **two-stage pipeline**: (1) train a class-agnostic or few-class detector (YOLOv8m/l) to localize products, then (2) classify crops using embeddings from reference images (CLIP/DINOv2). Ensemble with WBF gives +1-3% mAP. Realistic detection mAP@0.5 is 40-60%, classification mAP@0.5 is 20-45%, combined score 30-50%.

## Key Findings

### 1. State of the Art for Grocery Shelf Detection

The grocery shelf detection domain has specific characteristics that differentiate it from general object detection:

- **Dense packing**: Shelf images contain 50-200+ products in close proximity (SKU-110K averages 147 objects/image)
- **Fine-grained similarity**: Products in the same category differ only in subtle packaging details
- **Scale variation**: Products vary greatly in size within the same scene
- **Occlusion**: Products frequently overlap and obscure one another
- **Reflections**: Glass shelves, plastic packaging create reflections
- **Price tags**: Shelf labels and price markers create visual noise

**Best performing approaches in literature:**
- RetinaNet: 0.752 mAP on shelf detection
- Faster R-CNN: "mainstream technique" for retail, best accuracy
- YOLOv8m on SKU110K: F1=88.96% (class-agnostic detection)
- VGG-16 on GroZi-3.2k: 92.19% precision, 87.89% recall
- Large-scale (3,288 classes): only 36.02% mAP — illustrates the class count problem

**Critical insight**: With 357 classes and only 254 images (~0.7 images per class average), direct end-to-end detection+classification will fail. The literature shows even 3,288 classes with thousands of images only achieves ~36% mAP.

### 2. Best Models for 254 Images + 357 Categories

| Model | Strengths | Weaknesses | Recommendation |
|-------|-----------|------------|----------------|
| **YOLOv8m** | Fast, proven, 25.9M params, good augmentation pipeline | CNN attention limited for occlusion | **Primary choice for detection stage** |
| **YOLOv8l** | Better accuracy (52.9% mAP COCO), 43.7M params | More overfitting risk with small data | Good if validation loss is stable |
| **YOLOv8x** | Best accuracy (53.9% mAP), 68.2M params | Very high overfitting risk | Avoid with 254 images |
| **RT-DETRv2** | Transformer handles occlusion well, 54.3% mAP | 76M params, needs more VRAM, slower | Consider only if YOLOv8 plateaus |
| **RF-DETR** | DINOv2 backbone, best transfer learning, 60+ mAP COCO | Not pre-installed (needs pip), 29-128M params | **Best if you can get it running** |
| **Faster R-CNN** | Best for two-stage, mature for retail | Slower inference, complex setup | If time allows |

**Verdict**: YOLOv8m is the pragmatic choice — it's pre-installed, proven for retail, and medium-sized enough to avoid overfitting. RF-DETR would be ideal but requires `pip install rfdetr` which is blocked in sandbox.

### 3. The Two-Stage Strategy (Recommended)

Given the scoring formula `0.7 * detection_mAP@0.5 + 0.3 * classification_mAP@0.5`, detection matters more than classification. The optimal strategy:

#### Stage 1: Detection (70% of score)
Train a **class-agnostic or section-grouped detector** instead of 357-class detection:

- **Option A — Class-agnostic (1 class "product")**: All 254 images, all 22,300 annotations as "product". Maximizes detection training data. Simple, robust.
- **Option B — Section-grouped (4 classes)**: Egg, Frokost, Knekkebroed, Varmedrikker. Still plenty of training data per class.
- **Option C — Full 357-class**: Direct detection. Will likely have many classes with 0-2 training examples. Poor mAP.

**Recommendation**: Train Option A for maximum detection mAP, then classify in Stage 2.

#### Stage 2: Classification (30% of score)
Classify detected crops using reference images:

- Crop each detected product from the image
- Generate embeddings using a pre-trained model (timm has DINOv2, EfficientNet, etc.)
- Compare crop embeddings to reference image embeddings via cosine similarity
- Assign the closest matching category_id

**Reference image pipeline:**
1. Load all 327 product reference images (multi-angle)
2. Generate embeddings for each reference image using `timm` model
3. Average embeddings per product to get prototype vectors
4. At inference: detect -> crop -> embed -> nearest-neighbor classify

```python
# Pseudo-code for two-stage pipeline
import timm
import torch
from ultralytics import YOLO

# Stage 1: Detection
det_model = YOLO("best_detector.pt")
results = det_model.predict(image, conf=0.25)

# Stage 2: Classification with reference embeddings
cls_model = timm.create_model('vit_base_patch14_dinov2', pretrained=True, num_classes=0)
cls_model.eval()

for box in results[0].boxes:
    crop = image[y1:y2, x1:x2]
    crop_tensor = preprocess(crop)
    embedding = cls_model(crop_tensor)
    # Nearest neighbor against reference prototypes
    similarities = torch.cosine_similarity(embedding, reference_prototypes)
    category_id = category_ids[similarities.argmax()]
```

### 4. Data Augmentation Strategy

#### YOLOv8 Built-in Augmentations (recommended settings for small datasets)

```yaml
# Key augmentation parameters for 254-image grocery detection
mosaic: 1.0          # Combine 4 images — critical for small datasets
mixup: 0.15          # Blend two images — reduces overfitting
hsv_h: 0.015         # Hue shift — handles lighting variation
hsv_s: 0.7           # Saturation — handle different cameras
hsv_v: 0.4           # Brightness — shadows, reflections
degrees: 5.0         # Slight rotation — shelf angle variation
translate: 0.1       # Position shift
scale: 0.5           # Size variation
shear: 2.0           # Perspective skew
perspective: 0.0005  # 3D perspective distortion
fliplr: 0.5          # Horizontal flip — standard
flipud: 0.0          # NO vertical flip — products don't appear upside down
erasing: 0.4         # Random erasing — simulates occlusion
close_mosaic: 10     # Disable mosaic last 10 epochs for fine-tuning
```

#### Albumentations Pipeline (pre-installed, version 1.3.1)

```python
import albumentations as A

augmentation_pipeline = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.CLAHE(clip_limit=2.0, p=0.3),  # Handle uneven store lighting
    A.RandomShadow(p=0.2),            # Shelf shadows
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),  # Simulate occlusion
], bbox_params=A.BboxParams(format='coco', min_visibility=0.3))
```

#### Copy-Paste with Reference Images

The 327 multi-angle product reference images can be used for copy-paste augmentation:
1. Segment product from reference image background (simple threshold or GrabCut)
2. Paste onto shelf training images at random positions
3. Add corresponding bounding box annotations

**Copy-paste provides +5-10 AP in low data regime** (Google Brain, CVPR 2021). This is the single highest-impact augmentation technique for this challenge.

### 5. Classification with Reference Images

#### Approach 1: Embedding-based Nearest Neighbor (Recommended)

Available via `timm 0.9.12` (pre-installed):

| Model | Size | ImageNet Top-1 | Speed | Notes |
|-------|------|----------------|-------|-------|
| `vit_base_patch14_dinov2` | 86M | 86.5% | Medium | Best embeddings for products |
| `efficientnet_b3` | 12M | 82.0% | Fast | Good balance |
| `convnext_base` | 89M | 85.8% | Medium | Strong features |
| `vit_small_patch14_dinov2` | 22M | 81.1% | Fast | If memory is tight |

**DINOv2 is the best choice** for generating embeddings because:
- Self-supervised pretraining captures fine-grained visual features
- Explicitly designed for visual similarity tasks
- Same backbone as RF-DETR (which leads COCO benchmarks)
- Reference: k-NN with k=11 on retail products achieved 93.94% accuracy

#### Approach 2: CLIP Zero-Shot (if timm doesn't have suitable models)

CLIP can classify products using text prompts, but for retail products:
- Zero-shot with text: only 48.96-62.30% accuracy (insufficient)
- Few-shot with image prototypes: 89.90-93.94% accuracy (excellent)
- **Visual prototypes (reference images) dramatically outperform text prompts**

#### Approach 3: Fine-tune Classifier on Reference + Crop Data

1. Detect products in training images, crop with ground truth boxes
2. Combine with reference images as additional training data
3. Fine-tune EfficientNet-B3 or DINOv2-small on 357-class classification
4. Use heavy augmentation on reference images (rotation, color jitter, scale)

### 6. Ensemble Techniques

`ensemble-boxes 1.0.9` is pre-installed. WBF typically gives +1-3% mAP over single models.

#### Multi-Model Ensemble Strategy

```python
from ensemble_boxes import weighted_boxes_fusion

# Train 3 diverse models:
# 1. YOLOv8m (imgsz=640)
# 2. YOLOv8l (imgsz=640)
# 3. YOLOv8m (imgsz=1280, multi-scale)

# Normalize boxes to [0,1] range
boxes_list = [model1_boxes, model2_boxes, model3_boxes]
scores_list = [model1_scores, model2_scores, model3_scores]
labels_list = [model1_labels, model2_labels, model3_labels]
weights = [1.0, 1.0, 0.8]  # Weight by validation mAP

boxes, scores, labels = weighted_boxes_fusion(
    boxes_list, scores_list, labels_list,
    weights=weights,
    iou_thr=0.55,        # Merge similar boxes
    skip_box_thr=0.001   # Keep low-confidence detections
)
```

#### TTA + WBF (No Extra Model Training)

Test-Time Augmentation provides a "free" ensemble effect:
- Original image
- Horizontal flip
- Scale 0.8x and 1.2x
- Merge with WBF

**Expected improvement: +1-2% mAP@0.5** with 2-3x inference time.

### 7. Realistic mAP Benchmarks

| Scenario | Detection mAP@0.5 | Classification mAP@0.5 | Combined Score |
|----------|-------------------|------------------------|----------------|
| **Naive 357-class YOLO** | 15-25% | Same | 15-25% |
| **Two-stage (detect+classify)** | 50-65% | 25-40% | 42-57% |
| **Two-stage + ensemble + TTA** | 55-70% | 30-45% | 47-62% |
| **Theoretical ceiling** | 75-85% | 50-65% | 67-79% |

**Key reference points:**
- SKU110K (class-agnostic, thousands of images): F1=89%
- GroZi-3.2k (3,235 classes, larger dataset): 92% precision
- Large-scale retail (3,288 classes): 36% mAP — shows the scaling problem
- Checkout (200 classes, 53K images): 80% accuracy
- With only 254 images, expect 60-70% of these benchmarks

### 8. Training Configuration Recommendations

```python
from ultralytics import YOLO

# Detection model (class-agnostic or 4-section)
model = YOLO("yolov8m.pt")  # Start from COCO pretrained

results = model.train(
    data="grocery.yaml",
    epochs=200,              # More epochs for small dataset
    patience=50,             # Early stopping
    batch=16,                # L4 24GB can handle this at 640
    imgsz=640,               # Standard; try 1280 if time allows

    # Optimizer
    optimizer="AdamW",       # Better than SGD for small datasets
    lr0=0.001,               # Lower LR for fine-tuning
    lrf=0.01,                # Final LR ratio
    warmup_epochs=5,         # Warm up on small dataset
    weight_decay=0.0005,

    # Augmentation (aggressive for small dataset)
    mosaic=1.0,
    mixup=0.15,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=5.0,
    scale=0.5,
    translate=0.1,
    fliplr=0.5,
    erasing=0.4,
    close_mosaic=10,

    # Regularization
    dropout=0.1,             # Prevent overfitting

    # Other
    amp=True,                # Mixed precision for speed
    workers=4,               # Match 4 vCPU
    seed=42,
    save_period=10,
    val=True,
)
```

#### L4 GPU Memory Budget (24GB VRAM)

| Model | imgsz | Batch | VRAM Est. | Training Time (200 epochs) |
|-------|-------|-------|-----------|---------------------------|
| YOLOv8n | 640 | 32 | ~6GB | ~15 min |
| YOLOv8s | 640 | 24 | ~10GB | ~25 min |
| YOLOv8m | 640 | 16 | ~14GB | ~40 min |
| YOLOv8l | 640 | 8 | ~18GB | ~60 min |
| YOLOv8m | 1280 | 4 | ~20GB | ~90 min |
| YOLOv8x | 640 | 4 | ~22GB | ~80 min |

With 300s timeout, you can train ~75 epochs of YOLOv8m at batch 16, imgsz 640. **Pre-train offline and bring weights in the zip.**

### 9. Open Source Projects for Retail Detection

| Project | Stars | Status | Relevance |
|---------|-------|--------|-----------|
| [SKU110K_CVPR19](https://github.com/eg4000/SKU110K_CVPR19) | ~500 | Archived | Dense shelf detection baseline |
| [YOLOv8-retail](https://github.com/vmc-7645/YOLOv8-retail) | ~50 | Active | YOLOv8 retail detection |
| [shelf-product-identifier](https://github.com/albertferre/shelf-product-identifier) | ~30 | Recent | YOLOv8 + embeddings pipeline (exactly our approach) |
| [ShelfGuard](https://github.com/ElSalvatore-sys/shelfguard) | ~10 | Recent | YOLOv11 + albumentations for shelf monitoring |
| [copy-paste-aug](https://github.com/conradry/copy-paste-aug) | ~300 | Stable | Copy-paste augmentation implementation |
| [Weighted-Boxes-Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion) | ~2K | Active | WBF ensemble (already pre-installed) |

### 10. Known Challenges and Gotchas

- **Overlapping products**: Products on shelves frequently overlap — mosaic augmentation and NMS/WBF tuning help
- **Reflections**: Glass shelf covers and plastic wrap create specular reflections — HSV augmentation + CLAHE
- **Price tags**: Shelf labels, price stickers create false positives — need negative example training or post-filtering
- **Lighting variation**: Different store sections have different lighting — aggressive brightness/contrast augmentation
- **Category imbalance**: Some of 357 categories may have 0 training examples — embedding-based classification handles this naturally since reference images exist for all products
- **Similar products**: Same brand, different flavor — fine-grained classification requires strong embeddings (DINOv2 excels here)
- **Small products**: Items like spice packets are tiny in shelf images — multi-scale training (imgsz=1280) helps
- **300s sandbox timeout**: Cannot train from scratch in sandbox. **Must bring pre-trained weights.**
- **420MB weight limit**: YOLOv8m=49.7MB, DINOv2-base=346MB. Total ~396MB fits within 420MB limit.
- **No pip install**: Must use only pre-installed packages. RF-DETR and CLIP require additional packages.
- **ONNX export**: If using custom models, export to ONNX before submission (onnxruntime-gpu is pre-installed)

## Recommended Competition Strategy

### Phase 1: Offline Preparation (before submission)

1. **Train class-agnostic YOLOv8m detector** on all 254 images (all boxes = class 0 "product")
   - Also train YOLOv8l and YOLOv8m-1280 variants for ensemble
2. **Build reference embedding database** using DINOv2 (via timm)
   - Generate embeddings for all 327 product reference images
   - Create prototype vectors (mean per product category)
   - Save as `.pt` file
3. **Implement copy-paste augmentation** with reference images to boost training data
4. **Train 4-class section detector** as alternative (Egg, Frokost, Knekkebroed, Varmedrikker)
5. **Validate** two-stage pipeline on held-out images

### Phase 2: Submission Package (within 420MB)

```
submission.zip (max 420MB)
├── detect_model.pt          # YOLOv8m detector (~50MB)
├── classify_model.pt        # DINOv2-base via timm (~346MB) OR EfficientNet-B3 (~12MB)
├── reference_embeddings.pt  # Pre-computed reference prototypes (~2MB)
└── run.py                   # Inference script
```

### Phase 3: Inference Pipeline (within 300s)

```python
# 1. Load models
# 2. For each test image:
#    a. Run YOLOv8m detection (class-agnostic)
#    b. Optional: run YOLOv8l + WBF ensemble
#    c. Crop detected products
#    d. Generate embeddings for crops
#    e. Match to nearest reference prototype
#    f. Output: boxes + category_ids + scores
```

### Priority Order (if time-constrained)

1. **Must-have**: YOLOv8m class-agnostic detector + DINOv2 reference classifier
2. **High value**: Copy-paste augmentation with reference images during training
3. **Medium value**: Multi-model ensemble with WBF
4. **Medium value**: TTA at inference time
5. **Nice-to-have**: Pseudo-labeling on unlabeled test images
6. **Nice-to-have**: Fine-tune DINOv2 classifier on training crops + reference images

## Gotchas & Considerations

- **copy_paste in YOLOv8 only works with segmentation masks**, not bounding boxes. Must implement custom copy-paste separately using reference images.
- **DINOv2 via timm**: Verify `vit_base_patch14_dinov2` is available in timm 0.9.12. If not, fall back to `efficientnet_b3` or `convnext_base`.
- **Scoring weights**: Detection matters 2.3x more than classification (0.7 vs 0.3). Prioritize detection quality.
- **ONNX fallback**: If custom models don't load in sandbox, export to ONNX and use onnxruntime-gpu.
- **Memory limit**: 8GB RAM + 24GB VRAM. DINOv2-base + YOLOv8m should fit but watch batch sizes during inference.
- **albumentations 1.3.1** is an older version — some newer transforms may not be available. Test locally first.
- **supervision 0.18.0** has useful visualization/annotation tools but is not critical for the pipeline.

## Sources

1. [Best Object Detection Models 2025: RF-DETR, YOLOv12 & Beyond](https://blog.roboflow.com/best-object-detection-models/) — Model comparison and RF-DETR benchmarks
2. [Deep Learning for Retail Product Recognition: Challenges and Techniques](https://pmc.ncbi.nlm.nih.gov/articles/PMC7676964/) — Comprehensive retail detection survey with mAP benchmarks
3. [RT-DETRv2 vs YOLOv8 Technical Comparison](https://docs.ultralytics.com/compare/rtdetr-vs-yolov8/) — Detailed model comparison table
4. [YOLOv8 training with very small datasets](https://github.com/ultralytics/ultralytics/issues/6201) — Community tips for small dataset training
5. [Weighted-Boxes-Fusion GitHub](https://github.com/ZFTurbo/Weighted-Boxes-Fusion) — WBF implementation and usage examples
6. [Exploring Fine-grained Retail Product Discrimination with VLMs](https://arxiv.org/html/2409.14963) — CLIP/DINOv2 for retail product classification, k-NN 93.94% accuracy
7. [Simple Copy-Paste is a Strong Data Augmentation Method (CVPR 2021)](https://arxiv.org/abs/2012.07177) — +10 AP in low data regime
8. [YOLO Data Augmentation Guide](https://docs.ultralytics.com/guides/yolo-data-augmentation/) — Complete augmentation parameter reference
9. [shelf-product-identifier](https://github.com/albertferre/shelf-product-identifier) — YOLOv8 + embedding pipeline for shelf products
10. [RF-DETR: A SOTA Real-Time Object Detection Model](https://blog.roboflow.com/rf-detr/) — DINOv2 backbone, best for small datasets
11. [SKU-110K Dense Product Detection](https://docs.ultralytics.com/datasets/detect/sku-110k/) — Benchmark for dense shelf detection
12. [Real-time retail planogram compliance (Nature, 2025)](https://www.nature.com/articles/s41598-025-27773-5) — YOLOv8 + few-shot: 95.7% mAP@50, 98.39% Top-1 with 5 samples/class
13. [RetailKLIP: Finetuning OpenCLIP for Zero-shot Retail Classification](https://arxiv.org/html/2312.10282) — CLIP fine-tuning for retail products
14. [Hybrid Method for Multi-Stage Grocery Product Recognition](https://www.mdpi.com/2079-9292/12/17/3640) — Multi-stage detect+classify pipeline
