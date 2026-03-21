# NorgesGruppen shelf detection: a sandbox-compliant blueprint to beat 0.725 mAP

**The path from 0.725 to 0.80+ mAP runs through three upgrades: switching from COCO to Open Images V7 pretraining, adding tiled inference with WBF fusion, and bolting on a dedicated EfficientNet-B0 classifier to capture the 30% classification score.** All three fit comfortably inside the 420 MB / 3-file / 300-second sandbox. The OIV7 pretrained YOLOv8l checkpoint alone provides 600-class features that include dozens of grocery-relevant categories — a dramatically richer starting point than the 80-class COCO weights behind the current 0.725 baseline. Combined with domain pretraining on SKU-110K (built into ultralytics) and a reference-image classification pipeline, this strategy addresses both scoring components while remaining fully compliant with every sandbox constraint.

---

## Open Images V7 weights are the single biggest upgrade available

The ultralytics v8.1.0 asset release includes OIV7-pretrained YOLOv8 weights across all sizes, pretrained on **600 detection classes** that include Bottle, Can, Box, Drink, Food, Fruit, Vegetable, and dozens more grocery-relevant objects. These download from `https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8{size}-oiv7.pt` and load with a single line: `YOLO('yolov8l-oiv7.pt')`.

| Model | File size | Params | mAP₅₀₋₉₅ (OIV7 val) | A100 TRT (ms) |
|---|---|---|---|---|
| yolov8m-oiv7.pt | **50.3 MB** | 26.2M | 33.6 | 2.26 |
| yolov8l-oiv7.pt | **84.4 MB** | 44.1M | 34.9 | 2.43 |
| yolov8x-oiv7.pt | **131.5 MB** | 68.2M | ~36.0 | 3.53 |

When fine-tuned with `model.train(data='grocery.yaml', epochs=100, imgsz=1280)`, the detection head is automatically rebuilt for 357 classes while backbone weights transfer intact. This is fully compliant: ultralytics 8.1.0 natively supports these checkpoints, no blocked imports are needed, and the largest variant (131.5 MB) uses only 31% of the 420 MB weight budget.

**No Objects365 pretrained YOLOv8 weights exist** — ultralytics provides the Objects365.yaml dataset config but never published .pt checkpoints. RT-DETR is available (rtdetr-l.pt at 63.4 MB, rtdetr-x.pt at 129.5 MB), but these are COCO-only pretrained and carry known AMP training bugs and higher memory usage at 1280px. RT-DETR is a viable ensemble candidate but not the primary model. Standard COCO weights (yolov8l.pt at 83.7 MB) offer no advantage over OIV7. YOLOv5u P6 models (e.g., yolov5m6u.pt at 79 MB, designed for 1280px images) could supplement an ensemble but lack the vocabulary breadth of OIV7.

**Compliance verdict:** All OIV7 weights are ✅ sandbox-compliant — native ultralytics 8.1.0 support, .pt format, well under 420 MB individually, no blocked imports needed.

---

## SKU-110K and LVIS provide free domain pretraining built into ultralytics

Two datasets stand out for intermediate pretraining, both requiring zero setup effort because they are built directly into ultralytics as auto-downloading YAML configs.

**SKU-110K** (`data='SKU-110K.yaml'`) contains **11,762 shelf images with 1.73 million bounding box annotations** — an average of 147 products per image. It is class-agnostic (single "object" class), but that is precisely why it excels at domain adaptation: the backbone learns to detect densely packed products on retail shelves, understand shelf geometry, and separate visually similar items. Training on SKU-110K teaches *where* products are; the competition fine-tuning stage then teaches *what* they are. The dataset auto-downloads at 13.6 GB and auto-converts from CSV to YOLO format. License is academic/non-commercial, which is appropriate for a competition setting.

**LVIS** (`data='lvis.yaml'`) offers **160,000 images with 2 million annotations across 1,203 categories**, including dozens of grocery items (apple, banana, bread, bottle, cheese, chocolate, cereal, yogurt, etc.). Its long-tail distribution — where many categories have few examples — mirrors the real-world imbalance in the competition's 357 categories. LVIS uses COCO images with richer annotations, making it an excellent multi-class pretraining bridge between OIV7 and the competition data. Licensed under CC BY 4.0.

**Other datasets evaluated but less recommended:**

- **RPC (Retail Product Checkout):** 83,739 images, 200 SKU classes, COCO format, CC BY-NC-SA 4.0. Available on Kaggle. However, its top-down checkout-counter domain differs significantly from frontal shelf views, creating a domain gap that limits transfer value.
- **MVTec D2S:** 21,000 images, 60 grocery categories with pixel-precise masks. Lab-controlled backgrounds limit transfer to real shelf images. CC BY-NC-SA 4.0.
- **Roboflow Universe:** The largest grocery detection dataset is ~14,000 images, downloadable in YOLO format under CC BY 4.0. Useful as supplementary data but far smaller than SKU-110K.
- **RP2K:** 500,000+ product images across 2,000 SKUs — classification only, no detection bounding boxes. Unusable for detection pretraining directly.
- **Grozi-120 / GP-180:** Very small, outdated datasets with incomplete annotations. Not recommended.

**Optimal pretraining pipeline (all compliant, all using ultralytics built-ins):**

1. Start from `yolov8l-oiv7.pt` (600-class backbone features)
2. Intermediate pretrain on SKU-110K for 30-50 epochs at imgsz=640 (shelf domain adaptation)
3. Fine-tune on competition 248 images at imgsz=1280 with 357 classes

---

## Tiled inference with WBF fusion fits easily in 300 seconds on L4

The L4 GPU (Ada Lovelace, 24 GB VRAM, **242 TFLOPS FP16**) provides generous headroom for sophisticated inference. At imgsz=1280, YOLOv8l processes a single image in roughly **8-14 ms** — meaning 100 test images take under 2 seconds for a single pass. This leaves over **295 seconds** for tiling, multi-scale fusion, and classification.

**supervision 0.18.0's `InferenceSlicer` provides built-in SAHI tiling.** InferenceSlicer was added in supervision 0.14.0 and is fully available in 0.18.0 with the `overlap_ratio_wh` parameter API. The key code pattern:

```python
import cv2, numpy as np, supervision as sv
from ultralytics import YOLO

model = YOLO("best.pt")

def callback(image_slice: np.ndarray) -> sv.Detections:
    result = model(image_slice, conf=0.10, max_det=500, verbose=False)[0]
    return sv.Detections.from_ultralytics(result)

slicer = sv.InferenceSlicer(
    callback=callback,
    slice_wh=(640, 640),
    overlap_ratio_wh=(0.25, 0.25),
    iou_threshold=0.4,
    thread_workers=1  # Must be 1; threading blocked in sandbox
)
detections = slicer(image)
```

For a 4000×3000 image with 640×640 tiles and 25% overlap, stride is 480, producing roughly **8×6 = 48 tiles**. At ~12 ms each, that is **~0.6 seconds per image** — trivially within budget even for 100+ test images. If `InferenceSlicer` triggers the `threading` import block (it uses `concurrent.futures.ThreadPoolExecutor` internally), a manual tiling fallback using only numpy array slicing and a simple loop achieves identical results — no blocked imports required.

**Optimal tile parameters for 3000-4000px shelf images with 100-400px products:** Use **640×640 tiles with 0.25 overlap** for products near the minimum size, or **800×800 tiles with 0.20 overlap** for faster processing with slightly fewer tiles. Products occupying 100-400px in the original image will occupy 16-100% of a 640px tile, well within YOLO's detection sweet spot.

**WBF merging with `ensemble_boxes`** is the right fusion strategy (superior to NMS for overlapping tile detections). The `weighted_boxes_fusion` function requires boxes normalized to [0, 1]:

```python
from ensemble_boxes import weighted_boxes_fusion

# boxes_list: list of [N, 4] arrays, normalized [0,1]
# scores_list: list of [N] arrays
# labels_list: list of [N] arrays (integer class IDs)
fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
    boxes_list, scores_list, labels_list,
    weights=[1, 2],        # higher weight for tiled pass
    iou_thr=0.5,
    skip_box_thr=0.001,    # keep nearly all candidates for mAP
    conf_type='avg'
)
```

**Recommended multi-source inference strategy within 300s:**

| Source | Resolution | Purpose | Est. time/image |
|---|---|---|---|
| Full-image pass | imgsz=1280 | Large and medium products | ~10 ms |
| Tiled pass | 640×640 tiles, 0.25 overlap | Small/dense products | ~600 ms |
| TTA on full-image | augment=True, imgsz=1280 | +1-1.5 mAP boost | ~30 ms |
| **Total** | | | **~640 ms** |

For 100 images: ~64 seconds total, well under 300s. This leaves ~236 seconds for the classification pipeline.

**Critical inference parameters:** Set `conf=0.10` (or even `0.001` — mAP is computed across all thresholds, so lower recall thresholds help), `max_det=1000` (default 300 is insufficient for dense shelves with 50-200+ products), and `iou=0.7` for NMS (higher preserves more overlapping but distinct products). Enable FP16 with `half=True` for additional speed on the L4's tensor cores.

---

## A dedicated EfficientNet-B0 classifier unlocks the 30% classification score

The scoring formula — **70% detection mAP + 30% classification mAP** — means a separate classification pipeline that improves category accuracy directly boosts the final score. YOLO's built-in classification is trained jointly with detection and often struggles with fine-grained distinctions among 357 visually similar grocery products. A dedicated classifier operating on cropped detections can significantly outperform YOLO's class predictions.

**EfficientNet-B0 from timm 0.9.12** is the optimal classifier choice: only **~21 MB** for weights, producing **1280-dimensional embeddings** at 77.7% ImageNet top-1 accuracy. It fits comfortably alongside the detection model within the 420 MB budget. Load it offline (no network access needed):

```python
import timm, torch
model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
state_dict = torch.load('efficientnet_b0.pth', map_location='cuda')
model.load_state_dict(state_dict, strict=False)
model = model.eval().cuda()
```

**The reference image embedding pipeline** leverages the competition's 327 product reference images (7 views each = ~2,289 images). Pre-compute embeddings offline, save as `.npy`:

1. Extract 1280-dim embeddings for all 2,289 reference images → `embeddings.npy` (~11.7 MB FP32, ~5.9 MB FP16)
2. Save corresponding category IDs → `labels.npy` (~9 KB)
3. At inference: crop YOLO detections → resize to 224×224 → extract embeddings → cosine similarity against database → assign nearest category

The cosine similarity search for 1000 crops against 2,289 reference embeddings is a **[1000, 1280] × [1280, 2289] matrix multiply — under 1 ms on GPU**. EfficientNet-B0 processes 1000 crops in batches of 64 in roughly **2-3 seconds total** on L4. The entire classification pipeline adds ~5 seconds for 100 images.

**Hybrid strategy is strongest:** Fine-tune EfficientNet-B0 as a 357-class classifier on ~22,700 crops extracted from training annotations (~64 crops/class average). Use the fine-tuned classifier for high-confidence predictions; fall back to embedding nearest-neighbor matching for low-confidence cases and products only in the reference set. This covers both seen and unseen products.

| timm model | Size | Embedding dim | Under 100 MB? | Recommendation |
|---|---|---|---|---|
| efficientnet_b0 | ~21 MB | 1280 | ✅ | **Best choice** |
| mobilenetv3_large_100 | ~22 MB | 1280 | ✅ | Good alternative |
| efficientnet_b2 | ~36 MB | 1408 | ✅ | Better accuracy, larger |
| vit_small_patch16_224 | ~87 MB | 384 | ✅ | High accuracy but budget-heavy |
| resnet50 | ~98 MB | 2048 | ⚠️ Borderline | Too large for the value |

**Compliance:** timm 0.9.12 is pre-installed ✅. Weights load from local `.pth` file (no network) ✅. numpy is allowed for `.npy` loading ✅. No blocked imports needed ✅. Total classifier pipeline adds ~6 MB (embeddings) + ~21 MB (weights) = **27 MB** to weight budget ✅.

---

## Training strategy: three-stage transfer with corrected annotation filtering

**Stage 1 — Domain pretraining on SKU-110K (30-50 epochs, imgsz=640):**
Start from `yolov8l-oiv7.pt`. Train on SKU-110K (class-agnostic) with `freeze=15` (only neck + head trains). This stage adapts the OIV7 backbone to dense retail shelf scenes. Use default augmentation settings; the 11,762 images provide sufficient diversity.

**Stage 2 — Full training on all competition annotations (80-120 epochs, imgsz=1280):**
Load Stage 1 weights. Train on all 248 images with 357 classes. Key augmentation settings for this critically small dataset:

```python
model.train(
    data='grocery.yaml',
    epochs=100,
    imgsz=1280,
    mosaic=1.0,        # Essential — creates 4x diversity per batch
    mixup=0.15,        # Moderate regularization
    scale=0.5,         # ±50% scale variation catches size diversity
    fliplr=0.5,        # Horizontal flip (products are left-right symmetric)
    flipud=0.0,        # No vertical flip (products are upright)
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,  # Color jitter for lighting variation
    close_mosaic=10,   # Disable mosaic last 10 epochs for stability
    freeze=10,         # Freeze early backbone, train later layers
    lr0=0.01,
    batch=8            # Fits on RTX 4090 at imgsz=1280
)
```

**Stage 3 — Fine-tuning on corrected-only annotations (20-30 epochs, lower LR):**
Filter `annotations.json` to `corrected=True` entries using the json module (compliant — no yaml/pickle needed). Unfreeze all layers, use `lr0=0.001`, `mosaic=0.5`, `close_mosaic=5`. This polishes the model on verified ground truth for maximum precision.

```python
import json
with open('annotations.json', 'r') as f:
    data = json.load(f)
corrected = [a for a in data['annotations'] if a.get('corrected', False)]
```

**Why `freeze=10` for OIV7 fine-tuning:** YOLOv8's layers 0-9 are early backbone convolutions that capture universal edge/texture features — already well-trained by OIV7's 600 classes. Freezing these prevents catastrophic forgetting on a 248-image dataset while allowing layers 10+ (later backbone, FPN/PAN neck, detection head) to adapt to grocery-specific features. Progressive unfreezing (freeze=15 → freeze=10 → freeze=0) over three training phases yields the most stable convergence.

**Training at imgsz=1280 vs 640:** Products on 4000px shelf images that are 100-200px occupy only 2.5-5% of the image at native resolution. At imgsz=640, they shrink to ~16-32px — near YOLO's minimum detectable size. At **imgsz=1280, these products become 32-64px, well within YOLO's reliable detection range**. Train at 1280 if GPU allows (requires ~30-48 GB VRAM at batch=8; fits on A100 or two RTX 4090s with gradient accumulation). During inference, match the training resolution.

---

## Weight budget: 114 MB leaves 73% of the 420 MB limit unused

The optimal weight allocation uses all three file slots:

| File | Contents | Size | Format |
|---|---|---|---|
| `best.pt` | YOLOv8l fine-tuned from OIV7 | ~87 MB | .pt ✅ |
| `classifier.pth` | Fine-tuned EfficientNet-B0 | ~21 MB | .pth ✅ |
| `embeddings.npy` | 2,289 reference embeddings + labels | ~6 MB | .npy ✅ |
| **Total** | | **~114 MB** | **27% of 420 MB budget** |

**Use native .pt inference, not ONNX.** The ultralytics `model.predict()` API handles all pre/post-processing seamlessly with zero blocked imports. ONNX export in ultralytics 8.1.0 has known FP16 bugs (GitHub issue #12721 — `half=True` did not reliably reduce file size), and ONNX inference requires manual letterbox preprocessing, output decoding, and NMS — significant added complexity. FP16 inference is already available via `model.predict(half=True)` with native .pt files. Given the generous 300s budget and ample weight room, ONNX offers no meaningful advantage.

**If ensemble detection is desired,** the budget easily accommodates two YOLO models: yolov8l-oiv7 fine-tuned (~87 MB) + yolov8x-oiv7 fine-tuned (~131 MB) + embeddings.npy (~6 MB) = **~224 MB**. Run both models, merge with WBF (`weights=[1, 1.5]`), then classify. However, the separate classifier likely provides more marginal score improvement than a second detection model, because the 30% classification component is currently unaddressed in the baseline.

---

## Conclusion: the integrated pipeline

The highest-impact changes over the 0.725 baseline are, in order of expected contribution:

1. **OIV7 pretraining** replaces COCO's 80 grocery-irrelevant classes with 600 classes including bottles, cans, food, and produce — expect **+2-4% detection mAP** from richer features alone.
2. **Tiled inference with WBF** recovers small products missed at full-image resolution — expect **+3-5% detection mAP** from improved recall on dense shelves.
3. **Dedicated EfficientNet-B0 classifier** directly addresses the 30% classification score that the YOLO-only baseline largely wastes — expect **+5-10% on the classification component**, translating to **+1.5-3% overall score**.
4. **SKU-110K domain pretraining** teaches shelf-specific features (product boundaries, shelf grid patterns, dense-object separation) — expect **+1-2% detection mAP** from domain-adapted features.
5. **Corrected-only fine-tuning** in the final training stage removes noisy auto-generated labels — expect **+0.5-1% mAP** from cleaner supervision.

The complete inference pipeline — YOLO tiled detection → WBF fusion → crop → EfficientNet-B0 classification → category override — runs in under **70 seconds for 100 images on L4 GPU**, using 23% of the time budget and 27% of the weight budget. Every component uses only pre-installed packages (`ultralytics`, `supervision`, `ensemble_boxes`, `timm`, `torch`, `numpy`, `pathlib`, `json`) and avoids all blocked imports. The strategy is fully deterministic, requires no runtime network access, and produces COCO-format predictions with both bounding boxes and corrected category IDs.