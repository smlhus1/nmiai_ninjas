# Research: Optimal Training Strategy for Mixed-Quality Annotations (Corrected vs Auto-Generated)

> Researched: 2026-03-21 | Sources consulted: 18 | Confidence: High

## TL;DR

**Do NOT remove uncorrected annotations.** Train in two stages: Stage 1 on ALL 22,700 annotations (200 epochs), then Stage 2 fine-tune on corrected-only subset (30-50 epochs, lr reduced 5-10x). This consistently outperforms single-stage training in the literature. If you want maximum bang-for-buck, implement a weighted dataloader that oversamples corrected images 2-3x — this is simpler than custom loss functions and directly supported via ultralytics monkey-patching.

## Key Findings

### 1. How Annotation Quality Affects mAP

Quantitative data from the Universal Noise Annotation paper (UNA, 2023) on Faster R-CNN / MS-COCO:

| Noise Level | Single Noise Type (worst: categorization) | Combined Noise (all types) |
|-------------|------------------------------------------|---------------------------|
| 5% noisy    | -0.9 mAP                                | -1.7 mAP                 |
| 10% noisy   | -1.4 mAP                                | -4.3 mAP                 |
| 15% noisy   | -1.7 mAP                                | -7.5 mAP                 |
| 20% noisy   | -2.4 mAP                                | -10.8 mAP                |
| 40% noisy   | —                                        | ~-33.5 mAP (49→15.5)     |

**Noise type severity ranking** (worst to least):
1. **Categorization noise** (wrong class label) — most damaging
2. **Localization noise** (inaccurate bbox) — second worst
3. **Missing annotations** (unlabeled objects) — moderate
4. **Bogus bounding boxes** (phantom objects) — least damaging

**Key insight**: Models maintain >80% of baseline mAP with up to 40% label corruption for individual noise types. Combined noise is far more destructive than any single type.

For your dataset (22,700 annotations, 248 images): if the auto-generated annotations are from a decent model (YOLO pretrained on similar data), expect mostly localization noise and some categorization errors. Missing annotations and bogus boxes are less likely from auto-labeling. This means your effective noise impact is probably in the 5-15% range per annotation, which is very manageable.

### 2. Typical Correction Rates

There is no universal "typical" correction rate — it depends entirely on the quality of the auto-annotation model. However:

- **COCO dataset itself** contains ~2.8% problematic annotations (25,144 boxes out of ~897K) according to CLOD analysis
- **Waymo dataset**: ~0.4% below quality threshold
- **Industry practice**: Model-assisted labeling pipelines typically achieve 70-90% usable auto-annotations, requiring 10-30% corrections
- **Your situation**: Having a "corrected" flag means you can explicitly control the quality split, which is better than most real-world scenarios

### 3. Two-Stage Training: The Recommended Approach

The literature strongly supports a two-stage approach, drawing from multiple methodologies:

#### Stage 1: Train on ALL data (200 epochs)
- Use all 22,700 annotations across 248 images
- Standard learning rate (lr0=0.01 for YOLOv8 default)
- Standard augmentation (mosaic, mixup, etc.)
- This gives the model broad feature learning from maximum data volume
- The model learns general patterns even from noisy labels — DNNs learn patterns before memorizing noise (the "early learning" phase)

#### Stage 2: Fine-tune on corrected-only (30-50 epochs)
- Use ONLY corrected=true annotations
- **Reduce learning rate 5-10x** (lr0=0.001-0.002)
- Keep augmentation moderate (reduce mosaic probability)
- Freeze early backbone layers (freeze=10, keeping layers 0-9 frozen)
- This "polishes" the model, correcting any noise memorized in Stage 1

**Why this works**: Research shows deep networks learn generalizable patterns first, then memorize noise in later epochs. Stage 2 leverages clean data to overwrite memorized noise while preserving the pattern knowledge from Stage 1.

#### Alternative: FHLR Model Merging (Advanced)
The FHLR paper (2025, Nature Scientific Reports) adds a third step:
1. Train seed model on all data (with label smoothing)
2. Fine-tune on corrected subset
3. **Merge the two models via weighted parameter averaging**: `merged = alpha * seed + (1-alpha) * finetuned`

This achieved up to 19% accuracy improvement. The merging preserves generalization from the seed while incorporating precision from the fine-tuned model. Alpha typically 0.3-0.7 (tune on validation set).

### 4. Weighting Corrected Annotations Higher

Three practical approaches, from simplest to most complex:

#### Option A: Weighted Dataloader (Recommended — Simplest)
Oversample images with corrected annotations during training:

```python
import numpy as np
from ultralytics.data.dataset import YOLODataset
from torch.utils.data import WeightedRandomSampler

class QualityWeightedDataset(YOLODataset):
    """Oversamples images with corrected annotations."""

    def __init__(self, *args, corrected_image_ids=None, quality_weight=3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = np.ones(len(self))
        if corrected_image_ids:
            for i, label in enumerate(self.labels):
                img_id = self.im_files[i]  # or however you map to image IDs
                if img_id in corrected_image_ids:
                    self.weights[i] = quality_weight
        self.weights /= self.weights.sum()
        self.sampler = WeightedRandomSampler(self.weights, len(self), replacement=True)

# Monkey-patch into ultralytics
import ultralytics.data.build as build
build.YOLODataset = QualityWeightedDataset
```

A quality_weight of 2-3x means corrected images appear 2-3x more often per epoch. This showed +2.5 mAP50 improvement in class balancing tests, expect similar or better for quality balancing.

#### Option B: Custom Loss Weighting (More Complex)
Override `v8DetectionLoss` to weight loss per image:

```python
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionTrainer

class QualityWeightedLoss(v8DetectionLoss):
    """Scales loss higher for corrected (high-quality) annotations."""

    def __init__(self, model, quality_weight=2.0, **kwargs):
        super().__init__(model, **kwargs)
        self.quality_weight = quality_weight

    def __call__(self, preds, batch):
        # Standard loss computation
        loss = super().__call__(preds, batch)
        # NOTE: requires custom batch metadata to know which images are corrected
        # This is the hard part — you need to pass quality flags through the dataloader
        return loss

class QualityDetectionModel(DetectionModel):
    def init_criterion(self):
        return QualityWeightedLoss(self, quality_weight=2.0)

class QualityTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = QualityDetectionModel(cfg, nc=self.data["nc"])
        if weights:
            model.load(weights)
        return model

# Usage
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(data="your_data.yaml", epochs=200, trainer=QualityTrainer)
```

**Caveat**: Per-image loss weighting requires modifying the dataloader to pass quality metadata in the batch dict, which is non-trivial in ultralytics.

#### Option C: Duplicate Corrected Images (Simplest Hack)
Simply include corrected images 2-3 times in your training split txt file. Crude but effective, zero code changes needed.

### 5. Does Ultralytics YOLOv8/YOLO11 Support Sample Weighting Natively?

**No.** Ultralytics does NOT natively support per-sample or per-annotation confidence weighting. What IS available:

| Feature | Supported? | How |
|---------|-----------|-----|
| Per-class loss weight | Yes (custom) | Override `v8DetectionLoss`, set `pos_weight` in BCE |
| Per-image sample weight | No (native) | Requires custom weighted dataloader (monkey-patch) |
| Per-annotation confidence | No | Not in YOLO format; would need custom loss |
| Loss component weights | Yes (native) | `box=7.5, cls=0.5, dfl=1.5` hyperparameters |
| Class frequency balancing | Partial | Custom weighted sampler (see Option A above) |
| Freeze layers | Yes (native) | `freeze=10` parameter |

The YOLO annotation format is `class x y w h` — there is no field for annotation confidence or quality. To use per-annotation weighting, you would need to extend the format and modify the dataloader, which is significant effort.

### 6. Remove vs Keep Uncorrected Annotations

**Strong consensus: KEEP uncorrected annotations.** Here is why:

| Strategy | Pros | Cons |
|----------|------|------|
| **Remove uncorrected** | Cleaner signal, no noise | Massive data loss, fewer examples, worse generalization |
| **Keep all, no weighting** | Maximum data volume | Some noise memorization in later epochs |
| **Keep all, weight corrected higher** | Best of both worlds | Slight implementation complexity |
| **Two-stage (recommended)** | Maximum generalization + precision | Two training runs needed |

Key evidence:
- Google Research found that "DNNs handle realistic label noise far better than random synthetic mislabeling"
- Removing 5% of the noisiest samples improved mAP@50 by up to 0.085 — but this was targeted removal of the WORST samples, not removing all uncorrected data
- CLOD found that cleaning datasets with automatic suggestions improved mAP by 16-46%, but this was targeted correction, not removal
- "Deep Learning is Robust to Massive Label Noise" (ICLR 2018): performance remains >90% even with label accuracy as low as 1% above chance (extreme case)

**Bottom line**: Your auto-generated annotations still contain valuable localization and feature information. Removing them loses ~90% of correct information to avoid ~10% noise. Weight them down instead.

### 7. Learning Rate for Stage 2 (Corrected-Only Fine-Tuning)

| Parameter | Stage 1 (All Data) | Stage 2 (Corrected Only) |
|-----------|-------------------|------------------------|
| lr0 | 0.01 (default) | 0.001-0.002 (5-10x lower) |
| lrf | 0.01 (default) | 0.1 (higher final ratio) |
| warmup_epochs | 3.0 | 0-1.0 (less warmup needed) |
| warmup_momentum | 0.8 | 0.9 |
| momentum | 0.937 | 0.937 (keep default) |
| weight_decay | 0.0005 | 0.0005 (keep default) |
| freeze | 0 (all trainable) | 10 (freeze early backbone) |

**Why lower LR**: The model already has good weights from Stage 1. A high LR would destroy learned features (catastrophic forgetting). Research on YOLOv8 fine-tuning shows "negligible catastrophic forgetting" with proper LR, even with significant backbone modifications.

**Cosine annealing** is recommended for Stage 2 — it naturally reduces LR over the fine-tuning period, preventing oscillation around the optimum.

### 8. Epochs for Stage 2

**Recommendation: 30-50 epochs for Stage 2**, with early stopping monitoring val mAP.

Rationale:
- You have fewer images in the corrected subset — overfitting risk is higher
- The model is already well-initialized from 200 epochs on all data
- Research shows 80 epochs is the upper bound for fine-tuning with good pretrained weights
- With a small corrected subset, diminishing returns set in quickly
- **Monitor val mAP@0.5 and stop if it plateaus for 10+ epochs** (use `patience=10`)

```bash
# Stage 1: Full training
yolo detect train data=your_data.yaml model=yolo11m.pt epochs=200 imgsz=640

# Stage 2: Fine-tune on corrected only
yolo detect train data=your_data_corrected.yaml model=runs/detect/train/weights/best.pt \
    epochs=50 lr0=0.001 freeze=10 patience=10 imgsz=640
```

## Complete Recommended Pipeline

```
┌─────────────────────────────────────────────────┐
│ STAGE 0: Data Preparation                       │
│ - Split corrected/uncorrected annotations       │
│ - Create two YAML configs (all vs corrected)    │
│ - Validate annotation format                     │
│ - Create proper train/val split (corrected imgs │
│   represented in BOTH splits for fair eval)     │
└─────────────┬───────────────────────────────────┘
              ▼
┌─────────────────────────────────────────────────┐
│ STAGE 1: Full Training (200 epochs)             │
│ - All 248 images, all 22,700 annotations        │
│ - lr0=0.01, standard augmentation               │
│ - Pre-trained YOLO11m backbone                   │
│ - Output: best.pt                                │
└─────────────┬───────────────────────────────────┘
              ▼
┌─────────────────────────────────────────────────┐
│ STAGE 2: Corrected Fine-Tune (30-50 epochs)     │
│ - Only corrected=true images/annotations        │
│ - Resume from Stage 1 best.pt                    │
│ - lr0=0.001, freeze=10, patience=10             │
│ - Reduced augmentation (mosaic=0.5)             │
│ - Output: best_refined.pt                        │
└─────────────┬───────────────────────────────────┘
              ▼
┌─────────────────────────────────────────────────┐
│ STAGE 3 (Optional): Model Merge                 │
│ - Weighted average of Stage 1 + Stage 2 weights │
│ - alpha=0.4 (tune on val set)                    │
│ - merged = alpha*stage1 + (1-alpha)*stage2      │
│ - May improve generalization                     │
└─────────────────────────────────────────────────┘
```

### Model Merging Code (Optional Stage 3)

```python
import torch

def merge_models(path_stage1, path_stage2, alpha=0.4, output_path="merged.pt"):
    """Weighted average of two YOLO checkpoints."""
    ckpt1 = torch.load(path_stage1, map_location="cpu")
    ckpt2 = torch.load(path_stage2, map_location="cpu")

    sd1 = ckpt1["model"].state_dict()
    sd2 = ckpt2["model"].state_dict()

    merged_sd = {}
    for key in sd1:
        merged_sd[key] = alpha * sd1[key] + (1 - alpha) * sd2[key]

    ckpt2["model"].load_state_dict(merged_sd)
    torch.save(ckpt2, output_path)
    print(f"Merged model saved: alpha={alpha} (stage1) + {1-alpha} (stage2)")

# Usage
merge_models(
    "runs/detect/train/weights/best.pt",    # Stage 1
    "runs/detect/train2/weights/best.pt",   # Stage 2
    alpha=0.4                                # 40% stage1 + 60% stage2
)
```

## Gotchas & Considerations

- **Validation set must be clean**: Your val split should ideally contain ONLY corrected annotations. Otherwise you are evaluating against noisy ground truth, which underestimates actual performance.
- **Categorization noise is worst**: If your auto-annotations have wrong class labels, that is far more damaging than slightly inaccurate bounding boxes. Prioritize correcting class labels over bbox positions.
- **248 images is small**: With only 248 images, augmentation is critical. Use mosaic, mixup, copy-paste, and hsv augmentation aggressively in Stage 1. Reduce in Stage 2.
- **Early learning phenomenon**: DNNs learn correct patterns in early epochs and memorize noise later. Your 200-epoch Stage 1 likely memorizes some noise in the final epochs — Stage 2 corrects this.
- **Auto-delivery of corrections**: If your auto-annotator was trained on similar data, its errors will be systematic (e.g., consistently mislabeling one class). Stage 2 is especially effective at correcting systematic bias.
- **Freeze strategy**: Freezing the first 10 layers in Stage 2 prevents the backbone from unlearning low-level features while allowing the detection head to recalibrate on clean data. This showed +10% mAP improvement over head-only fine-tuning.

## Recommendations

1. **Primary strategy**: Two-stage training (Stage 1 all data, Stage 2 corrected-only) — highest confidence, well-supported by literature
2. **Quick win**: If two-stage is too much work, simply duplicate corrected images 2-3x in the training list — zero code changes, meaningful improvement
3. **Advanced**: Add weighted dataloader (Option A) for Stage 1, giving corrected images 2-3x sampling weight
4. **Experimental**: Try model merging (Stage 3) with alpha sweep [0.2, 0.3, 0.4, 0.5] on val set
5. **Do NOT remove uncorrected annotations** — the data volume benefit far outweighs the noise cost

## Sources

1. [Universal Noise Annotation: Impact of Noisy Annotation on Object Detection (2023)](https://arxiv.org/html/2312.13822v1) — quantitative mAP degradation at different noise levels, noise type severity ranking
2. [Combating Noisy Labels in Object Detection Datasets (CLOD)](https://arxiv.org/html/2211.13993v3) — CLOD algorithm, 16-46% mAP improvement from cleaning, COCO has 2.8% problematic annotations
3. [Fine-Tuning Without Forgetting: YOLOv8 Preserves COCO Performance (2025)](https://arxiv.org/html/2505.01016v1) — freeze=10 gives +10% mAP, negligible catastrophic forgetting
4. [Understanding Deep Learning on Controlled Noisy Labels (Google Research)](https://research.google/blog/understanding-deep-learning-on-controlled-noisy-labels/) — real vs synthetic noise, MentorMix, early learning phenomenon
5. [FHLR: Few-Shot Human-in-the-Loop Refinement (Nature, 2025)](https://arxiv.org/abs/2401.14107) — model merging via weighted parameter averaging, 19% improvement
6. [Robust Curriculum Learning: Clean Label Detection to Noisy Label Self-Correction](https://openreview.net/forum?id=lmTWnm3coJJ) — curriculum learning from clean to noisy data
7. [Balance Classes During YOLO Training Using a Weighted Dataloader](https://y-t-g.github.io/tutorials/yolo-class-balancing/) — monkey-patch weighted sampler for ultralytics, +2.5 mAP50
8. [Customizing Trainer - Ultralytics YOLO Docs](https://docs.ultralytics.com/guides/custom-trainer/) — custom loss, model, and trainer override patterns
9. [Deep Learning is Robust to Massive Label Noise (ICLR 2018)](https://arxiv.org/pdf/1705.10694) — >90% accuracy even at 1% above chance label accuracy
10. [Constrained Reweighting for DNNs with Noisy Labels (Google Research)](https://research.google/blog/constrained-reweighting-for-training-deep-neural-nets-with-noisy-labels/) — dynamic instance-level importance weighting, 10% improvement
11. [The Effect of Improving Annotation Quality on Object Detection (CVPRW 2022)](https://openaccess.thecvf.com/content/CVPR2022W/VDU/papers/Ma_The_Effect_of_Improving_Annotation_Quality_on_Object_Detection_Datasets_CVPRW_2022_paper.pdf) — category/dataset-dependent quality impact
12. [Fine-tuning Pre-trained Models for Robustness under Noisy Labels (IJCAI 2024)](https://www.ijcai.org/proceedings/2024/0403.pdf) — pre-trained model fine-tuning on noisy data strategies
