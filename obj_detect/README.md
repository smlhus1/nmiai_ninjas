# NorgesGruppen Object Detection — NM i AI 2026

Grocery shelf product detection and classification. **Score: 0.8857** on the competition leaderboard.

## Approach

3-stage pipeline: OIV7 pretrained YOLOv8 -> trained on 100% data -> ONNX ensemble with tiling + WBF.

### Detection (70% of score)
- **2x YOLOv8 ONNX ensemble**: YOLOv8l at 1280px + YOLOv8l at 640px
- **OIV7 pretrain** (Open Images V7, 600 grocery-relevant classes) — much better than COCO for this domain
- **Tiled inference** on 1280 model: 640x640 tiles with 10% overlap, catches small products
- **WBF fusion** (iou_thr=0.43, optimized for dense grocery shelves)
- Trained on **100% of data** (248 images, no val holdout) for maximum learning

### Classification (30% of score)
- YOLOv8 multi-class output (356 product categories) trained end-to-end
- Category IDs map directly to competition annotations

### Key decisions
- **ONNX over .pt**: PyTorch .pt files failed in sandbox due to Python 3.12->3.11 pickle incompatibility. ONNX eliminates all version issues.
- **conf=0.001**: Low confidence threshold maximizes recall for mAP evaluation
- **49K prediction cap**: Undocumented 50K limit in competition sandbox
- **No copy-paste augmentation in final models**: Caused OOM issues; pseudo-labeling used instead in training pipeline

## Weights

Pre-trained weights available on HuggingFace:
**https://huggingface.co/smlhus/nmiai-grocery-detection**

- `v7_final_l_1280.onnx` (176 MB) — YOLOv8l, 1280px input
- `v7_final_l_640.onnx` (176 MB) — YOLOv8l, 640px input
- `run.py` — submission entry point
- `submission_v7_onnx.zip` — ready-to-submit zip

## Reproduce

### 1. Training (Google Colab with A100)

```bash
# Full pipeline: setup -> pseudo-labeling -> model soup -> final models
# See colab_train_v7.py for step-by-step instructions
```

Key training params:
- `ultralytics==8.1.0`, `torch==2.6.0+cu124`
- OIV7 pretrained: `yolov8l-oiv7.pt`, `yolov8x-oiv7.pt`
- AdamW, lr=0.001, cos_lr, label_smoothing=0.1, freeze=10
- 150 epochs, batch=32 (1280px) / batch=64 (640px)

### 2. ONNX Export

```python
from ultralytics import YOLO
model = YOLO('best.pt')
model.export(format='onnx', imgsz=1280, opset=17)
```

### 3. Submission

Download weights from HuggingFace, place alongside `run.py`, zip:
```
submission.zip
+-- run.py
+-- detect_l.onnx    (v7_final_l_1280.onnx)
+-- detect_l2.onnx   (v7_final_l_640.onnx)
```

## Score History

| Version | Score | Format | Models | Notes |
|---------|-------|--------|--------|-------|
| v5 | 0.574 | ONNX | 1x l-640 | Preprocessing mismatch |
| v7 ONNX | **0.8857** | ONNX | 2x (l-1280 + l-640) | Best submission |
| v8 ONNX+cls | 0.8818 | ONNX | 2x + EfficientNet | Classifier hurt score |

## Files

| File | Description |
|------|-------------|
| `colab_train_v7.py` | Full training pipeline (Steps 0-5) |
| `colab_train_v6.py` | Earlier v6 training script |
| `run_submission_v6.py` | Submission run.py with torch.load patch |
| `prepare_data.py` | COCO to YOLO format conversion |
| `copy_paste_aug.py` | Copy-paste augmentation with product images |
| `validate.py` | Local mAP evaluation |
| `package.py` | Build submission zip |

## Sandbox Environment

- Python 3.11, NVIDIA L4 24GB, CUDA 12.4
- 300s timeout, 8GB RAM, 420MB submission limit
- Pre-installed: ultralytics 8.1.0, PyTorch 2.6.0, onnxruntime-gpu 1.20.0

## Scoring

`0.7 x detection_mAP@0.5 + 0.3 x classification_mAP@0.5`

248 training images, 356 product categories, ~22,700 annotations.
