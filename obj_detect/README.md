# NorgesGruppen Object Detection — NM i AI 2026

Grocery shelf product detection and classification. **Score: 0.8995** on the competition leaderboard.

## Approach

OIV7 pretrained YOLOv8l trained on 100% data, exported to ONNX. Single model at 1280px beats ensemble.

### Best submission (v9, 0.8995)
- **Single YOLOv8l ONNX** at 1280px — no ensemble, no tiling
- **OIV7 pretrain** (Open Images V7, 600 grocery-relevant classes) — much better than COCO
- Trained on **100% of data** (248 images, no val holdout)
- **conf=0.001**: Low confidence threshold maximizes recall for mAP
- 356 product categories trained end-to-end

### Key learnings
- **Single model > ensemble** for this task: ensemble confused classification (0.8857) vs single (0.8995)
- **ONNX mandatory**: .pt files fail in sandbox (Python 3.12->3.11 pickle incompatibility)
- **Tiling hurt**: Added false positives without improving detection enough
- **Classifier hurt**: EfficientNet re-classification (91% val acc) reduced score by 0.004
- **49K prediction cap**: Undocumented limit in sandbox

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
+-- run.py           (run_v9_best.py from HF)
+-- detect_l.onnx    (v7_final_l_1280.onnx from HF)
```

## Score History

| Version | Score | Format | Models | Notes |
|---------|-------|--------|--------|-------|
| v5 | 0.574 | ONNX | 1x l-640 | Preprocessing mismatch |
| v7 ONNX | 0.8857 | ONNX | 2x (l-1280 + l-640) | Ensemble + tiling + WBF |
| v8 ONNX+cls | 0.8818 | ONNX | 2x + EfficientNet | Classifier hurt score |
| **v9 single** | **0.8995** | **ONNX** | **1x l-1280** | **Best — single model wins** |

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
