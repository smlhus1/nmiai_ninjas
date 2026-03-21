# NorgesGruppen Object Detection — NM i AI 2026

Grocery shelf product detection and classification pipeline.

## Pipeline Overview

1. **Data Prep** — `prepare_data.py`: COCO → YOLO format, 1-class (product), 80/20 split
2. **Augmentation** — `copy_paste_aug.py`: paste reference product images into training scenes
3. **Detection** — `train_detector.py`: YOLOv8m/l/x, 1-class product detector
4. **Embeddings** — `build_embeddings.py`: DINOv2-base prototypes per category
5. **Classifier** — `train_classifier.py`: optional linear probe on DINOv2
6. **Inference** — `run.py`: detect → crop → embed → classify → COCO JSON
7. **Validate** — `validate.py`: local mAP@0.5 evaluation
8. **Package** — `package.py`: build submission.zip (≤420 MB)

## Quick Start

```bash
# 1. Extract datasets
py prepare_data.py --coco-zip NM_NGD_coco_dataset.zip --output dataset/

# 2. Train detector
py train_detector.py --data dataset/dataset.yaml --epochs 100

# 3. Build reference embeddings
py build_embeddings.py --ref-zip NM_NGD_product_images.zip --annotations dataset/annotations.json --output weights/

# 4. Validate locally
py validate.py --input dataset/val/images --gt dataset/annotations.json --weights weights/

# 5. Package submission
py package.py --output submission.zip
```

## Sandbox Environment

- Python 3.11, NVIDIA L4 24GB, CUDA 12.4
- 300s timeout, 8GB RAM, 420MB submission limit
- Pre-installed: ultralytics 8.1.0, PyTorch 2.6.0, timm 0.9.12

## Scoring

`0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5`
