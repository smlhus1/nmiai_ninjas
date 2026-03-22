# What We Learned — NorgesGruppen Object Detection

NM i AI 2026, March 19-22. From 0.574 to 0.8995 in 48 hours with 248 training images.

## The Journey

### Day 1-2: Getting something to work (0 → 0.574)

We started with zero object detection experience. The challenge: detect and classify 356 grocery products on store shelves.

**First attempts were disasters.** We tried a 1-class detector (just find "product") with a separate DINOv2 classifier for identification. The detector learned shelf rows instead of individual products — huge boxes covering entire shelves. Multi-class YOLO was the fix: forcing the model to learn 356 categories made it draw tight boxes around individual products.

**SKU-110K pretrain was a dead end.** It's a retail dataset, but annotated as shelf rows, not individual products. Our model inherited that behavior. COCO pretrain was better, but the real breakthrough was OIV7 (Open Images V7) — 600 classes including many grocery-relevant items. Immediate +2% mAP.

**The ONNX saga.** Our first working submission used ONNX because .pt files from torch 2.10 crashed in the sandbox (torch 2.6.0). But ONNX had a preprocessing mismatch — we did manual letterboxing wrong, losing ~15% mAP vs .pt. The fix was using `YOLO("model.onnx")` which handles preprocessing identically. Score: 0.574.

**Submission format ate 4 of 6 submissions.** We burned submissions debugging exit code 1 errors. Lessons:
- Always test in a clean subprocess, not the training notebook
- The sandbox security scanner is aggressive — test every import
- .pt files are NOT portable between Python 3.12 (Colab) and 3.11 (sandbox)

### Day 3: The real training (0.574 → 0.8857)

**Retrained with exact sandbox versions.** torch 2.6.0+cu124, ultralytics 8.1.0. Crucial: models must be trained with the SAME versions the sandbox runs.

**OOM was our constant enemy.** Copy-paste augmentation creates images with 1400+ instances — random batches spike VRAM. Solutions: lower batch size, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, gc.collect() between models.

**3-model ensemble + WBF.** YOLOv8l at 1280px, YOLOv8x at 640px, YOLOv8l at 640px. Weighted Boxes Fusion with iou_thr=0.43 (research showed this beats the default 0.55 for dense scenes). Tiled inference on the 1280 model caught small products. Score: 0.8857.

**torch.load patch is mandatory.** ultralytics 8.1.0 claims to patch `weights_only=False` but it doesn't work in the sandbox. You MUST add:
```python
_original_load = torch.serialization.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, 'weights_only': False})
```

BUT: this didn't help either — .pt files still failed. Python 3.12→3.11 pickle incompatibility is real.

### Day 4: Nuclear options (0.8857 → 0.8995)

**ONNX is the ONLY format that works.** After multiple .pt failures, we exported everything to ONNX. No pickle, no version issues, just works. This should have been our approach from day 1.

**Training on 100% data was massive.** Dropping the 20% validation holdout and training on all 248 images pushed val mAP from 0.741 to 0.959 (train=val, so overfit, but the model learned the data perfectly). This was the single biggest improvement.

**Pseudo-labeling pipeline.** Used the trained model to generate labels on augmented copies of training images (heavy color/noise augmentation). 528/600 images got pseudo-labels. Combined with original data for model soup training.

**Model soup.** Trained 3 models with different random seeds, averaged their weights. Each individual model: ~0.736 mAP. Soup should generalize better, but we couldn't verify on a real test set.

**The classifier that made things worse.** Fine-tuned EfficientNet-B0 on crops + product reference images. 91% validation accuracy! But re-classifying YOLO detections dropped the score from 0.8857 to 0.8818. Why? YOLO's classification uses spatial context (shelf layout, neighboring products). EfficientNet only sees the isolated crop — it's like identifying a product without seeing the shelf.

**Single model beat ensemble.** This was the biggest surprise. v7 ensemble (2 models + tiling + WBF): 0.8857. v9 single model (1x l-1280, no tiling): 0.8995. The ensemble confused classification — two models disagreeing on category_id, with WBF averaging them into something wrong.

## Technical Learnings

### What mattered most (in order)

1. **OIV7 pretrain** — grocery-relevant classes vs COCO's 80 general classes
2. **Training on 100% data** — no validation holdout for final submission
3. **ONNX export** — the only format that survives Python version differences
4. **High resolution (1280px)** — grocery shelves have tiny products
5. **conf=0.001** — mAP is computed across all confidence levels; low threshold = better recall
6. **Single model > ensemble** — cleaner classification output

### What didn't matter or hurt

1. **Ensemble/WBF** — confused classification more than it helped detection
2. **Tiling** — added false positives, marginal recall improvement
3. **Separate classifier** — spatial context matters, isolated crops lose it
4. **Copy-paste augmentation** — caused OOM, pseudo-labeling was better
5. **Corrected annotations** — the dataset had NO corrected annotations (field was always false)
6. **Model soup** — individual models were weaker than simple 100% data training

### Sandbox gotchas

- **Python 3.11 vs 3.12**: .pt files pickled in 3.12 DON'T load in 3.11. Use ONNX.
- **ultralytics patches don't work**: Despite having `weights_only=False` in patches.py, it fails in sandbox. Monkey-patch yourself — or just use ONNX.
- **Security scanner blocks**: os, sys, subprocess, pickle, shutil, yaml, gc, threading, multiprocessing. Also timm and safetensors imports (discovered the hard way).
- **50K prediction limit**: Undocumented. conf=0.001 with tiling easily generates 300K+ predictions.
- **8GB RAM**: Loading 3 YOLO models is fine (~600MB GPU), but peak CPU during pickle deserialization can spike.
- **FP16 ONNX needs GPU export**: `half=True` is silently ignored on CPU. Export on GPU or accept FP32 sizes.
- **ONNX is ~2x .pt size**: 3 FP32 ONNX models won't fit in 420MB. Plan for 1-2 models.

### Training setup that worked

```python
# Exact sandbox versions
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics==8.1.0

# Training params
model = YOLO('yolov8l-oiv7.pt')  # OIV7 pretrain
model.train(
    data='dataset.yaml',
    epochs=150, imgsz=1280, batch=32,  # A100 80GB
    optimizer='AdamW', lr0=0.001, lrf=0.01,
    cos_lr=True, label_smoothing=0.1, freeze=10,
    mosaic=1.0, mixup=0.15, erasing=0.3,
    patience=0,  # no early stopping for 100% data training
)

# Export
model.export(format='onnx', imgsz=1280, opset=17)
```

## Process Learnings

### What we'd do differently next time

1. **Start with ONNX from day 1.** Don't waste submissions debugging .pt compatibility.
2. **Test in exact sandbox environment.** Docker container matching Python 3.11 + all package versions. Our Colab L4 test (Python 3.12) passed but sandbox failed.
3. **Single model first, ensemble later.** We spent hours on ensemble infrastructure that hurt the score.
4. **Train on 100% data for ALL final submissions.** The val set was only 48 images — too small to be meaningful.
5. **Use product reference images during YOLO training** (copy-paste aug), not as a separate classifier.
6. **Budget submissions carefully.** We had 3/day and burned most on format errors.

### Tools that helped

- **Google Colab Pro+** (A100 80GB) — essential for 1280px training with batch=32
- **Google Drive** — persistent weight storage across Colab sessions
- **Claude Code + Colab MCP** — automated monitoring, building, testing
- **Discord notifications** — knew immediately when training finished
- **ultralytics** — saved weeks of implementation. YOLO("model.onnx") just works.

### The numbers

| Metric | Value |
|--------|-------|
| Training images | 248 |
| Product categories | 356 |
| Annotations | 22,731 |
| Total training time | ~12 hours (A100) |
| Colab compute used | ~100 units |
| Submissions used | 8 (3 failed, 5 scored) |
| Best score | **0.8995** |
| Score improvement | 0.574 → 0.8995 (+57%) |
| Final ranking | ~135th (object detection), 16th overall |

### Score progression

```
Day 1: 0.000 — Format errors, exit code 1
Day 2: 0.574 — First working ONNX submission
Day 3: 0.574 — More format errors with .pt files
Day 4: 0.8857 — 2x ONNX ensemble + tiling + WBF
Day 4: 0.8818 — Added classifier (made it worse)
Day 4: 0.8995 — Single model, no ensemble (best)
```

The biggest jumps came from: fixing preprocessing (ONNX via ultralytics), OIV7 pretrain, training on 100% data, and simplifying the pipeline.
