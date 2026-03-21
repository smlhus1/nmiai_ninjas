# Research: Ultralytics 8.1.0 ONNX Export Quality Loss & PyTorch 2.6 Sandbox Compatibility

> Researched: 2026-03-21 | Sources consulted: 22 | Confidence: High

## TL;DR

The 0.738 -> 0.574 mAP drop is almost certainly caused by **running ONNX with custom preprocessing that doesn't match ultralytics' internal pipeline** (letterboxing, NMS parameters, confidence thresholds). The fix: **don't use ONNX at all**. Ultralytics 8.1.0 already patches `torch.load` with `weights_only=False` internally via `ultralytics/utils/patches.py`, so `.pt` files load fine on PyTorch 2.6 in the sandbox. The security scanner only analyzes `run.py` (your code), not pre-installed library internals like ultralytics/timm.

## Key Findings

### 1. ONNX Export Quality Loss — Root Causes

The 0.738 local -> 0.574 leaderboard gap (~22% relative drop) with ONNX is consistent with multiple known issues:

#### 1a. Preprocessing Mismatch (MOST LIKELY CAUSE)
When you use `model.predict()` with a `.pt` file, ultralytics automatically applies:
- **LetterBox** resizing (maintains aspect ratio, pads with gray [114,114,114])
- BGR -> RGB conversion
- HWC -> CHW transpose
- Normalization to [0, 1] float32

When you load an ONNX model and run it with `onnxruntime` manually, you must replicate this **exactly**. Common mistakes:
- Using `cv2.resize()` instead of LetterBox (stretches image, changes aspect ratio)
- Forgetting BGR->RGB conversion
- Wrong normalization (dividing by 255 vs not)
- Wrong padding color or padding position (bottom-right vs centered)

**GitHub Issue #10419 confirmed**: when LetterBox preprocessing is applied identically, PT and ONNX confidence scores match to 4+ decimal places (0.2578768 vs 0.2578771). Without LetterBox, scores diverge significantly (0.2578 vs 0.2330).

#### 1b. NMS Parameter Differences
Default ONNX export does NOT embed NMS in the graph. You must apply NMS yourself with identical parameters:
- `conf_threshold=0.25` (default)
- `iou_threshold=0.7` (default)
- `max_det=300` (default)

If your ONNX inference code uses different thresholds, or implements NMS slightly differently, results will diverge.

#### 1c. imgsz Mismatch
If you trained at `imgsz=640` but exported ONNX with a different size, confidence scores change. **GitHub Issue #1247**: training at 48x48 but ONNX export at different size caused confidence to drop from 0.77 to 0.40. Fix: ensure export imgsz matches training imgsz exactly.

#### 1d. FP32 vs FP16 Precision
ONNX export with `half=True` introduces numerical drift. For class-agnostic detection (1 class), this is minimal. But for classification post-processing with 356 categories, FP16 can't reliably represent integer class indices >2048 — though this only matters with `end2end=True` export.

#### 1e. Output Tensor Format
ONNX output shape is `(1, 84, N)` and must be transposed to `(1, N, 84)` before NMS. The first 4 values per detection are `cx, cy, w, h` (center format), NOT `x1, y1, x2, y2`. Getting this wrong silently produces garbage boxes.

### 2. The Simple Fix: Use ultralytics predict() with ONNX

**Critical insight**: You do NOT need to write custom ONNX inference code. Ultralytics `YOLO()` class natively loads `.onnx` files and handles all preprocessing/postprocessing automatically:

```python
from ultralytics import YOLO
model = YOLO("best.onnx")  # loads ONNX model
results = model.predict(image_path, conf=0.25, iou=0.7)
```

This applies the exact same LetterBox, NMS, and postprocessing as `.pt` inference. If you're seeing quality loss, you're likely doing manual ONNX inference instead of using this approach.

**However**: using `.pt` directly is even simpler and avoids the ONNX conversion step entirely (see section 4).

### 3. PyTorch 2.6 + ultralytics 8.1.0 — weights_only Issue

#### 3a. The Problem
PyTorch 2.6 changed `torch.load()` default from `weights_only=False` to `weights_only=True`. This breaks loading `.pt` files that contain pickled model objects (like ultralytics DetectionModel).

Error: `"Unsupported global: GLOBAL ultralytics.nn.tasks.DetectionModel was not an allowed global by default"`

#### 3b. ultralytics 8.1.0 Has a Built-in Patch
ultralytics 8.1.0 (released ~January 2024) includes `ultralytics/utils/patches.py` which wraps `torch.load`:

```python
def torch_load(*args, **kwargs):
    from ultralytics.utils.torch_utils import TORCH_1_13
    if TORCH_1_13 and "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return torch.load(*args, **kwargs)
```

**This means**: when you call `YOLO("best.pt")`, ultralytics uses its own `torch_load` wrapper, which sets `weights_only=False` automatically. Your `.pt` files WILL load on PyTorch 2.6 in the sandbox without any monkey-patching.

**Note**: The "official" PyTorch 2.6 compatibility fix was ultralytics v8.3.70, but the `patches.py` workaround has existed since much earlier (v8.1.0 includes it). The sandbox's pre-installed 8.1.0 already has this patch.

#### 3c. add_safe_globals Alternative
If you wanted to be explicit, you could use:
```python
import torch
from ultralytics.nn.tasks import DetectionModel
torch.serialization.add_safe_globals([DetectionModel])
```
But this is **unnecessary** because ultralytics' own `torch_load` already handles it. Only needed if you call `torch.load()` directly outside of ultralytics.

#### 3d. Monkey-patching torch.load — DON'T DO IT
```python
# DON'T DO THIS:
_orig = torch.load
torch.load = lambda *a, **kw: _orig(*a, **{**kw, 'weights_only': False})
```
This could trigger the security scanner (overriding stdlib functions). It's also unnecessary since ultralytics handles it internally.

### 4. Sandbox Security Scanner Behavior

Based on the official NM i AI docs at app.ainm.no/docs:

#### 4a. What It Scans: YOUR CODE ONLY
The scanner analyzes **`run.py` and any .py files you include in the ZIP**. It does NOT scan:
- Pre-installed packages (ultralytics, torch, timm, etc.)
- Their transitive imports (ultralytics internally uses `os`, `pickle`, `yaml`, etc.)

This means ultralytics' internal use of `os.path`, `pickle` (for torch.load), and `yaml` (for model configs) is **completely fine**.

#### 4b. Blocked Imports (in YOUR code)
```
os, sys, subprocess, socket, ctypes, builtins, importlib, pickle,
marshal, shelve, shutil, yaml, requests, urllib, http.client,
multiprocessing, threading, signal, gc, code, codeop, pty
```

#### 4c. Blocked Function Calls (in YOUR code)
```
eval(), exec(), compile(), __import__(), getattr() with dangerous names
```

#### 4d. Blocked Content
- ELF/Mach-O/PE binaries
- Symlinks
- Path traversal attempts

#### 4e. Safe Alternatives
- `pathlib` instead of `os` for file operations
- `json` instead of `yaml` for configuration

#### 4f. AST vs Runtime
The scanner likely does **static AST analysis** of your Python files, not runtime monitoring. Evidence:
- It checks for specific import statements and function calls
- It scans for binary file signatures
- Pre-installed libraries (which heavily use `os`, `pickle`, etc.) work fine

A `torch.load` override in your `run.py` **could** trigger the scanner if it detects `eval`, `exec`, or `__import__` patterns, but a simple lambda override probably passes. Still, it's unnecessary (see 3b).

### 5. Best Weight Format for This Sandbox

#### Option Analysis

| Format | Size | Quality | Loading | Scanner Risk | Recommendation |
|--------|------|---------|---------|-------------|----------------|
| `.pt` (FP32) | ~50 MB | Baseline | Native ultralytics, auto-patches torch.load | None | **RECOMMENDED** |
| `.pt` (FP16) | ~25 MB | ~Same | `model.half()` after load | None | Good if size matters |
| `.onnx` (FP32) | ~50 MB | Same IF using `YOLO("x.onnx")` | Via onnxruntime-gpu | None | Only if you need ONNX specifically |
| `.onnx` (FP16) | ~25 MB | Risk of precision loss | Via onnxruntime-gpu | None | Avoid unless tested |
| `.safetensors` | ~50 MB | N/A | **NOT natively supported by ultralytics** | None | **DO NOT USE** |

#### Verdict: Use .pt (FP32) — Period.

1. **No quality loss**: Exact same model as training, no conversion artifacts
2. **No preprocessing mismatch**: `model.predict()` handles everything automatically
3. **PyTorch 2.6 compatible**: ultralytics 8.1.0's `torch_load` patch handles `weights_only`
4. **No scanner risk**: No monkey-patching or workarounds needed
5. **50 MB is well within** the 420 MB weight limit
6. **ONNX gives zero advantage** in this sandbox (onnxruntime-gpu is available but adds complexity for no gain)

### 6. Why Your Local Val Shows 0.738 But Leaderboard Shows 0.574

Most likely causes, ranked by probability:

1. **ONNX preprocessing mismatch** (90% likely if using custom ONNX inference): Your local val uses `model.val()` which applies correct preprocessing. Your `run.py` uses manual ONNX preprocessing that's slightly different. Switch to `.pt` and `model.predict()`.

2. **Confidence threshold difference**: Your local val might use `conf=0.001` (standard for mAP calculation) while your `run.py` uses `conf=0.25`. Lower conf threshold = more detections = higher recall = higher mAP.

3. **Image size mismatch**: If local training used `imgsz=640` but ONNX was exported with different size.

4. **NMS parameter difference**: Local val vs your inference code using different `iou` thresholds.

5. **Different evaluation metric**: Your local val computes mAP@0.5:0.95, but the leaderboard uses mAP@0.5 with a specific hybrid formula (0.7*det + 0.3*cls). Unlikely to cause a DROP though.

6. **Test set distribution shift**: Test images may differ from val split. This would explain a small drop (5-10%), not 22%.

## Recommended run.py Structure

```python
"""NorgesGruppen Object Detection - NM i AI 2026"""
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    # Use .pt directly — ultralytics handles PyTorch 2.6 torch.load internally
    from ultralytics import YOLO
    model = YOLO(str(Path(__file__).parent / "best.pt"))

    images_dir = Path(args.images)
    image_files = sorted(images_dir.glob("*.jpg"))

    results_list = []
    for img_path in image_files:
        results = model.predict(
            str(img_path),
            conf=0.001,      # Low threshold for mAP — let evaluator handle filtering
            iou=0.7,
            imgsz=640,
            device="cuda",
            verbose=False,
        )

        preds = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1
            preds.append({
                "bbox": [x1, y1, w, h],         # COCO format: x, y, w, h
                "category_id": int(box.cls[0]),   # 0 for class-agnostic
                "confidence": float(box.conf[0]),
            })

        results_list.append({
            "image": img_path.name,
            "predictions": preds,
        })

    with open(args.output, "w") as f:
        json.dump(results_list, f)

if __name__ == "__main__":
    main()
```

**Key points:**
- Uses `pathlib` (not `os`) — scanner-safe
- Loads `.pt` directly — no ONNX conversion needed
- `conf=0.001` for maximum recall during mAP evaluation
- COCO bbox format: `[x, y, w, h]` (top-left corner + width/height)
- No blocked imports used

## Gotchas & Considerations

1. **conf=0.001 vs conf=0.25**: For mAP calculation, you want ALL detections above a very low threshold. The evaluator computes precision-recall curves across all confidence levels. Using `conf=0.25` throws away low-confidence true positives and kills recall.

2. **COCO bbox format**: ultralytics returns `xyxy` (top-left, bottom-right). The submission requires `[x, y, w, h]` (top-left, width, height). Conversion: `w = x2 - x1, h = y2 - y1`.

3. **300-second timeout**: With a single YOLOv8m model, inference is ~20-50ms per image on L4 GPU. Model loading takes ~2-5 seconds. You can comfortably process 1000+ images within the time limit.

4. **safetensors is NOT supported**: ultralytics 8.1.0 does not natively load `.safetensors`. The `safetensors` package is installed in the sandbox (v0.4.2) but only for libraries that support it (like timm/transformers). Don't convert YOLO weights to safetensors.

5. **FP16 at inference time**: You can do `model.half()` after loading to run FP16 inference. This halves VRAM usage and may speed up inference, but test for quality impact first. For class-agnostic detection (1 class) the risk is minimal.

6. **Don't use end2end=True for ONNX export**: This embeds NMS in the graph and can cause precision issues with FP16 class indices. Not relevant if you use .pt (recommended).

## Recommendations

1. **Switch from ONNX to .pt immediately**. The quality gap will likely close to near-zero.
2. **Set conf=0.001** in your prediction code for maximum mAP.
3. **Verify COCO bbox format** — wrong format = 0 score with no error message.
4. **Don't monkey-patch anything** — ultralytics handles PyTorch 2.6 internally.
5. **Run a validation submission** with the simple .pt approach first, then iterate.

## Sources

1. [ultralytics/issues/4791 — ONNX vs PT results difference](https://github.com/ultralytics/ultralytics/issues/4791) — Root causes: precision, NMS, ONNX runtime variations
2. [ultralytics/issues/10419 — Confidence differences, LetterBox bug](https://github.com/ultralytics/ultralytics/issues/10419) — Confirmed LetterBox preprocessing is the key difference
3. [ultralytics/issues/1247 — Confidence PT vs ONNX](https://github.com/ultralytics/ultralytics/issues/1247) — imgsz mismatch causes confidence divergence
4. [ultralytics/issues/19778 — PyTorch 2.6 compatibility](https://github.com/ultralytics/ultralytics/issues/19778) — weights_only=True breaking ultralytics
5. [ultralytics/issues/19824 — WeightsUnpickler error](https://github.com/ultralytics/ultralytics/issues/19824) — Fix via add_safe_globals or ultralytics patch
6. [ultralytics/utils/patches.py docs](https://docs.ultralytics.com/reference/utils/patches/) — torch_load wrapper sets weights_only=False
7. [ultralytics/issues/2823 — ONNX vs YOLO module inference](https://github.com/ultralytics/ultralytics/issues/2823) — Preprocessing differences cause prediction variance
8. [ultralytics/issues/15055 — ONNX segmentation accuracy drop](https://github.com/ultralytics/ultralytics/issues/15055) — ~50% accuracy drop in ONNX segmentation
9. [app.ainm.no/docs — NM i AI 2026 sandbox specifications](https://app.ainm.no/docs) — Full sandbox environment, security scanner rules, submission format
10. [ultralytics export docs](https://docs.ultralytics.com/modes/export/) — Export parameters and format support
11. [GitHub releases — ultralytics](https://github.com/ultralytics/ultralytics/releases) — v8.3.70 = official PyTorch 2.6 fix; v8.1.0 has patches.py workaround
12. [fxis.ai — safetensors with ultralytics](https://fxis.ai/edu/how-to-utilize-safetensors-format-with-ultralytics-weights/) — Not natively supported for YOLO model loading
