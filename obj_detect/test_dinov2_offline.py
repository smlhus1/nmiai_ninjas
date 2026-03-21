"""
Test DINOv2 offline loading with timm 0.9.12.

Validates that we can:
1. Download DINOv2-base weights (online)
2. Save state_dict to a local .pt file
3. Load the model OFFLINE using multiple methods
4. Verify embeddings match (cosine similarity)

Also tests EfficientNet-B3 as fallback.

NOTE: Uses pathlib instead of os (sandbox restriction).
"""

import sys
from pathlib import Path

import torch
import timm
import timm.models

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME = "vit_base_patch14_dinov2.lvd142m"
EFFICIENTNET_NAME = "efficientnet_b3"
WEIGHTS_DIR = Path(__file__).parent / "weights"
DINOV2_WEIGHTS = WEIGHTS_DIR / "dinov2_base.pt"
EFFNET_WEIGHTS = WEIGHTS_DIR / "efficientnet_b3.pt"

# Dummy input for embedding comparison
DUMMY_INPUT = None  # created after torch is confirmed working


def banner(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1-D tensors."""
    return torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0)
    ).item()


def get_embedding(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Get embedding from model (flatten pooled output)."""
    model.eval()
    with torch.no_grad():
        out = model(x)
    return out.flatten()


# ── Step 1: Download model online and save weights ─────────────────────────
def step1_download_and_save():
    banner("STEP 1: Download DINOv2-base online & save state_dict")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if DINOV2_WEIGHTS.exists():
        print(f"  Weights already exist at {DINOV2_WEIGHTS}")
        print(f"  Size: {DINOV2_WEIGHTS.stat().st_size / 1e6:.1f} MB")
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0)
        model.load_state_dict(torch.load(DINOV2_WEIGHTS, map_location="cpu", weights_only=True), strict=False)
        return model

    print(f"  Downloading {MODEL_NAME} with pretrained=True ...")
    try:
        model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
        print(f"  OK - model loaded")
    except Exception as e:
        print(f"  FAILED to download: {e}")
        print("  Try running this script with network access first.")
        sys.exit(1)

    # Save state_dict
    torch.save(model.state_dict(), DINOV2_WEIGHTS)
    size_mb = DINOV2_WEIGHTS.stat().st_size / 1e6
    print(f"  Saved state_dict to {DINOV2_WEIGHTS} ({size_mb:.1f} MB)")
    return model


# ── Step 2: Block network access ──────────────────────────────────────────
def step2_block_network():
    banner("STEP 2: Simulating offline environment")
    # Set env vars that huggingface_hub respects
    # We use __builtins__ approach since we can't import os
    import builtins

    # Monkey-patch socket to truly block network
    import socket
    _original_getaddrinfo = socket.getaddrinfo
    _original_connect = socket.socket.connect

    def blocked_getaddrinfo(*args, **kwargs):
        raise ConnectionError("BLOCKED: No network access in sandbox")

    def blocked_connect(self, *args, **kwargs):
        raise ConnectionError("BLOCKED: No network access in sandbox")

    socket.getaddrinfo = blocked_getaddrinfo
    socket.socket.connect = blocked_connect

    print("  Network BLOCKED (socket monkey-patched)")
    print("  Any network attempt will raise ConnectionError")

    return _original_getaddrinfo, _original_connect


def step2_restore_network(originals):
    """Restore network after tests."""
    import socket
    socket.getaddrinfo, socket.socket.connect = originals
    print("  Network restored")


# ── Step 3: Test offline loading methods ───────────────────────────────────
def step3_test_offline_methods(reference_embedding: torch.Tensor, dummy_input: torch.Tensor):
    banner("STEP 3: Testing OFFLINE loading methods for DINOv2")
    results = {}

    # Method 1: pretrained=False + load_state_dict()
    print("\n--- Method 1: pretrained=False + load_state_dict() ---")
    try:
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0)
        state_dict = torch.load(DINOV2_WEIGHTS, map_location="cpu", weights_only=True)
        load_result = model.load_state_dict(state_dict, strict=False)
        emb = get_embedding(model, dummy_input)
        sim = cosine_sim(reference_embedding, emb)
        missing = len(load_result.missing_keys)
        unexpected = len(load_result.unexpected_keys)
        print(f"  OK - cosine similarity: {sim:.6f}")
        print(f"  Missing keys: {missing}, Unexpected keys: {unexpected}")
        if missing > 0:
            print(f"  Missing: {load_result.missing_keys[:5]}...")
        results["method1_load_state_dict"] = {
            "success": True, "similarity": sim,
            "missing_keys": missing, "unexpected_keys": unexpected
        }
    except Exception as e:
        print(f"  FAILED: {e}")
        results["method1_load_state_dict"] = {"success": False, "error": str(e)}

    # Method 2: pretrained_cfg_overlay with file= and custom_load=False
    print("\n--- Method 2: pretrained_cfg_overlay=dict(file=path, custom_load=False) ---")
    try:
        model = timm.create_model(
            MODEL_NAME,
            pretrained=True,
            num_classes=0,
            pretrained_cfg_overlay=dict(
                file=str(DINOV2_WEIGHTS),
                custom_load=False,
            ),
        )
        emb = get_embedding(model, dummy_input)
        sim = cosine_sim(reference_embedding, emb)
        print(f"  OK - cosine similarity: {sim:.6f}")
        results["method2_cfg_overlay"] = {"success": True, "similarity": sim}
    except Exception as e:
        print(f"  FAILED: {e}")
        results["method2_cfg_overlay"] = {"success": False, "error": str(e)}

    # Method 3: checkpoint_path parameter
    print("\n--- Method 3: checkpoint_path parameter ---")
    try:
        model = timm.create_model(
            MODEL_NAME,
            pretrained=False,
            num_classes=0,
            checkpoint_path=str(DINOV2_WEIGHTS),
        )
        emb = get_embedding(model, dummy_input)
        sim = cosine_sim(reference_embedding, emb)
        print(f"  OK - cosine similarity: {sim:.6f}")
        results["method3_checkpoint_path"] = {"success": True, "similarity": sim}
    except Exception as e:
        print(f"  FAILED: {e}")
        results["method3_checkpoint_path"] = {"success": False, "error": str(e)}

    return results


# ── Step 4: Test EfficientNet-B3 as fallback ──────────────────────────────
def step4_test_efficientnet(dummy_input_224: torch.Tensor):
    banner("STEP 4: Testing EfficientNet-B3 (fallback, no custom_load issue)")
    results = {}

    # First, ensure we have weights
    if not EFFNET_WEIGHTS.exists():
        print("  EfficientNet weights not found, need to download first")
        print("  (Would need network — skipping if blocked)")
        # Try creating without pretrained
        try:
            model = timm.create_model(EFFICIENTNET_NAME, pretrained=False, num_classes=0)
            print(f"  Created EfficientNet-B3 (random weights) — {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
            results["efficientnet_create"] = {"success": True, "note": "random weights only"}
        except Exception as e:
            print(f"  FAILED to create: {e}")
            results["efficientnet_create"] = {"success": False, "error": str(e)}
        return results

    # Load from saved weights
    print(f"  Loading from {EFFNET_WEIGHTS}")
    try:
        model = timm.create_model(EFFICIENTNET_NAME, pretrained=False, num_classes=0)
        state_dict = torch.load(EFFNET_WEIGHTS, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        emb = get_embedding(model, dummy_input_224)
        print(f"  OK - embedding shape: {emb.shape}")
        results["efficientnet_offline"] = {"success": True}
    except Exception as e:
        print(f"  FAILED: {e}")
        results["efficientnet_offline"] = {"success": False, "error": str(e)}

    return results


# ── Step 5: Save EfficientNet weights while online ────────────────────────
def save_efficientnet_weights():
    """Download and save EfficientNet-B3 weights (call while online)."""
    if EFFNET_WEIGHTS.exists():
        print(f"  EfficientNet weights already at {EFFNET_WEIGHTS}")
        return True

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {EFFICIENTNET_NAME} with pretrained=True ...")
    try:
        model = timm.create_model(EFFICIENTNET_NAME, pretrained=True, num_classes=0)
        torch.save(model.state_dict(), EFFNET_WEIGHTS)
        size_mb = EFFNET_WEIGHTS.stat().st_size / 1e6
        print(f"  Saved to {EFFNET_WEIGHTS} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print(f"timm version: {timm.__version__}")
    print(f"torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Create dummy inputs
    dummy_518 = torch.randn(1, 3, 518, 518)  # DINOv2 native resolution
    dummy_224 = torch.randn(1, 3, 224, 224)  # EfficientNet resolution

    # Step 1: Get reference model (online)
    ref_model = step1_download_and_save()
    ref_embedding = get_embedding(ref_model, dummy_518)
    print(f"  Reference embedding shape: {ref_embedding.shape}")
    del ref_model  # free memory

    # Save EfficientNet weights while we still have network
    banner("Saving EfficientNet-B3 weights (while online)")
    save_efficientnet_weights()

    # Step 2: Block network
    originals = step2_block_network()

    try:
        # Step 3: Test DINOv2 offline loading
        dinov2_results = step3_test_offline_methods(ref_embedding, dummy_518)

        # Step 4: Test EfficientNet-B3 offline
        effnet_results = step4_test_efficientnet(dummy_224)

    finally:
        # Restore network
        step2_restore_network(originals)

    # ── Summary ────────────────────────────────────────────────────────────
    banner("SUMMARY")
    print("\nDINOv2 offline loading methods:")
    for method, result in dinov2_results.items():
        status = "PASS" if result.get("success") else "FAIL"
        detail = ""
        if result.get("similarity"):
            sim = result["similarity"]
            match = "IDENTICAL" if sim > 0.9999 else "CLOSE" if sim > 0.99 else "DIFFERENT"
            detail = f" (sim={sim:.6f}, {match})"
        elif result.get("error"):
            detail = f" ({result['error'][:80]})"
        print(f"  [{status}] {method}{detail}")

    print("\nEfficientNet-B3 offline:")
    for method, result in effnet_results.items():
        status = "PASS" if result.get("success") else "FAIL"
        detail = f" ({result.get('note', result.get('error', ''))})"
        print(f"  [{status}] {method}{detail}")

    # Recommendation
    print("\n" + "-"*60)
    working_methods = [k for k, v in dinov2_results.items()
                       if v.get("success") and v.get("similarity", 0) > 0.99]
    if working_methods:
        best = working_methods[0]
        print(f"RECOMMENDATION: Use {best} for DINOv2 offline loading")
        print("DINOv2 is GO for the sandbox.")
    else:
        print("WARNING: No DINOv2 offline method produced matching embeddings!")
        if any(v.get("success") for v in effnet_results.values()):
            print("FALLBACK: Use EfficientNet-B3 instead.")
        else:
            print("CRITICAL: No offline backbone available!")


if __name__ == "__main__":
    main()
