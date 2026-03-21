"""
Build DINOv2 reference embeddings for product classification.

Loads reference product images (multiple angles per product), generates 768-dim
embeddings using DINOv2-base, computes prototype vector (mean) per category,
and saves reference_embeddings.pt + metadata.json.

Requires a mapping file (category_mapping.json) that maps category_id to
the product folder name (EAN/barcode) in the reference images zip.
If no mapping exists, generates one from annotations + folder listing.
"""

import argparse
import json
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


def parse_args():
    parser = argparse.ArgumentParser(description="Build DINOv2 reference embeddings")
    parser.add_argument("--ref-zip", type=str, default="NM_NGD_product_images.zip",
                        help="Path to reference product images zip")
    parser.add_argument("--annotations", type=str, default="dataset/annotations.json",
                        help="Path to COCO annotations (for category info)")
    parser.add_argument("--mapping", type=str, default=None,
                        help="Path to category_mapping.json (category_id -> folder name)")
    parser.add_argument("--output", type=str, default="weights",
                        help="Output directory for embeddings")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for embedding computation")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detect if not specified)")
    return parser.parse_args()


def load_dinov2(device: str) -> tuple:
    """Load DINOv2-base from timm with offline-safe method."""
    model = timm.create_model("vit_base_patch14_dinov2", pretrained=True, num_classes=0)
    model = model.to(device).eval()

    data_config = resolve_data_config(model.pretrained_cfg, model=model)
    transform = create_transform(**data_config, is_training=False)

    return model, transform


def build_folder_listing(ref_zip_path: Path) -> dict[str, list[str]]:
    """List all product folders and their image files in the reference zip."""
    folders = defaultdict(list)
    with zipfile.ZipFile(ref_zip_path, "r") as z:
        for name in z.namelist():
            parts = name.split("/")
            if len(parts) >= 2 and parts[1] and not parts[1].startswith("."):
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    folders[parts[0]].append(name)
    return dict(folders)


def auto_generate_mapping(annotations_path: Path, ref_folders: dict) -> dict:
    """
    Generate a best-effort mapping from category_id to reference folder.

    Strategy: folder names are EAN codes or custom IDs. If we have 344 folders
    and 356 categories, we map by index order. This is a heuristic — a manual
    mapping file is more reliable.
    """
    with open(annotations_path, "r", encoding="utf-8") as f:
        ann_data = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in ann_data["categories"]}
    folder_names = sorted(ref_folders.keys())

    # Try to build mapping — for now, we map sequentially
    # TODO: improve with name matching or a provided mapping file
    mapping = {}
    for i, folder in enumerate(folder_names):
        if i < len(categories):
            mapping[str(i)] = folder

    return mapping


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ref_zip = Path(args.ref_zip)
    annotations_path = Path(args.annotations)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ref_zip.exists():
        raise FileNotFoundError(f"Reference images zip not found: {ref_zip}")

    # --- Load model ---
    print(f"Loading DINOv2-base on {device}...")
    model, transform = load_dinov2(device)
    print("Model loaded.")

    # --- List reference folders ---
    ref_folders = build_folder_listing(ref_zip)
    print(f"Found {len(ref_folders)} product folders in reference zip")

    # --- Load or generate category mapping ---
    if args.mapping and Path(args.mapping).exists():
        with open(args.mapping, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        print(f"Loaded mapping with {len(mapping)} entries")
    else:
        print("No mapping file provided — generating auto mapping...")
        if not annotations_path.exists():
            raise FileNotFoundError(
                f"Need annotations for auto-mapping: {annotations_path}\n"
                "Run prepare_data.py first, or provide --mapping."
            )
        mapping = auto_generate_mapping(annotations_path, ref_folders)
        # Save for inspection
        auto_map_path = output_dir / "auto_category_mapping.json"
        with open(auto_map_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)
        print(f"Auto-mapping saved to {auto_map_path} — REVIEW AND CORRECT THIS!")

    # --- Load annotations for metadata ---
    if annotations_path.exists():
        with open(annotations_path, "r", encoding="utf-8") as f:
            ann_data = json.load(f)
        cat_names = {cat["id"]: cat["name"] for cat in ann_data["categories"]}
    else:
        cat_names = {}

    # --- Compute embeddings per category ---
    embeddings = {}
    metadata = {}
    skipped = 0

    with zipfile.ZipFile(ref_zip, "r") as z:
        for cat_id_str, folder_name in sorted(mapping.items(), key=lambda x: int(x[0])):
            cat_id = int(cat_id_str)

            if folder_name not in ref_folders:
                print(f"  Category {cat_id}: folder '{folder_name}' not in zip, skipping")
                skipped += 1
                continue

            image_paths = ref_folders[folder_name]
            if not image_paths:
                skipped += 1
                continue

            # Load and transform all angle images
            batch_tensors = []
            for img_path in image_paths:
                try:
                    img_data = z.read(img_path)
                    img = Image.open(BytesIO(img_data)).convert("RGB")
                    tensor = transform(img)
                    batch_tensors.append(tensor)
                except Exception as e:
                    print(f"  WARNING: Failed to load {img_path}: {e}")

            if not batch_tensors:
                skipped += 1
                continue

            # Batch embed
            batch = torch.stack(batch_tensors).to(device)

            # Process in sub-batches if needed
            all_feats = []
            for i in range(0, len(batch), args.batch_size):
                sub = batch[i:i + args.batch_size]
                with torch.no_grad():
                    feats = model(sub)  # (N, 768)
                all_feats.append(feats.cpu())

            feats = torch.cat(all_feats, dim=0)  # (N_angles, 768)

            # Prototype = mean of all angle embeddings, L2-normalized
            prototype = feats.mean(dim=0)
            prototype = prototype / prototype.norm()

            embeddings[cat_id] = prototype
            metadata[str(cat_id)] = {
                "name": cat_names.get(cat_id, f"category_{cat_id}"),
                "folder": folder_name,
                "n_images": len(batch_tensors),
            }

            if (cat_id + 1) % 50 == 0:
                print(f"  Processed {cat_id + 1} categories...")

    print(f"\nEmbeddings: {len(embeddings)} categories, skipped {skipped}")

    # --- Save ---
    emb_path = output_dir / "reference_embeddings.pt"
    torch.save(embeddings, emb_path)
    print(f"Saved embeddings to {emb_path}")

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata to {meta_path}")

    print("\nDone! Next: py train_detector.py --data dataset/dataset.yaml")


if __name__ == "__main__":
    main()
