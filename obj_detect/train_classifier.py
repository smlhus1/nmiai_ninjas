"""
Fine-tune DINOv2 with a linear classification head for product recognition.

Optional step — use only if k-NN embedding similarity is too weak.
Freezes DINOv2 backbone, trains only the linear head on:
  - Crops from training images (using COCO annotations)
  - Reference product images (multiple angles)

Exports fine-tuned model as classify_model.pt.
"""

import argparse
import json
import random
import zipfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader, Dataset


class CropDataset(Dataset):
    """Dataset of cropped product images with category labels."""

    def __init__(self, crops: list[tuple[Image.Image, int]], transform):
        self.crops = crops  # List of (PIL.Image, category_id)
        self.transform = transform

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        img, label = self.crops[idx]
        tensor = self.transform(img.convert("RGB"))
        return tensor, label


def extract_training_crops(
    coco_zip_path: Path,
    annotations_path: Path,
    max_per_category: int = 50,
) -> list[tuple[Image.Image, int]]:
    """Extract cropped products from training images using COCO annotations."""
    with open(annotations_path, "r", encoding="utf-8") as f:
        ann_data = json.load(f)

    images = {img["id"]: img for img in ann_data["images"]}
    annotations = ann_data["annotations"]

    # Group by image
    img_anns = defaultdict(list)
    for ann in annotations:
        img_anns[ann["image_id"]].append(ann)

    # Count per category, sample balanced
    cat_counts = Counter(a["category_id"] for a in annotations)

    crops = []
    cat_collected = Counter()

    with zipfile.ZipFile(coco_zip_path, "r") as z:
        for img_id, anns in img_anns.items():
            img_info = images[img_id]
            fname = img_info["file_name"]

            try:
                img_data = z.read(f"train/images/{fname}")
                pil_img = Image.open(BytesIO(img_data)).convert("RGB")
            except (KeyError, Exception):
                continue

            for ann in anns:
                cat_id = ann["category_id"]
                if cat_collected[cat_id] >= max_per_category:
                    continue

                x, y, w, h = ann["bbox"]
                # Add 5% padding
                pad_x = w * 0.05
                pad_y = h * 0.05
                x1 = max(0, int(x - pad_x))
                y1 = max(0, int(y - pad_y))
                x2 = min(pil_img.width, int(x + w + pad_x))
                y2 = min(pil_img.height, int(y + h + pad_y))

                if x2 - x1 < 5 or y2 - y1 < 5:
                    continue

                crop = pil_img.crop((x1, y1, x2, y2))
                crops.append((crop, cat_id))
                cat_collected[cat_id] += 1

    return crops


def extract_reference_crops(
    ref_zip_path: Path,
    mapping: dict,
) -> list[tuple[Image.Image, int]]:
    """Load reference product images as training crops."""
    crops = []

    with zipfile.ZipFile(ref_zip_path, "r") as z:
        for cat_id_str, folder_name in mapping.items():
            cat_id = int(cat_id_str)
            # Find all images in this folder
            for name in z.namelist():
                if name.startswith(f"{folder_name}/") and name.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                ):
                    try:
                        img_data = z.read(name)
                        img = Image.open(BytesIO(img_data)).convert("RGB")
                        crops.append((img, cat_id))
                    except Exception:
                        continue

    return crops


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DINOv2 linear classifier")
    parser.add_argument("--coco-zip", type=str, default="NM_NGD_coco_dataset.zip",
                        help="Path to COCO dataset zip")
    parser.add_argument("--ref-zip", type=str, default="NM_NGD_product_images.zip",
                        help="Path to reference product images zip")
    parser.add_argument("--annotations", type=str, default="dataset/annotations.json",
                        help="Path to COCO annotations")
    parser.add_argument("--mapping", type=str, default="weights/auto_category_mapping.json",
                        help="Category -> folder mapping")
    parser.add_argument("--n-classes", type=int, default=356,
                        help="Number of product categories")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for linear head")
    parser.add_argument("--max-crops-per-cat", type=int, default=50,
                        help="Max training crops per category from COCO images")
    parser.add_argument("--output", type=str, default="classify_model.pt",
                        help="Output model path")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Load DINOv2 backbone (frozen) ---
    print("Loading DINOv2-base backbone (frozen)...")
    backbone = timm.create_model("vit_base_patch14_dinov2", pretrained=True, num_classes=0)
    backbone = backbone.to(device).eval()

    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False

    data_config = resolve_data_config(backbone.pretrained_cfg, model=backbone)
    transform_train = create_transform(**data_config, is_training=True)
    transform_val = create_transform(**data_config, is_training=False)

    embed_dim = 768  # DINOv2-base

    # --- Linear classification head ---
    head = nn.Linear(embed_dim, args.n_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # --- Collect training data ---
    print("Extracting training crops from COCO images...")
    coco_crops = extract_training_crops(
        Path(args.coco_zip), Path(args.annotations), args.max_crops_per_cat
    )
    print(f"  COCO crops: {len(coco_crops)}")

    ref_crops = []
    mapping_path = Path(args.mapping)
    if mapping_path.exists():
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
        print("Extracting reference product crops...")
        ref_crops = extract_reference_crops(Path(args.ref_zip), mapping)
        print(f"  Reference crops: {len(ref_crops)}")

    all_crops = coco_crops + ref_crops
    random.shuffle(all_crops)

    # --- Train/val split ---
    n_val = int(len(all_crops) * args.val_ratio)
    val_crops = all_crops[:n_val]
    train_crops = all_crops[n_val:]

    print(f"Train: {len(train_crops)}, Val: {len(val_crops)}")

    train_dataset = CropDataset(train_crops, transform_train)
    val_dataset = CropDataset(val_crops, transform_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    # --- Training loop ---
    best_val_acc = 0.0
    best_state = None

    for epoch in range(args.epochs):
        # Train
        head.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                features = backbone(images)  # (B, 768)

            logits = head(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += len(labels)

        scheduler.step()

        # Validate
        head.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                features = backbone(images)
                logits = head(features)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += len(labels)

        train_acc = train_correct / max(1, train_total)
        val_acc = val_correct / max(1, val_total)
        avg_loss = train_loss / max(1, train_total)

        print(f"  Epoch {epoch + 1}/{args.epochs}: "
              f"loss={avg_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save full model (backbone + head) for inference
            full_model = timm.create_model("vit_base_patch14_dinov2", pretrained=False,
                                           num_classes=args.n_classes)
            # Copy backbone weights
            backbone_state = backbone.state_dict()
            full_state = full_model.state_dict()
            for k, v in backbone_state.items():
                if k in full_state:
                    full_state[k] = v
            # Copy head weights
            full_state["head.weight"] = head.weight.data.cpu()
            full_state["head.bias"] = head.bias.data.cpu()
            best_state = full_state

    # --- Save ---
    if best_state is not None:
        output_path = Path(args.output)
        torch.save(best_state, output_path)
        print(f"\nSaved classifier to {output_path}")
        print(f"  Best val accuracy: {best_val_acc:.3f}")
    else:
        print("\nWARNING: No model saved (no improvement during training)")


if __name__ == "__main__":
    main()
