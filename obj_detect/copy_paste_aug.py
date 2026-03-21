"""
Custom copy-paste augmentation for product detection training.

Takes reference product images and pastes them into training shelf images,
generating new annotations. This improves detection of under-represented
products and adds more training variety.

Uses PIL and albumentations — no os module.
"""

import argparse
import json
import random
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def parse_args():
    parser = argparse.ArgumentParser(description="Copy-paste augmentation for product detection")
    parser.add_argument("--coco-zip", type=str, default="NM_NGD_coco_dataset.zip",
                        help="COCO dataset zip with training images")
    parser.add_argument("--ref-zip", type=str, default="NM_NGD_product_images.zip",
                        help="Reference product images zip")
    parser.add_argument("--annotations", type=str, default="dataset/annotations.json",
                        help="COCO annotations path")
    parser.add_argument("--output-images", type=str, default="dataset/train_aug/images",
                        help="Output directory for augmented images")
    parser.add_argument("--output-labels", type=str, default="dataset/train_aug/labels",
                        help="Output directory for YOLO labels")
    parser.add_argument("--n-augmented", type=int, default=500,
                        help="Number of augmented images to generate")
    parser.add_argument("--pastes-per-image", type=int, default=5,
                        help="Number of products to paste per image (min)")
    parser.add_argument("--max-pastes", type=int, default=15,
                        help="Max products to paste per image")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_reference_products(ref_zip_path: Path) -> dict[str, list[Image.Image]]:
    """Load reference product images grouped by folder (EAN)."""
    products = defaultdict(list)

    with zipfile.ZipFile(ref_zip_path, "r") as z:
        for name in z.namelist():
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            parts = name.split("/")
            if len(parts) >= 2:
                folder = parts[0]
                try:
                    img_data = z.read(name)
                    img = Image.open(BytesIO(img_data)).convert("RGBA")
                    products[folder].append(img)
                except Exception:
                    continue

    return dict(products)


def random_transform_product(img: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Apply random transformations to a product image before pasting."""
    # Resize to target size
    img = img.resize(target_size, Image.LANCZOS)

    # Random rotation (small angles for shelf products)
    angle = random.uniform(-8, 8)
    img = img.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))

    # Random brightness/contrast
    if random.random() < 0.5:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.7, 1.3))

    if random.random() < 0.5:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))

    # Random slight blur (simulates focus variation)
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    return img


def get_product_scale(img_width: int, img_height: int) -> tuple[int, int]:
    """Generate a realistic product size relative to the shelf image."""
    # Typical product takes 3-8% of image width, 5-15% of height
    w = int(img_width * random.uniform(0.03, 0.10))
    h = int(img_height * random.uniform(0.05, 0.18))
    # Ensure minimum size
    w = max(20, w)
    h = max(30, h)
    return w, h


def check_overlap(new_box, existing_boxes, max_iou=0.3):
    """Check if new box overlaps too much with existing boxes."""
    nx, ny, nw, nh = new_box

    for ex, ey, ew, eh in existing_boxes:
        # Compute IoU
        xa = max(nx, ex)
        ya = max(ny, ey)
        xb = min(nx + nw, ex + ew)
        yb = min(ny + nh, ey + eh)

        inter = max(0, xb - xa) * max(0, yb - ya)
        area_new = nw * nh
        area_ext = ew * eh
        union = area_new + area_ext - inter

        if union > 0 and inter / union > max_iou:
            return True

    return False


def paste_products(
    background: Image.Image,
    products: list[Image.Image],
    existing_boxes: list[tuple],
    n_paste: int,
    max_attempts: int = 50,
) -> tuple[Image.Image, list[tuple]]:
    """
    Paste product images onto background.
    Returns augmented image and list of new bboxes [x, y, w, h].
    """
    bg = background.copy().convert("RGBA")
    bg_w, bg_h = bg.size
    new_boxes = []

    for _ in range(n_paste):
        if not products:
            break

        product = random.choice(products)
        pw, ph = get_product_scale(bg_w, bg_h)

        # Transform product
        transformed = random_transform_product(product.copy(), (pw, ph))
        tw, th = transformed.size

        # Find non-overlapping position
        placed = False
        for _ in range(max_attempts):
            x = random.randint(0, max(0, bg_w - tw))
            y = random.randint(0, max(0, bg_h - th))

            new_box = (x, y, tw, th)
            if not check_overlap(new_box, existing_boxes + new_boxes):
                # Paste with alpha
                bg.paste(transformed, (x, y), transformed)
                new_boxes.append(new_box)
                placed = True
                break

    # Convert back to RGB
    result = Image.new("RGB", bg.size, (255, 255, 255))
    result.paste(bg, mask=bg.split()[3] if bg.mode == "RGBA" else None)

    return result, new_boxes


def coco_bbox_to_yolo(bbox, img_width, img_height):
    """Convert [x, y, w, h] pixels to YOLO [cx, cy, w, h] normalized."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_width
    cy = (y + h / 2) / img_height
    nw = w / img_width
    nh = h / img_height
    return (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, nw)),
        max(0.0, min(1.0, nh)),
    )


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_images = Path(args.output_images)
    out_labels = Path(args.output_labels)
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # --- Load annotations ---
    ann_path = Path(args.annotations)
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotations not found: {ann_path}. Run prepare_data.py first.")

    with open(ann_path, "r", encoding="utf-8") as f:
        ann_data = json.load(f)

    images = {img["id"]: img for img in ann_data["images"]}
    img_anns = defaultdict(list)
    for ann in ann_data["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    image_ids = list(images.keys())

    # --- Load reference products ---
    ref_zip = Path(args.ref_zip)
    print(f"Loading reference products from {ref_zip}...")
    ref_products = load_reference_products(ref_zip)
    all_product_images = []
    for folder_imgs in ref_products.values():
        all_product_images.extend(folder_imgs)
    print(f"  Loaded {len(all_product_images)} reference images from {len(ref_products)} products")

    if not all_product_images:
        print("ERROR: No reference product images loaded!")
        return

    # --- Generate augmented images ---
    coco_zip = Path(args.coco_zip)
    print(f"Generating {args.n_augmented} augmented images...")

    with zipfile.ZipFile(coco_zip, "r") as z:
        for aug_idx in range(args.n_augmented):
            # Pick random background image
            img_id = random.choice(image_ids)
            img_info = images[img_id]
            fname = img_info["file_name"]

            try:
                img_data = z.read(f"train/images/{fname}")
                bg_image = Image.open(BytesIO(img_data)).convert("RGB")
            except (KeyError, Exception):
                continue

            bg_w, bg_h = bg_image.size

            # Existing annotations as boxes
            existing_boxes = [
                (ann["bbox"][0], ann["bbox"][1], ann["bbox"][2], ann["bbox"][3])
                for ann in img_anns.get(img_id, [])
            ]

            # Number of products to paste
            n_paste = random.randint(args.pastes_per_image, args.max_pastes)

            # Paste products
            aug_image, new_boxes = paste_products(
                bg_image, all_product_images, existing_boxes, n_paste
            )

            # Combine all boxes (existing + new) — all class 0 for 1-class detection
            all_boxes = existing_boxes + new_boxes

            # Write augmented image
            aug_name = f"aug_{aug_idx:05d}.jpg"
            aug_image.save(out_images / aug_name, quality=95)

            # Write YOLO label
            label_lines = []
            for box in all_boxes:
                cx, cy, nw, nh = coco_bbox_to_yolo(box, bg_w, bg_h)
                if nw > 0 and nh > 0:
                    label_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            label_path = out_labels / f"aug_{aug_idx:05d}.txt"
            label_path.write_text("\n".join(label_lines), encoding="utf-8")

            if (aug_idx + 1) % 50 == 0:
                print(f"  Generated {aug_idx + 1}/{args.n_augmented}")

    print(f"\nDone! Augmented images in {out_images}")
    print(f"Labels in {out_labels}")
    print(f"\nTo use: add {out_images} to your dataset.yaml train paths")


if __name__ == "__main__":
    main()
