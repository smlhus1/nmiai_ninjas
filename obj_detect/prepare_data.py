"""
Convert COCO annotations to 1-class YOLO format for class-agnostic detection training.

Reads annotations.json from the COCO dataset zip, converts all 22k+ annotations
to a single "product" class (id=0), creates stratified 80/20 train/val split,
and writes YOLO-format dataset with dataset.yaml for ultralytics.
"""

import argparse
import json
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from COCO annotations")
    parser.add_argument("--coco-zip", type=str, default="NM_NGD_coco_dataset.zip",
                        help="Path to COCO dataset zip")
    parser.add_argument("--output", type=str, default="dataset",
                        help="Output directory for YOLO dataset")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Validation split ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def coco_bbox_to_yolo(bbox, img_width, img_height):
    """Convert COCO [x, y, w, h] (pixels) to YOLO [cx, cy, w, h] (normalized)."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_width
    cy = (y + h / 2) / img_height
    nw = w / img_width
    nh = h / img_height
    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))
    return cx, cy, nw, nh


def main():
    args = parse_args()
    random.seed(args.seed)

    coco_zip = Path(args.coco_zip)
    output_dir = Path(args.output)

    if not coco_zip.exists():
        raise FileNotFoundError(f"COCO zip not found: {coco_zip}")

    # --- Load COCO annotations from zip ---
    with zipfile.ZipFile(coco_zip, "r") as z:
        ann_data = json.loads(z.read("train/annotations.json"))

    images = {img["id"]: img for img in ann_data["images"]}
    categories = ann_data["categories"]
    annotations = ann_data["annotations"]

    print(f"Loaded {len(images)} images, {len(annotations)} annotations, {len(categories)} categories")

    # --- Group annotations by image ---
    img_annotations = defaultdict(list)
    for ann in annotations:
        img_annotations[ann["image_id"]].append(ann)

    # --- Stratified split by annotation density ---
    # Stratify by number of annotations per image (binned)
    image_ids = sorted(images.keys())
    ann_counts = {img_id: len(img_annotations.get(img_id, [])) for img_id in image_ids}

    # Bin images by annotation count for stratification
    bins = defaultdict(list)
    for img_id in image_ids:
        count = ann_counts[img_id]
        if count == 0:
            bin_key = 0
        elif count <= 30:
            bin_key = 1
        elif count <= 80:
            bin_key = 2
        elif count <= 150:
            bin_key = 3
        else:
            bin_key = 4
        bins[bin_key].append(img_id)

    train_ids = []
    val_ids = []

    for bin_key in sorted(bins.keys()):
        bin_images = bins[bin_key]
        random.shuffle(bin_images)
        n_val = max(1, int(len(bin_images) * args.val_ratio))
        val_ids.extend(bin_images[:n_val])
        train_ids.extend(bin_images[n_val:])

    print(f"Split: {len(train_ids)} train, {len(val_ids)} val")

    # --- Extract images from zip and write YOLO format ---
    for split_name, split_ids in [("train", train_ids), ("val", val_ids)]:
        img_dir = output_dir / split_name / "images"
        lbl_dir = output_dir / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

    # Also save full annotations for later use (embeddings, validation)
    ann_out = output_dir / "annotations.json"
    with open(ann_out, "w", encoding="utf-8") as f:
        json.dump(ann_data, f, ensure_ascii=False)
    print(f"Saved full annotations to {ann_out}")

    with zipfile.ZipFile(coco_zip, "r") as z:
        for split_name, split_ids in [("train", train_ids), ("val", val_ids)]:
            img_dir = output_dir / split_name / "images"
            lbl_dir = output_dir / split_name / "labels"
            n_ann_total = 0

            for img_id in split_ids:
                img_info = images[img_id]
                fname = img_info["file_name"]
                w, h = img_info["width"], img_info["height"]

                # Extract image
                src_path = f"train/images/{fname}"
                try:
                    img_data = z.read(src_path)
                except KeyError:
                    print(f"  WARNING: {src_path} not found in zip, skipping")
                    continue

                dst_img = img_dir / fname
                dst_img.write_bytes(img_data)

                # Write YOLO label (all class 0 = "product")
                stem = Path(fname).stem
                label_path = lbl_dir / f"{stem}.txt"
                anns = img_annotations.get(img_id, [])
                n_ann_total += len(anns)

                lines = []
                for ann in anns:
                    cx, cy, nw, nh = coco_bbox_to_yolo(ann["bbox"], w, h)
                    if nw > 0 and nh > 0:
                        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

                label_path.write_text("\n".join(lines), encoding="utf-8")

            print(f"  {split_name}: {len(split_ids)} images, {n_ann_total} annotations")

    # --- Write dataset.yaml ---
    yaml_path = output_dir / "dataset.yaml"
    yaml_content = f"""# YOLO dataset config — 1-class product detection
path: {output_dir.resolve().as_posix()}
train: train/images
val: val/images

nc: 1
names:
  0: product
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"Wrote {yaml_path}")

    # --- Write split info for reproducibility ---
    split_info = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "train_image_ids": train_ids,
        "val_image_ids": val_ids,
    }
    split_path = output_dir / "split_info.json"
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"Wrote {split_path}")

    print("\nDone! Next steps:")
    print(f"  py train_detector.py --data {yaml_path}")


if __name__ == "__main__":
    main()
