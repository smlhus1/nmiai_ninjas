"""
Convert COCO annotations to multi-class YOLO format (356 categories).
This is the "just train YOLOv8 directly" approach.
"""

import argparse
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare multi-class YOLO dataset")
    parser.add_argument("--coco-zip", type=str, default="NM_NGD_coco_dataset.zip")
    parser.add_argument("--output", type=str, default="dataset_multiclass")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def coco_bbox_to_yolo(bbox, img_width, img_height):
    """Convert COCO [x, y, w, h] (pixels) to YOLO [cx, cy, w, h] (normalized)."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_width
    cy = (y + h / 2) / img_height
    nw = w / img_width
    nh = h / img_height
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

    with zipfile.ZipFile(coco_zip, "r") as z:
        ann_data = json.loads(z.read("train/annotations.json"))

    images = {img["id"]: img for img in ann_data["images"]}
    categories = ann_data["categories"]
    annotations = ann_data["annotations"]
    nc = len(categories)

    print(f"Loaded {len(images)} images, {len(annotations)} annotations, {nc} categories")

    # Count annotations per category
    cat_counts = defaultdict(int)
    for ann in annotations:
        cat_counts[ann["category_id"]] += 1

    non_empty = sum(1 for c in cat_counts.values() if c > 0)
    print(f"Categories with annotations: {non_empty}/{nc}")
    print(f"Avg annotations per category: {len(annotations)/nc:.1f}")
    print(f"Min: {min(cat_counts.values()) if cat_counts else 0}, Max: {max(cat_counts.values()) if cat_counts else 0}")

    # Group annotations by image
    img_annotations = defaultdict(list)
    for ann in annotations:
        img_annotations[ann["image_id"]].append(ann)

    # Same stratified split as 1-class version for fair comparison
    image_ids = sorted(images.keys())
    ann_counts = {img_id: len(img_annotations.get(img_id, [])) for img_id in image_ids}

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

    # Extract images and write YOLO format with original category_ids
    with zipfile.ZipFile(coco_zip, "r") as z:
        for split_name, split_ids in [("train", train_ids), ("val", val_ids)]:
            img_dir = output_dir / split_name / "images"
            lbl_dir = output_dir / split_name / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            n_ann_total = 0

            for img_id in split_ids:
                img_info = images[img_id]
                fname = img_info["file_name"]
                w, h = img_info["width"], img_info["height"]

                src_path = f"train/images/{fname}"
                try:
                    img_data = z.read(src_path)
                except KeyError:
                    print(f"  WARNING: {src_path} not found in zip, skipping")
                    continue

                dst_img = img_dir / fname
                dst_img.write_bytes(img_data)

                anns = img_annotations.get(img_id, [])
                n_ann_total += len(anns)

                lines = []
                for ann in anns:
                    cat_id = ann["category_id"]
                    cx, cy, nw, nh = coco_bbox_to_yolo(ann["bbox"], w, h)
                    if nw > 0 and nh > 0:
                        lines.append(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

                label_path = lbl_dir / f"{Path(fname).stem}.txt"
                label_path.write_text("\n".join(lines), encoding="utf-8")

            print(f"  {split_name}: {len(split_ids)} images, {n_ann_total} annotations")

    # Write dataset.yaml with all categories
    names_block = "\n".join(f"  {cat['id']}: \"{cat['name']}\"" for cat in categories)
    yaml_content = f"""# YOLO dataset config — {nc}-class product detection
path: {output_dir.resolve().as_posix()}
train: train/images
val: val/images

nc: {nc}
names:
{names_block}
"""
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"Wrote {yaml_path}")

    print(f"\nDone! Next steps:")
    print(f"  py train_detector.py --data {yaml_path} --epochs 100")


if __name__ == "__main__":
    main()
