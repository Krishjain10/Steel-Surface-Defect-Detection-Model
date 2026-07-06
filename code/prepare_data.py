"""
Converts PASCAL VOC XML annotations to YOLO format.

Usage:
    python code/prepare_data.py --img_dir train/images --ann_dir train/annotations --out_dir yolo_train
"""

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

random.seed(42)
np.random.seed(42)

# 10 defect classes — order must match training config
CLASS_NAMES = [
    "punching_hole", "welding_line", "crescent_gap", "water_spot", "oil_spot",
    "silk_spot", "inclusion", "rolled_pit", "crease", "waist_folding",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def normalize_class_name(raw: str) -> str:
    """Fix labelling inconsistency: 'waist folding' (space) → 'waist_folding' (underscore).
    Found 101 train + 21 val entries with this issue during annotation audit."""
    return raw.strip().replace(" ", "_")


def parse_voc_xml(xml_path: Path):
    """Extract filename, image dimensions, and bounding boxes from a VOC XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")
    size = root.find("size")
    img_w = int(size.findtext("width"))
    img_h = int(size.findtext("height"))

    objects = []
    for obj in root.iter("object"):
        raw_name = obj.findtext("name")
        class_name = normalize_class_name(raw_name)
        bbox = obj.find("bndbox")
        xmin = int(float(bbox.findtext("xmin")))
        ymin = int(float(bbox.findtext("ymin")))
        xmax = int(float(bbox.findtext("xmax")))
        ymax = int(float(bbox.findtext("ymax")))
        objects.append((class_name, xmin, ymin, xmax, ymax))

    return filename, img_w, img_h, objects


def voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    """VOC absolute pixels (xmin,ymin,xmax,ymax) → YOLO normalised (xcenter,ycenter,w,h)."""
    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return x_center, y_center, w, h


def main():
    parser = argparse.ArgumentParser(description="Convert VOC XML annotations to YOLO format.")
    parser.add_argument("--img_dir", type=str, required=True, help="Source images directory")
    parser.add_argument("--ann_dir", type=str, required=True, help="VOC XML annotations directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Output dir (creates images/ and labels/ inside)")
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    ann_dir = Path(args.ann_dir)
    out_img_dir = Path(args.out_dir) / "images"
    out_lbl_dir = Path(args.out_dir) / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(ann_dir.glob("*.xml"))
    if not xml_files:
        print(f"[WARNING] No XML files found in {ann_dir}")
        return

    total_images = 0
    total_boxes = 0
    skipped_classes: dict[str, int] = {}

    for xml_path in xml_files:
        filename, img_w, img_h, objects = parse_voc_xml(xml_path)

        # find the matching image file
        src_img = img_dir / filename
        if not src_img.exists():
            found = False
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = img_dir / (xml_path.stem + ext)
                if candidate.exists():
                    src_img = candidate
                    found = True
                    break
            if not found:
                print(f"[WARNING] Image not found for {xml_path.name}, skipping.")
                continue

        shutil.copy2(src_img, out_img_dir / src_img.name)

        # convert each box to YOLO format
        lines = []
        for class_name, xmin, ymin, xmax, ymax in objects:
            if class_name not in CLASS_TO_IDX:
                skipped_classes[class_name] = skipped_classes.get(class_name, 0) + 1
                continue
            cls_idx = CLASS_TO_IDX[class_name]
            xc, yc, w, h = voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
            lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            total_boxes += 1

        # always write a label file — empty file = image has no valid boxes
        label_path = out_lbl_dir / (xml_path.stem + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

        total_images += 1

    print("=" * 50)
    print("VOC → YOLO Conversion Summary")
    print(f"  Images processed : {total_images}")
    print(f"  Boxes written    : {total_boxes}")
    if skipped_classes:
        print("  Skipped classes:")
        for name, count in sorted(skipped_classes.items()):
            print(f"    '{name}': {count}")
    else:
        print("  Skipped classes  : none")
    print("=" * 50)


if __name__ == "__main__":
    main()
