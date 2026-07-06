"""
Data-quality analysis for the steel defect dataset.

Generates:
  - class_distribution.png (bar chart of class counts per split)
  - ambiguous_classes_grid.png (visual comparison of confusable classes)
  - near-duplicate check via perceptual hashing (printed to console)

Usage:
    python code/data_quality.py \
        --train_ann_dir train/annotations --val_ann_dir val/annotations \
        --train_img_dir train/images --val_img_dir val/images
"""

import argparse
import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import imagehash
import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

random.seed(42)
np.random.seed(42)

CLASS_NAMES = [
    "punching_hole", "welding_line", "crescent_gap", "water_spot", "oil_spot",
    "silk_spot", "inclusion", "rolled_pit", "crease", "waist_folding",
]


def normalize_class_name(raw: str) -> str:
    """Fixes 'waist folding' → 'waist_folding' and any similar spacing issues."""
    return raw.strip().replace(" ", "_")


def count_classes(ann_dir: Path) -> Counter:
    """Count bounding-box instances per class across all XML annotations."""
    counter: Counter = Counter()
    for xml_path in sorted(ann_dir.glob("*.xml")):
        tree = ET.parse(xml_path)
        for obj in tree.getroot().iter("object"):
            name = normalize_class_name(obj.findtext("name"))
            counter[name] += 1
    return counter


def plot_class_distribution(train_counts: Counter, val_counts: Counter, save_path: Path):
    """Grouped bar chart showing train vs val instance counts per class."""
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(CLASS_NAMES))
    width = 0.35

    train_vals = [train_counts.get(c, 0) for c in CLASS_NAMES]
    val_vals = [val_counts.get(c, 0) for c in CLASS_NAMES]

    bars_train = ax.bar(x - width / 2, train_vals, width, label="Train", color="#4C72B0")
    bars_val = ax.bar(x + width / 2, val_vals, width, label="Val", color="#DD8452")

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Instance Count", fontsize=12)
    ax.set_title("Class Distribution — Train vs Val", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=11)

    # add count labels on top of each bar
    for bars in [bars_train, bars_val]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {save_path}")


def check_near_duplicates(img_dirs: list[Path], hash_size: int = 16, threshold: int = 5):
    """Hash every image with pHash, flag pairs with hamming distance <= threshold.
    This catches potential train/val leakage from near-duplicate images."""
    hashes: list[tuple[str, imagehash.ImageHash]] = []

    for img_dir in img_dirs:
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            try:
                img = Image.open(img_path).convert("L")
                h = imagehash.phash(img, hash_size=hash_size)
                hashes.append((str(img_path), h))
            except Exception as e:
                print(f"[WARNING] Could not hash {img_path}: {e}")

    # brute-force pairwise — fine for ~2K images
    duplicates = []
    n = len(hashes)
    for i in range(n):
        for j in range(i + 1, n):
            dist = hashes[i][1] - hashes[j][1]
            if dist <= threshold:
                duplicates.append((hashes[i][0], hashes[j][0], dist))

    print("=" * 50)
    print("Near-Duplicate / Leakage Check")
    print(f"  Total images hashed: {n}")
    if duplicates:
        print(f"  Pairs found: {len(duplicates)}")
        for a, b, d in duplicates[:20]:
            print(f"    dist={d}  {a}  ↔  {b}")
        if len(duplicates) > 20:
            print(f"    ... and {len(duplicates) - 20} more.")
    else:
        print("  No near-duplicates found.")
    print("=" * 50)
    return duplicates


def _get_crops_for_class(ann_dir: Path, img_dir: Path, target_class: str, max_crops: int = 4):
    """Grab a few random bounding-box crops for a given class (for the visual grid)."""
    crops = []
    xml_files = sorted(ann_dir.glob("*.xml"))
    random.shuffle(xml_files)

    for xml_path in xml_files:
        if len(crops) >= max_crops:
            break
        tree = ET.parse(xml_path)
        root = tree.getroot()
        filename = root.findtext("filename")
        img_path = img_dir / filename
        if not img_path.exists():
            continue

        img = None
        for obj in root.iter("object"):
            name = normalize_class_name(obj.findtext("name"))
            if name != target_class:
                continue
            if img is None:
                img = np.array(Image.open(img_path).convert("RGB"))
            bbox = obj.find("bndbox")
            xmin = max(0, int(float(bbox.findtext("xmin"))))
            ymin = max(0, int(float(bbox.findtext("ymin"))))
            xmax = min(img.shape[1], int(float(bbox.findtext("xmax"))))
            ymax = min(img.shape[0], int(float(bbox.findtext("ymax"))))
            crop = img[ymin:ymax, xmin:xmax]
            if crop.size > 0:
                crops.append(crop)
            if len(crops) >= max_crops:
                break

    return crops


def plot_ambiguous_grid(ann_dir: Path, img_dir: Path, save_path: Path):
    """Side-by-side crops of commonly confused class pairs:
    water_spot vs oil_spot, inclusion vs rolled_pit."""
    pairs = [("water_spot", "oil_spot"), ("inclusion", "rolled_pit")]
    cols = 4

    fig, axes = plt.subplots(nrows=len(pairs) * 2, ncols=cols, figsize=(cols * 3, len(pairs) * 2 * 3))

    row = 0
    for cls_a, cls_b in pairs:
        for cls_name in (cls_a, cls_b):
            crops = _get_crops_for_class(ann_dir, img_dir, cls_name, max_crops=cols)
            for c in range(cols):
                ax = axes[row, c]
                if c < len(crops):
                    ax.imshow(crops[c], cmap="gray")
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(cls_name.replace("_", " ").title(), fontsize=11, fontweight="bold")
            row += 1

    fig.suptitle("Ambiguous Class Pairs — Cropped Examples", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Data-quality analysis for the steel defect dataset.")
    parser.add_argument("--train_ann_dir", type=str, required=True)
    parser.add_argument("--val_ann_dir", type=str, required=True)
    parser.add_argument("--train_img_dir", type=str, default=None, help="Defaults to sibling images/ of ann dir")
    parser.add_argument("--val_img_dir", type=str, default=None, help="Defaults to sibling images/ of ann dir")
    parser.add_argument("--out_dir", type=str, default=".", help="Where to save output PNGs")
    args = parser.parse_args()

    train_ann = Path(args.train_ann_dir)
    val_ann = Path(args.val_ann_dir)
    train_img = Path(args.train_img_dir) if args.train_img_dir else train_ann.parent / "images"
    val_img = Path(args.val_img_dir) if args.val_img_dir else val_ann.parent / "images"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. class distribution
    print("\n[STEP 1] Class distributions")
    train_counts = count_classes(train_ann)
    val_counts = count_classes(val_ann)

    print("\nTrain:")
    for cls in CLASS_NAMES:
        print(f"  {cls:20s}: {train_counts.get(cls, 0)}")
    print(f"  {'TOTAL':20s}: {sum(train_counts.values())}")

    print("\nVal:")
    for cls in CLASS_NAMES:
        print(f"  {cls:20s}: {val_counts.get(cls, 0)}")
    print(f"  {'TOTAL':20s}: {sum(val_counts.values())}")

    plot_class_distribution(train_counts, val_counts, out_dir / "class_distribution.png")

    # 2. check for train/val leakage via perceptual hashing
    print("\n[STEP 2] Near-duplicate check")
    img_dirs = [d for d in [train_img, val_img] if d.is_dir()]
    if img_dirs:
        check_near_duplicates(img_dirs)
    else:
        print("[WARNING] Image directories not found, skipping.")

    # 3. visual comparison of confusable classes
    print("\n[STEP 3] Ambiguous-class grid")
    if train_img.is_dir():
        plot_ambiguous_grid(train_ann, train_img, out_dir / "ambiguous_classes_grid.png")
    else:
        print(f"[WARNING] {train_img} not found, skipping.")

    print("\nDone.")


if __name__ == "__main__":
    main()
