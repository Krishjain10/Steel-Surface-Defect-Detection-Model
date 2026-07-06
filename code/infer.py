"""
Runs YOLOv8 inference on steel-surface images and outputs predictions.csv.

Usage:
    python code/infer.py test/images predictions.csv
    python code/infer.py test/images predictions.csv --conf 0.30
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

random.seed(42)
np.random.seed(42)

# must match the order used during training
CLASS_NAMES = [
    "punching_hole", "welding_line", "crescent_gap", "water_spot", "oil_spot",
    "silk_spot", "inclusion", "rolled_pit", "crease", "waist_folding",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 inference on steel-surface images.")
    parser.add_argument("input_dir", type=str, help="Folder with test images")
    parser.add_argument("output_csv", type=str, help="Output CSV path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)

    if not input_dir.is_dir():
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        sys.exit(1)

    # resolve weights relative to this script so it works from any working directory
    weights_path = (Path(__file__).parent / ".." / "weights" / "best.pt").resolve()
    if not weights_path.exists():
        print(f"[ERROR] Model weights not found at: {weights_path}")
        sys.exit(1)

    print(f"Loading model from {weights_path}")
    model = YOLO(str(weights_path))

    image_files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_files:
        print(f"[WARNING] No image files found in {input_dir}")

    rows: list[dict] = []
    total_processed = 0
    total_detections = 0
    zero_detection_count = 0

    for img_path in image_files:
        # wrap in try/except so one corrupt image doesn't crash the whole run
        try:
            results = model.predict(source=str(img_path), conf=args.conf, verbose=False)
        except Exception as e:
            print(f"[WARNING] Skipping {img_path.name}: {e}")
            continue

        total_processed += 1
        result = results[0]
        boxes = result.boxes

        # images with no detections just produce zero rows — no crash, no placeholder
        if boxes is None or len(boxes) == 0:
            zero_detection_count += 1
            continue

        for box in boxes:
            cls_idx = int(box.cls.item())
            conf = round(float(box.conf.item()), 2)
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # absolute pixel coords

            rows.append({
                "filename": img_path.name,
                "class": CLASS_NAMES[cls_idx],
                "confidence": conf,
                "xmin": x1, "ymin": y1,
                "xmax": x2, "ymax": y2,
            })
            total_detections += 1

    # write CSV with exact column order required by the assignment
    df = pd.DataFrame(rows, columns=["filename", "class", "confidence", "xmin", "ymin", "xmax", "ymax"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("=" * 50)
    print("Inference Summary")
    print(f"  Images processed       : {total_processed}")
    print(f"  Total detections       : {total_detections}")
    print(f"  Images w/ 0 detections : {zero_detection_count}")
    print(f"  Output saved to        : {output_csv}")
    print("=" * 50)


if __name__ == "__main__":
    main()
