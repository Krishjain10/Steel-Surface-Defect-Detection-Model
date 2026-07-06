# Steel Surface Defect Detection — Submission

> **Fine-tuned YOLOv8n (Ultralytics) on Kaggle T4 GPU, 50 epochs, seed=42.**

---

## Table of Contents

1. [Setup & Usage](#setup--usage)
2. [Approach](#approach)
3. [Data-Quality Judgment](#data-quality-judgment)
4. [Self-Evaluation](#self-evaluation)
5. [Sample Runs](#sample-runs)
6. [Limitations & Future Improvements](#limitations--future-improvements)
7. [Tools and Resources Used](#tools-and-resources-used)

---

## Setup & Usage

### Install dependencies

```bash
pip install -r requirements.txt
```

### Trained weights

The fine-tuned YOLOv8n weights (`best.pt`, ~6.2 MB) are included in `weights/best.pt`. These are required to reproduce `predictions.csv` and must not be modified.

### Run inference on the test set

```bash
python code/infer.py test/images predictions.csv
```

This reads every `.jpg` / `.jpeg` / `.png` in `test/images/`, loads the trained YOLOv8n weights from `weights/best.pt`, and writes detection results to `predictions.csv`.

Optional: adjust the confidence threshold (default 0.25):

```bash
python code/infer.py test/images predictions.csv --conf 0.30
```

### Data preparation (for reference — not needed for inference)

To reproduce the VOC-XML → YOLO format conversion used before training:

```bash
python code/prepare_data.py --img_dir train/images --ann_dir train/annotations --out_dir yolo_train
python code/prepare_data.py --img_dir val/images   --ann_dir val/annotations   --out_dir yolo_val
```

### Data quality analysis (for reference)

```bash
python code/data_quality.py \
    --train_ann_dir train/annotations \
    --val_ann_dir   val/annotations \
    --train_img_dir train/images \
    --val_img_dir   val/images \
    --out_dir       .
```

---

## Approach

### Why YOLOv8n fine-tuning?

A **pretrained YOLOv8n** (nano variant) was chosen over training from scratch or a classical computer-vision pipeline for the following reasons:

1. **Transfer learning on a small dataset.** The training set contains ~2,479 bounding-box annotations across 10 classes. Training a detector from scratch on this scale would severely overfit. YOLOv8n's COCO-pretrained backbone provides strong low-level features (edges, textures, shapes) that transfer well to grayscale industrial-surface imagery.

2. **Speed of iteration.** YOLOv8n trains in ~15 minutes on a Kaggle T4 GPU for 50 epochs, enabling rapid experimentation within time constraints. Heavier architectures (YOLOv8m/l, Faster R-CNN) would have required significantly longer training cycles for marginal gains.

3. **Simplicity.** The Ultralytics API provides a batteries-included pipeline (augmentation, learning-rate scheduling, NMS, metric logging) with minimal boilerplate, reducing implementation risk.

4. **State-of-the-art not expected.** Per the assignment brief, a reasonable and well-justified approach matters more than squeezing out the last mAP point.

### Training configuration

| Parameter | Value |
|-----------|-------|
| Base model | `yolov8n.pt` (pretrained on COCO) |
| Image size | 640 px |
| Epochs | 50 |
| Random seed | 42 |
| Hardware | Kaggle T4 GPU |

---

## Data-Quality Judgment

### 1. Labelling inconsistency — `waist_folding`

A full audit of every unique raw `<name>` string in the VOC XML annotations revealed that **101 train + 21 val `<object>` entries** used `"waist folding"` (with a space) instead of the canonical `"waist_folding"` (with an underscore). No other naming inconsistencies were found across either split.

**Fix:** All class name strings are normalised with `.strip().replace(" ", "_")` before matching against the 10 official class names. This is baked into both `prepare_data.py` (for training-data conversion) and `data_quality.py` (for analysis).

### 2. Class imbalance

The training set exhibits a **~13.5× imbalance** between the most common and rarest classes:

| Class | Train Instances |
|-------|:---------:|
| silk_spot | 636 |
| oil_spot | 378 |
| welding_line | 364 |
| inclusion | 243 |
| water_spot | 241 |
| punching_hole | 231 |
| crescent_gap | 186 |
| waist_folding | 101 |
| rolled_pit | 52 |
| crease | 47 |

This directly impacts model performance on the rare classes (see Self-Evaluation below).

### 3. Visual ambiguity — label noise

Several class pairs are visually very similar on grayscale steel surfaces:
- **`water_spot` vs `oil_spot`** — both appear as diffuse, low-contrast stains. On `sample_00278.jpg`, the model produced two overlapping boxes on the same region: `oil_spot` (conf 0.49) and `water_spot` (conf 0.29), indicating genuine uncertainty.
- **`inclusion` vs `rolled_pit`** — both manifest as small dark spots. On `sample_01630.jpg`, the model output three overlapping boxes: `rolled_pit` (0.33), `oil_spot` (0.27), and `inclusion` (0.27) — a three-way confusion on one region.

These ambiguities are inherent to the labelling task, not just a model weakness.

### 4. How the validation set was used

The validation set was used **only** for metric reporting (mAP50, per-class AP50) and early-stopping selection of the best checkpoint — never for training. No hyperparameters were tuned on the validation set beyond the default YOLOv8 configuration, so the risk of overfitting to val is minimal.

### 5. Note on `punching_hole`

`punching_hole` is arguably not a true defect — it is typically an intentional manufacturing feature (a deliberately punched hole for identification or alignment). However, since the task requires detection of all 10 classes, it was kept in training. The model performs very well on this class (AP50 = 0.959), likely because punching holes have a distinctive, consistent visual appearance.

---

## Self-Evaluation

### Overall metrics (validation set)

| Metric | Value |
|--------|:-----:|
| **mAP50** | **0.676** |
| **mAP50-95** | **0.344** |

### Per-class AP50 (validation set)

| Class | AP50 | Val Instances |
|-------|:----:|:----:|
| punching_hole | 0.959 | 51 |
| crescent_gap | 0.949 | 40 |
| welding_line | 0.896 | 79 |
| water_spot | 0.831 | 55 |
| waist_folding | 0.800 | 21 |
| oil_spot | 0.709 | 100 |
| silk_spot | 0.685 | 124 |
| rolled_pit | 0.360 | 14 |
| crease | 0.351 | 11 |
| inclusion | 0.217 | 56 |

### Failure-mode analysis

Two distinct failure modes are visible among the weak classes:

**1. Data scarcity → `rolled_pit` (AP50 = 0.360) and `crease` (AP50 = 0.351)**

These are the two rarest classes in the training set (~52 and ~47 instances respectively). With so few positive examples, the model struggles to learn robust representations. This is a straightforward data-quantity problem — more labelled examples would likely improve performance significantly.

**2. Genuine visual ambiguity → `inclusion` (AP50 = 0.217)**

`inclusion` scores *worst of all 10 classes despite having more training instances than several better-performing classes* (243 train instances — more than `water_spot` at 241, which scored 4× higher on the val set). On the validation set, inclusion has 56 instances, again more than water_spot's 55.

This strongly suggests the bottleneck is **not** data quantity but **visual ambiguity**: inclusion defects likely share similar appearance with `rolled_pit` (small dark spots in the surface) or are confused with background texture noise. Resolving this would require either (a) higher-resolution input, (b) more discriminative features (a larger backbone), or (c) cleaned/refined annotations to reduce inter-class overlap.

### Guarding against self-deception

- **No val leakage:** The model was evaluated only on the provided validation split, which was never seen during training. The provided split was used as-is; no custom re-splitting was performed.
- **Near-duplicate check:** `data_quality.py` computes perceptual hashes across train and val images to detect potential data leakage from near-duplicate images appearing in both splits.
- **Per-class reporting:** Overall mAP can mask poor performance on rare/hard classes. The per-class AP50 table above exposes exactly where the model fails, rather than hiding behind an aggregate number.

---

## Sample Runs

Three representative examples from running `infer.py` on the test set (343 images, 549 total detections, 34 images with zero detections).

### (a) Clean correct detection

```
Image:        sample_00156.jpg
Model output: oil_spot, confidence=0.96, bbox=[450.1, 0.0, 868.3, 998.1]
Assessment:   High-confidence single detection. The large, clearly visible oil stain
              spanning most of the image height is an easy case for the model.
              Consistent with oil_spot's overall strong performance (AP50 = 0.709).
```

### (b) Hard / ambiguous case

```
Image:        sample_00278.jpg
Model output: oil_spot,   confidence=0.49, bbox=[632.0, 458.6, 766.4, 565.9]
              water_spot, confidence=0.29, bbox=[631.7, 452.4, 766.9, 569.2]
Assessment:   Two overlapping bounding boxes on the same region, one labelled oil_spot
              and the other water_spot. The nearly identical coordinates confirm the
              model is uncertain between these visually similar classes. This is the
              water_spot/oil_spot confusion the assignment brief explicitly flags.
```

### (c) Failure case

```
Image:        sample_01630.jpg
Model output: rolled_pit, confidence=0.33, bbox=[1315.4, 778.0, 1538.7, 921.5]
              oil_spot,   confidence=0.27, bbox=[1316.2, 775.4, 1541.2, 923.6]
              inclusion,  confidence=0.27, bbox=[1312.6, 784.4, 1542.2, 905.0]
Assessment:   Three-way confusion — the model produced three nearly identical boxes
              classified as three different classes (rolled_pit, oil_spot, inclusion),
              all at very low confidence. This highlights the visual ambiguity among
              these classes: the model cannot distinguish them and is essentially
              guessing. Ground truth is unknown (test set), but this is clearly
              unreliable output regardless of the true label.
```

---

## Limitations & Future Improvements

1. **Small dataset for rare classes.** `rolled_pit` (52 train) and `crease` (47 train) have very few training instances. Data augmentation strategies tailored to rare classes (e.g., copy-paste augmentation, mosaic with oversampling) or collecting more labelled data would likely yield the largest single improvement.

2. **Visual ambiguity among classes.** `inclusion` / `rolled_pit` and `water_spot` / `oil_spot` are visually similar, leading to inter-class confusion. Higher input resolution, attention mechanisms, or a two-stage detector might help disentangle these pairs.

3. **No test-set ground truth.** All reported metrics are on the validation set only. Without test-set annotations, we cannot confirm that val-set performance generalises. Overfitting to validation-set characteristics is possible.

4. **Single-model, no ensemble.** Only a single YOLOv8n model was trained. Ensembling multiple models (e.g., different seeds, different image sizes, or YOLOv8n + YOLOv8s) with weighted-box fusion would likely improve robustness and mAP.

5. **Fixed confidence threshold.** A single global threshold (0.25) is used for all classes. Per-class threshold tuning (e.g., lowering the threshold for rare classes with higher false-negative rates) could improve recall without significantly hurting precision.

6. **Nano-variant only.** YOLOv8n was chosen for training speed. A larger variant (YOLOv8s or YOLOv8m) with the same 50-epoch schedule would likely improve accuracy, especially on the ambiguous classes, at the cost of longer training time.

---

## Tools and Resources Used

| Tool / Resource | Purpose |
|-----------------|---------|
| **Ultralytics YOLOv8** | Object detection framework — model architecture, training loop, inference, NMS |
| **Kaggle T4 GPU** | Training environment (notebook runtime) |
| **Python 3.10+** | Language runtime |
| **pandas** | CSV manipulation for predictions output |
| **matplotlib** | Visualisation (class distribution chart, ambiguous-class grid) |
| **imagehash** | Perceptual hashing for near-duplicate / leakage detection |
| **Pillow** | Image loading and manipulation |

### Rejected alternative: Faster R-CNN (torchvision)

Faster R-CNN (with a ResNet-50-FPN backbone from `torchvision.models.detection`) was considered as an alternative. It is a two-stage detector that could potentially offer higher accuracy on small or dense objects due to its region-proposal mechanism. However, it was **rejected** for this submission because:

- **Slower training iteration:** Faster R-CNN takes significantly longer to train per epoch, limiting the number of experiments possible within the assignment timeframe.
- **More implementation complexity:** The Ultralytics YOLOv8 API handles augmentation, scheduling, and evaluation out of the box, whereas a torchvision Faster R-CNN pipeline requires more manual setup (custom dataset class, collate function, LR scheduler configuration, NMS parameter tuning).
- **Marginal expected benefit:** The assignment brief explicitly notes that state-of-the-art performance is not expected. Given the small dataset and time constraints, a simpler, faster-to-iterate approach (YOLOv8n) was the pragmatic choice.
