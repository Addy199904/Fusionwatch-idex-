"""
FusionWatch — Day 1 Setup Script
=================================
Run this script in your project root after unzipping your DOTA files.

Expected folder structure after unzipping:
    data/
    ├── images/
    │   └── part1/          ← from part1-001.zip
    │       ├── P0001.png
    │       ├── P0002.png
    │       └── ...
    └── labelTxt-v1.5/      ← from labelTxt-v1.5-*.zip
        ├── P0001.txt
        ├── P0002.txt
        └── ...

This script:
    1. Checks your environment and installs requirements
    2. Converts DOTA annotations → YOLO format
    3. Splits into train/val sets
    4. Generates dataset YAML for YOLOv8 training
    5. Runs a quick sanity check with one image

Usage:
    python day1_setup.py
"""

import os
import sys
import shutil
import random
import math
from pathlib import Path
from collections import defaultdict

# ── 1. Install requirements ───────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Checking / installing requirements")
print("=" * 60)

import subprocess
packages = ["ultralytics", "opencv-python", "Pillow", "tqdm", "matplotlib", "numpy"]
for pkg in packages:
    result = subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"],
                             capture_output=True)
    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"  {pkg}: {status}")

print()

# ── 2. DOTA Class Definitions ─────────────────────────────────────────────────
# Full DOTA v1.5 classes — we keep all, but flag the defence-relevant ones
DOTA_CLASSES = [
    "plane",            # 0  ← DEFENCE RELEVANT
    "ship",             # 1  ← DEFENCE RELEVANT
    "storage-tank",     # 2
    "baseball-diamond", # 3
    "tennis-court",     # 4
    "basketball-court", # 5
    "ground-track-field",# 6
    "harbor",           # 7  ← DEFENCE RELEVANT
    "bridge",           # 8
    "large-vehicle",    # 9  ← DEFENCE RELEVANT (trucks, APCs)
    "small-vehicle",    # 10 ← DEFENCE RELEVANT
    "helicopter",       # 11 ← DEFENCE RELEVANT
    "roundabout",       # 12
    "soccer-ball-field",# 13
    "swimming-pool",    # 14
    "container-crane",  # 15 (v1.5 only)
]

DEFENCE_CLASSES = [0, 1, 7, 9, 10, 11]  # indices of defence-relevant classes
CLASS2IDX = {c: i for i, c in enumerate(DOTA_CLASSES)}

print("=" * 60)
print("STEP 2: DOTA class mapping")
print("=" * 60)
print(f"  Total classes : {len(DOTA_CLASSES)}")
print(f"  Defence-relevant : {[DOTA_CLASSES[i] for i in DEFENCE_CLASSES]}")
print()

# ── 3. Paths — EDIT THESE if your folder structure differs ───────────────────
BASE_DIR    = Path("data")
IMAGES_DIR  = BASE_DIR / "images" / "part1"      # unzipped part1-001.zip
LABELS_DIR  = BASE_DIR / "labelTxt-v1.5"         # unzipped labelTxt-v1.5-*.zip
OUTPUT_DIR  = Path("fusionwatch_dataset")

print("=" * 60)
print("STEP 3: Checking data paths")
print("=" * 60)

if not IMAGES_DIR.exists():
    print(f"  ERROR: Images folder not found: {IMAGES_DIR}")
    print(f"  Please unzip part1-001.zip into data/images/part1/")
    print(f"  Then re-run this script.")
    sys.exit(1)

if not LABELS_DIR.exists():
    print(f"  ERROR: Labels folder not found: {LABELS_DIR}")
    print(f"  Please unzip all labelTxt-v1.5-*.zip into data/labelTxt-v1.5/")
    print(f"  Then re-run this script.")
    sys.exit(1)

images = list(IMAGES_DIR.glob("*.png")) + list(IMAGES_DIR.glob("*.jpg"))
labels = list(LABELS_DIR.glob("*.txt"))
print(f"  Images found : {len(images)}")
print(f"  Labels found : {len(labels)}")
print()

# ── 4. Convert DOTA OBB annotations → YOLO HBB format ───────────────────────
print("=" * 60)
print("STEP 4: Converting DOTA → YOLO format")
print("=" * 60)
print("  (DOTA uses oriented bounding boxes — converting to axis-aligned for YOLOv8)")
print()

def dota_to_yolo(label_path, img_w, img_h):
    """
    DOTA format: x1 y1 x2 y2 class difficulty
    YOLO format: class_id cx cy w h  (all normalised 0-1)
    """
    yolo_lines = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("imagesource") or line.startswith("gsd"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                coords = list(map(float, parts[:8]))
                class_name = parts[8].lower()
                # difficulty = int(parts[9]) if len(parts) > 9 else 0
            except ValueError:
                continue

            if class_name not in CLASS2IDX:
                continue

            class_id = CLASS2IDX[class_name]

            # Convert oriented bbox to axis-aligned
            xs = coords[0::2]
            ys = coords[1::2]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            # Normalise
            cx = ((x_min + x_max) / 2) / img_w
            cy = ((y_min + y_max) / 2) / img_h
            w  = (x_max - x_min) / img_w
            h  = (y_max - y_min) / img_h

            # Clamp to [0, 1]
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            w  = max(0, min(1, w))
            h  = max(0, min(1, h))

            if w > 0.001 and h > 0.001:  # skip degenerate boxes
                yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return yolo_lines


# Set up output dirs
for split in ["train", "val"]:
    (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

# Match images to labels
label_map = {l.stem: l for l in labels}
matched   = [(img, label_map[img.stem]) for img in images if img.stem in label_map]
print(f"  Matched image+label pairs: {len(matched)}")

# Random train/val split (85/15)
random.seed(42)
random.shuffle(matched)
split_idx  = int(len(matched) * 0.85)
train_pairs = matched[:split_idx]
val_pairs   = matched[split_idx:]
print(f"  Train: {len(train_pairs)}   Val: {len(val_pairs)}")
print()

from PIL import Image
from tqdm import tqdm

stats = defaultdict(int)
skipped = 0

def process_pairs(pairs, split):
    global skipped
    for img_path, lbl_path in tqdm(pairs, desc=f"  Processing {split}"):
        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception:
            skipped += 1
            continue

        yolo_lines = dota_to_yolo(lbl_path, img_w, img_h)
        if not yolo_lines:
            skipped += 1
            continue

        # Count classes
        for line in yolo_lines:
            cid = int(line.split()[0])
            stats[DOTA_CLASSES[cid]] += 1

        # Copy image
        dst_img = OUTPUT_DIR / split / "images" / img_path.name
        shutil.copy2(img_path, dst_img)

        # Write label
        dst_lbl = OUTPUT_DIR / split / "labels" / (img_path.stem + ".txt")
        with open(dst_lbl, "w") as f:
            f.write("\n".join(yolo_lines))

process_pairs(train_pairs, "train")
process_pairs(val_pairs,   "val")

print(f"\n  Skipped (no annotations / corrupt): {skipped}")
print(f"\n  Object counts across dataset:")
for cls, cnt in sorted(stats.items(), key=lambda x: -x[1]):
    flag = " ← DEFENCE" if cls in [DOTA_CLASSES[i] for i in DEFENCE_CLASSES] else ""
    print(f"    {cls:<25} {cnt:>6}{flag}")
print()

# ── 5. Generate dataset YAML ──────────────────────────────────────────────────
print("=" * 60)
print("STEP 5: Writing dataset YAML")
print("=" * 60)

yaml_content = f"""# FusionWatch — DOTA v1.5 Dataset Config
# Generated by day1_setup.py

path: {OUTPUT_DIR.resolve()}
train: train/images
val:   val/images

nc: {len(DOTA_CLASSES)}
names:
{chr(10).join(f"  {i}: {c}" for i, c in enumerate(DOTA_CLASSES))}

# Defence-relevant class indices: {DEFENCE_CLASSES}
# plane=0, ship=1, harbor=7, large-vehicle=9, small-vehicle=10, helicopter=11
"""

yaml_path = Path("fusionwatch_dota.yaml")
with open(yaml_path, "w") as f:
    f.write(yaml_content)
print(f"  Saved: {yaml_path}")
print()

# ── 6. Sanity check — visualise one sample ────────────────────────────────────
print("=" * 60)
print("STEP 6: Sanity check — visualising one sample")
print("=" * 60)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sample_imgs = list((OUTPUT_DIR / "train" / "images").glob("*.png"))
if sample_imgs:
    sample_img  = random.choice(sample_imgs)
    sample_lbl  = OUTPUT_DIR / "train" / "labels" / (sample_img.stem + ".txt")

    img = cv2.imread(str(sample_img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(img)

    COLOURS = {
        "plane": "red", "ship": "orange", "large-vehicle": "cyan",
        "small-vehicle": "yellow", "helicopter": "magenta", "harbor": "lime"
    }

    with open(sample_lbl) as f:
        for line in f:
            cid, cx, cy, bw, bh = map(float, line.strip().split())
            cid = int(cid)
            cls = DOTA_CLASSES[cid]
            colour = COLOURS.get(cls, "white")
            x1 = (cx - bw/2) * w
            y1 = (cy - bh/2) * h
            rect = patches.Rectangle((x1, y1), bw*w, bh*h,
                                      linewidth=1.5, edgecolor=colour, facecolor="none")
            ax.add_patch(rect)
            ax.text(x1, y1-2, cls, color=colour, fontsize=6, fontweight="bold")

    ax.set_title(f"Sanity Check: {sample_img.name}\n(Coloured = defence-relevant)", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("sanity_check.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved sanity_check.png — open this to verify annotations look correct")
else:
    print("  No images found in output — check your data paths above")

print()
print("=" * 60)
print("DAY 1 COMPLETE")
print("=" * 60)
print()
print("Next step — run training (Day 2):")
print()
print("  python day2_train.py")
print()
print("Or manually:")
print()
print("  from ultralytics import YOLO")
print("  model = YOLO('yolov8m.pt')")
print("  model.train(data='fusionwatch_dota.yaml', epochs=50, imgsz=1024, batch=4)")
print()
