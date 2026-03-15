"""
FusionWatch — Day 2 Training Script
=====================================
Run AFTER day1_setup.py has completed successfully.

What this does:
    - Loads YOLOv8m pretrained on COCO
    - Fine-tunes on your DOTA v1.5 satellite imagery
    - Saves best model weights
    - Runs inference on 5 validation images and saves visualisations

Requirements:
    - fusionwatch_dota.yaml must exist (created by day1_setup.py)
    - fusionwatch_dataset/ must exist with train/val splits
    - GPU strongly recommended (runs on CPU but slow)

Usage:
    python day2_train.py
"""

from ultralytics import YOLO
from pathlib import Path
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import yaml

# ── Config — adjust based on your hardware ───────────────────────────────────
YAML_PATH  = "fusionwatch_dota.yaml"
MODEL_SIZE = "yolov8m.pt"   # n=nano(fast/weak), s=small, m=medium(recommended), l=large
EPOCHS     = 50             # 50 is good for PoC; increase to 100 for better results
IMG_SIZE   = 1024           # DOTA images are large — 1024 works well
BATCH_SIZE = 4              # reduce to 2 if you get OOM errors; increase to 8 with good GPU
WORKERS    = 4              # number of dataloader workers
DEVICE     = 0 if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("FusionWatch — YOLOv8 Training on DOTA v1.5")
print("=" * 60)
print(f"  Model     : {MODEL_SIZE}")
print(f"  Epochs    : {EPOCHS}")
print(f"  Image size: {IMG_SIZE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Device    : {'GPU (CUDA)' if DEVICE == 0 else 'CPU (slow — consider Colab)'}")
print()

if DEVICE == "cpu":
    print("  WARNING: No GPU detected. Training will be very slow.")
    print("  Consider using Google Colab (free GPU) — copy this script there.")
    print()

# ── Check YAML exists ─────────────────────────────────────────────────────────
if not Path(YAML_PATH).exists():
    print(f"ERROR: {YAML_PATH} not found. Run day1_setup.py first.")
    exit(1)

# ── Load model and train ──────────────────────────────────────────────────────
print("Starting training...")
print("(First run downloads yolov8m.pt weights ~52MB — needs internet once)")
print()

model = YOLO(MODEL_SIZE)

results = model.train(
    data       = YAML_PATH,
    epochs     = EPOCHS,
    imgsz      = IMG_SIZE,
    batch      = BATCH_SIZE,
    workers    = WORKERS,
    device     = DEVICE,
    project    = "fusionwatch_runs",
    name       = "dota_v15_yolov8m",
    exist_ok   = True,

    # Augmentation — important for satellite imagery
    hsv_h      = 0.015,     # hue shift (small — satellite colour calibrated)
    hsv_s      = 0.5,       # saturation
    hsv_v      = 0.3,       # brightness
    degrees    = 45,        # rotation — military objects appear at any angle
    translate  = 0.1,
    scale      = 0.5,
    flipud     = 0.5,       # vertical flip (satellite — no up/down concept)
    fliplr     = 0.5,

    # Training params
    lr0        = 0.01,
    lrf        = 0.01,
    momentum   = 0.937,
    weight_decay = 0.0005,
    warmup_epochs = 3,
    patience   = 20,        # early stopping

    # Logging
    plots      = True,
    save       = True,
    val        = True,
    verbose    = True,
)

print()
print("=" * 60)
print("Training complete!")
print("=" * 60)

best_model_path = Path("fusionwatch_runs/dota_v15_yolov8m/weights/best.pt")
print(f"  Best model saved: {best_model_path}")
print()

# ── Run inference on validation samples ──────────────────────────────────────
print("Running inference on 5 validation samples...")

with open(YAML_PATH) as f:
    cfg = yaml.safe_load(f)

dataset_path = Path(cfg["path"])
val_images   = list((dataset_path / "val" / "images").glob("*.png"))
samples      = random.sample(val_images, min(5, len(val_images)))

model_best = YOLO(str(best_model_path))

DOTA_CLASSES = list(cfg["names"].values()) if isinstance(cfg["names"], dict) else cfg["names"]
DEFENCE_IDX  = [0, 1, 7, 9, 10, 11]  # plane, ship, harbor, large-vehicle, small-vehicle, helicopter

COLOURS = {
    0:  (255, 50,  50),   # plane — red
    1:  (255, 165, 0),    # ship — orange
    7:  (50,  255, 50),   # harbor — green
    9:  (0,   200, 255),  # large-vehicle — cyan
    10: (255, 255, 0),    # small-vehicle — yellow
    11: (255, 50,  255),  # helicopter — magenta
}
DEFAULT_COLOUR = (200, 200, 200)

Path("demo").mkdir(exist_ok=True)

for i, img_path in enumerate(samples):
    img = cv2.imread(str(img_path))
    results_inf = model_best.predict(img, imgsz=IMG_SIZE, conf=0.25, iou=0.5, verbose=False)

    for r in results_inf:
        boxes  = r.boxes
        for box in boxes:
            cid   = int(box.cls[0])
            conf  = float(box.conf[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            colour = COLOURS.get(cid, DEFAULT_COLOUR)
            label  = f"{DOTA_CLASSES[cid]} {conf:.2f}"
            is_def = cid in DEFENCE_IDX

            thickness = 3 if is_def else 1
            cv2.rectangle(img, (x1,y1), (x2,y2), colour, thickness)
            cv2.putText(img, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

    out_path = f"demo/detection_sample_{i+1}_{img_path.stem}.jpg"
    cv2.imwrite(out_path, img)
    print(f"  Saved: {out_path}")

print()
print("=" * 60)
print("DAY 2 COMPLETE")
print("=" * 60)
print()
print("Your PoC detection outputs are in the demo/ folder.")
print()
print("Metrics from training:")
metrics = results.results_dict
for k in ["metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"]:
    if k in metrics:
        print(f"  {k}: {metrics[k]:.4f}")
print()
print("Next: run day3_change_detection.py")
