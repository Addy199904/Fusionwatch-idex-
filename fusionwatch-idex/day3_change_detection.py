"""
FusionWatch — Day 3: Change Detection Demo
==========================================
Demonstrates temporal change detection between two satellite images.

For the PoC, we simulate temporal change by:
    - Taking a real DOTA image (Day 1 baseline)
    - Adding synthetic "new objects" to simulate Day 7
    - Running detection on both
    - Comparing detections to flag changes

In production: replace with real multi-temporal imagery (Sentinel-1/2 pairs).

Usage:
    python day3_change_detection.py
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import json
import random
import math
from datetime import datetime, timedelta

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH  = "fusionwatch_runs/dota_v15_yolov8m/weights/best.pt"
DATASET_DIR = Path("fusionwatch_dataset/val/images")
OUTPUT_DIR  = Path("demo/change_detection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOTA_CLASSES = [
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge",
    "large-vehicle", "small-vehicle", "helicopter", "roundabout",
    "soccer-ball-field", "swimming-pool", "container-crane"
]
DEFENCE_IDX = [0, 1, 7, 9, 10, 11]

print("=" * 60)
print("FusionWatch — Change Detection Demo")
print("=" * 60)

# ── Load model ────────────────────────────────────────────────────────────────
if not Path(MODEL_PATH).exists():
    print(f"ERROR: Model not found at {MODEL_PATH}")
    print("Run day2_train.py first, or update MODEL_PATH to your weights.")
    exit(1)

model = YOLO(MODEL_PATH)

# ── Pick a sample with detected objects ──────────────────────────────────────
val_images = list(DATASET_DIR.glob("*.png"))
random.shuffle(val_images)

baseline_img_path = None
baseline_results  = None

print("Finding a good sample image with detections...")
for img_path in val_images[:20]:
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    results = model.predict(img, imgsz=1024, conf=0.3, iou=0.5, verbose=False)
    detections = results[0].boxes
    if detections is not None and len(detections) >= 3:
        baseline_img_path = img_path
        baseline_results  = results[0]
        print(f"  Using: {img_path.name}  ({len(detections)} objects detected)")
        break

if baseline_img_path is None:
    print("No suitable image found. Using first available image.")
    baseline_img_path = val_images[0]
    baseline_results  = model.predict(
        cv2.imread(str(baseline_img_path)), imgsz=1024, conf=0.25, verbose=False)[0]

# ── Create simulated Day 7 image ──────────────────────────────────────────────
def simulate_new_activity(img_original, n_new_vehicles=8, n_new_structures=3):
    """
    Simulate military activity by stamping small rectangles
    representing new vehicles and temporary structures.
    Returns modified image.
    """
    img = img_original.copy()
    h, w = img.shape[:2]

    new_objects = []

    # New vehicles (small dark rectangles — top-down view of trucks)
    for _ in range(n_new_vehicles):
        x = random.randint(50, w - 80)
        y = random.randint(50, h - 50)
        vw = random.randint(12, 20)
        vh = random.randint(6, 10)
        angle = random.uniform(0, 180)
        centre = (x, y)

        # Draw rotated rectangle
        rect = cv2.boxPoints(((x, y), (vw, vh), angle))
        rect = np.int0(rect)
        colour = (random.randint(40, 80), random.randint(40, 80), random.randint(40, 80))
        cv2.fillPoly(img, [rect], colour)
        new_objects.append({"type": "large-vehicle", "cx": x/w, "cy": y/h})

    # New structures (slightly larger irregular patches)
    for _ in range(n_new_structures):
        x = random.randint(80, w - 100)
        y = random.randint(80, h - 80)
        sw = random.randint(25, 50)
        sh = random.randint(20, 35)
        colour = (random.randint(120, 180), random.randint(120, 180), random.randint(100, 140))
        cv2.rectangle(img, (x, y), (x+sw, y+sh), colour, -1)
        new_objects.append({"type": "temporary-structure", "cx": (x+sw/2)/w, "cy": (y+sh/2)/h})

    return img, new_objects

# ── Run change detection pipeline ────────────────────────────────────────────
baseline_img   = cv2.imread(str(baseline_img_path))
day7_img, sim_new = simulate_new_activity(baseline_img, n_new_vehicles=12, n_new_structures=4)
h, w = baseline_img.shape[:2]

# Detect on Day 7 image
day7_results = model.predict(day7_img, imgsz=1024, conf=0.25, iou=0.5, verbose=False)[0]

# ── Compare detections ────────────────────────────────────────────────────────
def boxes_to_list(boxes_result):
    items = []
    if boxes_result.boxes is None:
        return items
    for box in boxes_result.boxes:
        cid = int(box.cls[0])
        conf = float(box.conf[0])
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        cx = ((x1+x2)/2) / w
        cy = ((y1+y2)/2) / h
        items.append({
            "class": DOTA_CLASSES[cid],
            "class_id": cid,
            "confidence": round(conf, 3),
            "cx": round(cx, 4),
            "cy": round(cy, 4),
            "bbox": [x1, y1, x2, y2],
            "defence_relevant": cid in DEFENCE_IDX,
        })
    return items

baseline_detections = boxes_to_list(baseline_results)
day7_detections     = boxes_to_list(day7_results)

# Count by class
def count_by_class(detections):
    counts = {}
    for d in detections:
        cls = d["class"]
        counts[cls] = counts.get(cls, 0) + 1
    return counts

baseline_counts = count_by_class(baseline_detections)
day7_counts     = count_by_class(day7_detections)

all_classes = set(list(baseline_counts.keys()) + list(day7_counts.keys()))
changes = {}
for cls in all_classes:
    b = baseline_counts.get(cls, 0)
    d = day7_counts.get(cls, 0)
    delta = d - b
    if delta != 0:
        changes[cls] = {"baseline": b, "day7": d, "delta": delta}

# Risk score (simple heuristic)
risk_score = 0
for cls, ch in changes.items():
    if ch["delta"] > 0:
        cls_idx = DOTA_CLASSES.index(cls) if cls in DOTA_CLASSES else -1
        weight = 15 if cls_idx in DEFENCE_IDX else 3
        risk_score += ch["delta"] * weight
risk_score = min(100, risk_score)

# ── Build alert JSON ──────────────────────────────────────────────────────────
t0 = datetime(2026, 3, 13, 6, 0, 0)
t7 = t0 + timedelta(days=7)

alert = {
    "alert_id": f"FW-{random.randint(10000,99999)}",
    "system": "FusionWatch v0.1 PoC",
    "baseline_timestamp": t0.isoformat() + "Z",
    "current_timestamp":  t7.isoformat() + "Z",
    "image_id": baseline_img_path.stem,
    "risk_score": risk_score,
    "priority": "HIGH" if risk_score >= 70 else "MEDIUM" if risk_score >= 40 else "LOW",
    "baseline_detections": len(baseline_detections),
    "current_detections":  len(day7_detections),
    "changes": changes,
    "defence_relevant_changes": {k: v for k, v in changes.items()
                                  if DOTA_CLASSES.index(k) in DEFENCE_IDX
                                  if k in DOTA_CLASSES},
    "uav_cued": risk_score >= 60,
    "uav_coordinates": {
        "lat": round(34.1024 + random.uniform(-0.05, 0.05), 6),
        "lon": round(72.4567 + random.uniform(-0.05, 0.05), 6),
    } if risk_score >= 60 else None,
}

alert_path = OUTPUT_DIR / "fusionwatch_alert.json"
with open(alert_path, "w") as f:
    json.dump(alert, f, indent=2)

# ── Visualise ─────────────────────────────────────────────────────────────────
COLOURS_MAP = {
    "plane": (50,50,255), "ship": (0,165,255), "large-vehicle": (255,255,0),
    "small-vehicle": (0,255,255), "helicopter": (255,0,255), "harbor": (0,255,0),
}
DEFAULT_C = (200,200,200)

def draw_detections(img, detections, title):
    out = img.copy()
    for d in detections:
        x1,y1,x2,y2 = d["bbox"]
        col = COLOURS_MAP.get(d["class"], DEFAULT_C)
        thick = 3 if d["defence_relevant"] else 1
        cv2.rectangle(out, (x1,y1), (x2,y2), col, thick)
        cv2.putText(out, f"{d['class']} {d['confidence']:.2f}",
                    (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    # Title bar
    cv2.rectangle(out, (0,0), (w, 32), (30,30,30), -1)
    cv2.putText(out, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1)
    return out

vis_baseline = draw_detections(baseline_img, baseline_detections,
    f"Day 0 Baseline — {len(baseline_detections)} objects")
vis_day7     = draw_detections(day7_img, day7_detections,
    f"Day 7 Current — {len(day7_detections)} objects  |  RISK: {risk_score}/100 {alert['priority']}")

# Resize to same height and concatenate side by side
target_h = 600
def resize_h(img, h):
    ratio = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1]*ratio), h))

vis_b = resize_h(vis_baseline, target_h)
vis_d = resize_h(vis_day7,     target_h)
combined = np.hstack([vis_b, vis_d])

# Alert overlay at bottom
alert_banner_h = 80
banner = np.zeros((alert_banner_h, combined.shape[1], 3), dtype=np.uint8)
banner[:] = (20, 20, 40)

col = (0,80,255) if alert["priority"]=="HIGH" else (0,165,255)
cv2.putText(banner, f"FUSIONWATCH ALERT — PRIORITY: {alert['priority']}   RISK: {risk_score}/100",
            (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

change_str = "  |  ".join([f"{cls} {'+' if v['delta']>0 else ''}{v['delta']}"
                             for cls, v in list(changes.items())[:5]])
cv2.putText(banner, f"Changes: {change_str}",
            (20, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

uav_str = f"UAV CUED: YES — Coords {alert['uav_coordinates']}" if alert["uav_cued"] else "UAV: NOT TRIGGERED"
cv2.putText(banner, uav_str, (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,120) if alert["uav_cued"] else (150,150,150), 1)

final = np.vstack([combined, banner])
out_img = OUTPUT_DIR / "change_detection_demo.jpg"
cv2.imwrite(str(out_img), final, [cv2.IMWRITE_JPEG_QUALITY, 92])

print()
print("=" * 60)
print("CHANGE DETECTION RESULTS")
print("=" * 60)
print(f"  Baseline detections  : {len(baseline_detections)}")
print(f"  Day 7 detections     : {len(day7_detections)}")
print(f"  Changes detected     : {len(changes)} classes")
print(f"  Risk Score           : {risk_score}/100 — {alert['priority']}")
print(f"  UAV Cued             : {alert['uav_cued']}")
print()
print("  Class-level changes:")
for cls, ch in changes.items():
    arrow = "↑" if ch["delta"] > 0 else "↓"
    print(f"    {cls:<20} {ch['baseline']} → {ch['day7']}  ({arrow}{abs(ch['delta'])})")
print()
print(f"  Alert JSON : {alert_path}")
print(f"  Demo image : {out_img}")
print()
print("=" * 60)
print("DAY 3 COMPLETE — PoC Core Built")
print("=" * 60)
print()
print("You now have:")
print("  ✓ Object detection on satellite imagery")
print("  ✓ Temporal change detection")
print("  ✓ Risk scoring")
print("  ✓ UAV cueing trigger")
print("  ✓ Structured JSON alert output")
print("  ✓ Visual demo output")
print()
print("Push to GitHub, record your screen, update the annexures.")
