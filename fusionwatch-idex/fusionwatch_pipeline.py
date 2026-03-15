"""
FusionWatch v0.1 — Satellite Intelligence Pipeline
====================================================
Full pipeline: GeoTIFF satellite image → GPS-tagged detections → QR-coded alerts

Usage:
    python fusionwatch_pipeline.py --image path/to/image.tif
    python fusionwatch_pipeline.py --image path/to/image.tif --model best.pt --conf 0.275
    python fusionwatch_pipeline.py --demo   (runs on a test image if no GeoTIFF available)

Requirements:
    pip install ultralytics rasterio numpy opencv-python qrcode pillow

Author: Adarsh T Swathiraj
Submitted: iDEX Challenge #4 — AI Based Satellite Image Analysis
"""

import os
import json
import argparse
import base64
import io
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import cv2

# ── Constants ─────────────────────────────────────────────────────────────────

VERSION = "0.1.0"

DOTA_CLASSES = [
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge",
    "large-vehicle", "small-vehicle", "helicopter", "roundabout",
    "soccer-ball-field", "swimming-pool", "container-crane"
]

# Defence-relevant class indices (DOTA v1.5)
DEFENCE_CLASSES = [0, 1, 7, 9, 10, 11]
# plane=0, ship=1, harbor=7, large-vehicle=9, small-vehicle=10, helicopter=11

# Risk weights per class
CLASS_RISK_WEIGHT = {
    "plane":         20,
    "helicopter":    25,
    "ship":          15,
    "harbor":        10,
    "large-vehicle": 12,
    "small-vehicle":  5,
}

# Confidence threshold → risk override
CONF_RISK_MAP = [
    (0.85, "CRITICAL"),
    (0.70, "HIGH"),
    (0.50, "MEDIUM"),
    (0.00, "LOW"),
]

# Visualisation colours (BGR) per class
COLOURS = {
    0:  (0,   0,   255),   # plane        — red
    1:  (0,   165, 255),   # ship         — orange
    7:  (0,   255, 0),     # harbor       — green
    9:  (255, 255, 0),     # large-vehicle — yellow
    10: (0,   255, 255),   # small-vehicle — cyan
    11: (255, 0,   255),   # helicopter   — magenta
}
DEFAULT_COLOUR = (200, 200, 200)


# ── Utility Functions ─────────────────────────────────────────────────────────

def dd_to_dms(lat: float, lon: float) -> str:
    """Convert decimal degrees to degrees-minutes-seconds string."""
    def convert(deg: float) -> str:
        d = int(abs(deg))
        m = int((abs(deg) - d) * 60)
        s = ((abs(deg) - d) * 60 - m) * 60
        return f"{d}\u00b0{m}'{s:.2f}\""

    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{convert(lat)}{ns} {convert(lon)}{ew}"


def get_risk(cls_name: str, confidence: float) -> str:
    """Determine risk level from class and confidence."""
    for threshold, level in CONF_RISK_MAP:
        if confidence >= threshold:
            # Upgrade to CRITICAL for high-value targets at high confidence
            if level == "HIGH" and cls_name in ["plane", "helicopter"] and confidence > 0.80:
                return "CRITICAL"
            return level
    return "LOW"


def metres_per_degree(lat: float) -> float:
    """Approximate metres per degree of latitude at given latitude."""
    return 111132.92 - 559.82 * np.cos(2 * np.radians(lat)) + 1.175 * np.cos(4 * np.radians(lat))


# ── Stage 1: Load GeoTIFF ─────────────────────────────────────────────────────

def load_geotiff(path: str):
    """
    Load a GeoTIFF satellite image.
    Returns: (rgb_image: np.ndarray HxWx3 uint8, transform, crs, meta)
    Falls back to plain image loading if rasterio unavailable.
    """
    try:
        import rasterio
        with rasterio.open(path) as src:
            meta = {
                'width':      src.width,
                'height':     src.height,
                'crs':        str(src.crs),
                'bounds':     src.bounds,
                'pixel_size': src.res,
            }
            transform = src.transform
            count = src.count

            if count >= 3:
                r = src.read(1).astype(np.float32)
                g = src.read(2).astype(np.float32)
                b = src.read(3).astype(np.float32)
            elif count == 1:
                # SAR or panchromatic — replicate to 3 channels
                band = src.read(1).astype(np.float32)
                r = g = b = band
            else:
                raise ValueError(f"Unsupported band count: {count}")

            # Normalise to uint8
            def norm(arr):
                p2, p98 = np.percentile(arr[arr > 0], [2, 98]) if np.any(arr > 0) else (0, 1)
                arr = np.clip(arr, p2, p98)
                arr = ((arr - p2) / (p98 - p2 + 1e-8) * 255).astype(np.uint8)
                return arr

            rgb = np.stack([norm(r), norm(g), norm(b)], axis=-1)
            print(f"  Loaded GeoTIFF: {meta['width']}×{meta['height']}px")
            print(f"  CRS: {meta['crs']}")
            print(f"  Pixel size: {meta['pixel_size'][0]*111000:.1f}m")
            return rgb, transform, meta

    except ImportError:
        print("  rasterio not installed — loading as plain image (no GPS metadata)")
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot open image: {path}")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create a dummy affine transform centred on Visakhapatnam
        # (used for demo purposes when real GeoTIFF metadata unavailable)
        from collections import namedtuple
        FakeTransform = namedtuple('FakeTransform', ['a', 'e', 'c', 'f'])
        H, W = rgb.shape[:2]
        pixel_deg = 0.0001  # ~10m per pixel
        transform = FakeTransform(
            a=pixel_deg, e=-pixel_deg,
            c=83.1500,   f=17.8000
        )
        transform.__mul__ = lambda self, xy: (
            self.c + xy[0] * self.a,
            self.f + xy[1] * self.e
        )
        meta = {'width': W, 'height': H, 'crs': 'EPSG:4326 (estimated)',
                'pixel_size': (pixel_deg, pixel_deg)}
        return rgb, transform, meta


# ── Stage 2: Tiling ───────────────────────────────────────────────────────────

def tile_image(image: np.ndarray, transform, tile_size: int = 800, overlap: int = 100):
    """
    Slice large satellite image into overlapping tiles.
    Each tile carries its GPS offset for coordinate back-conversion.
    """
    H, W = image.shape[:2]
    stride = tile_size - overlap
    tiles = []

    rows = range(0, max(H - tile_size + 1, 1), stride)
    cols = range(0, max(W - tile_size + 1, 1), stride)

    for row in rows:
        row_end = min(row + tile_size, H)
        row_start = row_end - tile_size
        if row_start < 0:
            row_start = 0

        for col in cols:
            col_end = min(col + tile_size, W)
            col_start = col_end - tile_size
            if col_start < 0:
                col_start = 0

            tile = image[row_start:row_end, col_start:col_end]

            # Pad if smaller than tile_size (edge tiles)
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded

            # GPS origin of this tile's top-left corner
            try:
                origin_lon, origin_lat = transform * (col_start, row_start)
                pixel_width  = transform.a
                pixel_height = transform.e
            except (TypeError, AttributeError):
                origin_lon = transform.c + col_start * transform.a
                origin_lat = transform.f + row_start * transform.e
                pixel_width  = transform.a
                pixel_height = transform.e

            tiles.append({
                'image':        tile,
                'col_offset':   col_start,
                'row_offset':   row_start,
                'origin_lon':   origin_lon,
                'origin_lat':   origin_lat,
                'pixel_width':  pixel_width,
                'pixel_height': pixel_height,
            })

    print(f"  Tiled: {W}×{H}px → {len(tiles)} tiles ({tile_size}×{tile_size}px, overlap={overlap}px)")
    return tiles


# ── Stage 3: Inference ────────────────────────────────────────────────────────

def run_inference_on_tiles(tiles: list, model, conf: float = 0.275, iou: float = 0.45):
    """Run YOLOv8 inference on all tiles, return raw geo-tagged detections."""
    from tqdm import tqdm

    all_detections = []
    images = [t['image'] for t in tiles]

    # Batch predict for speed
    batch_size = 8
    for i in tqdm(range(0, len(images), batch_size), desc="  Detecting"):
        batch_images = images[i:i+batch_size]
        batch_tiles  = tiles[i:i+batch_size]

        results = model.predict(
            batch_images,
            imgsz=800,
            conf=conf,
            iou=iou,
            classes=DEFENCE_CLASSES,
            verbose=False,
        )

        for result, tile_data in zip(results, batch_tiles):
            for box in result.boxes:
                cid  = int(box.cls[0])
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Pixel centre in full-image coordinates
                cx_local = (x1 + x2) / 2
                cy_local = (y1 + y2) / 2
                cx_full  = cx_local + tile_data['col_offset']
                cy_full  = cy_local + tile_data['row_offset']

                # Convert to GPS
                lon = tile_data['origin_lon'] + cx_local * tile_data['pixel_width']
                lat = tile_data['origin_lat'] + cy_local * tile_data['pixel_height']

                # Real-world size in metres
                pixel_size_m = abs(tile_data['pixel_width']) * metres_per_degree(lat)
                width_m  = (x2 - x1) * pixel_size_m
                height_m = (y2 - y1) * pixel_size_m

                all_detections.append({
                    'class_id':    cid,
                    'class':       DOTA_CLASSES[cid],
                    'confidence':  round(conf_val, 4),
                    'lat':         round(lat,  7),
                    'lon':         round(lon,  7),
                    'width_m':     round(width_m,  1),
                    'height_m':    round(height_m, 1),
                    'bbox_pixel':  [int(cx_full - (x2-x1)/2), int(cy_full - (y2-y1)/2),
                                    int(cx_full + (x2-x1)/2), int(cy_full + (y2-y1)/2)],
                    'tile_bbox':   [int(x1), int(y1), int(x2), int(y2)],
                    'col_offset':  tile_data['col_offset'],
                    'row_offset':  tile_data['row_offset'],
                })

    return all_detections


# ── Stage 4: NMS Across Tile Boundaries ───────────────────────────────────────

def merge_detections(detections: list, dist_threshold_m: float = 15.0) -> list:
    """
    Remove duplicate detections from overlapping tiles using GPS-space NMS.
    Keeps highest-confidence detection when two detections are within
    dist_threshold_m metres of each other and share the same class.
    """
    if not detections:
        return []

    detections.sort(key=lambda x: x['confidence'], reverse=True)
    kept = []

    for det in detections:
        is_duplicate = False
        for kept_det in kept:
            if det['class'] != kept_det['class']:
                continue
            dlat_m = (det['lat'] - kept_det['lat']) * metres_per_degree(det['lat'])
            dlon_m = (det['lon'] - kept_det['lon']) * metres_per_degree(det['lat']) * np.cos(np.radians(det['lat']))
            dist_m = np.sqrt(dlat_m**2 + dlon_m**2)
            if dist_m < dist_threshold_m:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(det)

    print(f"  NMS: {len(detections)} raw → {len(kept)} unique detections")
    return kept


# ── Stage 5: QR Code Generation ───────────────────────────────────────────────

def generate_qr(detection: dict, alert_id: str, output_dir: str) -> str:
    """
    Generate a scannable QR code for a detection alert.
    Encodes geo: URI — opens in Google Maps / Apple Maps on any phone.
    Returns base64-encoded PNG for embedding in JSON.
    """
    try:
        import qrcode

        qr_data = "\n".join([
            f"FW-ALERT:{alert_id}",
            f"geo:{detection['lat']},{detection['lon']}",
            f"CLASS:{detection['class'].upper()}",
            f"CONF:{detection['confidence']}",
            f"SIZE:{detection['width_m']}m x {detection['height_m']}m",
            f"RISK:{detection['risk']}",
            f"DMS:{dd_to_dms(detection['lat'], detection['lon'])}",
        ])

        qr = qrcode.QRCode(
            version=2,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=6,
            border=2,
        )
        qr.add_data(qr_data)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")

        # Save PNG
        qr_path = Path(output_dir) / f"qr_{alert_id}.png"
        img.save(str(qr_path))

        # Return base64 for JSON embedding
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()

    except ImportError:
        return "qrcode_library_not_installed"


# ── Stage 6: Build Alert ──────────────────────────────────────────────────────

def build_alert(detection: dict, alert_id: str, output_dir: str) -> dict:
    """Build structured intelligence alert from a detection."""
    risk = get_risk(detection['class'], detection['confidence'])
    detection['risk'] = risk

    qr_b64 = generate_qr(detection, alert_id, output_dir)

    return {
        "alert_id":   alert_id,
        "system":     f"FusionWatch v{VERSION}",
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "detection": {
            "class":       detection['class'],
            "confidence":  detection['confidence'],
            "location": {
                "lat":         detection['lat'],
                "lon":         detection['lon'],
                "dms":         dd_to_dms(detection['lat'], detection['lon']),
                "maps_link":   f"https://maps.google.com/?q={detection['lat']},{detection['lon']}",
            },
            "size": {
                "width_m":  detection['width_m'],
                "height_m": detection['height_m'],
            },
            "risk":          risk,
            "bbox_pixel":    detection['bbox_pixel'],
        },
        "uav_cued":   risk in ["HIGH", "CRITICAL"],
        "qr_code_b64": qr_b64,
    }


# ── Stage 7: Annotated Image ──────────────────────────────────────────────────

def draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    """Draw detection boxes on full image with FusionWatch header."""
    vis = image.copy()
    if len(vis.shape) == 3 and vis.shape[2] == 3:
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    # Draw boxes
    for det in detections:
        cid = det['class_id']
        x1, y1, x2, y2 = det['bbox_pixel']
        col = COLOURS.get(cid, DEFAULT_COLOUR)
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        label = f"{det['class']} {det['confidence']:.2f} | {det['risk']}"
        cv2.putText(vis, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    # Header bar
    header = np.zeros((55, vis.shape[1], 3), dtype=np.uint8)
    header[:] = (20, 20, 40)
    defence_count = len(detections)
    high_risk = sum(1 for d in detections if d.get('risk') in ['HIGH', 'CRITICAL'])
    cv2.putText(header,
                f"FusionWatch v{VERSION}  |  {defence_count} defence objects  |  {high_risk} HIGH/CRITICAL  |  UAV cued: {high_risk}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(header,
                f"Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  |  Model: YOLOv8m DOTA-v1.5 100ep  |  mAP50: 0.546",
                (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return np.vstack([header, vis])


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_fusionwatch(
    image_path: str,
    model_path: str = 'best.pt',
    conf: float = 0.275,
    iou: float = 0.45,
    tile_size: int = 800,
    overlap: int = 100,
    output_dir: str = 'fusionwatch_output',
) -> dict:
    """
    Full FusionWatch pipeline.

    Args:
        image_path:  Path to GeoTIFF or image file
        model_path:  Path to trained YOLOv8 weights (best.pt)
        conf:        Detection confidence threshold (default: 0.275)
        iou:         NMS IoU threshold (default: 0.45)
        tile_size:   Tile size in pixels (default: 800)
        overlap:     Tile overlap in pixels (default: 100)
        output_dir:  Directory for output files

    Returns:
        dict with alerts, summary, and file paths
    """
    from ultralytics import YOLO

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"\n{'='*60}")
    print(f"FusionWatch v{VERSION} — Satellite Intelligence Pipeline")
    print(f"{'='*60}")
    print(f"Image  : {image_path}")
    print(f"Model  : {model_path}")
    print(f"Conf   : {conf}  |  IoU: {iou}  |  Tile: {tile_size}px")
    print(f"Output : {output_dir}/")
    print(f"{'='*60}\n")

    # Stage 1: Load
    print("[1/6] Loading satellite image...")
    image, transform, meta = load_geotiff(image_path)

    # Stage 2: Tile
    print("[2/6] Tiling image...")
    tiles = tile_image(image, transform, tile_size, overlap)

    # Stage 3: Detect
    print("[3/6] Running YOLOv8 inference...")
    model = YOLO(model_path)
    raw_detections = run_inference_on_tiles(tiles, model, conf, iou)
    print(f"  Raw detections: {len(raw_detections)}")

    # Stage 4: Merge
    print("[4/6] Merging tile detections (NMS)...")
    final_detections = merge_detections(raw_detections)

    # Stage 5: Build alerts + QR codes
    print("[5/6] Building alerts and QR codes...")
    alerts = []
    for i, det in enumerate(final_detections):
        alert_id = f"FW-{run_id}-{i:04d}"
        alert = build_alert(det, alert_id, output_dir)
        alerts.append(alert)

    uav_cued = sum(1 for a in alerts if a['uav_cued'])
    risk_summary = {}
    for a in alerts:
        r = a['detection']['risk']
        risk_summary[r] = risk_summary.get(r, 0) + 1

    # Stage 6: Annotated image
    print("[6/6] Generating annotated output image...")
    annotated = draw_detections(image, final_detections)
    annotated_path = f"{output_dir}/fusionwatch_{run_id}_annotated.jpg"
    cv2.imwrite(annotated_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Save full alert JSON
    summary = {
        "run_id":          run_id,
        "system":          f"FusionWatch v{VERSION}",
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "image":           str(image_path),
        "model":           str(model_path),
        "image_meta":      {k: str(v) for k, v in meta.items()},
        "total_detections": len(final_detections),
        "uav_cued_count":  uav_cued,
        "risk_summary":    risk_summary,
        "class_summary":   {},
        "alerts":          alerts,
    }
    for det in final_detections:
        cls = det['class']
        summary['class_summary'][cls] = summary['class_summary'].get(cls, 0) + 1

    json_path = f"{output_dir}/fusionwatch_{run_id}_alerts.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"FUSIONWATCH COMPLETE")
    print(f"{'='*60}")
    print(f"  Total detections : {len(final_detections)}")
    print(f"  UAV cued targets : {uav_cued}")
    print(f"  Risk breakdown   : {risk_summary}")
    print(f"  Class breakdown  : {summary['class_summary']}")
    print(f"\n  Outputs saved to : {output_dir}/")
    print(f"  Annotated image  : {annotated_path}")
    print(f"  Alert JSON       : {json_path}")
    print(f"  QR codes         : {output_dir}/qr_FW-*.png")
    print(f"{'='*60}\n")

    return summary


# ── Demo Mode ─────────────────────────────────────────────────────────────────

def run_demo(model_path: str = 'best.pt'):
    """
    Demo mode: runs pipeline on a generated synthetic test image.
    Use this to verify the pipeline works without a GeoTIFF.
    """
    print("\nRunning FusionWatch in DEMO mode...")
    print("(No GeoTIFF provided — using synthetic test image)\n")

    # Create a simple test image
    test_img = np.random.randint(50, 150, (800, 800, 3), dtype=np.uint8)

    # Add some rectangular "objects" to make it interesting
    for _ in range(5):
        x = np.random.randint(100, 700)
        y = np.random.randint(100, 700)
        cv2.rectangle(test_img, (x, y), (x+40, y+20), (200, 200, 200), -1)

    test_path = 'fusionwatch_demo_input.jpg'
    cv2.imwrite(test_path, test_img)

    result = run_fusionwatch(
        image_path=test_path,
        model_path=model_path,
        output_dir='fusionwatch_demo_output',
    )

    os.remove(test_path)
    return result


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=f'FusionWatch v{VERSION} — Satellite Intelligence Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fusionwatch_pipeline.py --image vizag.tif
  python fusionwatch_pipeline.py --image vizag.tif --model best.pt --conf 0.30
  python fusionwatch_pipeline.py --demo
  python fusionwatch_pipeline.py --image vizag.tif --output my_results/
        """
    )
    parser.add_argument('--image',   type=str, help='Path to GeoTIFF or image file')
    parser.add_argument('--model',   type=str, default='best.pt', help='Path to YOLOv8 weights (default: best.pt)')
    parser.add_argument('--conf',    type=float, default=0.275, help='Confidence threshold (default: 0.275)')
    parser.add_argument('--iou',     type=float, default=0.45,  help='NMS IoU threshold (default: 0.45)')
    parser.add_argument('--tile',    type=int,   default=800,   help='Tile size in pixels (default: 800)')
    parser.add_argument('--overlap', type=int,   default=100,   help='Tile overlap in pixels (default: 100)')
    parser.add_argument('--output',  type=str,   default='fusionwatch_output', help='Output directory')
    parser.add_argument('--demo',    action='store_true', help='Run in demo mode with synthetic test image')

    args = parser.parse_args()

    if args.demo:
        run_demo(model_path=args.model)
        return

    if not args.image:
        parser.print_help()
        print("\nError: --image required (or use --demo for demo mode)")
        sys.exit(1)

    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        print("Download best.pt from the FusionWatch GitHub repository")
        sys.exit(1)

    run_fusionwatch(
        image_path=args.image,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        tile_size=args.tile,
        overlap=args.overlap,
        output_dir=args.output,
    )


if __name__ == '__main__':
    main()
