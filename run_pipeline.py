import argparse
import time
from asset_manager import assets
from data_ingest import DataIngestor
from ai_engine import FusionWatchAI
from context_analyst import ContextAnalyst
from intel_export import IntelligenceExporter

def main():
    parser = argparse.ArgumentParser(description="FusionWatch C4ISR Automated Pipeline")
    parser.add_argument("--image", required=True, help="Path to Sentinel-2 True Color GeoTIFF")
    parser.add_argument("--inject-demo", action="store_true", help="Injects simulated targets")
    # NEW: Accept the geofence bounding box
    parser.add_argument("--bounds", nargs=4, type=float, metavar=('N', 'S', 'E', 'W'), help="Geofence bounds")
    args = parser.parse_args()

    print("\n=======================================================")
    print("    FUSIONWATCH TACTICAL INTELLIGENCE PIPELINE v0.1    ")
    print("=======================================================\n")

    pipeline_start = time.time()

    try:
        ingestor = DataIngestor()
        ai = FusionWatchAI(assets.get_model_path("best.pt"))
        analyst = ContextAnalyst()
        exporter = IntelligenceExporter()

        image_path = assets.verify_image(args.image)
        img_array, transform = ingestor.load_geotiff(image_path)

        raw_detections, inf_time = ai.scan_image(img_array, image_path.name)

        if not raw_detections and args.inject_demo:
            print("\n[DEMO OVERRIDE] Injecting simulated tactical targets for pitch demonstration...")
            raw_detections = [
                {"class": "SHIP", "confidence": 0.94, "pixel_coords": {"cx": 400, "cy": 300}},
                {"class": "SHIP", "confidence": 0.88, "pixel_coords": {"cx": 415, "cy": 315}},
                {"class": "HELICOPTER", "confidence": 0.76, "pixel_coords": {"cx": 600, "cy": 150}},
                {"class": "LARGE-VEHICLE", "confidence": 0.82, "pixel_coords": {"cx": 200, "cy": 500}}
            ]

        if not raw_detections:
            print("\n[PIPELINE] Zero anomalies detected. Securing logs.")
            return

        enriched_targets = analyst.pixels_to_gps(raw_detections, transform)
        
        # --- NEW: APPLY GEOFENCE CULLING ---
        final_targets = []
        if args.bounds:
            n, s, e, w = args.bounds
            print(f"[GEOFENCE] Culling targets outside bounds: N:{n:.4f} S:{s:.4f} E:{e:.4f} W:{w:.4f}")
            for tgt in enriched_targets:
                lat, lon = tgt["location"]["lat"], tgt["location"]["lon"]
                # Check if target is inside the drawn box
                if s <= lat <= n and w <= lon <= e:
                    final_targets.append(tgt)
            print(f"[GEOFENCE] {len(enriched_targets) - len(final_targets)} targets discarded.")
        else:
            final_targets = enriched_targets

        if not final_targets:
            print("\n[PIPELINE] Zero anomalies found inside Geofence.")
            return

        exporter.export_intelligence_package(final_targets, image_path.name)

        total_time = time.time() - pipeline_start
        print(f"\n[PIPELINE] SUCCESS. Full intelligence cycle completed in {total_time:.2f}s.")

    except Exception as e:
        print(f"\n[PIPELINE CRITICAL ERROR] {e}")

if __name__ == "__main__":
    main()