class ContextAnalyst:
    def __init__(self):
        print("[ANALYST] Spatial intelligence module online.")

    def pixels_to_gps(self, raw_detections, transform):
        enriched_targets = []
        for tgt in raw_detections:
            cx = tgt["pixel_coords"]["cx"]
            cy = tgt["pixel_coords"]["cy"]
            
            # Matrix multiplication for exact GPS
            lon, lat = transform * (cx, cy)
            
            # Basic Risk Logic
            risk = "LOW"
            if tgt["class"] in ['PLANE', 'HELICOPTER', 'AIRCRAFT']: risk = "CRITICAL"
            elif tgt["class"] in ['SHIP', 'HARBOR']: risk = "HIGH"
            elif tgt["class"] in ['LARGE-VEHICLE', 'SMALL-VEHICLE']: risk = "MEDIUM"

            enriched_targets.append({
                "class": tgt["class"],
                "confidence": tgt["confidence"],
                "location": {"lat": round(lat, 6), "lon": round(lon, 6)},
                "risk": risk
            })
            
        return enriched_targets