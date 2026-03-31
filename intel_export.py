import json
import mgrs
from datetime import datetime, timezone
from asset_manager import assets

class IntelligenceExporter:
    def __init__(self):
        self.m = mgrs.MGRS()
        print("[EXPORTER] MGRS Translation & Intelligence formatting online.")

    def dd_to_mgrs(self, lat, lon):
        try:
            raw_mgrs = self.m.toMGRS(lat, lon).decode('utf-8') if isinstance(self.m.toMGRS(lat, lon), bytes) else self.m.toMGRS(lat, lon)
            if len(raw_mgrs) == 15:
                return f"{raw_mgrs[:5]} {raw_mgrs[5:10]} {raw_mgrs[10:]}"
            return raw_mgrs
        except Exception as e:
            return f"MGRS_ERR: {e}"

    def generate_9_line(self, tgt_id, cls_name, mgrs_grid, risk):
        return f""">>> CAS 9-LINE BRIEFING <<<
LINE 1: IP               - N/A
LINE 2: Heading          - N/A
LINE 3: Target Elevation - 12m MSL
LINE 4: Target Descript  - 1 x {cls_name}
LINE 5: Target Location  - {mgrs_grid}
LINE 6: Type Mark        - NONE
LINE 7: Loc Friendlies   - N/A
LINE 8: Egress           - WEST
LINE 9: Remarks          - Automated AI Detection ({risk} Risk)
TGT ID: {tgt_id}"""

    def export_intelligence_package(self, enriched_targets, source_image_name):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        mission_id = f"FW-{timestamp}"
        
        final_targets = []
        for i, tgt in enumerate(enriched_targets):
            mgrs_grid = self.dd_to_mgrs(tgt["location"]["lat"], tgt["location"]["lon"])
            cas_9_line = self.generate_9_line(f"TGT-{i:03d}", tgt["class"], mgrs_grid, tgt["risk"])
            
            final_targets.append({
                "target_id": f"TGT-{i:03d}",
                "class": tgt["class"],
                "confidence": tgt["confidence"],
                "risk": tgt["risk"],
                "location": {
                    "lat": tgt["location"]["lat"],
                    "lon": tgt["location"]["lon"],
                    "mgrs": mgrs_grid
                },
                "cas_9_line": cas_9_line
            })

        payload = {
            "mission_id": mission_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat() + "Z",
            "source_feed": source_image_name,
            "target_count": len(final_targets),
            "targets": final_targets
        }
        
        json_path = assets.get_output_dir() / f"{mission_id}_alerts.json"
        with open(json_path, 'w') as f:
            json.dump(payload, f, indent=4)
            
        print(f"[EXPORTER] Intelligence package secured -> {json_path.name}")
        return json_path