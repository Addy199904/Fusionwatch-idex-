import time
import numpy as np
from ultralytics import YOLO
from asset_manager import assets

class FusionWatchAI:
    def __init__(self, model_path):
        print(f"[AI ENGINE] Loading neural weights from {model_path}...")
        self.model = YOLO(model_path)
        self.names = self.model.names
        print("[AI ENGINE] Neural network online.")

    def scan_image(self, image_data, image_name="feed"):
        print(f"[AI ENGINE] Initiating SLICED scan on {image_name}...")
        start_time = time.time()
        
        # --- SLICED INFERENCE ARCHITECTURE ---
        # We slice the massive satellite image into 800x800 patches.
        # This prevents YOLO from squashing/compressing the image, preserving the 
        # actual pixel data so tiny ships remain visible to the convolutional layers.
        
        tile_size = 800
        h, w, _ = image_data.shape
        raw_detections = []
        
        # Loop through the image in a grid
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                # Extract the 800x800 high-resolution tile
                tile = image_data[y:y+tile_size, x:x+tile_size]
                
                # Skip edge slivers that are too small for the model
                if tile.shape[0] < 100 or tile.shape[1] < 100: 
                    continue 

                # Scan the high-res tile
                results = self.model(tile, imgsz=tile_size, conf=0.15, verbose=False)
                
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    cls_name = self.names[cls_id].upper()
                    
                    # Get coordinates relative to the tile
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Offset the coordinates back to the global master image
                    global_cx = ((x1 + x2) / 2) + x
                    global_cy = ((y1 + y2) / 2) + y
                    
                    raw_detections.append({
                        "class": cls_name,
                        "confidence": round(conf, 3),
                        "pixel_coords": {"cx": global_cx, "cy": global_cy}
                    })
                    
        inference_time_ms = (time.time() - start_time) * 1000
        print(f"[AI ENGINE] Sliced Scan complete. {len(raw_detections)} anomalies found in {inference_time_ms:.1f}ms.")
        return raw_detections, inference_time_ms

    def scan_raw(self, image_path):
        """
        Runs inference on a raw image (PNG/JPG) and returns bounding box corners 
        so the UI can draw them dynamically based on user filters.
        """
        print(f"[AI ENGINE] Running raw algorithm diagnostics on {image_path}...")
        start_time = time.time()
        
        results = self.model(image_path, imgsz=1280, conf=0.15, verbose=False)
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        raw_detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            cls_name = self.names[cls_id].upper()
            
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            raw_detections.append({
                "class": cls_name,
                "confidence": round(conf, 3),
                "pixel_coords": {
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2, 
                    "cx": round(cx, 1), "cy": round(cy, 1)
                }
            })
            
        print(f"[AI ENGINE] Diagnostics complete. {len(raw_detections)} anomalies found.")
        return raw_detections, inference_time_ms