import os
import rasterio
import numpy as np

class DataIngestor:
    def __init__(self):
        print("[INGESTOR] Ready to receive satellite feeds.")

    def load_geotiff(self, file_path):
        print(f"[INGESTOR] Reading GeoTIFF: {os.path.basename(file_path)}")
        with rasterio.open(file_path) as dataset:
            transform = dataset.transform
            crs = dataset.crs
            
            img_array = dataset.read([1, 2, 3]) 
            img_array = np.transpose(img_array, (1, 2, 0))
            
            # Normalize 16-bit to 8-bit for YOLO
            if img_array.dtype == np.uint16 or img_array.max() > 255:
                img_array = (img_array / img_array.max() * 255).astype(np.uint8)
                
            print(f"[INGESTOR] Feed loaded. Size: {img_array.shape[:2]}, CRS: {crs}")
            return img_array, transform