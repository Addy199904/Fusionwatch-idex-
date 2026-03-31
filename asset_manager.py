import os
from pathlib import Path

class AssetManager:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.model_dir = self.base_dir
        self.output_dir = self.base_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model_name="best.pt") -> Path:
        model_path = self.model_dir / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"[SYSTEM ERROR] Weights missing at {model_path}")
        return model_path

    def get_output_dir(self) -> Path:
        return self.output_dir

    def verify_image(self, image_path: str) -> Path:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"[SYSTEM ERROR] Feed not found at {path}")
        return path

assets = AssetManager()