# config.py
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent  # Up to D:\...\Bouldering_CV_Analysis
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATHS = {
    "lite": MODELS_DIR / "pose_landmarker_lite.task",
    "full": MODELS_DIR / "pose_landmarker_full.task",
    "heavy": MODELS_DIR / "pose_landmarker_heavy.task",
}

def get_model_path(model_type: str) -> str:
    return str(MODEL_PATHS.get(model_type, MODEL_PATHS["lite"]))

