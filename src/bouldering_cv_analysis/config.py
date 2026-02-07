"""Centralized configuration for the project."""

from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = PROJECT_ROOT / "models"
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH, override=False)

MODEL_PATHS = {
    "lite": MODELS_DIR / "pose_landmarker_lite.task",
    "full": MODELS_DIR / "pose_landmarker_full.task",
    "heavy": MODELS_DIR / "pose_landmarker_heavy.task",
}


def _resolve_path(value: str | None, default: Path) -> Path:
    """Resolve a given path string to an absolute Path object. If the input value is None or empty, returns the default path.
    If the input value is a relative path, it is resolved relative to the project root."""

    if not value:
        return default
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def get_model_path(model_type: str) -> Path:
    return MODEL_PATHS.get(model_type, MODEL_PATHS["lite"])


@dataclass(frozen=True)
class Settings:
    model_type: str
    model_bundle_path: str
    assets_dir: str
    detection_confidence: dict[str, float]


def get_settings() -> Settings:
    model_type = os.getenv("MODEL_TYPE", "lite").strip().lower()
    model_bundle_path = MODEL_PATHS[model_type]
    detection_confidence = {
        "min_pose_detection_confidence": float(os.getenv("min_pose_detection_confidence", 0.5)),
        "min_pose_presence_confidence": float(os.getenv("min_pose_presence_confidence", 0.5)),
        "min_tracking_confidence": float(os.getenv("min_tracking_confidence", 0.5)),
    }

    return Settings(
        model_type=model_type,
        model_bundle_path=str(model_bundle_path),
        assets_dir=str(ASSETS_DIR),
        detection_confidence=detection_confidence,
    )

