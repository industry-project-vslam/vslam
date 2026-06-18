from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default
    return Path(value).expanduser().resolve()


def _float_from_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


@dataclass(frozen=True)
class Settings:
    detector_model: Path
    classifier_model: Path
    class_map: Path
    detector_confidence: float
    posture_confidence: float
    crop_padding: float
    detector_image_size: int
    classifier_image_size: int
    min_box_size: int
    device: str
    max_image_bytes: int


def load_settings() -> Settings:
    root = repo_root()
    detector_model = root / "code" / "object_detection" / "runs" / "human_detection" / "weights" / "best.pt"
    classifier_dir = root / "code" / "object_detection" / "runs" / "posture_classification" / "mobilenet_v3_small"

    return Settings(
        detector_model=_path_from_env("DETECTOR_MODEL", detector_model),
        classifier_model=_path_from_env("CLASSIFIER_MODEL", classifier_dir / "best.pt"),
        class_map=_path_from_env("CLASS_MAP", classifier_dir / "class_to_idx.json"),
        detector_confidence=_float_from_env("DETECTOR_CONFIDENCE", 0.23),
        posture_confidence=_float_from_env("POSTURE_CONFIDENCE", 0.50),
        crop_padding=_float_from_env("CROP_PADDING", 0.12),
        detector_image_size=_int_from_env("DETECTOR_IMAGE_SIZE", 320),
        classifier_image_size=_int_from_env("CLASSIFIER_IMAGE_SIZE", 224),
        min_box_size=_int_from_env("MIN_BOX_SIZE", 12),
        device=os.getenv("DEVICE", "cpu"),
        max_image_bytes=_int_from_env("MAX_IMAGE_BYTES", 5 * 1024 * 1024),
    )
