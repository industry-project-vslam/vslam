from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torchvision import models
from ultralytics import YOLO

from .config import Settings


@dataclass(frozen=True)
class Detection:
    bbox: list[int]
    human_confidence: float
    posture: str
    posture_confidence: float
    unknown: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox,
            "human_confidence": self.human_confidence,
            "posture": self.posture,
            "posture_confidence": self.posture_confidence,
            "unknown": self.unknown,
        }


class ImageDecodeError(ValueError):
    pass


class PostureDetectionPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = torch.device(settings.device)
        self.detector = self._load_detector(settings.detector_model)
        self.classifier, self.class_names = self._load_classifier(settings.classifier_model, settings.class_map)

    def predict_bytes(self, image_bytes: bytes) -> dict[str, Any]:
        start = time.perf_counter()
        frame = self.decode_image(image_bytes)
        decoded_at = time.perf_counter()
        detections = self.predict_frame(frame)
        done = time.perf_counter()

        height, width = frame.shape[:2]
        channels = 1 if frame.ndim == 2 else frame.shape[2]
        return {
            "image": {
                "width": int(width),
                "height": int(height),
                "channels": int(channels),
            },
            "detections": [detection.to_dict() for detection in detections],
            "timing_ms": {
                "decode": round((decoded_at - start) * 1000, 3),
                "inference": round((done - decoded_at) * 1000, 3),
                "total": round((done - start) * 1000, 3),
            },
        }

    @staticmethod
    def decode_image(image_bytes: bytes) -> np.ndarray:
        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise ImageDecodeError("Could not decode image bytes. Send JPEG or PNG bytes.")
        return frame

    def predict_frame(self, frame: np.ndarray) -> list[Detection]:
        frame = self._ensure_bgr(frame)
        results = self.detector.predict(
            source=frame,
            imgsz=self.settings.detector_image_size,
            conf=self.settings.detector_confidence,
            device=self.settings.device,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []

        boxes = results[0].boxes
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        height, width = frame.shape[:2]

        detections: list[Detection] = []
        for coords, det_conf in zip(xyxy, confidences):
            x1, y1, x2, y2 = self._clip_box(coords, width, height)
            if x2 - x1 < self.settings.min_box_size or y2 - y1 < self.settings.min_box_size:
                continue

            crop_box = self._pad_box(x1, y1, x2, y2, width, height)
            crop = frame[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2]]
            posture, posture_confidence, unknown = self._classify_crop(crop)
            if unknown:
                posture = "unknown"

            detections.append(
                Detection(
                    bbox=[x1, y1, x2, y2],
                    human_confidence=round(float(det_conf), 6),
                    posture=posture,
                    posture_confidence=round(posture_confidence, 6),
                    unknown=unknown,
                )
            )

        return detections

    def _classify_crop(self, crop: np.ndarray) -> tuple[str, float, bool]:
        if crop.size == 0:
            return "unknown", 0.0, True

        image = cv2.cvtColor(self._ensure_bgr(crop), cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image,
            (self.settings.classifier_image_size, self.settings.classifier_image_size),
            interpolation=cv2.INTER_AREA,
        )
        tensor = torch.from_numpy(image).to(self.device, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        tensor = ((tensor - mean) / std).unsqueeze(0)

        with torch.inference_mode():
            probs = torch.softmax(self.classifier(tensor), dim=1)[0]
            top = torch.topk(probs, k=1)

        top_probability = float(top.values[0].item())
        top_index = int(top.indices[0].item())
        unknown = top_probability < self.settings.posture_confidence
        return self.class_names[top_index], top_probability, unknown

    def _load_detector(self, path: Path) -> YOLO:
        if not path.exists():
            raise FileNotFoundError(f"Missing detector model: {path}")
        return YOLO(str(path))

    def _load_classifier(self, model_path: Path, class_map_path: Path) -> tuple[torch.nn.Module, list[str]]:
        if not model_path.exists():
            raise FileNotFoundError(f"Missing classifier model: {model_path}")
        if not class_map_path.exists():
            raise FileNotFoundError(f"Missing class map: {class_map_path}")

        class_to_idx = json.loads(class_map_path.read_text(encoding="utf-8"))
        class_names = [name for name, _idx in sorted(class_to_idx.items(), key=lambda item: item[1])]

        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=self.device)

        model_name = checkpoint.get("model_name", "mobilenet_v3_small")
        model = self._build_classifier(model_name, len(class_names))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)
        model.eval()
        return model, class_names

    @staticmethod
    def _build_classifier(model_name: str, num_classes: int) -> torch.nn.Module:
        if model_name == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(weights=None)
        elif model_name == "mobilenet_v3_large":
            model = models.mobilenet_v3_large(weights=None)
        else:
            raise ValueError(f"Unsupported classifier model: {model_name}")

        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
        return model

    @staticmethod
    def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    @staticmethod
    def _clip_box(coords: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = coords[:4]
        return (
            max(0, min(int(round(x1)), width - 1)),
            max(0, min(int(round(y1)), height - 1)),
            max(0, min(int(round(x2)), width - 1)),
            max(0, min(int(round(y2)), height - 1)),
        )

    def _pad_box(self, x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
        pad_x = int((x2 - x1) * self.settings.crop_padding)
        pad_y = int((y2 - y1) * self.settings.crop_padding)
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(width, x2 + pad_x),
            min(height, y2 + pad_y),
        )
