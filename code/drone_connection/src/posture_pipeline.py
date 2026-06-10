from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import models
from ultralytics import YOLO


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_detector_model() -> Path:
    return default_repo_root() / "code" / "object_detection" / "runs" / "human_detection" / "weights" / "best.pt"


def default_classifier_model() -> Path:
    return (
        default_repo_root()
        / "code"
        / "object_detection"
        / "runs"
        / "posture_classification"
        / "mobilenet_v3_small"
        / "best.pt"
    )


def default_class_map() -> Path:
    return (
        default_repo_root()
        / "code"
        / "object_detection"
        / "runs"
        / "posture_classification"
        / "mobilenet_v3_small"
        / "class_to_idx.json"
    )


@dataclass
class PosturePipelineSettings:
    detector_model: Path = default_detector_model()
    classifier_model: Path = default_classifier_model()
    class_map: Path = default_class_map()
    detector_confidence: float = 0.23
    posture_confidence: float = 0.50
    crop_padding: float = 0.12
    detector_image_size: int = 320
    classifier_image_size: int = 224
    analyze_every_n_frames: int = 2
    min_box_size: int = 12
    device: str = "cpu"


@dataclass
class Annotation:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    posture_confidence: float
    detector_confidence: float
    unknown: bool


class PosturePipeline:
    def __init__(self, settings: PosturePipelineSettings | None = None):
        self.settings = settings or PosturePipelineSettings()
        self.device = torch.device(self.settings.device)
        self.detector = self._load_detector()
        self.classifier, self.class_names = self._load_classifier()
        self.frame_index = 0
        self.last_annotations: list[Annotation] = []

    def process(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return frame

        output = self._ensure_bgr(frame).copy()
        self.frame_index += 1

        stride = max(int(self.settings.analyze_every_n_frames), 1)
        if self.frame_index == 1 or self.frame_index % stride == 0:
            self.last_annotations = self._analyze(output)

        self._draw_annotations(output, self.last_annotations)
        return output

    def _load_detector(self) -> YOLO:
        if not self.settings.detector_model.exists():
            raise FileNotFoundError(f"Missing detector model: {self.settings.detector_model}")
        return YOLO(str(self.settings.detector_model))

    def _load_classifier(self) -> tuple[torch.nn.Module, list[str]]:
        if not self.settings.classifier_model.exists():
            raise FileNotFoundError(f"Missing classifier model: {self.settings.classifier_model}")
        if not self.settings.class_map.exists():
            raise FileNotFoundError(f"Missing class map: {self.settings.class_map}")

        class_to_idx = json.loads(self.settings.class_map.read_text(encoding="utf-8"))
        class_names = [name for name, _idx in sorted(class_to_idx.items(), key=lambda item: item[1])]

        try:
            checkpoint = torch.load(self.settings.classifier_model, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.settings.classifier_model, map_location=self.device)

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

    def _analyze(self, frame: np.ndarray) -> list[Annotation]:
        results = self.detector.predict(
            source=frame,
            imgsz=self.settings.detector_image_size,
            conf=self.settings.detector_confidence,
            device=self.settings.device,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []

        annotations: list[Annotation] = []
        boxes = results[0].boxes
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()

        height, width = frame.shape[:2]
        for coords, det_conf in zip(xyxy, confidences):
            x1, y1, x2, y2 = self._clip_box(coords, width, height)
            if x2 - x1 < self.settings.min_box_size or y2 - y1 < self.settings.min_box_size:
                continue

            crop_box = self._pad_box(x1, y1, x2, y2, width, height)
            crop = frame[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2]]
            label, cls_conf, unknown = self._classify_crop(crop)
            if unknown:
                label = "unknown"

            annotations.append(
                Annotation(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    label=label,
                    posture_confidence=cls_conf,
                    detector_confidence=float(det_conf),
                    unknown=unknown,
                )
            )

        return annotations

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

        top1_prob = float(top.values[0].item())
        top1_idx = int(top.indices[0].item())
        unknown = top1_prob < self.settings.posture_confidence
        return self.class_names[top1_idx], top1_prob, unknown

    def _draw_annotations(self, frame: np.ndarray, annotations: list[Annotation]) -> None:
        for ann in annotations:
            color = (0, 165, 255) if ann.unknown else (0, 200, 0)
            cv2.rectangle(frame, (ann.x1, ann.y1), (ann.x2, ann.y2), color, 2)

            label = f"{ann.label} {ann.posture_confidence:.2f}"
            cv2.putText(frame, f"det {ann.detector_confidence:.2f}", (ann.x1, ann.y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            y_text = max(ann.y1, text_height + baseline + 4)
            cv2.rectangle(
                frame,
                (ann.x1, y_text - text_height - baseline - 4),
                (ann.x1 + text_width + 8, y_text + baseline),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (ann.x1 + 4, y_text - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

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
