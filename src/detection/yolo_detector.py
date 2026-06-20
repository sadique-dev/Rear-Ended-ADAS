"""Ultralytics YOLO wrapper for vehicle detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from src.utils.config import PROJECT_ROOT, ModelConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

# COCO class IDs for vehicle categories used by this ADAS system.
COCO_VEHICLE_NAMES: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass(frozen=True)
class Detection:
    """Single vehicle detection from one inference pass.

    Attributes:
        bbox: Bounding box as ``(x1, y1, x2, y2)`` in pixel coordinates.
        confidence: Detection confidence score in ``[0.0, 1.0]``.
        class_id: COCO class index (e.g. 2 = car).
        class_name: Human-readable class label (e.g. ``"car"``).
    """

    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str


class YoloDetector:
    """Pre-trained YOLO vehicle detector backed by Ultralytics.

    Loads a YOLO weights file (downloaded automatically on first use),
    runs single-frame inference, and returns only vehicle-class detections
    above the configured confidence threshold.

    The public ``detect`` method is the stable interface consumed by the
    tracking module. A different backend (e.g. ONNX, TensorRT) can replace
    this class without changing downstream code.

    Example:
        >>> detector = YoloDetector(config.model)
        >>> detections = detector.detect(frame)
    """

    def __init__(
        self,
        model_config: ModelConfig,
        model_path: str | None = None,
    ) -> None:
        """Load the YOLO model and prepare it for inference.

        Args:
            model_config: Detection settings from application config
                (model name, confidence, IoU, allowed class IDs).
            model_path: Optional override for the model weights path or
                name. When ``None``, uses ``model_config.name``.

        Raises:
            ValueError: If configured class IDs are not supported vehicles.
            RuntimeError: If the model fails to load.
        """
        self._confidence = model_config.confidence
        self._iou = model_config.iou
        self._class_ids = self._validate_class_ids(model_config.classes)

        weights = model_path if model_path is not None else model_config.name
        weights = self._resolve_weights_path(weights)
        self._device = self._resolve_device()

        logger.info("Loading YOLO model: %s", weights)
        try:
            self._model = YOLO(weights)
        except Exception as exc:
            logger.error("Failed to load YOLO model %s: %s", weights, exc)
            raise RuntimeError(f"Failed to load YOLO model: {weights}") from exc

        logger.info(
            "YOLO model ready on %s (confidence=%.2f, iou=%.2f, classes=%s)",
            self._device,
            self._confidence,
            self._iou,
            self._class_ids,
        )

    @property
    def device(self) -> str:
        """Inference device string (``cuda`` or ``cpu``)."""
        return self._device

    @property
    def confidence_threshold(self) -> float:
        """Minimum confidence score for returned detections."""
        return self._confidence

    @property
    def iou_threshold(self) -> float:
        """Non-maximum suppression IoU threshold."""
        return self._iou

    @property
    def class_ids(self) -> list[int]:
        """COCO class IDs used for vehicle filtering."""
        return self._class_ids

    @property
    def model(self) -> YOLO:
        """Underlying Ultralytics model instance for tracker integration."""
        return self._model

    @staticmethod
    def _resolve_weights_path(weights: str) -> str:
        """Resolve model weights from an explicit path or the models/ directory.

        Ultralytics auto-downloads weights when the file is not found locally.
        """
        path = Path(weights)
        if path.is_file():
            return str(path)

        models_dir_path = PROJECT_ROOT / "models" / weights
        if models_dir_path.is_file():
            return str(models_dir_path)

        return weights

    @staticmethod
    def _resolve_device() -> str:
        """Select CUDA when available; otherwise fall back to CPU."""
        try:
            import torch

            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                logger.info("GPU detected: %s", device_name)
                return "cuda"
        except ImportError:
            pass

        logger.info("No GPU available — using CPU for inference")
        return "cpu"

    @staticmethod
    def _validate_class_ids(class_ids: list[int]) -> list[int]:
        """Ensure configured classes are known vehicle categories."""
        invalid = [cid for cid in class_ids if cid not in COCO_VEHICLE_NAMES]
        if invalid:
            raise ValueError(
                f"Unsupported vehicle class IDs: {invalid}. "
                f"Allowed: {list(COCO_VEHICLE_NAMES.keys())}"
            )
        return class_ids

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run vehicle detection on a single BGR frame.

        Args:
            frame: Input image as a NumPy array with shape
                ``(height, width, 3)`` and dtype ``uint8``.

        Returns:
            List of ``Detection`` objects for vehicles above the confidence
            threshold. Returns an empty list when no vehicles are found.

        Raises:
            ValueError: If the input frame has an invalid shape or dtype.
            RuntimeError: If inference fails unexpectedly.
        """
        self._validate_frame(frame)

        try:
            results = self._model.predict(
                source=frame,
                conf=self._confidence,
                iou=self._iou,
                classes=self._class_ids,
                device=self._device,
                verbose=False,
            )
        except Exception as exc:
            logger.error("YOLO inference failed: %s", exc)
            raise RuntimeError("YOLO inference failed") from exc

        return self._parse_results(results)

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        """Validate frame shape and dtype before inference."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Frame must have shape (H, W, 3), got {frame.shape}"
            )
        if frame.dtype != np.uint8:
            raise ValueError(f"Frame dtype must be uint8, got {frame.dtype}")

    def _parse_results(self, results) -> list[Detection]:
        """Convert Ultralytics result tensors into ``Detection`` objects."""
        if not results:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        detections: list[Detection] = []

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            # Ultralytics already filters by `classes` and `conf`, but we
            # map names explicitly to keep output self-contained.
            class_name = COCO_VEHICLE_NAMES.get(class_id, "unknown")
            x1, y1, x2, y2 = (int(coord) for coord in box)

            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(confidence),
                    class_id=int(class_id),
                    class_name=class_name,
                )
            )

        logger.debug("Detected %d vehicle(s)", len(detections))
        return detections

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"YoloDetector(device={self._device!r}, "
            f"confidence={self._confidence}, classes={self._class_ids})"
        )
