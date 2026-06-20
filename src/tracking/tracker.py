"""Multi-object vehicle tracking via Ultralytics ByteTrack."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.detection.yolo_detector import COCO_VEHICLE_NAMES, YoloDetector
from src.utils.config import TrackerConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Ultralytics tracker YAML config filenames keyed by config ``tracker.type``.
TRACKER_CONFIG_FILES: dict[str, str] = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml",
}


@dataclass(frozen=True)
class TrackedDetection:
    """Vehicle detection with a persistent multi-frame track ID.

    Attributes:
        track_id: Unique identifier maintained by ByteTrack across frames.
        bbox: Bounding box as ``(x1, y1, x2, y2)`` in pixel coordinates.
        confidence: Detection confidence score in ``[0.0, 1.0]``.
        class_id: COCO class index (e.g. 2 = car).
        class_name: Human-readable class label (e.g. ``"car"``).
    """

    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str


class VehicleTracker:
    """Track vehicles across frames using Ultralytics built-in ByteTrack.

    ByteTrack is integrated at the YOLO model level via ``model.track()``.
    This class wraps that API, reuses the ``YOLO`` instance from
    ``YoloDetector`` (avoiding a second model load), and returns structured
    ``TrackedDetection`` objects.

    The public ``track`` method is the stable interface for downstream
    modules (distance estimation, lead-vehicle selection). A different
    tracker backend can replace this class without changing callers.

    Example:
        >>> detector = YoloDetector(config.model)
        >>> tracker = VehicleTracker.from_detector(detector, config.tracker)
        >>> tracked = tracker.track(frame)
    """

    def __init__(
        self,
        detector: YoloDetector,
        tracker_config: TrackerConfig,
    ) -> None:
        """Initialize the tracker using an existing YOLO detector.

        Args:
            detector: Configured ``YoloDetector`` whose underlying model
                will run detection + ByteTrack association.
            tracker_config: Tracker type and persistence settings from YAML.

        Raises:
            ValueError: If the configured tracker type is unsupported.
        """
        self._model = detector.model
        self._device = detector.device
        self._confidence = detector.confidence_threshold
        self._iou = detector.iou_threshold
        self._class_ids = detector.class_ids
        self._persist = tracker_config.persist
        self._tracker_yaml = self._resolve_tracker_config(tracker_config.type)

        logger.info(
            "Vehicle tracker ready (type=%s, persist=%s, device=%s)",
            tracker_config.type,
            self._persist,
            self._device,
        )

    @classmethod
    def from_detector(
        cls,
        detector: YoloDetector,
        tracker_config: TrackerConfig,
    ) -> VehicleTracker:
        """Create a tracker that shares the detector's YOLO model.

        Args:
            detector: Initialized ``YoloDetector`` instance.
            tracker_config: Tracker settings from application config.

        Returns:
            Configured ``VehicleTracker`` instance.
        """
        return cls(detector=detector, tracker_config=tracker_config)

    @property
    def persist(self) -> bool:
        """Whether track IDs persist across consecutive ``track`` calls."""
        return self._persist

    @staticmethod
    def _resolve_tracker_config(tracker_type: str) -> str:
        """Map config tracker type to an Ultralytics YAML filename."""
        if tracker_type not in TRACKER_CONFIG_FILES:
            raise ValueError(
                f"Unsupported tracker type: {tracker_type!r}. "
                f"Allowed: {list(TRACKER_CONFIG_FILES.keys())}"
            )
        return TRACKER_CONFIG_FILES[tracker_type]

    def track(self, frame: np.ndarray) -> list[TrackedDetection]:
        """Detect and track vehicles in a single BGR frame.

        Ultralytics runs YOLO inference and ByteTrack association in one
        pass. Detections without a track ID (e.g. first association frame)
        are excluded from the returned list.

        Args:
            frame: Input image as a NumPy array with shape
                ``(height, width, 3)`` and dtype ``uint8``.

        Returns:
            List of ``TrackedDetection`` objects with persistent track IDs.
            Returns an empty list when no tracked vehicles are present.

        Raises:
            ValueError: If the input frame has an invalid shape or dtype.
            RuntimeError: If tracking inference fails unexpectedly.
        """
        self._validate_frame(frame)

        try:
            results = self._model.track(
                source=frame,
                conf=self._confidence,
                iou=self._iou,
                classes=self._class_ids,
                device=self._device,
                persist=self._persist,
                tracker=self._tracker_yaml,
                verbose=False,
            )
        except Exception as exc:
            logger.error("ByteTrack inference failed: %s", exc)
            raise RuntimeError("ByteTrack inference failed") from exc

        tracked = self._parse_tracked_results(results)
        logger.debug("Tracking %d vehicle(s)", len(tracked))
        return tracked

    def reset(self) -> None:
        """Reset internal tracker state when starting a new video sequence.

        Call this between separate videos so track IDs do not carry over
        from a previous sequence.
        """
        if getattr(self._model, "predictor", None) is not None:
            self._model.predictor = None
        logger.debug("Tracker state reset")

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        """Validate frame shape and dtype before tracking."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Frame must have shape (H, W, 3), got {frame.shape}"
            )
        if frame.dtype != np.uint8:
            raise ValueError(f"Frame dtype must be uint8, got {frame.dtype}")

    def _parse_tracked_results(self, results) -> list[TrackedDetection]:
        """Convert Ultralytics track results into ``TrackedDetection`` objects."""
        if not results:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        # Track IDs are absent until ByteTrack assigns them.
        if result.boxes.id is None:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        track_ids = result.boxes.id.cpu().numpy().astype(int)

        tracked: list[TrackedDetection] = []

        for box, confidence, class_id, track_id in zip(
            boxes, confidences, class_ids, track_ids
        ):
            class_name = COCO_VEHICLE_NAMES.get(int(class_id), "unknown")
            x1, y1, x2, y2 = (int(coord) for coord in box)

            tracked.append(
                TrackedDetection(
                    track_id=int(track_id),
                    bbox=(x1, y1, x2, y2),
                    confidence=float(confidence),
                    class_id=int(class_id),
                    class_name=class_name,
                )
            )

        return tracked

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"VehicleTracker(tracker={self._tracker_yaml!r}, "
            f"persist={self._persist}, device={self._device!r})"
        )
