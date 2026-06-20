"""Detection overlay rendering using OpenCV drawing primitives."""

from __future__ import annotations

import cv2
import numpy as np

from src.detection.yolo_detector import Detection
from src.risk.risk_classifier import RiskAssessment
from src.tracking.tracker import TrackedDetection
from src.utils.logger import get_logger

logger = get_logger(__name__)

# BGR colors assigned per vehicle class for consistent visual identity.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "car": (0, 255, 0),          # Green
    "motorcycle": (0, 255, 255),  # Yellow
    "bus": (255, 128, 0),         # Blue-orange
    "truck": (0, 0, 255),         # Red
}

DEFAULT_COLOR: tuple[int, int, int] = (255, 255, 255)
LEAD_VEHICLE_COLOR: tuple[int, int, int] = (255, 0, 255)  # Magenta
LEAD_VEHICLE_BOX_THICKNESS = 3


class DetectionOverlay:
    """Draw YOLO vehicle detections on video frames.

    Accepts ``Detection`` objects from the detector module and returns an
    annotated copy of the input frame. Each vehicle class is rendered with
    a distinct bounding-box color, class label, and confidence score.

    This class handles detection-level drawing only. Risk overlays, HUD
    panels, and TTC gauges will be added in later visualization modules.

    Example:
        >>> overlay = DetectionOverlay()
        >>> annotated = overlay.draw_detections(frame, detections)
    """

    def __init__(
        self,
        box_thickness: int = 2,
        font_scale: float = 0.6,
        font_thickness: int = 1,
    ) -> None:
        """Initialize overlay drawing parameters.

        Args:
            box_thickness: Line thickness for bounding boxes in pixels.
            font_scale: OpenCV font scale for label text.
            font_thickness: Line thickness for label text.
        """
        self._box_thickness = box_thickness
        self._font_scale = font_scale
        self._font_thickness = font_thickness
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> np.ndarray:
        """Draw all detections on a copy of the input frame.

        Args:
            frame: Source BGR image with shape ``(height, width, 3)``.
            detections: Vehicle detections from ``YoloDetector.detect``.

        Returns:
            New annotated frame; the original ``frame`` is not modified.

        Raises:
            ValueError: If the input frame has an invalid shape or dtype.
        """
        self._validate_frame(frame)
        annotated = frame.copy()

        for detection in detections:
            self._draw_single_detection(annotated, detection)

        logger.debug("Drew %d detection overlay(s)", len(detections))
        return annotated

    def draw_tracked_detections(
        self,
        frame: np.ndarray,
        tracked_detections: list[TrackedDetection],
    ) -> np.ndarray:
        """Draw tracked vehicles with track IDs on a copy of the input frame.

        Args:
            frame: Source BGR image with shape ``(height, width, 3)``.
            tracked_detections: Tracked vehicles from ``VehicleTracker.track``.

        Returns:
            New annotated frame; the original ``frame`` is not modified.

        Raises:
            ValueError: If the input frame has an invalid shape or dtype.
        """
        self._validate_frame(frame)
        annotated = frame.copy()

        for tracked in tracked_detections:
            self._draw_single_tracked_detection(annotated, tracked)

        logger.debug("Drew %d tracked overlay(s)", len(tracked_detections))
        return annotated

    def draw_tracked_with_lead(
        self,
        frame: np.ndarray,
        tracked_detections: list[TrackedDetection],
        lead_vehicle: TrackedDetection | None,
        lead_distance_m: float | None = None,
        lead_relative_speed_mps: float | None = None,
        lead_ttc_display: str | None = None,
        risk_assessment: RiskAssessment | None = None,
    ) -> np.ndarray:
        """Draw tracked vehicles and highlight the selected lead vehicle.

        Non-lead tracks use standard class colors. The lead vehicle is drawn
        with a distinct color, thicker border, and ``LEAD VEHICLE`` label.
        Optional distance and relative speed readouts appear below the box.

        Args:
            frame: Source BGR image with shape ``(height, width, 3)``.
            tracked_detections: All tracked vehicles in the frame.
            lead_vehicle: Selected lead vehicle, or ``None`` if unavailable.
            lead_distance_m: Optional smoothed distance to the lead vehicle.
            lead_relative_speed_mps: Optional smoothed relative speed (m/s).
            lead_ttc_display: Optional TTC display string (e.g. ``"4.2 s"``).
            risk_assessment: Optional collision risk classification result.

        Returns:
            New annotated frame; the original ``frame`` is not modified.
        """
        self._validate_frame(frame)
        annotated = frame.copy()

        for tracked in tracked_detections:
            if lead_vehicle is not None and tracked.track_id == lead_vehicle.track_id:
                continue
            self._draw_single_tracked_detection(annotated, tracked)

        if lead_vehicle is not None:
            self._draw_lead_vehicle(
                annotated,
                lead_vehicle,
                lead_distance_m,
                lead_relative_speed_mps,
                lead_ttc_display,
                risk_assessment,
            )

        logger.debug(
            "Drew %d tracked overlay(s) with lead=%s",
            len(tracked_detections),
            lead_vehicle.track_id if lead_vehicle else None,
        )
        return annotated

    def _draw_lead_vehicle(
        self,
        frame: np.ndarray,
        lead_vehicle: TrackedDetection,
        lead_distance_m: float | None = None,
        lead_relative_speed_mps: float | None = None,
        lead_ttc_display: str | None = None,
        risk_assessment: RiskAssessment | None = None,
    ) -> None:
        """Draw the lead vehicle with a highlighted bounding box and label."""
        box_color = (
            risk_assessment.display_color_bgr
            if risk_assessment is not None
            else LEAD_VEHICLE_COLOR
        )
        x1, y1, x2, y2 = lead_vehicle.bbox
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            LEAD_VEHICLE_BOX_THICKNESS,
        )
        label = (
            f"ID:{lead_vehicle.track_id} | "
            f"{lead_vehicle.class_name} | "
            f"conf:{lead_vehicle.confidence:.2f}"
        )
        self._draw_label(frame, label, lead_vehicle.bbox, box_color)

        lines_below: list[str] = []
        line_colors: list[tuple[int, int, int]] = []

        if lead_distance_m is not None:
            lines_below.append(f"Distance: {lead_distance_m:.1f} m")
            line_colors.append(box_color)
        if lead_relative_speed_mps is not None:
            lines_below.append(f"Relative Speed: {lead_relative_speed_mps:.1f} m/s")
            line_colors.append(box_color)
        if lead_ttc_display is not None:
            ttc_text = (
                lead_ttc_display
                if lead_ttc_display.endswith(" s") or lead_ttc_display == "INF"
                else f"{lead_ttc_display} s"
            )
            lines_below.append(f"TTC: {ttc_text}")
            line_colors.append(box_color)
        if risk_assessment is not None:
            lines_below.append(f"Risk: {risk_assessment.level.value}")
            line_colors.append(risk_assessment.display_color_bgr)

        for line_index, (text, color) in enumerate(zip(lines_below, line_colors)):
            offset_bbox = self._offset_bbox_below(lead_vehicle.bbox, line_index)
            self._draw_text_below_bbox(frame, text, offset_bbox, color)

    @staticmethod
    def _offset_bbox_below(
        bbox: tuple[int, int, int, int],
        line_index: int,
    ) -> tuple[int, int, int, int]:
        """Shift a synthetic bbox downward to stack multiple labels."""
        x1, y1, x2, y2 = bbox
        offset = 28 * line_index
        return (x1, y1 + offset, x2, y2 + offset)

    def _draw_text_below_bbox(
        self,
        frame: np.ndarray,
        text: str,
        bbox: tuple[int, int, int, int],
        color: tuple[int, int, int],
    ) -> None:
        """Draw text on a filled strip immediately below a bounding box."""
        x1, _, x2, y2 = bbox
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            self._font,
            self._font_scale,
            self._font_thickness,
        )

        label_top = min(y2 + 4, frame.shape[0] - text_height - baseline - 8)
        label_bottom = label_top + text_height + baseline + 4
        label_right = min(max(x2, x1 + text_width + 4), frame.shape[1] - 1)

        cv2.rectangle(
            frame,
            (x1, label_top),
            (label_right, label_bottom),
            color,
            thickness=-1,
        )
        cv2.putText(
            frame,
            text,
            (x1 + 2, label_bottom - baseline - 2),
            self._font,
            self._font_scale,
            (255, 255, 255),
            self._font_thickness,
            lineType=cv2.LINE_AA,
        )

    def _draw_single_tracked_detection(
        self,
        frame: np.ndarray,
        tracked: TrackedDetection,
    ) -> None:
        """Draw one tracked bounding box and label including the track ID."""
        color = self._color_for_class(tracked.class_name)
        self._draw_bounding_box(frame, tracked.bbox, color)
        label = (
            f"ID:{tracked.track_id} "
            f"{tracked.class_name} {tracked.confidence:.2f}"
        )
        self._draw_label(frame, label, tracked.bbox, color)

    def _draw_single_detection(
        self,
        frame: np.ndarray,
        detection: Detection,
    ) -> None:
        """Draw one detection bounding box and label on ``frame``."""
        color = self._color_for_class(detection.class_name)
        self._draw_bounding_box(frame, detection.bbox, color)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        self._draw_label(frame, label, detection.bbox, color)

    @staticmethod
    def _color_for_class(class_name: str) -> tuple[int, int, int]:
        """Return the BGR color associated with a vehicle class."""
        return CLASS_COLORS.get(class_name, DEFAULT_COLOR)

    def _draw_bounding_box(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        color: tuple[int, int, int],
    ) -> None:
        """Draw a rectangle for the detection bounding box."""
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self._box_thickness)

    def _draw_label(
        self,
        frame: np.ndarray,
        label: str,
        bbox: tuple[int, int, int, int],
        color: tuple[int, int, int],
    ) -> None:
        """Draw a filled label background with text above the bounding box."""
        x1, y1, _, _ = bbox

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            self._font,
            self._font_scale,
            self._font_thickness,
        )

        # Place label above the box; clamp to the top edge of the frame.
        label_top = max(y1 - text_height - baseline - 4, 0)
        label_bottom = label_top + text_height + baseline + 4
        label_right = min(x1 + text_width + 4, frame.shape[1] - 1)

        cv2.rectangle(
            frame,
            (x1, label_top),
            (label_right, label_bottom),
            color,
            thickness=-1,
        )

        # Dark text on light-ish backgrounds is hard to read; force white text.
        cv2.putText(
            frame,
            label,
            (x1 + 2, label_bottom - baseline - 2),
            self._font,
            self._font_scale,
            (255, 255, 255),
            self._font_thickness,
            lineType=cv2.LINE_AA,
        )

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        """Validate frame shape and dtype before drawing."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Frame must have shape (H, W, 3), got {frame.shape}"
            )
        if frame.dtype != np.uint8:
            raise ValueError(f"Frame dtype must be uint8, got {frame.dtype}")
