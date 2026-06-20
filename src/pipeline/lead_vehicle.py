"""Lead vehicle selection within the ego-lane region of interest."""

from __future__ import annotations

import cv2
import numpy as np

from src.tracking.tracker import TrackedDetection
from src.utils.config import LeadVehicleConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LeadVehicleSelector:
    """Select the primary vehicle ahead for collision monitoring.

    Filters tracked detections to a configurable trapezoidal ROI that
    approximates the ego lane, then chooses the vehicle whose bounding box
    has the largest area as a monocular proxy for closest distance.

    Example:
        >>> selector = LeadVehicleSelector(config.lead_vehicle)
        >>> lead = selector.select(tracked_detections, frame_width=1280, frame_height=720)
    """

    def __init__(self, config: LeadVehicleConfig) -> None:
        """Initialize the selector with ROI settings from config.

        Args:
            config: Lead vehicle selection settings including normalized ROI
                polygon points.
        """
        self._roi_points_normalized = config.roi.points

    @property
    def roi_points_normalized(self) -> list[tuple[float, float]]:
        """ROI polygon vertices as normalized ``(x, y)`` coordinates."""
        return self._roi_points_normalized

    def select(
        self,
        tracked_detections: list[TrackedDetection],
        frame_width: int,
        frame_height: int,
    ) -> TrackedDetection | None:
        """Select the lead vehicle from tracked detections.

        Args:
            tracked_detections: Active tracks from ``VehicleTracker.track``.
            frame_width: Width of the current video frame in pixels.
            frame_height: Height of the current video frame in pixels.

        Returns:
            The selected ``TrackedDetection`` inside the ROI with the largest
            bounding-box area, or ``None`` when no suitable vehicle exists.

        Raises:
            ValueError: If frame dimensions are invalid.
        """
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError(
                f"Invalid frame dimensions: {frame_width}x{frame_height}"
            )

        if not tracked_detections:
            logger.debug("No tracked detections available for lead selection")
            return None

        roi_polygon = self._scale_roi_to_frame(frame_width, frame_height)
        candidates = [
            tracked
            for tracked in tracked_detections
            if self._is_inside_roi(tracked, roi_polygon)
        ]

        if not candidates:
            logger.debug(
                "No tracked vehicles inside ROI (%d total tracks)",
                len(tracked_detections),
            )
            return None

        lead = max(candidates, key=self._bounding_box_area)
        logger.debug(
            "Selected lead vehicle track_id=%d (area=%d px^2, %d ROI candidates)",
            lead.track_id,
            self._bounding_box_area(lead),
            len(candidates),
        )
        return lead

    def _scale_roi_to_frame(
        self,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:
        """Convert normalized ROI points to pixel coordinates."""
        pixel_points = [
            (int(x_norm * frame_width), int(y_norm * frame_height))
            for x_norm, y_norm in self._roi_points_normalized
        ]
        return np.array(pixel_points, dtype=np.int32)

    @staticmethod
    def _bottom_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        """Return the bottom-center point of a bounding box."""
        x1, _, x2, y2 = bbox
        return ((x1 + x2) / 2.0, float(y2))

    def _is_inside_roi(
        self,
        tracked: TrackedDetection,
        roi_polygon: np.ndarray,
    ) -> bool:
        """Check whether a track's bottom-center lies inside the ROI polygon."""
        bottom_center = self._bottom_center(tracked.bbox)
        # pointPolygonTest returns >= 0 when the point is inside or on the edge.
        result = cv2.pointPolygonTest(
            roi_polygon,
            bottom_center,
            measureDist=False,
        )
        return result >= 0

    @staticmethod
    def _bounding_box_area(tracked: TrackedDetection) -> int:
        """Compute bounding-box area as a closeness proxy."""
        x1, y1, x2, y2 = tracked.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def get_roi_polygon_pixels(
        self,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:
        """Return the ROI polygon scaled to pixel coordinates for visualization.

        Args:
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.

        Returns:
            OpenCV-compatible polygon array with shape ``(4, 1, 2)``.
        """
        polygon = self._scale_roi_to_frame(frame_width, frame_height)
        return polygon.reshape((-1, 1, 2))
