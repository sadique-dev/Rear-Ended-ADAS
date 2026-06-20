"""Monocular distance estimation using the pinhole camera model."""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.tracking.tracker import TrackedDetection
from src.utils.config import CameraConfig, EstimationConfig
from src.utils.logger import get_logger
from src.utils.smoothing import ExponentialMovingAverage

logger = get_logger(__name__)


@dataclass(frozen=True)
class DistanceEstimate:
    """Distance estimate for a lead vehicle.

    Attributes:
        raw_distance_m: Instantaneous distance from the pinhole model (metres).
        smoothed_distance_m: EMA-smoothed distance (metres).
        track_id: ByteTrack ID of the estimated vehicle.
        class_name: Vehicle class used for width lookup.
    """

    raw_distance_m: float
    smoothed_distance_m: float
    track_id: int
    class_name: str


class DistanceEstimator:
    """Estimate camera-to-vehicle distance from bounding-box width.

    Uses the pinhole camera model with class-specific real-world vehicle
    widths and configurable focal length. Applies per-track exponential
    moving average smoothing to reduce jitter.

    Distance formula::

        D = (W_real × f_px) / w_bbox

    where:

    - ``D``     = distance to the vehicle (metres)
    - ``W_real`` = assumed real-world vehicle width (metres)
    - ``f_px``  = focal length in pixels
    - ``w_bbox`` = bounding-box width in pixels

    Focal length is read from config when set; otherwise derived from
    horizontal FOV and frame width::

        f_px = (frame_width / 2) / tan(FOV_horizontal / 2)

    Example:
        >>> estimator = DistanceEstimator(config.camera, config.estimation)
        >>> result = estimator.estimate(lead_vehicle, frame_width=1280)
    """

    def __init__(
        self,
        camera_config: CameraConfig,
        estimation_config: EstimationConfig,
    ) -> None:
        """Initialize the distance estimator.

        Args:
            camera_config: Camera intrinsics, focal length, and vehicle widths.
            estimation_config: Smoothing and distance clamp parameters.
        """
        self._fov_horizontal_deg = camera_config.fov_horizontal_deg
        self._focal_length_px = camera_config.focal_length_px
        self._vehicle_widths = camera_config.vehicle_widths_m
        self._default_vehicle_width_m = camera_config.assumed_vehicle_width_m
        self._ema_alpha = estimation_config.distance_ema_alpha
        self._distance_min_m = estimation_config.distance_min_m
        self._distance_max_m = estimation_config.distance_max_m
        self._smoothers: dict[int, ExponentialMovingAverage] = {}

    def estimate(
        self,
        lead_vehicle: TrackedDetection | None,
        frame_width: int,
    ) -> DistanceEstimate | None:
        """Estimate distance to the selected lead vehicle.

        Args:
            lead_vehicle: Selected lead ``TrackedDetection``, or ``None``.
            frame_width: Current frame width in pixels (used for focal length
                derivation when ``focal_length_px`` is not set in config).

        Returns:
            ``DistanceEstimate`` with raw and smoothed distances, or ``None``
            when no lead vehicle is provided or bbox width is invalid.

        Raises:
            ValueError: If ``frame_width`` is not positive.
        """
        if lead_vehicle is None:
            logger.debug("No lead vehicle — skipping distance estimation")
            return None

        if frame_width <= 0:
            raise ValueError(f"frame_width must be positive, got {frame_width}")

        bbox_width_px = self._bounding_box_width(lead_vehicle.bbox)
        if bbox_width_px <= 0:
            logger.warning(
                "Invalid bbox width for track_id=%d", lead_vehicle.track_id
            )
            return None

        focal_length_px = self._resolve_focal_length_px(frame_width)
        vehicle_width_m = self._vehicle_width_for_class(lead_vehicle.class_name)

        # Pinhole model: D = (W_real × f_px) / w_bbox
        raw_distance_m = (vehicle_width_m * focal_length_px) / bbox_width_px
        raw_distance_m = self._clamp_distance(raw_distance_m)

        smoothed_distance_m = self._smooth_distance(
            track_id=lead_vehicle.track_id,
            raw_distance_m=raw_distance_m,
        )

        logger.debug(
            "Distance track_id=%d class=%s raw=%.2fm smoothed=%.2fm "
            "(w_bbox=%dpx, f=%.1fpx, W=%.2fm)",
            lead_vehicle.track_id,
            lead_vehicle.class_name,
            raw_distance_m,
            smoothed_distance_m,
            bbox_width_px,
            focal_length_px,
            vehicle_width_m,
        )

        return DistanceEstimate(
            raw_distance_m=raw_distance_m,
            smoothed_distance_m=smoothed_distance_m,
            track_id=lead_vehicle.track_id,
            class_name=lead_vehicle.class_name,
        )

    def reset(self) -> None:
        """Clear all per-track smoothing state."""
        self._smoothers.clear()
        logger.debug("Distance estimator smoothing state reset")

    def _resolve_focal_length_px(self, frame_width: int) -> float:
        """Return focal length in pixels from config or FOV derivation."""
        if self._focal_length_px is not None:
            return self._focal_length_px

        half_fov_rad = math.radians(self._fov_horizontal_deg / 2.0)
        return (frame_width / 2.0) / math.tan(half_fov_rad)

    def _vehicle_width_for_class(self, class_name: str) -> float:
        """Look up the configured real-world width for a vehicle class."""
        width = self._vehicle_widths.get(class_name)
        if width is None:
            logger.warning(
                "Unknown class %r — using default width %.2fm",
                class_name,
                self._default_vehicle_width_m,
            )
            return self._default_vehicle_width_m
        return width

    @staticmethod
    def _bounding_box_width(bbox: tuple[int, int, int, int]) -> int:
        """Return bounding-box width in pixels."""
        x1, _, x2, _ = bbox
        return max(0, x2 - x1)

    def _clamp_distance(self, distance_m: float) -> float:
        """Clamp distance to configured physical limits."""
        return max(self._distance_min_m, min(distance_m, self._distance_max_m))

    def _smooth_distance(self, track_id: int, raw_distance_m: float) -> float:
        """Apply per-track EMA smoothing to a raw distance measurement."""
        if track_id not in self._smoothers:
            self._smoothers[track_id] = ExponentialMovingAverage(self._ema_alpha)
        return self._smoothers[track_id].update(raw_distance_m)
