"""Time-To-Collision (TTC) estimation from distance and relative speed."""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.estimation.distance_estimator import DistanceEstimate
from src.estimation.speed_estimator import SpeedEstimate
from src.utils.config import TTCConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TTCEstimate:
    """Time-To-Collision estimate for the lead vehicle.

    Attributes:
        raw_ttc_seconds: Unclamped TTC in seconds, or ``math.inf`` when
            invalid (not approaching, speed too low, or missing inputs).
        display_ttc: Human-readable TTC string for overlays (e.g. ``"4.2 s"``
            or ``"INF"``).
        is_valid: ``True`` when TTC was computed from a closing scenario.
        track_id: ByteTrack ID of the lead vehicle.
    """

    raw_ttc_seconds: float
    display_ttc: str
    is_valid: bool
    track_id: int


class TTCEstimator:
    """Estimate Time-To-Collision for an approaching lead vehicle.

    TTC is computed only when the lead vehicle is closing (negative relative
    speed above a noise threshold), using the constant closing-speed model:

        TTC = Distance / |RelativeSpeed|     (when RelativeSpeed < 0)

    When the vehicle is not approaching, relative speed is near zero, or
    inputs are unavailable, TTC is marked invalid and displayed as infinity.

    Example:
        >>> estimator = TTCEstimator(config.ttc)
        >>> ttc = estimator.estimate(distance, speed)
    """

    def __init__(self, ttc_config: TTCConfig) -> None:
        """Initialize the TTC estimator.

        Args:
            ttc_config: Minimum closing speed, maximum display value, and
                infinity label from application config.
        """
        self._minimum_speed_mps = ttc_config.minimum_speed_mps
        self._maximum_ttc_seconds = ttc_config.maximum_ttc_seconds
        self._display_infinity_as = ttc_config.display_infinity_as

    def estimate(
        self,
        distance: DistanceEstimate | None,
        speed: SpeedEstimate | None,
    ) -> TTCEstimate | None:
        """Compute TTC from smoothed distance and relative speed.

        Args:
            distance: Smoothed distance from ``DistanceEstimator``, or ``None``.
            speed: Smoothed relative speed from ``SpeedEstimator``, or ``None``.

        Returns:
            ``TTCEstimate`` when distance and speed are available (valid or
            invalid), or ``None`` when either input is missing.
        """
        if distance is None or speed is None:
            logger.debug("Missing distance or speed — skipping TTC estimation")
            return None

        if distance.track_id != speed.track_id:
            logger.warning(
                "Track ID mismatch for TTC (distance=%d, speed=%d)",
                distance.track_id,
                speed.track_id,
            )
            return self._invalid_estimate(distance.track_id, reason="track mismatch")

        return self._compute_ttc(
            track_id=distance.track_id,
            distance_m=distance.smoothed_distance_m,
            relative_speed_mps=speed.smoothed_speed_mps,
        )

    def _compute_ttc(
        self,
        track_id: int,
        distance_m: float,
        relative_speed_mps: float,
    ) -> TTCEstimate:
        """Apply TTC formula and handle edge cases."""
        # TTC only defined when the lead vehicle is approaching (negative speed).
        if relative_speed_mps >= 0.0:
            logger.debug(
                "TTC invalid track_id=%d — not approaching (speed=%.2f m/s)",
                track_id,
                relative_speed_mps,
            )
            return self._invalid_estimate(track_id, reason="not approaching")

        closing_speed_mps = abs(relative_speed_mps)

        # Guard against division by zero and noise-floor speeds.
        if closing_speed_mps < self._minimum_speed_mps:
            logger.debug(
                "TTC invalid track_id=%d — closing speed below threshold "
                "(%.2f < %.2f m/s)",
                track_id,
                closing_speed_mps,
                self._minimum_speed_mps,
            )
            return self._invalid_estimate(track_id, reason="speed below threshold")

        raw_ttc_seconds = distance_m / closing_speed_mps
        display_ttc = self._format_display_ttc(raw_ttc_seconds)

        logger.debug(
            "TTC track_id=%d valid raw=%.2fs display=%s (d=%.2fm, v=%.2f m/s)",
            track_id,
            raw_ttc_seconds,
            display_ttc,
            distance_m,
            relative_speed_mps,
        )

        return TTCEstimate(
            raw_ttc_seconds=raw_ttc_seconds,
            display_ttc=display_ttc,
            is_valid=True,
            track_id=track_id,
        )

    def _invalid_estimate(self, track_id: int, reason: str) -> TTCEstimate:
        """Build an invalid TTC result displayed as infinity."""
        logger.debug("TTC invalid track_id=%d (%s)", track_id, reason)
        return TTCEstimate(
            raw_ttc_seconds=math.inf,
            display_ttc=self._display_infinity_as,
            is_valid=False,
            track_id=track_id,
        )

    def _format_display_ttc(self, raw_ttc_seconds: float) -> str:
        """Format TTC for on-screen display with clamping."""
        if math.isinf(raw_ttc_seconds) or raw_ttc_seconds <= 0.0:
            return self._display_infinity_as

        clamped = min(raw_ttc_seconds, self._maximum_ttc_seconds)
        return f"{clamped:.1f} s"
