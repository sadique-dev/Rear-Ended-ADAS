"""Relative speed estimation from temporal distance differentiation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from src.estimation.distance_estimator import DistanceEstimate
from src.utils.config import EstimationConfig
from src.utils.logger import get_logger
from src.utils.smoothing import ExponentialMovingAverage

logger = get_logger(__name__)


@dataclass
class _TrackSpeedState:
    """Per-track distance history and speed smoothing state."""

    history: deque[tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=30)
    )
    speed_smoother: ExponentialMovingAverage | None = None
    last_seen_timestamp: float = 0.0


@dataclass(frozen=True)
class SpeedEstimate:
    """Relative speed estimate for a tracked lead vehicle.

    Attributes:
        raw_speed_mps: Instantaneous relative speed from finite difference.
        smoothed_speed_mps: EMA-smoothed relative speed.
        track_id: ByteTrack ID of the estimated vehicle.

    Sign convention:
        - Negative → lead vehicle is approaching (distance decreasing).
        - Positive → lead vehicle is receding (distance increasing).
        - Near zero → similar speed to the ego vehicle.
    """

    raw_speed_mps: float
    smoothed_speed_mps: float
    track_id: int


class SpeedEstimator:
    """Estimate relative speed from smoothed distance over time.

    Maintains a short per-track history of ``(timestamp, smoothed_distance)``
    and computes relative speed via finite differencing:

        v_rel = (D_current - D_previous) / (t_current - t_previous)

    Positive distance change means the object is moving away; negative
    change means it is approaching. Raw speeds are smoothed with an EMA
    filter and small values below a noise floor are zeroed.

    Example:
        >>> estimator = SpeedEstimator(config.estimation)
        >>> speed = estimator.estimate(distance, timestamp_seconds=1.0)
    """

    def __init__(self, estimation_config: EstimationConfig) -> None:
        """Initialize the speed estimator.

        Args:
            estimation_config: Smoothing, warmup, and noise-floor parameters.
        """
        self._speed_ema_alpha = estimation_config.speed_ema_alpha
        self._speed_min_mps = estimation_config.speed_min_mps
        self._warmup_frames = estimation_config.speed_warmup_frames
        self._track_states: dict[int, _TrackSpeedState] = {}

    def estimate(
        self,
        distance: DistanceEstimate | None,
        timestamp_seconds: float,
    ) -> SpeedEstimate | None:
        """Update history and compute relative speed for the lead vehicle.

        Args:
            distance: Smoothed distance estimate from ``DistanceEstimator``,
                or ``None`` when no lead vehicle is available.
            timestamp_seconds: Elapsed video time for the current frame.

        Returns:
            ``SpeedEstimate`` when sufficient history exists, otherwise
            ``None`` during warm-up or when distance is unavailable.

        Raises:
            ValueError: If ``timestamp_seconds`` is negative.
        """
        if timestamp_seconds < 0.0:
            raise ValueError(
                f"timestamp_seconds must be non-negative, got {timestamp_seconds}"
            )

        if distance is None:
            logger.debug("No distance estimate — skipping speed estimation")
            return None

        state = self._track_states.setdefault(distance.track_id, _TrackSpeedState())
        state.last_seen_timestamp = timestamp_seconds
        state.history.append((timestamp_seconds, distance.smoothed_distance_m))

        if len(state.history) < self._warmup_frames:
            logger.debug(
                "Speed warm-up track_id=%d (%d/%d samples)",
                distance.track_id,
                len(state.history),
                self._warmup_frames,
            )
            return None

        raw_speed_mps = self._compute_raw_speed(state.history)
        if raw_speed_mps is None:
            return None

        smoothed_speed_mps = self._smooth_speed(state, raw_speed_mps)
        raw_speed_mps = self._apply_noise_floor(raw_speed_mps)
        smoothed_speed_mps = self._apply_noise_floor(smoothed_speed_mps)

        logger.debug(
            "Speed track_id=%d raw=%.2f m/s smoothed=%.2f m/s",
            distance.track_id,
            raw_speed_mps,
            smoothed_speed_mps,
        )

        return SpeedEstimate(
            raw_speed_mps=raw_speed_mps,
            smoothed_speed_mps=smoothed_speed_mps,
            track_id=distance.track_id,
        )

    def prune_inactive_tracks(self, active_track_ids: set[int]) -> None:
        """Remove history for tracks that are no longer present.

        Args:
            active_track_ids: Track IDs seen in the current frame.
        """
        inactive_ids = set(self._track_states) - active_track_ids
        for track_id in inactive_ids:
            del self._track_states[track_id]
            logger.debug("Cleared speed history for inactive track_id=%d", track_id)

    def reset(self) -> None:
        """Clear all per-track speed history and smoothers."""
        self._track_states.clear()
        logger.debug("Speed estimator state reset")

    @staticmethod
    def _compute_raw_speed(
        history: deque[tuple[float, float]],
    ) -> float | None:
        """Compute relative speed from the two most recent distance samples."""
        if len(history) < 2:
            return None

        prev_time, prev_distance = history[-2]
        curr_time, curr_distance = history[-1]
        delta_time = curr_time - prev_time

        if delta_time <= 0.0:
            logger.warning(
                "Non-positive delta_time (%.4fs) — skipping speed update",
                delta_time,
            )
            return None

        # v_rel = ΔDistance / ΔTime
        # Negative → approaching; positive → receding.
        return (curr_distance - prev_distance) / delta_time

    def _smooth_speed(
        self,
        state: _TrackSpeedState,
        raw_speed_mps: float,
    ) -> float:
        """Apply EMA smoothing to a raw relative speed measurement."""
        if state.speed_smoother is None:
            state.speed_smoother = ExponentialMovingAverage(self._speed_ema_alpha)
        return state.speed_smoother.update(raw_speed_mps)

    def _apply_noise_floor(self, speed_mps: float) -> float:
        """Zero speeds below the configured noise floor."""
        if abs(speed_mps) < self._speed_min_mps:
            return 0.0
        return speed_mps
