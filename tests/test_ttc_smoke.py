"""Smoke tests for Time-To-Collision estimation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.detection import YoloDetector
from src.estimation import (
    DistanceEstimate,
    DistanceEstimator,
    SpeedEstimate,
    SpeedEstimator,
    TTCEstimator,
)
from src.io import VideoReader
from src.pipeline import LeadVehicleSelector
from src.tracking import VehicleTracker
from src.utils.config import TTCConfig, load_config
from src.visualization import DetectionOverlay


def _ttc_config(
    minimum_speed_mps: float = 0.5,
    maximum_ttc_seconds: float = 15.0,
) -> TTCConfig:
    """Build TTC config for unit tests."""
    return TTCConfig(
        minimum_speed_mps=minimum_speed_mps,
        maximum_ttc_seconds=maximum_ttc_seconds,
        display_infinity_as="INF",
    )


def _distance(track_id: int, smoothed_m: float) -> DistanceEstimate:
    """Synthetic distance estimate."""
    return DistanceEstimate(
        raw_distance_m=smoothed_m,
        smoothed_distance_m=smoothed_m,
        track_id=track_id,
        class_name="car",
    )


def _speed(track_id: int, smoothed_mps: float) -> SpeedEstimate:
    """Synthetic speed estimate."""
    return SpeedEstimate(
        raw_speed_mps=smoothed_mps,
        smoothed_speed_mps=smoothed_mps,
        track_id=track_id,
    )


@pytest.fixture
def ttc_estimator() -> TTCEstimator:
    """Default TTC estimator for tests."""
    return TTCEstimator(_ttc_config())


def test_ttc_decreases_as_vehicle_approaches(ttc_estimator):
    """Shorter distance at the same closing speed should yield lower TTC."""
    closing_speed = -2.0  # approaching at 2 m/s

    far = ttc_estimator.estimate(_distance(1, 20.0), _speed(1, closing_speed))
    near = ttc_estimator.estimate(_distance(1, 10.0), _speed(1, closing_speed))

    assert far is not None and near is not None
    assert far.is_valid and near.is_valid
    assert near.raw_ttc_seconds < far.raw_ttc_seconds
    assert far.raw_ttc_seconds == pytest.approx(10.0)
    assert near.raw_ttc_seconds == pytest.approx(5.0)


def test_ttc_infinity_when_vehicle_moves_away(ttc_estimator):
    """Receding vehicles should produce invalid TTC (infinity)."""
    result = ttc_estimator.estimate(_distance(1, 15.0), _speed(1, 2.0))

    assert result is not None
    assert not result.is_valid
    assert math.isinf(result.raw_ttc_seconds)
    assert result.display_ttc == "INF"


def test_division_by_zero_handled_safely(ttc_estimator):
    """Near-zero closing speed must not raise and should be invalid."""
    result = ttc_estimator.estimate(_distance(1, 10.0), _speed(1, -0.1))

    assert result is not None
    assert not result.is_valid
    assert math.isinf(result.raw_ttc_seconds)


def test_unrealistic_ttc_values_are_clamped(ttc_estimator):
    """Display TTC should clamp to the configured maximum."""
    # distance=30, speed=-1 -> raw TTC=30s, display clamped to 15s
    result = ttc_estimator.estimate(_distance(1, 30.0), _speed(1, -1.0))

    assert result is not None
    assert result.is_valid
    assert result.raw_ttc_seconds == pytest.approx(30.0)
    assert result.display_ttc == "15.0 s"


def test_ttc_formula_at_threshold_speed():
    """Closing speed exactly at minimum should still compute TTC."""
    estimator = TTCEstimator(_ttc_config(minimum_speed_mps=0.5))
    result = estimator.estimate(_distance(1, 5.0), _speed(1, -0.5))

    assert result is not None
    assert result.is_valid
    assert result.raw_ttc_seconds == pytest.approx(10.0)


def test_missing_inputs_returns_none(ttc_estimator):
    """Missing distance or speed should return None without error."""
    assert ttc_estimator.estimate(None, _speed(1, -2.0)) is None
    assert ttc_estimator.estimate(_distance(1, 10.0), None) is None


def test_full_pipeline_runs_without_errors():
    """End-to-end pipeline through TTC should run cleanly."""
    config = load_config()
    detector = YoloDetector(config.model)
    tracker = VehicleTracker.from_detector(detector, config.tracker)
    selector = LeadVehicleSelector(config.lead_vehicle)
    distance_estimator = DistanceEstimator(config.camera, config.estimation)
    speed_estimator = SpeedEstimator(config.estimation)
    ttc_estimator = TTCEstimator(config.ttc)
    overlay = DetectionOverlay()

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for timestamp in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        tracked = tracker.track(frame)
        speed_estimator.prune_inactive_tracks({t.track_id for t in tracked})
        lead = selector.select(tracked, 640, 480)
        distance = distance_estimator.estimate(lead, 640)
        speed = speed_estimator.estimate(distance, timestamp)
        ttc = ttc_estimator.estimate(distance, speed)

    annotated = overlay.draw_tracked_with_lead(
        frame,
        tracked,
        lead,
        lead_distance_m=distance.smoothed_distance_m if distance else None,
        lead_relative_speed_mps=speed.smoothed_speed_mps if speed else None,
        lead_ttc_display=ttc.display_ttc if ttc else None,
    )

    assert annotated.shape == frame.shape


def test_pipeline_with_vehicle_video_if_available():
    """Process frames from a real clip when available."""
    sample_video = Path("data/samples/drive.mp4")
    if not sample_video.is_file():
        pytest.skip("No sample video available")

    config = load_config()
    detector = YoloDetector(config.model)
    tracker = VehicleTracker.from_detector(detector, config.tracker)
    selector = LeadVehicleSelector(config.lead_vehicle)
    distance_estimator = DistanceEstimator(config.camera, config.estimation)
    speed_estimator = SpeedEstimator(config.estimation)
    ttc_estimator = TTCEstimator(config.ttc)

    with VideoReader(sample_video) as reader:
        for frame_index, frame_data in enumerate(reader):
            if frame_index >= 10:
                break
            tracked = tracker.track(frame_data.frame)
            speed_estimator.prune_inactive_tracks({t.track_id for t in tracked})
            lead = selector.select(tracked, reader.width, reader.height)
            distance = distance_estimator.estimate(lead, reader.width)
            speed = speed_estimator.estimate(distance, frame_data.timestamp_seconds)
            ttc_estimator.estimate(distance, speed)

    assert True
