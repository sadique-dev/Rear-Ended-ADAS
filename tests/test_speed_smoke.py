"""Smoke tests for relative speed estimation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.detection import YoloDetector
from src.estimation import DistanceEstimate, DistanceEstimator, SpeedEstimator
from src.io import VideoReader
from src.pipeline import LeadVehicleSelector
from src.tracking import VehicleTracker
from src.utils.config import EstimationConfig, load_config
from src.visualization import DetectionOverlay


def _estimation_config(
    warmup_frames: int = 2,
    speed_ema_alpha: float = 0.3,
    speed_min_mps: float = 0.01,
) -> EstimationConfig:
    """Build estimation config tuned for speed unit tests."""
    return EstimationConfig(
        distance_ema_alpha=0.3,
        distance_min_m=1.0,
        distance_max_m=100.0,
        speed_ema_alpha=speed_ema_alpha,
        speed_min_mps=speed_min_mps,
        speed_warmup_frames=warmup_frames,
        ttc_max_display_s=10.0,
    )


def _distance(track_id: int, smoothed_m: float) -> DistanceEstimate:
    """Create a synthetic distance estimate."""
    return DistanceEstimate(
        raw_distance_m=smoothed_m,
        smoothed_distance_m=smoothed_m,
        track_id=track_id,
        class_name="car",
    )


@pytest.fixture
def speed_estimator() -> SpeedEstimator:
    """Speed estimator with minimal warm-up for tests."""
    return SpeedEstimator(_estimation_config(warmup_frames=2))


def test_approaching_vehicle_gives_negative_speed(speed_estimator):
    """Decreasing distance should produce negative relative speed."""
    speed_estimator.estimate(_distance(1, 20.0), timestamp_seconds=0.0)
    result = speed_estimator.estimate(_distance(1, 18.0), timestamp_seconds=1.0)

    assert result is not None
    assert result.raw_speed_mps < 0.0
    assert result.raw_speed_mps == pytest.approx(-2.0)


def test_receding_vehicle_gives_positive_speed(speed_estimator):
    """Increasing distance should produce positive relative speed."""
    speed_estimator.estimate(_distance(1, 18.0), timestamp_seconds=0.0)
    result = speed_estimator.estimate(_distance(1, 20.0), timestamp_seconds=1.0)

    assert result is not None
    assert result.raw_speed_mps > 0.0
    assert result.raw_speed_mps == pytest.approx(2.0)


def test_constant_distance_gives_near_zero_speed(speed_estimator):
    """Stable distance should produce near-zero relative speed."""
    speed_estimator.estimate(_distance(1, 15.0), timestamp_seconds=0.0)
    result = speed_estimator.estimate(_distance(1, 15.0), timestamp_seconds=1.0)

    assert result is not None
    assert result.raw_speed_mps == pytest.approx(0.0)
    assert result.smoothed_speed_mps == pytest.approx(0.0)


def test_smoothing_reduces_sudden_spikes():
    """EMA smoothing should dampen a large raw speed spike."""
    estimator = SpeedEstimator(
        _estimation_config(warmup_frames=2, speed_ema_alpha=0.2)
    )

    estimator.estimate(_distance(1, 20.0), timestamp_seconds=0.0)
    baseline = estimator.estimate(_distance(1, 19.0), timestamp_seconds=1.0)
    assert baseline is not None

    spike = estimator.estimate(_distance(1, 10.0), timestamp_seconds=2.0)
    assert spike is not None

    assert abs(spike.smoothed_speed_mps) < abs(spike.raw_speed_mps)


def test_prune_inactive_tracks_clears_history(speed_estimator):
    """Inactive track history should be removed when tracks disappear."""
    speed_estimator.estimate(_distance(1, 20.0), timestamp_seconds=0.0)
    speed_estimator.estimate(_distance(2, 15.0), timestamp_seconds=0.0)
    speed_estimator.prune_inactive_tracks({1})

    assert 1 in speed_estimator._track_states
    assert 2 not in speed_estimator._track_states


def test_full_pipeline_runs_without_errors():
    """End-to-end pipeline through speed estimation should run cleanly."""
    config = load_config()
    detector = YoloDetector(config.model)
    tracker = VehicleTracker.from_detector(detector, config.tracker)
    selector = LeadVehicleSelector(config.lead_vehicle)
    distance_estimator = DistanceEstimator(config.camera, config.estimation)
    speed_estimator = SpeedEstimator(config.estimation)
    overlay = DetectionOverlay()

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for timestamp in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        tracked = tracker.track(frame)
        speed_estimator.prune_inactive_tracks({t.track_id for t in tracked})
        lead = selector.select(tracked, frame_width=640, frame_height=480)
        distance = distance_estimator.estimate(lead, frame_width=640)
        speed = speed_estimator.estimate(distance, timestamp_seconds=timestamp)

    display_distance = distance.smoothed_distance_m if distance else None
    display_speed = speed.smoothed_speed_mps if speed else None
    annotated = overlay.draw_tracked_with_lead(
        frame,
        tracked,
        lead,
        lead_distance_m=display_distance,
        lead_relative_speed_mps=display_speed,
    )

    assert annotated.shape == frame.shape


def test_pipeline_with_vehicle_video_if_available():
    """Process multiple frames from a real clip when available."""
    sample_video = Path("data/samples/drive.mp4")
    if not sample_video.is_file():
        pytest.skip("No sample video available")

    config = load_config()
    detector = YoloDetector(config.model)
    tracker = VehicleTracker.from_detector(detector, config.tracker)
    selector = LeadVehicleSelector(config.lead_vehicle)
    distance_estimator = DistanceEstimator(config.camera, config.estimation)
    speed_estimator = SpeedEstimator(config.estimation)
    overlay = DetectionOverlay()

    with VideoReader(sample_video) as reader:
        for frame_index, frame_data in enumerate(reader):
            if frame_index >= 10:
                break
            tracked = tracker.track(frame_data.frame)
            speed_estimator.prune_inactive_tracks({t.track_id for t in tracked})
            lead = selector.select(tracked, reader.width, reader.height)
            distance = distance_estimator.estimate(lead, reader.width)
            speed_estimator.estimate(distance, frame_data.timestamp_seconds)

    assert True
