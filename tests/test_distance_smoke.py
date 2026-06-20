"""Smoke tests for monocular distance estimation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.detection import YoloDetector
from src.estimation import DistanceEstimator
from src.io import VideoReader
from src.pipeline import LeadVehicleSelector
from src.tracking import TrackedDetection, VehicleTracker
from src.utils.config import (
    CameraConfig,
    EstimationConfig,
    VehicleWidthsConfig,
    load_config,
)
from src.visualization import DetectionOverlay


def _camera_config(focal_length_px: float = 800.0) -> CameraConfig:
    """Build a camera config with fixed focal length for deterministic tests."""
    return CameraConfig(
        fov_horizontal_deg=90.0,
        focal_length_px=focal_length_px,
        vehicle_widths_m=VehicleWidthsConfig(
            car=1.8,
            motorcycle=0.8,
            bus=2.5,
            truck=2.5,
        ),
        assumed_vehicle_width_m=1.8,
    )


def _estimation_config(alpha: float = 0.3) -> EstimationConfig:
    """Build estimation config for distance tests."""
    return EstimationConfig(
        distance_ema_alpha=alpha,
        distance_min_m=1.0,
        distance_max_m=100.0,
        speed_ema_alpha=0.2,
        speed_min_mps=0.5,
        speed_warmup_frames=5,
        ttc_max_display_s=10.0,
    )


def _make_lead(
    bbox: tuple[int, int, int, int],
    track_id: int = 1,
    class_name: str = "car",
) -> TrackedDetection:
    """Create a synthetic lead vehicle for unit tests."""
    return TrackedDetection(
        track_id=track_id,
        bbox=bbox,
        confidence=0.9,
        class_id=2,
        class_name=class_name,
    )


@pytest.fixture
def estimator() -> DistanceEstimator:
    """Distance estimator with fixed intrinsics for repeatable tests."""
    return DistanceEstimator(_camera_config(), _estimation_config())


def test_larger_bbox_produces_shorter_distance(estimator):
    """A wider bounding box should map to a closer vehicle."""
    large_bbox = _make_lead((100, 100, 300, 200))   # width = 200 px
    small_bbox = _make_lead((100, 100, 200, 200))   # width = 100 px

    large_result = estimator.estimate(large_bbox, frame_width=640)
    small_result = estimator.estimate(small_bbox, frame_width=640)

    assert large_result is not None
    assert small_result is not None
    assert large_result.raw_distance_m < small_result.raw_distance_m


def test_smaller_bbox_produces_larger_distance(estimator):
    """A narrower bounding box should map to a farther vehicle."""
    # D = (1.8 * 800) / w_bbox
    narrow = estimator.estimate(_make_lead((0, 0, 80, 100)), frame_width=640)
    wide = estimator.estimate(_make_lead((0, 0, 160, 100)), frame_width=640)

    assert narrow is not None and wide is not None
    assert narrow.raw_distance_m > wide.raw_distance_m
    assert narrow.raw_distance_m == pytest.approx(18.0)
    assert wide.raw_distance_m == pytest.approx(9.0)


def test_smoothing_reduces_sudden_jumps():
    """EMA smoothing should produce a smaller step than a raw measurement jump."""
    estimator = DistanceEstimator(
        _camera_config(),
        _estimation_config(alpha=0.3),
    )
    lead = _make_lead((100, 100, 200, 200), track_id=7)

    first = estimator.estimate(lead, frame_width=640)
    assert first is not None

    # Simulate the vehicle appearing farther away (narrower bbox).
    farther_lead = _make_lead((100, 100, 160, 200), track_id=7)
    second = estimator.estimate(farther_lead, frame_width=640)
    assert second is not None

    raw_jump = abs(second.raw_distance_m - first.raw_distance_m)
    smooth_jump = abs(second.smoothed_distance_m - first.smoothed_distance_m)

    assert smooth_jump < raw_jump


def test_full_pipeline_runs_without_errors():
    """Detection through distance estimation and overlay should run end-to-end."""
    config = load_config()
    detector = YoloDetector(config.model)
    tracker = VehicleTracker.from_detector(detector, config.tracker)
    selector = LeadVehicleSelector(config.lead_vehicle)
    estimator = DistanceEstimator(config.camera, config.estimation)
    overlay = DetectionOverlay()

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracked = tracker.track(frame)
    lead = selector.select(tracked, frame_width=640, frame_height=480)
    distance = estimator.estimate(lead, frame_width=640)
    display_distance = distance.smoothed_distance_m if distance else None
    annotated = overlay.draw_tracked_with_lead(
        frame,
        tracked,
        lead,
        lead_distance_m=display_distance,
    )

    assert annotated.shape == frame.shape


def test_pipeline_with_vehicle_frame_if_available():
    """Run full pipeline on a real frame when a vehicle asset exists."""
    frame = None
    try:
        import ultralytics

        asset_path = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
        if asset_path.is_file():
            frame = cv2.imread(str(asset_path))
    except ImportError:
        pass

    if frame is None:
        sample_video = Path("data/samples/drive.mp4")
        if sample_video.is_file():
            with VideoReader(sample_video) as reader:
                frame_data = next(iter(reader), None)
                if frame_data is not None:
                    frame = frame_data.frame

    if frame is None:
        pytest.skip("No vehicle test frame available")

    config = load_config()
    detector = YoloDetector(config.model)
    tracker = VehicleTracker.from_detector(detector, config.tracker)
    selector = LeadVehicleSelector(config.lead_vehicle)
    estimator = DistanceEstimator(config.camera, config.estimation)
    overlay = DetectionOverlay()

    tracker.reset()
    estimator.reset()
    tracked = tracker.track(frame)
    height, width = frame.shape[:2]
    lead = selector.select(tracked, frame_width=width, frame_height=height)
    distance = estimator.estimate(lead, frame_width=width)
    display_distance = distance.smoothed_distance_m if distance else None
    annotated = overlay.draw_tracked_with_lead(
        frame,
        tracked,
        lead,
        lead_distance_m=display_distance,
    )

    assert annotated.shape == frame.shape
