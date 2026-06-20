"""Smoke tests for ByteTrack vehicle tracking."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.detection import YoloDetector
from src.io import VideoReader
from src.tracking import VehicleTracker
from src.utils.config import load_config
from src.visualization import DetectionOverlay


@pytest.fixture(scope="module")
def app_config():
    """Load application config once for the test module."""
    return load_config()


@pytest.fixture(scope="module")
def detector(app_config):
    """Shared YOLO detector instance."""
    return YoloDetector(app_config.model)


@pytest.fixture(scope="module")
def tracker(detector, app_config):
    """Shared vehicle tracker instance."""
    return VehicleTracker.from_detector(detector, app_config.tracker)


def _vehicle_test_frame() -> np.ndarray | None:
    """Return a frame likely to contain vehicles, or None if unavailable."""
    try:
        import ultralytics

        asset_path = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
        if asset_path.is_file():
            frame = cv2.imread(str(asset_path))
            if frame is not None:
                return frame
    except ImportError:
        pass

    sample_video = Path("data/samples/drive.mp4")
    if sample_video.is_file():
        with VideoReader(sample_video) as reader:
            frame_data = next(iter(reader), None)
            if frame_data is not None:
                return frame_data.frame

    return None


def test_tracker_initializes(tracker, app_config):
    """Tracker should initialize with config-driven settings."""
    assert tracker.persist == app_config.tracker.persist
    assert "bytetrack" in repr(tracker).lower() or app_config.tracker.type in repr(tracker)


def test_track_on_empty_frame_returns_list(tracker):
    """Tracking a blank frame should return an empty list without error."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracked = tracker.track(frame)
    assert isinstance(tracked, list)
    assert tracked == []


def test_track_id_consistency_on_consecutive_frames(tracker):
    """Track IDs should remain stable when the same scene is tracked twice."""
    frame = _vehicle_test_frame()
    if frame is None:
        pytest.skip("No vehicle test frame available (bus.jpg or sample video)")

    tracker.reset()
    first_pass = tracker.track(frame)
    second_pass = tracker.track(frame)

    if not first_pass or not second_pass:
        pytest.skip("No vehicles detected in test frame")

    first_ids = {detection.track_id for detection in first_pass}
    second_ids = {detection.track_id for detection in second_pass}

    assert first_ids == second_ids


def test_full_pipeline_runs_without_errors(tracker, app_config):
    """Detection + tracking + overlay pipeline should run end-to-end."""
    frame = _vehicle_test_frame()
    if frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

    tracked = tracker.track(frame)
    overlay = DetectionOverlay()
    annotated = overlay.draw_tracked_detections(frame, tracked)

    assert annotated.shape == frame.shape
    assert annotated.dtype == np.uint8


def test_tracker_reset_clears_state(tracker):
    """Reset should allow a clean tracker session without errors."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracker.track(frame)
    tracker.reset()
    tracked = tracker.track(frame)
    assert isinstance(tracked, list)
