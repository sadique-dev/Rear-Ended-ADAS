"""Smoke tests for lead vehicle selection."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.detection import YoloDetector
from src.io import VideoReader
from src.pipeline import LeadVehicleSelector
from src.tracking import TrackedDetection, VehicleTracker
from src.utils.config import LeadVehicleConfig, LeadVehicleROIConfig, load_config
from src.visualization import DetectionOverlay


FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def _make_track(
    track_id: int,
    bbox: tuple[int, int, int, int],
    class_name: str = "car",
) -> TrackedDetection:
    """Create a synthetic tracked detection for unit tests."""
    return TrackedDetection(
        track_id=track_id,
        bbox=bbox,
        confidence=0.9,
        class_id=2,
        class_name=class_name,
    )


@pytest.fixture
def tight_roi_selector() -> LeadVehicleSelector:
    """Selector with a small central ROI for controlled unit tests."""
    config = LeadVehicleConfig(
        roi=LeadVehicleROIConfig(
            points=[
                (0.40, 0.40),
                (0.60, 0.40),
                (0.60, 0.60),
                (0.40, 0.60),
            ]
        )
    )
    return LeadVehicleSelector(config)


def test_no_lead_vehicle_when_all_tracks_outside_roi(tight_roi_selector):
    """Tracks outside the ROI must not produce a lead vehicle."""
    outside_tracks = [
        _make_track(1, (10, 10, 80, 80)),
        _make_track(2, (550, 20, 620, 90)),
    ]

    lead = tight_roi_selector.select(
        outside_tracks,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
    )

    assert lead is None


def test_selects_largest_bbox_inside_roi(tight_roi_selector):
    """The vehicle with the largest bounding-box area in the ROI should win."""
    tracks = [
        _make_track(1, (270, 220, 310, 280)),  # area = 2400, inside ROI
        _make_track(2, (240, 180, 400, 280)),  # area = 16000, inside ROI
        _make_track(3, (10, 10, 80, 80)),      # outside ROI
    ]

    lead = tight_roi_selector.select(
        tracks,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
    )

    assert lead is not None
    assert lead.track_id == 2


def test_full_pipeline_runs_without_errors():
    """Detection + tracking + lead selection + overlay should run end-to-end."""
    config = load_config()
    detector = YoloDetector(config.model)
    tracker = VehicleTracker.from_detector(detector, config.tracker)
    selector = LeadVehicleSelector(config.lead_vehicle)
    overlay = DetectionOverlay()

    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    tracked = tracker.track(frame)
    lead = selector.select(
        tracked,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
    )
    annotated = overlay.draw_tracked_with_lead(frame, tracked, lead)

    assert annotated.shape == frame.shape
    assert annotated.dtype == np.uint8


def test_pipeline_with_vehicle_frame_if_available():
    """Run the pipeline on a real frame when a vehicle asset exists."""
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
    overlay = DetectionOverlay()

    tracker.reset()
    tracked = tracker.track(frame)
    height, width = frame.shape[:2]
    lead = selector.select(tracked, frame_width=width, frame_height=height)
    annotated = overlay.draw_tracked_with_lead(frame, tracked, lead)

    assert annotated.shape == frame.shape
