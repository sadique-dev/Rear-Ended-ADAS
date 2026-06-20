"""Smoke tests for collision risk classification."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.detection import YoloDetector
from src.estimation import (
    DistanceEstimator,
    SpeedEstimator,
    TTCEstimate,
    TTCEstimator,
)
from src.pipeline import LeadVehicleSelector
from src.risk import RiskClassifier, RiskLevel
from src.tracking import VehicleTracker
from src.utils.config import RiskConfig, RiskMessagesConfig, load_config
from src.visualization import DetectionOverlay


def _risk_config(
    danger_max: float = 2.0,
    caution_max: float = 5.0,
) -> RiskConfig:
    """Build risk config for unit tests."""
    return RiskConfig(
        danger_ttc_max_seconds=danger_max,
        caution_ttc_max_seconds=caution_max,
        messages=RiskMessagesConfig(
            safe="Safe Following Distance",
            caution="Reduce Speed",
            danger="Collision Warning!",
        ),
    )


def _ttc(
    raw_seconds: float,
    is_valid: bool = True,
    track_id: int = 1,
) -> TTCEstimate:
    """Synthetic TTC estimate."""
    display = f"{raw_seconds:.1f} s" if is_valid else "INF"
    return TTCEstimate(
        raw_ttc_seconds=raw_seconds if is_valid else math.inf,
        display_ttc=display,
        is_valid=is_valid,
        track_id=track_id,
    )


@pytest.fixture
def classifier() -> RiskClassifier:
    """Default risk classifier for tests."""
    return RiskClassifier(_risk_config())


def test_safe_classification(classifier):
    """TTC above caution threshold should be SAFE."""
    result = classifier.classify(_ttc(6.0))

    assert result.level == RiskLevel.SAFE
    assert result.message == "Safe Following Distance"
    assert result.display_color_bgr == (0, 255, 0)


def test_caution_classification(classifier):
    """TTC within caution band should be CAUTION."""
    result = classifier.classify(_ttc(4.0))

    assert result.level == RiskLevel.CAUTION
    assert result.message == "Reduce Speed"
    assert result.display_color_bgr == (0, 255, 255)


def test_danger_classification(classifier):
    """TTC at or below danger threshold should be DANGER."""
    result = classifier.classify(_ttc(1.5))

    assert result.level == RiskLevel.DANGER
    assert result.message == "Collision Warning!"
    assert result.display_color_bgr == (0, 0, 255)


def test_invalid_ttc_returns_safe(classifier):
    """Invalid or infinite TTC should classify as SAFE."""
    invalid = classifier.classify(_ttc(0.0, is_valid=False))
    none_result = classifier.classify(None)

    assert invalid.level == RiskLevel.SAFE
    assert none_result.level == RiskLevel.SAFE


def test_threshold_boundaries(classifier):
    """Verify exact threshold boundary behaviour."""
    assert classifier.classify(_ttc(2.0)).level == RiskLevel.DANGER
    assert classifier.classify(_ttc(2.001)).level == RiskLevel.CAUTION
    assert classifier.classify(_ttc(5.0)).level == RiskLevel.CAUTION
    assert classifier.classify(_ttc(5.001)).level == RiskLevel.SAFE


def test_full_pipeline_runs_without_errors():
    """End-to-end pipeline through risk classification should run cleanly."""
    config = load_config()
    detector = YoloDetector(config.model)
    tracker = VehicleTracker.from_detector(detector, config.tracker)
    selector = LeadVehicleSelector(config.lead_vehicle)
    distance_estimator = DistanceEstimator(config.camera, config.estimation)
    speed_estimator = SpeedEstimator(config.estimation)
    ttc_estimator = TTCEstimator(config.ttc)
    risk_classifier = RiskClassifier(config.risk)
    overlay = DetectionOverlay()

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for timestamp in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        tracked = tracker.track(frame)
        speed_estimator.prune_inactive_tracks({t.track_id for t in tracked})
        lead = selector.select(tracked, 640, 480)
        distance = distance_estimator.estimate(lead, 640)
        speed = speed_estimator.estimate(distance, timestamp)
        ttc = ttc_estimator.estimate(distance, speed)
        risk = risk_classifier.classify(ttc)

    annotated = overlay.draw_tracked_with_lead(
        frame,
        tracked,
        lead,
        lead_distance_m=distance.smoothed_distance_m if distance else None,
        lead_relative_speed_mps=speed.smoothed_speed_mps if speed else None,
        lead_ttc_display=ttc.display_ttc if ttc else None,
        risk_assessment=risk,
    )

    assert annotated.shape == frame.shape
    assert risk.level in {RiskLevel.SAFE, RiskLevel.CAUTION, RiskLevel.DANGER}
