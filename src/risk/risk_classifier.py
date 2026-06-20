"""Collision risk classification based on Time-To-Collision."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from src.estimation.ttc_estimator import TTCEstimate
from src.utils.config import RiskConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

# BGR display colors for each risk level.
RISK_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "SAFE": (0, 255, 0),       # Green
    "CAUTION": (0, 255, 255),  # Yellow
    "DANGER": (0, 0, 255),     # Red
}


class RiskLevel(str, Enum):
    """Discrete collision risk levels."""

    SAFE = "SAFE"
    CAUTION = "CAUTION"
    DANGER = "DANGER"


@dataclass(frozen=True)
class RiskAssessment:
    """Collision risk classification result.

    Attributes:
        level: Risk level (``SAFE``, ``CAUTION``, or ``DANGER``).
        display_color_bgr: OpenCV BGR color tuple for overlays.
        message: Human-readable warning message for the driver.
    """

    level: RiskLevel
    display_color_bgr: tuple[int, int, int]
    message: str


class RiskClassifier:
    """Classify collision risk from a TTC estimate.

    Maps Time-To-Collision into three actionable levels using configurable
    thresholds:

    - **DANGER** — ``TTC <= danger_ttc_max_seconds``
    - **CAUTION** — ``danger < TTC <= caution_ttc_max_seconds``
    - **SAFE** — ``TTC > caution_ttc_max_seconds`` or invalid / infinite TTC

    Example:
        >>> classifier = RiskClassifier(config.risk)
        >>> assessment = classifier.classify(ttc_estimate)
    """

    def __init__(self, risk_config: RiskConfig) -> None:
        """Initialize the classifier with thresholds from config.

        Args:
            risk_config: TTC boundary values and warning messages.
        """
        self._danger_ttc_max = risk_config.danger_ttc_max_seconds
        self._caution_ttc_max = risk_config.caution_ttc_max_seconds
        self._messages = risk_config.messages

    def classify(self, ttc: TTCEstimate | None) -> RiskAssessment:
        """Classify collision risk from a TTC estimate.

        Args:
            ttc: TTC estimate from ``TTCEstimator``, or ``None`` when TTC
                could not be computed.

        Returns:
            ``RiskAssessment`` with level, display color, and message.
            Invalid or infinite TTC yields ``SAFE``.
        """
        if ttc is None or not ttc.is_valid or math.isinf(ttc.raw_ttc_seconds):
            logger.debug("TTC invalid or missing — classifying as SAFE")
            return self._build_assessment(RiskLevel.SAFE)

        ttc_seconds = ttc.raw_ttc_seconds

        if ttc_seconds <= self._danger_ttc_max:
            level = RiskLevel.DANGER
        elif ttc_seconds <= self._caution_ttc_max:
            level = RiskLevel.CAUTION
        else:
            level = RiskLevel.SAFE

        logger.debug(
            "Risk classified track_id=%d ttc=%.2fs -> %s",
            ttc.track_id,
            ttc_seconds,
            level.value,
        )
        return self._build_assessment(level)

    def _build_assessment(self, level: RiskLevel) -> RiskAssessment:
        """Construct a ``RiskAssessment`` for the given level."""
        message_map = {
            RiskLevel.SAFE: self._messages.safe,
            RiskLevel.CAUTION: self._messages.caution,
            RiskLevel.DANGER: self._messages.danger,
        }
        return RiskAssessment(
            level=level,
            display_color_bgr=RISK_COLORS_BGR[level.value],
            message=message_map[level],
        )
