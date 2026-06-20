"""Full ADAS processing pipeline orchestrating all system modules."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.detection import YoloDetector
from src.estimation import DistanceEstimator, SpeedEstimator, TTCEstimator
from src.io import FrameData, VideoReader, VideoWriter
from src.pipeline.lead_vehicle import LeadVehicleSelector
from src.risk import RiskAssessment, RiskClassifier
from src.tracking import TrackedDetection, VehicleTracker
from src.utils.config import AppConfig
from src.utils.logger import get_logger
from src.visualization import DetectionOverlay

logger = get_logger(__name__)


@dataclass(frozen=True)
class PipelineStats:
    """Summary statistics collected after a pipeline run.

    Attributes:
        total_frames: Number of frames processed.
        processing_time_seconds: Wall-clock elapsed time in seconds.
        average_fps: Effective processing throughput (frames / seconds).
    """

    total_frames: int
    processing_time_seconds: float
    average_fps: float


@dataclass(frozen=True)
class FrameResult:
    """Output of processing a single video frame.

    Attributes:
        frame_index: Zero-based index of the processed frame.
        annotated_frame: BGR image with overlays drawn.
        tracked_count: Number of tracked vehicles in the frame.
        lead_vehicle: Selected lead vehicle, if any.
        risk: Collision risk assessment for the lead vehicle.
    """

    frame_index: int
    annotated_frame: np.ndarray
    tracked_count: int
    lead_vehicle: TrackedDetection | None
    risk: RiskAssessment


class ADASPipeline:
    """End-to-end Rear-End ADAS collision warning pipeline.

    Wires together video I/O, detection, tracking, estimation, risk
    classification, and visualization into a single processing loop.

    Example:
        >>> pipeline = ADASPipeline(config, input_path, output_path)
        >>> stats = pipeline.run()
        >>> pipeline.cleanup()
    """

    def __init__(
        self,
        config: AppConfig,
        input_path: str | Path,
        output_path: str | Path,
        model_path: str | None = None,
        display: bool = False,
        max_frames: int | None = None,
    ) -> None:
        """Configure the pipeline (components are created in ``initialize``).

        Args:
            config: Application configuration loaded from YAML.
            input_path: Path to the input video file.
            output_path: Path for the annotated output video.
            model_path: Optional override for YOLO model weights.
            display: When ``True``, show a live preview window.
            max_frames: Optional cap on frames processed (useful for tests).
        """
        self._config = config
        self._input_path = Path(input_path)
        self._output_path = Path(output_path)
        self._model_path = model_path
        self._display = display
        self._max_frames = max_frames

        self._reader: VideoReader | None = None
        self._writer: VideoWriter | None = None
        self._detector: YoloDetector | None = None
        self._tracker: VehicleTracker | None = None
        self._lead_selector: LeadVehicleSelector | None = None
        self._distance_estimator: DistanceEstimator | None = None
        self._speed_estimator: SpeedEstimator | None = None
        self._ttc_estimator: TTCEstimator | None = None
        self._risk_classifier: RiskClassifier | None = None
        self._overlay: DetectionOverlay | None = None

    def initialize(self) -> None:
        """Load models, open video I/O, and prepare all processing modules."""
        if self._reader is not None:
            return

        logger.info("Initializing ADAS pipeline")
        logger.info("Input:  %s", self._input_path.resolve())
        logger.info("Output: %s", self._output_path.resolve())

        self._detector = YoloDetector(self._config.model, model_path=self._model_path)
        self._tracker = VehicleTracker.from_detector(self._detector, self._config.tracker)
        self._lead_selector = LeadVehicleSelector(self._config.lead_vehicle)
        self._distance_estimator = DistanceEstimator(
            self._config.camera,
            self._config.estimation,
        )
        self._speed_estimator = SpeedEstimator(self._config.estimation)
        self._ttc_estimator = TTCEstimator(self._config.ttc)
        self._risk_classifier = RiskClassifier(self._config.risk)
        self._overlay = DetectionOverlay()

        self._reader = VideoReader(self._input_path)
        self._writer = VideoWriter.from_reader_metadata(
            output_path=self._output_path,
            width=self._reader.width,
            height=self._reader.height,
            fps=self._reader.fps,
            codec=self._config.io.output_codec,
            output_fps=self._config.io.output_fps,
        )

        logger.info(
            "Pipeline ready (%dx%d @ %.2f FPS)",
            self._reader.width,
            self._reader.height,
            self._reader.fps,
        )

    def process_frame(self, frame_data: FrameData) -> FrameResult:
        """Run the full ADAS stack on a single frame.

        Args:
            frame_data: Decoded frame with index and timestamp metadata.

        Returns:
            ``FrameResult`` containing the annotated frame and metadata.

        Raises:
            RuntimeError: If ``initialize`` has not been called.
        """
        self._ensure_initialized()

        tracked = self._tracker.track(frame_data.frame)
        active_ids = {track.track_id for track in tracked}
        self._speed_estimator.prune_inactive_tracks(active_ids)

        lead = self._lead_selector.select(
            tracked,
            frame_width=self._reader.width,
            frame_height=self._reader.height,
        )
        distance = self._distance_estimator.estimate(lead, self._reader.width)
        speed = self._speed_estimator.estimate(
            distance,
            frame_data.timestamp_seconds,
        )
        ttc = self._ttc_estimator.estimate(distance, speed)
        risk = self._risk_classifier.classify(ttc)

        annotated = self._overlay.draw_tracked_with_lead(
            frame_data.frame,
            tracked,
            lead,
            lead_distance_m=distance.smoothed_distance_m if distance else None,
            lead_relative_speed_mps=speed.smoothed_speed_mps if speed else None,
            lead_ttc_display=ttc.display_ttc if ttc else None,
            risk_assessment=risk,
        )

        return FrameResult(
            frame_index=frame_data.index,
            annotated_frame=annotated,
            tracked_count=len(tracked),
            lead_vehicle=lead,
            risk=risk,
        )

    def run(self) -> PipelineStats:
        """Process the entire input video and write the annotated output.

        Returns:
            ``PipelineStats`` with frame count, elapsed time, and average FPS.

        Raises:
            RuntimeError: If the pipeline is not initialized or has no frames.
        """
        if self._reader is None:
            self.initialize()

        self._ensure_initialized()
        start_time = time.perf_counter()
        total_frames = 0

        try:
            for frame_data in self._reader:
                if self._max_frames is not None and total_frames >= self._max_frames:
                    break

                result = self.process_frame(frame_data)
                self._writer.write(result.annotated_frame)
                total_frames += 1

                if self._display:
                    cv2.imshow("Rear-End ADAS", result.annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("Processing interrupted by user")
                        break

                if total_frames % 30 == 0:
                    logger.info("Processed %d frames", total_frames)

        finally:
            if self._display:
                cv2.destroyAllWindows()

        elapsed = time.perf_counter() - start_time
        if total_frames == 0:
            raise RuntimeError("No frames were processed")

        average_fps = total_frames / elapsed if elapsed > 0 else 0.0
        stats = PipelineStats(
            total_frames=total_frames,
            processing_time_seconds=elapsed,
            average_fps=average_fps,
        )

        self._log_stats(stats)
        return stats

    def cleanup(self) -> None:
        """Release video handles and reset temporal estimator state."""
        if self._reader is not None:
            self._reader.release()
            self._reader = None

        if self._writer is not None:
            self._writer.release()
            self._writer = None

        if self._tracker is not None:
            self._tracker.reset()

        if self._distance_estimator is not None:
            self._distance_estimator.reset()

        if self._speed_estimator is not None:
            self._speed_estimator.reset()

        logger.info("Pipeline resources released")

    def _ensure_initialized(self) -> None:
        """Raise if core components are not ready."""
        required = (
            self._reader,
            self._writer,
            self._tracker,
            self._lead_selector,
            self._distance_estimator,
            self._speed_estimator,
            self._ttc_estimator,
            self._risk_classifier,
            self._overlay,
        )
        if any(component is None for component in required):
            raise RuntimeError("Pipeline not initialized — call initialize() first")

    @staticmethod
    def _log_stats(stats: PipelineStats) -> None:
        """Log and print pipeline summary statistics."""
        logger.info(
            "Processing complete — frames=%d, time=%.2fs, avg_fps=%.2f",
            stats.total_frames,
            stats.processing_time_seconds,
            stats.average_fps,
        )
        print("\n--- ADAS Processing Statistics ---")
        print(f"Total Frames     : {stats.total_frames}")
        print(f"Processing Time  : {stats.processing_time_seconds:.2f} s")
        print(f"Average FPS      : {stats.average_fps:.2f}")
        print("--------------------------------\n")
