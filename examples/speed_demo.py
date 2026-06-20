"""Demonstrate relative speed estimation on the lead vehicle.

Pipeline:
    VideoReader -> YOLO + ByteTrack -> Lead -> Distance -> Speed -> Overlay

Usage:
    python examples/speed_demo.py --input data/samples/drive.mp4
    python examples/speed_demo.py --input data/samples/drive.mp4 --max-frames 60 --no-display
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection import YoloDetector
from src.estimation import DistanceEstimator, SpeedEstimator
from src.io import VideoReader
from src.pipeline import LeadVehicleSelector
from src.tracking import VehicleTracker
from src.utils.config import DEFAULT_CONFIG_PATH, load_config
from src.utils.logger import setup_logging
from src.visualization import DetectionOverlay


def main() -> int:
    """Run the pipeline through relative speed estimation."""
    parser = argparse.ArgumentParser(
        description="Estimate relative speed to the lead vehicle in dashcam video."
    )
    parser.add_argument("--input", "-i", required=True, help="Input video path.")
    parser.add_argument(
        "--config",
        "-c",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional YOLO model override.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30,
        help="Frames to process (default: 30 for speed warm-up).",
    )
    parser.add_argument(
        "--save",
        default="output.jpg",
        help="Path to save the last annotated frame.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Skip the OpenCV preview window.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(level=config.logging.level, log_file=config.logging.log_file)

    detector = YoloDetector(config.model, model_path=args.model)
    tracker = VehicleTracker.from_detector(detector, config.tracker)
    selector = LeadVehicleSelector(config.lead_vehicle)
    distance_estimator = DistanceEstimator(config.camera, config.estimation)
    speed_estimator = SpeedEstimator(config.estimation)
    overlay = DetectionOverlay()

    last_annotated = None
    last_distance = None
    last_speed = None

    with VideoReader(args.input) as reader:
        for frame_index, frame_data in enumerate(reader):
            if frame_index >= args.max_frames:
                break

            tracked = tracker.track(frame_data.frame)
            active_ids = {track.track_id for track in tracked}
            speed_estimator.prune_inactive_tracks(active_ids)

            lead = selector.select(
                tracked,
                frame_width=reader.width,
                frame_height=reader.height,
            )
            distance = distance_estimator.estimate(lead, frame_width=reader.width)
            speed = speed_estimator.estimate(
                distance,
                timestamp_seconds=frame_data.timestamp_seconds,
            )

            display_distance = (
                distance.smoothed_distance_m if distance is not None else None
            )
            display_speed = (
                speed.smoothed_speed_mps if speed is not None else None
            )
            last_annotated = overlay.draw_tracked_with_lead(
                frame_data.frame,
                tracked,
                lead,
                lead_distance_m=display_distance,
                lead_relative_speed_mps=display_speed,
            )
            last_distance = distance
            last_speed = speed

            if speed is not None:
                print(
                    f"Frame {frame_data.index:4d} | "
                    f"dist={display_distance:.1f} m | "
                    f"speed={display_speed:.2f} m/s"
                )

    if last_annotated is None:
        print("No frames processed.")
        return 1

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), last_annotated)

    if last_speed is not None and last_distance is not None:
        print(
            f"Final: track_id={last_speed.track_id} | "
            f"distance={last_distance.smoothed_distance_m:.2f} m | "
            f"raw_speed={last_speed.raw_speed_mps:.2f} m/s | "
            f"smoothed_speed={last_speed.smoothed_speed_mps:.2f} m/s"
        )
    else:
        print("Speed: unavailable (warm-up or no lead vehicle)")

    print(f"Saved annotated frame to: {save_path.resolve()}")

    if not args.no_display:
        window_name = "Relative Speed Preview (press any key to close)"
        cv2.imshow(window_name, last_annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    tracker.reset()
    distance_estimator.reset()
    speed_estimator.reset()
    return 0


if __name__ == "__main__":
    sys.exit(main())
