"""Demonstrate Time-To-Collision estimation on the lead vehicle.

Pipeline:
    VideoReader -> Track -> Lead -> Distance -> Speed -> TTC -> Overlay

Usage:
    python examples/ttc_demo.py --input data/samples/drive.mp4
    python examples/ttc_demo.py --input data/samples/drive.mp4 --max-frames 60 --no-display
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
from src.estimation import (
    DistanceEstimator,
    SpeedEstimator,
    TTCEstimator,
)
from src.io import VideoReader
from src.pipeline import LeadVehicleSelector
from src.tracking import VehicleTracker
from src.utils.config import DEFAULT_CONFIG_PATH, load_config
from src.utils.logger import setup_logging
from src.visualization import DetectionOverlay


def main() -> int:
    """Run the pipeline through TTC estimation."""
    parser = argparse.ArgumentParser(
        description="Estimate Time-To-Collision for the lead vehicle."
    )
    parser.add_argument("--input", "-i", required=True, help="Input video path.")
    parser.add_argument(
        "--config",
        "-c",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config file.",
    )
    parser.add_argument("--model", default=None, help="Optional YOLO override.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30,
        help="Frames to process (default: 30).",
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
    ttc_estimator = TTCEstimator(config.ttc)
    overlay = DetectionOverlay()

    last_annotated = None
    last_ttc = None

    with VideoReader(args.input) as reader:
        for frame_index, frame_data in enumerate(reader):
            if frame_index >= args.max_frames:
                break

            tracked = tracker.track(frame_data.frame)
            speed_estimator.prune_inactive_tracks({t.track_id for t in tracked})

            lead = selector.select(tracked, reader.width, reader.height)
            distance = distance_estimator.estimate(lead, reader.width)
            speed = speed_estimator.estimate(
                distance,
                frame_data.timestamp_seconds,
            )
            ttc = ttc_estimator.estimate(distance, speed)

            last_annotated = overlay.draw_tracked_with_lead(
                frame_data.frame,
                tracked,
                lead,
                lead_distance_m=(
                    distance.smoothed_distance_m if distance else None
                ),
                lead_relative_speed_mps=(
                    speed.smoothed_speed_mps if speed else None
                ),
                lead_ttc_display=ttc.display_ttc if ttc else None,
            )
            last_ttc = ttc

            if ttc is not None:
                raw_display = (
                    f"{ttc.raw_ttc_seconds:.2f}s"
                    if ttc.is_valid
                    else "INF"
                )
                print(
                    f"Frame {frame_data.index:4d} | "
                    f"valid={ttc.is_valid} | "
                    f"raw={raw_display} | "
                    f"display={ttc.display_ttc}"
                )

    if last_annotated is None:
        print("No frames processed.")
        return 1

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), last_annotated)

    if last_ttc is not None:
        print(
            f"Final TTC: valid={last_ttc.is_valid} | "
            f"raw={last_ttc.raw_ttc_seconds} | display={last_ttc.display_ttc}"
        )
    else:
        print("TTC: unavailable")

    print(f"Saved annotated frame to: {save_path.resolve()}")

    if not args.no_display:
        cv2.imshow("TTC Preview (press any key to close)", last_annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    tracker.reset()
    distance_estimator.reset()
    speed_estimator.reset()
    return 0


if __name__ == "__main__":
    sys.exit(main())
