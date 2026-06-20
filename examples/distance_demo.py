"""Demonstrate monocular distance estimation on the lead vehicle.

Pipeline:
    VideoReader -> YOLO + ByteTrack -> Lead Selection -> Distance -> Overlay

Usage:
    python examples/distance_demo.py --input data/samples/drive.mp4
    python examples/distance_demo.py --input data/samples/drive.mp4 --save output.jpg --no-display
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
from src.estimation import DistanceEstimator
from src.io import VideoReader
from src.pipeline import LeadVehicleSelector
from src.tracking import VehicleTracker
from src.utils.config import DEFAULT_CONFIG_PATH, load_config
from src.utils.logger import setup_logging
from src.visualization import DetectionOverlay


def main() -> int:
    """Run the pipeline through distance estimation on one frame."""
    parser = argparse.ArgumentParser(
        description="Estimate distance to the lead vehicle in dashcam video."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input video file.",
    )
    parser.add_argument(
        "--config",
        "-c",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional YOLO model override (e.g. yolov8n.pt).",
    )
    parser.add_argument(
        "--save",
        default="output.jpg",
        help="Path to save annotated frame (default: output.jpg).",
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
    overlay = DetectionOverlay()

    with VideoReader(args.input) as reader:
        frame_data = next(iter(reader), None)
        if frame_data is None:
            print("No frames found in video.")
            return 1

        tracked = tracker.track(frame_data.frame)
        lead = selector.select(
            tracked,
            frame_width=reader.width,
            frame_height=reader.height,
        )
        distance = distance_estimator.estimate(lead, frame_width=reader.width)

        display_distance = (
            distance.smoothed_distance_m if distance is not None else None
        )
        annotated = overlay.draw_tracked_with_lead(
            frame_data.frame,
            tracked,
            lead,
            lead_distance_m=display_distance,
        )

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), annotated)

    if distance is not None:
        print(
            f"Lead track_id={distance.track_id} | "
            f"raw={distance.raw_distance_m:.2f} m | "
            f"smoothed={distance.smoothed_distance_m:.2f} m"
        )
    else:
        print("Distance: unavailable (no lead vehicle in ROI)")

    print(f"Saved annotated frame to: {save_path.resolve()}")

    if not args.no_display:
        window_name = "Distance Estimation Preview (press any key to close)"
        cv2.imshow(window_name, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    tracker.reset()
    distance_estimator.reset()
    return 0


if __name__ == "__main__":
    sys.exit(main())
