"""Run YOLO vehicle detection on the first frame of an input video.

Usage:
    python examples/detect_single_frame.py --input data/samples/drive.mp4
    python examples/detect_single_frame.py --input data/samples/drive.mp4 --model yolov8s.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from project root without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection import YoloDetector
from src.io import VideoReader
from src.utils.config import DEFAULT_CONFIG_PATH, load_config
from src.utils.logger import setup_logging


def main() -> int:
    """Load one frame from a video and print vehicle detections."""
    parser = argparse.ArgumentParser(
        description="Detect vehicles on the first frame of a video."
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
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(level=config.logging.level, log_file=config.logging.log_file)

    detector = YoloDetector(config.model, model_path=args.model)

    with VideoReader(args.input) as reader:
        first_frame = next(iter(reader), None)

    if first_frame is None:
        print("No frames found in video.")
        return 1

    detections = detector.detect(first_frame.frame)

    print(f"Frame index : {first_frame.index}")
    print(f"Timestamp   : {first_frame.timestamp_seconds:.3f}s")
    print(f"Device      : {detector.device}")
    print(f"Detections  : {len(detections)}")
    print("-" * 60)

    for index, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det.bbox
        print(
            f"{index}. {det.class_name:10s} | "
            f"conf={det.confidence:.2f} | "
            f"bbox=({x1}, {y1}, {x2}, {y2}) | "
            f"class_id={det.class_id}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
