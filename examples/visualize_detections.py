"""Visualize YOLO detections on a single video frame.

Reads one frame, runs detection, draws bounding boxes, displays the result,
and optionally saves it as a JPEG image.

Usage:
    python examples/visualize_detections.py --input data/samples/drive.mp4
    python examples/visualize_detections.py --input data/samples/drive.mp4 --save output.jpg
    python examples/visualize_detections.py --input data/samples/drive.mp4 --save output.jpg --no-display
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

# Allow running from project root without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection import YoloDetector
from src.io import VideoReader
from src.utils.config import DEFAULT_CONFIG_PATH, load_config
from src.utils.logger import setup_logging
from src.visualization import DetectionOverlay


def main() -> int:
    """Detect vehicles on one frame and visualize the results."""
    parser = argparse.ArgumentParser(
        description="Draw YOLO vehicle detections on a single video frame."
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
    overlay = DetectionOverlay()

    with VideoReader(args.input) as reader:
        frame_data = next(iter(reader), None)

    if frame_data is None:
        print("No frames found in video.")
        return 1

    detections = detector.detect(frame_data.frame)
    annotated = overlay.draw_detections(frame_data.frame, detections)

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), annotated)
    print(f"Saved annotated frame to: {save_path.resolve()}")
    print(f"Detections drawn: {len(detections)}")

    if not args.no_display:
        window_name = "ADAS Detection Preview (press any key to close)"
        cv2.imshow(window_name, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
