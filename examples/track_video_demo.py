"""Run detection + ByteTrack tracking on video frames with visualization.

Pipeline:
    VideoReader -> YOLO + ByteTrack -> Detection Overlay -> display / save

Usage:
    python examples/track_video_demo.py --input data/samples/drive.mp4
    python examples/track_video_demo.py --input data/samples/drive.mp4 --save-video out.mp4
    python examples/track_video_demo.py --input data/samples/drive.mp4 --save-frame output.jpg --no-display
"""

from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection import YoloDetector
from src.io import VideoReader, VideoWriter
from src.tracking import VehicleTracker
from src.utils.config import DEFAULT_CONFIG_PATH, load_config
from src.utils.logger import setup_logging
from src.visualization import DetectionOverlay


def main() -> int:
    """Run the detection + tracking pipeline on a video."""
    parser = argparse.ArgumentParser(
        description="Detect and track vehicles with ByteTrack overlays."
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
        "--max-frames",
        type=int,
        default=1,
        help="Number of frames to process (default: 1 for quick demo).",
    )
    parser.add_argument(
        "--save-frame",
        default="output.jpg",
        help="Path to save the last annotated frame (default: output.jpg).",
    )
    parser.add_argument(
        "--save-video",
        default=None,
        help="Optional path to save annotated output video.",
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
    overlay = DetectionOverlay()

    last_annotated = None

    with VideoReader(args.input) as reader:
        writer_context = (
            VideoWriter.from_reader_metadata(
                output_path=args.save_video,
                width=reader.width,
                height=reader.height,
                fps=reader.fps,
                codec=config.io.output_codec,
                output_fps=config.io.output_fps,
            )
            if args.save_video is not None
            else nullcontext()
        )

        with writer_context as writer:
            for frame_index, frame_data in enumerate(reader):
                if frame_index >= args.max_frames:
                    break

                tracked = tracker.track(frame_data.frame)
                last_annotated = overlay.draw_tracked_detections(
                    frame_data.frame,
                    tracked,
                )

                if writer is not None:
                    writer.write(last_annotated)

                print(
                    f"Frame {frame_data.index:4d} | "
                    f"tracks={len(tracked)} | "
                    f"IDs={[t.track_id for t in tracked]}"
                )

    if last_annotated is None:
        print("No frames processed.")
        return 1

    save_frame_path = Path(args.save_frame)
    save_frame_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_frame_path), last_annotated)
    print(f"Saved annotated frame to: {save_frame_path.resolve()}")

    if args.save_video is not None:
        print(f"Saved annotated video to: {Path(args.save_video).resolve()}")

    if not args.no_display:
        window_name = "ADAS Tracking Preview (press any key to close)"
        cv2.imshow(window_name, last_annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    tracker.reset()
    return 0


if __name__ == "__main__":
    sys.exit(main())
