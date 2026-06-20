"""Full ADAS pipeline demonstration — processes a video end-to-end.

Usage:
    python examples/final_demo.py --input data/samples/drive.mp4 --output data/outputs/final_demo.mp4
    python examples/final_demo.py --input data/samples/drive.mp4 --output data/outputs/final_demo.mp4 --display
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import ADASPipeline
from src.utils.config import DEFAULT_CONFIG_PATH, load_config
from src.utils.logger import setup_logging


def main() -> int:
    """Run the complete ADAS pipeline on an input video."""
    parser = argparse.ArgumentParser(
        description="Full Rear-End ADAS collision warning demonstration."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input video file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path for annotated output video.",
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
        help="Optional YOLO model override.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show live preview window during processing.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    setup_logging(level=config.logging.level, log_file=config.logging.log_file)

    pipeline = ADASPipeline(
        config=config,
        input_path=input_path,
        output_path=output_path,
        model_path=args.model,
        display=args.display,
    )

    try:
        stats = pipeline.run()
        print(f"Output saved to: {output_path.resolve()}")
        print(f"Processed {stats.total_frames} frames at {stats.average_fps:.2f} FPS")
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1
    finally:
        pipeline.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
