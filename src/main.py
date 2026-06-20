"""CLI entry point for the Rear-End ADAS Collision Warning System."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src import __version__
from src.pipeline import ADASPipeline
from src.utils.config import DEFAULT_CONFIG_PATH, load_config
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="rear-end-adas",
        description=(
            "Camera-Based Rear-End ADAS Collision Warning System. "
            "Detects vehicles, estimates distance and TTC, and renders "
            "collision risk overlays on driving video."
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input video file (e.g. data/samples/drive.mp4).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help=(
            "Path for annotated output video. "
            "Defaults to data/outputs/<input_stem>_adas.mp4."
        ),
    )
    parser.add_argument(
        "--config",
        "-c",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to YAML config file (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override YOLO model name or path (e.g. yolov8n.pt).",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show live preview window during processing.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def _resolve_output_path(input_path: Path, output_arg: str | None) -> Path:
    """Derive output video path when the user does not specify one."""
    if output_arg is not None:
        return Path(output_arg)
    return Path("data/outputs") / f"{input_path.stem}_adas.mp4"


def main(argv: list[str] | None = None) -> int:
    """Load configuration, run the ADAS pipeline, and return exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    output_path = _resolve_output_path(input_path, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: failed to load configuration: {exc}", file=sys.stderr)
        return 1

    setup_logging(level=config.logging.level, log_file=config.logging.log_file)

    pipeline = ADASPipeline(
        config=config,
        input_path=input_path,
        output_path=output_path,
        model_path=args.model,
        display=args.display,
    )

    try:
        pipeline.run()
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 1
    finally:
        pipeline.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
