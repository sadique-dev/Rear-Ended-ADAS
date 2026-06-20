"""Integration tests for the full ADAS pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.pipeline import ADASPipeline
from src.utils.config import load_config


def _create_test_video(path: Path, frame_count: int = 3) -> None:
    """Write a minimal MP4 clip for pipeline integration tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (640, 480),
    )
    for index in range(frame_count):
        gray = 40 + index * 20
        writer.write(
            np.full((480, 640, 3), gray, dtype=np.uint8)
        )
    writer.release()


@pytest.fixture
def test_video(tmp_path: Path) -> Path:
    """Temporary test video path."""
    video_path = tmp_path / "integration_test.mp4"
    _create_test_video(video_path, frame_count=3)
    return video_path


@pytest.fixture
def output_video(tmp_path: Path) -> Path:
    """Temporary output video path."""
    return tmp_path / "integration_output.mp4"


def test_pipeline_runs_without_crash(test_video: Path, output_video: Path):
    """Full pipeline should process a video and write output."""
    config = load_config()
    pipeline = ADASPipeline(
        config=config,
        input_path=test_video,
        output_path=output_video,
        max_frames=3,
    )

    try:
        stats = pipeline.run()
    finally:
        pipeline.cleanup()

    assert stats.total_frames == 3
    assert stats.processing_time_seconds > 0.0
    assert stats.average_fps > 0.0
    assert output_video.is_file()
    assert output_video.stat().st_size > 0


def test_pipeline_initialize_process_frame_cleanup(
    test_video: Path,
    output_video: Path,
):
    """Individual pipeline lifecycle methods should work correctly."""
    config = load_config()
    pipeline = ADASPipeline(
        config=config,
        input_path=test_video,
        output_path=output_video,
    )

    pipeline.initialize()

    try:
        from src.io import VideoReader

        with VideoReader(test_video) as reader:
            frame_data = next(iter(reader))

        result = pipeline.process_frame(frame_data)

        assert result.annotated_frame.shape == frame_data.frame.shape
        assert result.frame_index == frame_data.index
        assert result.risk is not None
    finally:
        pipeline.cleanup()


def test_pipeline_respects_max_frames(test_video: Path, output_video: Path):
    """max_frames should limit how many frames are processed."""
    config = load_config()
    pipeline = ADASPipeline(
        config=config,
        input_path=test_video,
        output_path=output_video,
        max_frames=2,
    )

    try:
        stats = pipeline.run()
    finally:
        pipeline.cleanup()

    assert stats.total_frames == 2
