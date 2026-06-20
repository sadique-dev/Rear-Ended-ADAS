"""Video input handling via OpenCV VideoCapture."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Some containers report 0 FPS when metadata is missing; 30 FPS is a safe default.
DEFAULT_FPS = 30.0


@dataclass(frozen=True)
class FrameData:
    """Single decoded video frame with timing metadata.

    Attributes:
        frame: BGR image as a NumPy array with shape (height, width, 3).
        index: Zero-based frame index in the source video.
        timestamp_seconds: Elapsed time from the start of the video in seconds.
    """

    frame: np.ndarray
    index: int
    timestamp_seconds: float


class VideoReader:
    """Read frames sequentially from a video file using OpenCV.

    The reader validates that the file exists, opens a ``VideoCapture`` handle,
    extracts stream metadata (FPS, resolution, frame count), and yields one
    ``FrameData`` object per frame. Use as a context manager to guarantee the
    capture device is released.

    Example:
        >>> with VideoReader("data/samples/drive.mp4") as reader:
        ...     for frame_data in reader:
        ...         process(frame_data.frame)
    """

    def __init__(self, source: str | Path) -> None:
        """Initialize the reader and open the video source.

        Args:
            source: Path to an existing video file.

        Raises:
            FileNotFoundError: If the video file does not exist.
            RuntimeError: If OpenCV cannot open the video stream.
            ValueError: If required metadata (width, height) is invalid.
        """
        self._source = Path(source)
        self._capture: cv2.VideoCapture | None = None

        self._validate_source_path()
        self._open_capture()
        self._read_metadata()

        logger.info(
            "Opened video: %s (%dx%d @ %.2f FPS, %d frames)",
            self._source.name,
            self._width,
            self._height,
            self._fps,
            self._total_frames,
        )

    @property
    def source(self) -> Path:
        """Path to the input video file."""
        return self._source

    @property
    def fps(self) -> float:
        """Frames per second reported by the video container."""
        return self._fps

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return self._height

    @property
    def total_frames(self) -> int:
        """Total frame count reported by the container.

        Note:
            This value can be approximate for some codecs. It is used for
            progress reporting, not as a hard loop bound.
        """
        return self._total_frames

    def _validate_source_path(self) -> None:
        """Ensure the source path exists and is a regular file."""
        if not self._source.exists():
            logger.error("Video file not found: %s", self._source)
            raise FileNotFoundError(f"Video file not found: {self._source}")

        if not self._source.is_file():
            logger.error("Source path is not a file: %s", self._source)
            raise FileNotFoundError(f"Source path is not a file: {self._source}")

    def _open_capture(self) -> None:
        """Open the OpenCV capture handle."""
        capture = cv2.VideoCapture(str(self._source))
        if not capture.isOpened():
            logger.error("Failed to open video: %s", self._source)
            raise RuntimeError(f"Failed to open video: {self._source}")

        self._capture = capture

    def _read_metadata(self) -> None:
        """Extract and validate FPS, resolution, and frame count."""
        assert self._capture is not None

        raw_fps = self._capture.get(cv2.CAP_PROP_FPS)
        self._width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total_frames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if self._width <= 0 or self._height <= 0:
            self.release()
            logger.error(
                "Invalid video dimensions: %dx%d for %s",
                self._width,
                self._height,
                self._source,
            )
            raise ValueError(
                f"Invalid video dimensions: {self._width}x{self._height}"
            )

        if raw_fps <= 0:
            # Missing FPS metadata is common in some MP4 files.
            logger.warning(
                "Video reports invalid FPS (%.2f); using default %.1f",
                raw_fps,
                DEFAULT_FPS,
            )
            self._fps = DEFAULT_FPS
        else:
            self._fps = float(raw_fps)

        if self._total_frames < 0:
            logger.warning("Frame count unavailable; reporting as 0")
            self._total_frames = 0

    def __iter__(self) -> Iterator[FrameData]:
        """Yield decoded frames with index and timestamp metadata.

        Yields:
            ``FrameData`` for each successfully decoded frame.

        Raises:
            RuntimeError: If the reader has been released or was never opened.
        """
        if self._capture is None or not self._capture.isOpened():
            raise RuntimeError("VideoReader is not open. Use as a context manager.")

        frame_index = 0

        while True:
            success, frame = self._capture.read()
            if not success:
                break

            timestamp = frame_index / self._fps
            yield FrameData(
                frame=frame,
                index=frame_index,
                timestamp_seconds=timestamp,
            )
            frame_index += 1

        logger.debug("Finished reading %d frames from %s", frame_index, self._source)

    def release(self) -> None:
        """Release the underlying OpenCV capture handle."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.debug("Released video capture: %s", self._source)

    def __enter__(self) -> VideoReader:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and release resources."""
        self.release()

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"VideoReader(source={self._source!s}, "
            f"size={self._width}x{self._height}, fps={self._fps:.2f})"
        )
