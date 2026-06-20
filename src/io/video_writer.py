"""Video output handling via OpenCV VideoWriter."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VideoWriter:
    """Write annotated frames to an MP4 video file.

    The writer creates parent directories as needed, initializes an OpenCV
    ``VideoWriter`` with the specified codec, and validates that each written
    frame matches the configured resolution. Use as a context manager to
    ensure the file is finalized properly.

    Example:
        >>> with VideoWriter("data/outputs/out.mp4", 1920, 1080, 30.0) as writer:
        ...     writer.write(annotated_frame)
    """

    def __init__(
        self,
        output_path: str | Path,
        width: int,
        height: int,
        fps: float,
        codec: str = "mp4v",
    ) -> None:
        """Initialize the video writer and open the output file.

        Args:
            output_path: Destination path for the output video (typically .mp4).
            width: Frame width in pixels. Must match every written frame.
            height: Frame height in pixels. Must match every written frame.
            fps: Output frames per second. Should match the input video FPS
                unless intentionally overridden via config.
            codec: FourCC codec string. ``mp4v`` produces widely compatible MP4
                files on Windows and Linux.

        Raises:
            ValueError: If width, height, or fps are invalid.
            RuntimeError: If OpenCV cannot open the output file for writing.
        """
        self._output_path = Path(output_path)
        self._width = width
        self._height = height
        self._fps = fps
        self._codec = codec
        self._writer: cv2.VideoWriter | None = None
        self._frames_written = 0

        self._validate_parameters()
        self._open_writer()

        logger.info(
            "Opened video writer: %s (%dx%d @ %.2f FPS, codec=%s)",
            self._output_path,
            self._width,
            self._height,
            self._fps,
            self._codec,
        )

    @property
    def output_path(self) -> Path:
        """Path to the output video file."""
        return self._output_path

    @property
    def frames_written(self) -> int:
        """Number of frames successfully written so far."""
        return self._frames_written

    @classmethod
    def from_reader_metadata(
        cls,
        output_path: str | Path,
        width: int,
        height: int,
        fps: float,
        codec: str = "mp4v",
        output_fps: float | None = None,
    ) -> VideoWriter:
        """Create a writer using metadata extracted from a ``VideoReader``.

        This factory method is the preferred way to construct a writer in the
        pipeline so output settings mirror the input stream.

        Args:
            output_path: Destination path for the output video.
            width: Frame width from the input video.
            height: Frame height from the input video.
            fps: FPS from the input video.
            codec: FourCC codec string from application config.
            output_fps: Optional override from config. When ``None``, uses
                the input ``fps`` unchanged.

        Returns:
            Configured ``VideoWriter`` instance.
        """
        effective_fps = output_fps if output_fps is not None else fps
        return cls(
            output_path=output_path,
            width=width,
            height=height,
            fps=effective_fps,
            codec=codec,
        )

    def _validate_parameters(self) -> None:
        """Validate constructor parameters before opening the writer."""
        if self._width <= 0 or self._height <= 0:
            raise ValueError(
                f"Invalid frame dimensions: {self._width}x{self._height}"
            )
        if self._fps <= 0:
            raise ValueError(f"FPS must be positive, got {self._fps}")
        if len(self._codec) != 4:
            raise ValueError(
                f"Codec must be a 4-character FourCC string, got {self._codec!r}"
            )

    def _open_writer(self) -> None:
        """Create parent directories and open the OpenCV VideoWriter."""
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*self._codec)
        writer = cv2.VideoWriter(
            str(self._output_path),
            fourcc,
            self._fps,
            (self._width, self._height),
        )

        if not writer.isOpened():
            logger.error("Failed to open video writer: %s", self._output_path)
            raise RuntimeError(
                f"Failed to open video writer: {self._output_path}"
            )

        self._writer = writer

    def write(self, frame: np.ndarray) -> None:
        """Write a single BGR frame to the output video.

        Args:
            frame: Annotated frame with shape (height, width, 3) and dtype
                ``uint8``.

        Raises:
            RuntimeError: If the writer has been released or was never opened.
            ValueError: If the frame shape does not match the configured
                resolution or dtype is not ``uint8``.
        """
        if self._writer is None or not self._writer.isOpened():
            raise RuntimeError("VideoWriter is not open. Use as a context manager.")

        if frame.dtype != np.uint8:
            raise ValueError(f"Frame dtype must be uint8, got {frame.dtype}")

        frame_height, frame_width = frame.shape[:2]
        if frame_width != self._width or frame_height != self._height:
            raise ValueError(
                f"Frame size mismatch: expected {self._width}x{self._height}, "
                f"got {frame_width}x{frame_height}"
            )

        self._writer.write(frame)
        self._frames_written += 1

    def release(self) -> None:
        """Finalize and release the underlying OpenCV writer."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info(
                "Closed video writer: %s (%d frames written)",
                self._output_path,
                self._frames_written,
            )

    def __enter__(self) -> VideoWriter:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and release resources."""
        self.release()

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"VideoWriter(path={self._output_path!s}, "
            f"size={self._width}x{self._height}, fps={self._fps:.2f}, "
            f"frames_written={self._frames_written})"
        )
