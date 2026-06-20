"""Structured logging configuration for the ADAS system."""

import logging
import sys
from pathlib import Path


# Standard format used across all modules for consistent log output.
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
) -> None:
    """Configure root logger for console and optional file output.

    Args:
        level: Logging level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to write logs to disk. When None, logs
            are written to stderr only.

    Raises:
        ValueError: If an invalid logging level string is provided.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(
            f"Invalid logging level: {level!r}. "
            "Use DEBUG, INFO, WARNING, ERROR, or CRITICAL."
        )

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers when setup_logging is called more than once.
    root_logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with the given name.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        Configured ``logging.Logger`` instance.
    """
    return logging.getLogger(name)
