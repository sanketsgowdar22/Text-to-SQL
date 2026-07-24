"""
Common utility functions.

Provides helpers used across multiple features.
"""

import logging
import sys
from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(UTC)


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure application-wide logging.

    Sets up a structured log format with timestamps, levels,
    and module names. Outputs to stdout for container compatibility.

    Args:
        log_level: The minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if numeric_level <= logging.DEBUG else logging.WARNING
    )


def truncate_string(value: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    Truncate a string to a maximum length with a suffix.

    Args:
        value: The string to truncate.
        max_length: Maximum allowed length (including suffix).
        suffix: The suffix to append when truncated.

    Returns:
        The original string if short enough, or truncated with suffix.
    """
    if len(value) <= max_length:
        return value
    return value[: max_length - len(suffix)] + suffix
