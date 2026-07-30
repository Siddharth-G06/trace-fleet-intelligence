"""
Centralized logging utility for TRACE fleet intelligence system.

Provides a consistent log format across all modules, writing to both
console and a rotating file handler under logs/trace.log.
"""

import logging
import os
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────
_LOG_DIR = Path("logs")
_LOG_FILE = _LOG_DIR / "trace.log"
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Ensure logs/ directory exists at import time
_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Module-level registry to avoid adding duplicate handlers
_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str) -> logging.Logger:
    """Return a named logger configured with console and file handlers.

    The logger writes to both stdout and ``logs/trace.log`` using a
    unified format:  ``timestamp | level | module_name | message``.

    Loggers are cached by *name* so that handlers are not duplicated when
    the same module calls ``get_logger`` more than once.

    Args:
        name: The logger name, typically ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger` instance.

    Example::

        logger = get_logger(__name__)
        logger.info("Pipeline started")
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)

    # Only configure if handlers haven't been attached yet
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

        # ── Console handler ────────────────────────────────────────────────
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)

        # ── File handler ───────────────────────────────────────────────────
        file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        # Prevent propagation to the root logger (avoids duplicate output)
        logger.propagate = False

    _loggers[name] = logger
    return logger
