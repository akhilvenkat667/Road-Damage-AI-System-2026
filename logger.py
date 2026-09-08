"""
logger.py
---------
Purpose:
    Provides a single, reusable, pre-configured logger for the entire
    backend. Every module imports `get_logger(__name__)` from here instead
    of configuring `logging` independently, which guarantees consistent
    log formatting and a single log file destination
    (backend/logs/app.log).

    Logs are written both to the console (for live debugging during
    development) and to a rotating log file (so logs do not grow forever
    and disk space is not exhausted in production).
"""

import logging
from logging.handlers import RotatingFileHandler

import config

# Make sure the logs directory exists before we try to write to it.
config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure_root_logger() -> None:
    """
    Configure the root logger exactly once with a console handler and a
    rotating file handler. Subsequent calls are no-ops thanks to the
    module-level `_configured` flag, which prevents duplicate log lines
    that would otherwise appear if this function ran more than once
    (e.g. due to Flask's debug-mode reloader importing modules twice).
    """
    global _configured
    if _configured:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler: useful when running `python app.py` in a terminal.
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Rotating file handler: keeps up to 5 log files of 2MB each so the
    # logs folder never grows without bound.
    file_handler = RotatingFileHandler(
        config.LOG_FILE_PATH, maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-scoped logger configured with the shared handlers.

    Args:
        name: Typically `__name__` of the calling module, so log lines
              are traceable to their source file (e.g. "detector",
              "routes", "database").

    Returns:
        A configured `logging.Logger` instance.
    """
    _configure_root_logger()
    return logging.getLogger(name)
