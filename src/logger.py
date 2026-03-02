# logger.py

import logging
import os
from logging.handlers import RotatingFileHandler

# Configuration
LOG_FILE    = os.getenv("LOG_FILE", "app.log")
LOG_LEVEL   = os.getenv("LOG_LEVEL", "DEBUG").upper()

# Each individual log file is capped at 5 MB.
# Up to 3 rotated backups are kept (app.log.1, app.log.2, app.log.3).
# Total max disk usage: ~20 MB.
MAX_BYTES   = int(os.getenv("LOG_MAX_BYTES",  str(5 * 1024 * 1024)))  # 5 MB
BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "3"))

# 2026-03-02 12:10:00,000 WARNING: api | Session XYZ not found
LOG_FORMAT  = "%(asctime)s %(levelname)s: %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Root logger setup (runs once at import time)

def _configure_root_logger() -> None:
    root = logging.getLogger()

    # Avoid adding duplicate handlers if this module is imported more than once
    # (e.g. during Streamlit's hot-reload cycle).
    if root.handlers:
        return

    root.setLevel(logging.DEBUG)  # Root accepts everything; handlers filter.

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    # Rotating file handler
    file_handler = RotatingFileHandler(
        filename=LOG_FILE,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, LOG_LEVEL, logging.WARNING))

    # Console handler (mirrors file, useful during development)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.WARNING))

    root.addHandler(file_handler)
    root.addHandler(console_handler)


_configure_root_logger()

# Public API

def get_logger(name: str) -> logging.Logger:
    """
    Return a named child logger.

    Usage (at the top of each module):
        from logger import get_logger
        logger = get_logger(__name__)

    The `name` argument drives the %(name)s field in the log format, producing
    entries like:
        2026-03-02 12:10:00,000 WARNING: api | Session XYZ not found
    """
    return logging.getLogger(name)