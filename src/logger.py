"""Logging subsystem for Wild-ID with rotation and privacy guarantees.

Guarantees:
- Rotates logs with bounded file size and backup count.
- Never records image binary content, base64 strings, or raw pixel matrices.
- Uses UTF-8 encoding safely across platforms.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from src.config import LOG_DIR, LOG_MAX_BYTES, LOG_BACKUP_COUNT


def setup_logger(name: str = "wild_id") -> logging.Logger:
    """Configure and return a rotating file and console logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if setup is called multiple times
    if logger.handlers:
        return logger

    # Ensure log directory exists
    log_dir_path = Path(LOG_DIR).resolve()
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_file = log_dir_path / "wild_id.log"

    # Rotating file handler (bounded in-memory and on-disk)
    file_handler = RotatingFileHandler(
        filename=str(log_file),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8"
    )
    file_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        fmt="[%(levelname)s] %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    return logger


# Global default logger instance
logger = setup_logger("wild_id")


def log_inference_event(
    action: str,
    image_metadata: dict,
    status: str,
    duration_ms: float = None,
    detail: str = ""
):
    """
    Log an inference or processing event safely without any image pixel content.

    image_metadata should strictly contain non-sensitive structural attributes:
      e.g. {'width': 640, 'height': 480, 'mode': 'RGB', 'size_bytes': 10240}
    """
    # Defensive scrub: Ensure no raw image data or base64 ever passes into log
    scrubbed_meta = {}
    for k, v in image_metadata.items():
        if k in ("width", "height", "mode", "format", "size_bytes", "downscaled", "filename"):
            scrubbed_meta[k] = v

    meta_str = ", ".join(f"{k}={v}" for k, v in scrubbed_meta.items())
    timing_str = f" in {duration_ms:.1f}ms" if duration_ms is not None else ""
    msg = f"ACTION={action} | STATUS={status}{timing_str} | META=[{meta_str}]"
    if detail:
        msg += f" | {detail}"

    logger.info(msg)
