"""
MVP - Simple logging helper.
"""

from __future__ import annotations

import logging


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """MVP - Create a logger with uniform format.

    Params:
        name: logger name, usually __name__ of the caller module.
        level: logging level string (e.g., 'DEBUG', 'INFO').

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    return logger
