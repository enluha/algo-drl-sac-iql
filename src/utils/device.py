from __future__ import annotations

import logging
import os
from typing import Optional

import torch


def get_torch_device(prefer: Optional[str] = None) -> torch.device:
    """Return the desired torch.device respecting availability."""

    prefer = (prefer or "cuda").lower()
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_num_threads(n: int) -> None:
    """Limit BLAS/torch thread usage for reproducibility."""

    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)


def log_device(logger: logging.Logger) -> None:
    """Log device information, including CUDA metadata if available."""

    available = torch.cuda.is_available()
    logger.info("CUDA available: %s", available)
    if available:
        idx = torch.cuda.current_device()
        logger.info(
            "CUDA device %d: %s (cc %s)",
            idx,
            torch.cuda.get_device_name(idx),
            ".".join(str(x) for x in torch.cuda.get_device_capability(idx)),
        )
