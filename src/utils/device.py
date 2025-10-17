from __future__ import annotations

import logging
import os
from typing import Optional

import torch


def get_torch_device(prefer: Optional[str] = None) -> torch.device:
    """Return the preferred torch.device considering availability."""
    if prefer is None:
        prefer = "cuda"
    prefer = prefer.lower()
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_num_threads(n: int) -> None:
    """Set threading limits for PyTorch and common BLAS backends."""
    torch.set_num_threads(max(1, int(n)))
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)


def log_device(logger: logging.Logger) -> None:
    """Log CUDA availability and device metadata."""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        name = torch.cuda.get_device_name(device)
        capability = torch.cuda.get_device_capability(device)
        logger.info(
            "Using CUDA device %s (capability %s.%s)",
            name,
            capability[0],
            capability[1],
        )
    else:
        logger.info("Using CPU device")
