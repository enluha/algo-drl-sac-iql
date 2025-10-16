"""
MVP - Seed control for reproducibility.
"""

from __future__ import annotations

import random

import numpy as np


def fix_seeds(seed: int) -> None:
    """MVP - Seed Python and NumPy RNGs.

    Params:
        seed: integer seed used across libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
