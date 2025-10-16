"""
MVP - Embargo utilities for time-series split hygiene.
"""

from __future__ import annotations

import numpy as np


def apply_embargo(valid_idx: np.ndarray, all_idx: np.ndarray, embargo_len: int) -> np.ndarray:
    """MVP - Mask out indices within embargo_len after validation end.

    Params:
        valid_idx: validation indices.
        all_idx: candidate indices for training.
        embargo_len: embargo length in steps (currently unused).

    Returns:
        Unmodified `all_idx` in MVP implementation.
    """
    # Simplified MVP: return indices unchanged; enforce embargo later (NON MVP).
    return all_idx
