"""
MVP - Purged K-Fold index generator (simplified).
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np


def purged_kfold_indices(n: int, n_splits: int, embargo: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """MVP - Yield (train_idx, valid_idx) with simple purge rules.

    Params:
        n: total number of observations.
        n_splits: number of folds.
        embargo: embargo length (unused in MVP).

    Returns:
        Iterator of (train_idx, valid_idx) numpy arrays.

    Notes:
        NON MVP - implement overlap-aware purging using label horizons.
    """
    fold = n // n_splits
    for fold_id in range(n_splits):
        valid_idx = np.arange(fold_id * fold, (fold_id + 1) * fold)
        train_idx = np.setdiff1d(np.arange(n), valid_idx)
        yield train_idx, valid_idx
