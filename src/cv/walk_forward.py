"""
MVP - Walk-forward window generator using calendar spans.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd


def walk_forward_splits(
    index: pd.DatetimeIndex,
    train: str,
    valid: str,
    test: str,
    step: str,
    embargo: str,
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    MVP - Produce list of (train_idx, valid_idx, test_idx) as DatetimeIndex (preserve tz).

    Parameters
    ----------
    index : pd.DatetimeIndex
        Full timeline index for the dataset.
    train, valid, test, step, embargo : str
        Pandas offset strings configuring window lengths (embargo unused for MVP).

    Returns
    -------
    list[tuple[DatetimeIndex, DatetimeIndex, DatetimeIndex]]
    """
    idx = pd.DatetimeIndex(index)  # copy to avoid modifying caller and preserve tz
    splits: List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]] = []
    current_start = idx.min()

    while True:
        train_start = current_start
        train_end = train_start + pd.Timedelta(train)
        valid_end = train_end + pd.Timedelta(valid)
        test_end = valid_end + pd.Timedelta(test)

        if test_end > idx.max():
            break

        train_mask = (idx >= train_start) & (idx < train_end)
        valid_mask = (idx >= train_end) & (idx < valid_end)
        test_mask = (idx >= valid_end) & (idx < test_end)

        splits.append((idx[train_mask], idx[valid_mask], idx[test_mask]))
        current_start += pd.Timedelta(step)

    return splits
