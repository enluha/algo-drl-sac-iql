"""
Purged cross-validation utilities with embargo support.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class PurgedKFold:
    n_splits: int
    purge_bars: int = 0
    embargo_bars: int = 0

    def split(
        self,
        index: Sequence,
        t1: pd.Series,
    ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        if self.n_splits <= 1:
            raise ValueError("n_splits must be >= 2")

        idx = pd.Index(index)
        n = len(idx)
        folds = np.array_split(np.arange(n), self.n_splits)

        for test_idx in folds:
            test_lo, test_hi = test_idx[0], test_idx[-1]
            test_start = idx[test_lo]
            test_end = idx[test_hi]

            train_idx = np.setdiff1d(np.arange(n), test_idx, assume_unique=True)
            if len(train_idx) == 0:
                continue

            if self.purge_bars > 0:
                spill_mask = (
                    (t1.iloc[train_idx] >= (test_start - pd.Timedelta(self.purge_bars, unit="h")))
                    & (idx[train_idx] <= (test_end + pd.Timedelta(self.purge_bars, unit="h")))
                )
                train_idx = train_idx[~spill_mask.to_numpy()]

            if self.embargo_bars > 0:
                embargo_start = max(0, test_lo - self.embargo_bars)
                embargo_end = min(n, test_hi + self.embargo_bars + 1)
                embargo_range = np.r_[
                    np.arange(embargo_start, test_lo),
                    np.arange(test_hi + 1, embargo_end),
                ]
                train_idx = np.setdiff1d(train_idx, embargo_range, assume_unique=True)

            yield train_idx, test_idx


@dataclass
class CombinatorialPurgedCV:
    n_splits: int
    n_test_folds: int
    purge_bars: int = 0
    embargo_bars: int = 0

    def split(
        self,
        index: Sequence,
        t1: pd.Series,
    ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        idx = pd.Index(index)
        folds = list(np.array_split(np.arange(len(idx)), self.n_splits))

        for test_combo in itertools.combinations(range(self.n_splits), self.n_test_folds):
            test_idx = np.sort(np.concatenate([folds[i] for i in test_combo]))
            mask = np.ones(len(idx), dtype=bool)
            mask[test_idx] = False
            candidate_train = np.nonzero(mask)[0]

            test_lo, test_hi = test_idx[0], test_idx[-1]
            test_start = idx[test_lo]
            test_end = idx[test_hi]

            train_idx = candidate_train.copy()
            if self.purge_bars > 0:
                spill_mask = (
                    (t1.iloc[train_idx] >= (test_start - pd.Timedelta(self.purge_bars, unit="h")))
                    & (idx[train_idx] <= (test_end + pd.Timedelta(self.purge_bars, unit="h")))
                )
                train_idx = train_idx[~spill_mask.to_numpy()]

            if self.embargo_bars > 0:
                embargo_start = max(0, test_lo - self.embargo_bars)
                embargo_end = min(len(idx), test_hi + self.embargo_bars + 1)
                embargo_range = np.r_[
                    np.arange(embargo_start, test_lo),
                    np.arange(test_hi + 1, embargo_end),
                ]
                train_idx = np.setdiff1d(train_idx, embargo_range, assume_unique=True)

            yield train_idx, test_idx
