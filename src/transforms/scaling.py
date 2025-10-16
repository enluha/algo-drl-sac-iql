"""
MVP - Standard scaling fitted only on training data per window.
"""

from __future__ import annotations

import numpy as np


class RollingStandardScaler:
    """MVP - Fit mean/std on training window, transform arrays."""

    def __init__(self) -> None:
        self.mu_: np.ndarray | None = None
        self.sd_: np.ndarray | None = None

    def fit(self, X_train: np.ndarray) -> "RollingStandardScaler":
        """MVP - Compute per-column mean/std on `X_train`.

        Params:
            X_train: 2D numpy array of training features.

        Returns:
            Self for chaining.
        """
        self.mu_ = np.nanmean(X_train, axis=0)
        self.sd_ = np.nanstd(X_train, axis=0) + 1e-12
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """MVP - Standardize `X` using stored statistics.

        Params:
            X: 2D numpy array to transform.

        Returns:
            Transformed array with zero mean/unit variance per feature.
        """
        if self.mu_ is None or self.sd_ is None:
            raise RuntimeError("Scaler must be fitted before calling transform")
        return (X - self.mu_) / self.sd_
