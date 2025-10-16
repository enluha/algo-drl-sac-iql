"""
MVP - Supervised (PLS) or unsupervised (IncrementalPCA) reducers.
"""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import IncrementalPCA


class RollingPLS:
    """MVP - Fit PLS on `(X_train, y_train)` and transform `X`."""

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.model: PLSRegression | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RollingPLS":
        """MVP - Fit PLS on standardized arrays.

        Params:
            X_train: standardized training features.
            y_train: training target vector.

        Returns:
            Self for chaining.
        """
        self.model = PLSRegression(n_components=self.n_components)
        self.model.fit(X_train, y_train)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """MVP - Return transformed latent factors."""
        if self.model is None:
            raise RuntimeError("PLS model must be fitted before transform")
        return self.model.transform(X)


class RollingIncrementalPCA:
    """MVP - Fit IncrementalPCA on `X_train`; transform `X`."""

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.model = IncrementalPCA(n_components=n_components)

    def fit(self, X_train: np.ndarray, y_unused: np.ndarray | None = None) -> "RollingIncrementalPCA":
        """MVP - Fit PCA on standardized `X_train` (ignores `y_unused`)."""
        self.model.fit(X_train)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """MVP - Transform `X` using fitted PCA components."""
        return self.model.transform(X)
