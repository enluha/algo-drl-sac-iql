"""
Reducer utilities with sign-stable PLS and PCA fallback.
"""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import IncrementalPCA


class StablePLS:
    def __init__(self, n_components: int) -> None:
        self.model = PLSRegression(n_components=n_components)
        self.sign_ref_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StablePLS":
        self.model.fit(X, y)
        loadings = self.model.x_weights_
        proj = X @ loadings
        corr = np.sign(np.where(np.nanmean(proj * y[:, None], axis=0) >= 0, 1, -1))
        corr[corr == 0] = 1
        self.sign_ref_ = corr.reshape(1, -1)
        self.model.x_weights_ *= self.sign_ref_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X @ self.model.x_weights_


def make_reducer(kind: str, n_components: int):
    if kind.lower() == "pls":
        return StablePLS(n_components)
    return IncrementalPCA(n_components=n_components)
