"""
MVP - ElasticNet regressor wrapper (scikit-learn).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import ElasticNet


class ElasticNetRegressor:
    """MVP - Regularized linear regressor for returns forecast."""

    def __init__(self, alpha: float = 0.001, l1_ratio: float = 0.1, max_iter: int = 1000):
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "ElasticNetRegressor":
        """MVP - Fit regressor on training data."""
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """MVP - Predict returns for feature matrix `X`."""
        return self.model.predict(X)
