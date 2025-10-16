"""
MVP - Logistic classifier wrapper.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class LogisticClassifier:
    """MVP - Classify P(up) for meta-labeling or alternative signal."""

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        max_iter: int = 1000,
        class_weight: str | dict | None = None,
    ) -> None:
        self.model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver="lbfgs",
            max_iter=max_iter,
            class_weight=class_weight,
        )

    def fit(self, X_train: np.ndarray, y_label: np.ndarray) -> "LogisticClassifier":
        """MVP - Fit classifier on labels {-1,0,1} or {0,1}."""
        self.model.fit(X_train, y_label)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """MVP - Return probability for class 1 (up)."""
        proba = self.model.predict_proba(X)
        return proba[:, 1]
