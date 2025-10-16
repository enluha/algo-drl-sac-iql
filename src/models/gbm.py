"""
Gradient boosting wrappers (HistGradientBoosting-based) for optional heads.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


class HGBClassifier:
    def __init__(self, **kw):
        self.model = HistGradientBoostingClassifier(**kw)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HGBClassifier":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray):
        return self.model.predict_proba(X)


class HGBRegressor:
    def __init__(self, **kw):
        self.model = HistGradientBoostingRegressor(**kw)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HGBRegressor":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray):
        return self.model.predict(X)
