"""
MVP - Scaler -> Reducer (PLS/PCA) -> Model pipeline.
"""

from __future__ import annotations

import numpy as np


class ForecastPipeline:
    """MVP - Combined pipeline for regression forecasts.

    Attributes:
        scaler: RollingStandardScaler instance.
        reducer: RollingPLS or RollingIncrementalPCA.
        model: ElasticNetRegressor or LogisticClassifier.
    """

    def __init__(self, scaler, reducer, model) -> None:
        self.scaler = scaler
        self.reducer = reducer
        self.model = model

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "ForecastPipeline":
        """MVP - Fit scaler -> reducer -> model on training arrays."""
        X_scaled = self.scaler.fit(X_train).transform(X_train)
        try:
            self.reducer.fit(X_scaled, y_train)
        except TypeError:
            self.reducer.fit(X_scaled)
        latent = self.reducer.transform(X_scaled)
        self.model.fit(latent, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """MVP - Transform and predict on new arrays."""
        latent = self.reducer.transform(self.scaler.transform(X))
        return self.model.predict(latent)

    def predict_proba(self, X: np.ndarray):
        latent = self.reducer.transform(self.scaler.transform(X))
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(latent)
        raise AttributeError("Underlying model does not support predict_proba")
