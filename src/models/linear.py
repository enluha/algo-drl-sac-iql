"""
Factory utilities for linear heads with optional calibration.
"""

from __future__ import annotations

from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import ElasticNet, LogisticRegression


def make_model(task: str, config: dict[str, Any]):
    task = task.lower()
    if task == "classification":
        lg_cfg = config.get("logistic", {})
        base = LogisticRegression(
            C=lg_cfg.get("C", 1.0),
            penalty=lg_cfg.get("penalty", "l2"),
            max_iter=lg_cfg.get("max_iter", 2000),
            class_weight=lg_cfg.get("class_weight"),
            solver="lbfgs",
        )
        calib_cfg = config.get("calibration", {})
        if calib_cfg.get("enable", False):
            return CalibratedClassifierCV(
                base, method=calib_cfg.get("method", "isotonic"), cv=3
            )
        return base

    if task == "regression":
        en_cfg = config.get("elasticnet", {})
        return ElasticNet(
            alpha=en_cfg.get("alpha", 0.001),
            l1_ratio=en_cfg.get("l1_ratio", 0.1),
            max_iter=en_cfg.get("max_iter", 5000),
        )

    raise ValueError(f"Unsupported task '{task}' for make_model")
