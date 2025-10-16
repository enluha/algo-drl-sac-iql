"""
MVP - Volatility-targeted position sizing.
"""

from __future__ import annotations

import numpy as np


def vol_targeted_size(signal: np.ndarray, vol: np.ndarray, risk_aversion: float, w_max: float) -> np.ndarray:
    """MVP - Compute bounded weights from discrete signals and volatility.

    Params:
        signal: array of {-1, 0, 1} trading signals.
        vol: volatility series aligned with `signal`.
        risk_aversion: scaling factor for target volatility.
        w_max: cap on absolute position size.

    Returns:
        Array of weights clipped to [-w_max, w_max].
    """
    vol = np.clip(vol, 1e-6, None)
    weights = signal / (risk_aversion * vol)
    return np.clip(weights, -w_max, w_max)
