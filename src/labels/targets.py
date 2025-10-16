"""
MVP - Regression target utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def k_ahead_log_return(close: pd.Series, k: int) -> pd.Series:
    """MVP - Compute k-step-ahead log return.

    Params:
        close: price series.
        k: forward horizon in bars.

    Returns:
        Series of forward log returns with NaNs at the tail for alignment.
    """
    return np.log(close.shift(-k) / close)


def ewma_vol(returns: pd.Series, span: int = 20) -> pd.Series:
    """EWMA volatility helper shared by labeling routines."""
    return returns.ewm(span=span, adjust=False).std()


def make_labels(
    y_reg: pd.Series,
    vol_span: int,
    mode: str = "direction",
    vol_k_sigma: float = 0.5,
) -> pd.Series:
    """
    Build classification labels from regression targets.

    Parameters
    ----------
    y_reg : pd.Series
        Forward returns (log) series.
    vol_span : int
        EWMA span for volatility normalization (vol_threshold mode).
    mode : {"direction", "vol_threshold"}
        Labeling regime.
    vol_k_sigma : float, default 0.5
        Sigma multiplier for vol_threshold.

    Returns
    -------
    pd.Series
        Label series aligned with y_reg.
    """
    if mode == "direction":
        return (y_reg > 0.0).astype("int")

    if mode == "vol_threshold":
        sigma = ewma_vol(y_reg, span=vol_span)
        up = (y_reg > vol_k_sigma * sigma).astype("int")
        dn = (y_reg < -vol_k_sigma * sigma).astype("int")
        lab = up.where(up == 1, np.nan)
        lab = lab.where(~dn.astype(bool), 0)
        return lab

    raise ValueError("Unknown labeling mode")
