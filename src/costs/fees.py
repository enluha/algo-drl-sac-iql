"""
MVP - Commission and slippage cost model (bps).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def roundtrip_cost_bps(slippage_bps: float, commission_bps: float) -> float:
    """MVP - Approximate roundtrip bps cost (enter+exit full).

    Params:
        slippage_bps: slippage per side in basis points.
        commission_bps: commission per side in basis points.

    Returns:
        Total roundtrip cost in basis points.
    """
    return 2.0 * (slippage_bps + commission_bps)


def apply_trade_costs(weights: np.ndarray, slippage_bps: float, commission_bps: float) -> pd.Series:
    """MVP - Costs proportional to |position|.

    Params:
        weights: position series in [-1, 1].
        slippage_bps: slippage per side in basis points.
        commission_bps: commission per side in basis points.

    Returns:
        Pandas Series of per-step transaction costs expressed as returns.
    """
    series = pd.Series(weights)
    delta = series.diff().abs().fillna(abs(series.iloc[0]))
    per_side = (slippage_bps + commission_bps) / 1e4
    return delta * per_side
