"""
Triple-barrier event labeling aligned with max-holding horizon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(
    close: pd.Series,
    vol: pd.Series,
    pt_mult: float,
    sl_mult: float,
    t_max: int,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute triple-barrier labels.

    Parameters
    ----------
    close : pd.Series
        Price series indexed by timestamps.
    vol : pd.Series
        Rolling volatility (same index) used to size barriers.
    pt_mult : float
        Take-profit multiplier on volatility.
    sl_mult : float
        Stop-loss multiplier on volatility.
    t_max : int
        Maximum holding period in bars.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        y in {-1, 0, +1} and t1 (timestamp of barrier touch or timeout).
    """
    close = close.astype(float)
    vol = vol.reindex(close.index).astype(float).fillna(method="ffill")

    idx = close.index
    up_barrier = close * (1.0 + pt_mult * vol)
    dn_barrier = close * (1.0 - sl_mult * vol)

    t1 = idx.to_series().shift(-t_max)
    y = pd.Series(0, index=idx, dtype=np.int8)

    for i, t0 in enumerate(idx):
        horizon_end = t1.iat[i]
        if pd.isna(horizon_end):
            break  # remaining events do not have full horizon

        window = close.loc[t0:horizon_end]
        if window.empty:
            continue

        cond_up = window >= up_barrier.iat[i]
        cond_dn = window <= dn_barrier.iat[i]

        up_hit_idx = (
            window.index[np.flatnonzero(cond_up.values)[0]]
            if cond_up.values.any()
            else None
        )
        dn_hit_idx = (
            window.index[np.flatnonzero(cond_dn.values)[0]]
            if cond_dn.values.any()
            else None
        )

        if up_hit_idx is not None and (
            dn_hit_idx is None or up_hit_idx <= dn_hit_idx
        ):
            y.iat[i] = 1
            t1.iat[i] = up_hit_idx
        elif dn_hit_idx is not None and (
            up_hit_idx is None or dn_hit_idx <= up_hit_idx
        ):
            y.iat[i] = -1
            t1.iat[i] = dn_hit_idx
        # else y remains 0 until timeout

    return y, t1
