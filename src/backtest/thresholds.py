"""
Threshold optimiser based on expected net P&L.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def pick_threshold(
    pred: pd.Series,
    ret_fwd: pd.Series,
    costs_bps: dict,
    grid: Iterable[float],
    min_trades: int = 30,
    long_only: bool = True,
) -> dict:
    """
    Choose probability threshold tau maximising expected net pnl:
    E[ side * ret_fwd ] - costs.
    """
    pred = pred.astype(float)
    ret_fwd = ret_fwd.reindex(pred.index).astype(float).fillna(0.0)

    fee = (costs_bps.get("commission_bps", 0.0) + costs_bps.get("slippage_bps", 0.0)) * 1e-4
    short_extra = costs_bps.get("short_extra_bps", 0.0) * 1e-4

    best = {"tau": 0.5, "ev": -np.inf, "turns": 0.0}

    for tau in grid:
        long_mask = pred >= tau
        if long_only:
            side = pd.Series(0, index=pred.index, dtype=np.int8)
            side[long_mask] = 1
        else:
            short_mask = pred <= (1 - tau)
            side = pd.Series(0, index=pred.index, dtype=np.int8)
            side[long_mask] = 1
            side[short_mask] = -1

        trades = side.ne(0)
        n_trades = trades.sum()
        if n_trades < min_trades:
            continue

        gross = (side * ret_fwd).mean()
        turns = n_trades / len(side)
        short_penalty = short_extra * (side.eq(-1).sum() / len(side))
        net = gross - fee * turns - short_penalty

        if net > best["ev"]:
            best = {"tau": float(tau), "ev": float(net), "turns": float(turns)}

    return best
