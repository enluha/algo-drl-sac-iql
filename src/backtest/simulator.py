"""
Enhanced simulator with explicit position decisions and timeouts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def decide_side(prob: float, tau: float, long_only: bool = True) -> int:
    if long_only:
        return 1 if prob >= tau else 0
    if prob >= tau:
        return 1
    if prob <= (1.0 - tau):
        return -1
    return 0


def simulate(
    prices: pd.Series,
    predictions: pd.Series,
    tau: float,
    t_max: int,
    costs_cfg: dict,
    long_only: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    prices = prices.astype(float)
    preds = predictions.astype(float).reindex(prices.index)

    fee = (costs_cfg.get("commission_bps", 0.0) + costs_cfg.get("slippage_bps", 0.0)) * 1e-4
    short_extra = costs_cfg.get("short_extra_bps", 0.0) * 1e-4

    positions = []
    trade_costs = []
    entry_idx = None
    current = 0

    for i, prob in enumerate(preds):
        desired = decide_side(prob, tau, long_only=long_only)

        if current != 0 and entry_idx is not None and (i - entry_idx) >= t_max:
            desired = 0 if desired == current else desired

        if desired != current:
            change = abs(desired - current)
            cost = fee * change
            if desired == -1 or current == -1:
                cost += short_extra
            trade_costs.append(cost)
            current = desired
            entry_idx = i if current != 0 else None
        else:
            trade_costs.append(0.0)
            if current != 0 and entry_idx is None:
                entry_idx = i

        positions.append(current)

    position_series = pd.Series(positions, index=preds.index, dtype=float)
    cost_series = pd.Series(trade_costs, index=preds.index, dtype=float)

    returns = prices.pct_change().fillna(0.0)
    gross = position_series.shift(1).fillna(0.0) * returns
    net = gross - cost_series
    equity = (1.0 + net).cumprod()

    ledger = pd.DataFrame(
        {
            "position": position_series,
            "gross": gross,
            "costs": cost_series,
            "net": net,
            "price": prices,
        }
    )

    return ledger, equity
