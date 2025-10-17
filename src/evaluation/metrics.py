from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


EPS = 1e-12
HOURS_PER_YEAR = 24 * 365


def summarize(ledger: pd.DataFrame, equity: pd.Series) -> Dict[str, float]:
    returns = ledger["net_ret"].fillna(0.0)
    mean_ret = returns.mean()
    std_ret = returns.std(ddof=0) + EPS
    sharpe = (mean_ret * HOURS_PER_YEAR) / (std_ret * np.sqrt(HOURS_PER_YEAR))

    downside = returns[returns < 0]
    downside_std = np.sqrt((downside.pow(2).mean() or 0.0) + EPS)
    sortino = (mean_ret * HOURS_PER_YEAR) / (downside_std * np.sqrt(HOURS_PER_YEAR))

    total_hours = len(equity)
    years = max(total_hours / HOURS_PER_YEAR, EPS)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1

    running_max = equity.cummax()
    drawdowns = equity / running_max - 1.0
    max_dd = drawdowns.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    exposure = ledger["weight"].abs().mean()
    weight_diff = ledger["weight"].diff().abs().fillna(0.0)
    trades = int((weight_diff > 1e-3).sum())
    turnover = weight_diff.sum()

    gross = ledger["raw_ret"].abs().sum() + EPS
    cost_share = ledger["cost"].sum() / gross

    return {
        "sharpe_hourly": float(sharpe),
        "sortino_hourly": float(sortino),
        "cagr": float(cagr),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar if np.isfinite(calmar) else 0.0),
        "exposure": float(exposure),
        "trade_count": trades,
        "turnover": float(turnover),
        "cost_share": float(cost_share),
    }
