from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd


ANNUALIZATION_HOURS = 24 * 365


def _sharpe(returns: pd.Series) -> float:
    if returns.std() == 0:
        return 0.0
    return math.sqrt(ANNUALIZATION_HOURS) * returns.mean() / returns.std()


def _sortino(returns: pd.Series) -> float:
    downside = returns[returns < 0]
    if downside.empty or downside.std() == 0:
        return 0.0
    return math.sqrt(ANNUALIZATION_HOURS) * returns.mean() / downside.std()


def _cagr(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0 if equity.iloc[0] != 0 else 0.0
    days = (equity.index[-1] - equity.index[0]).total_seconds() / (3600 * 24)
    if days <= 0:
        return 0.0
    years = days / 365
    return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0


def _max_drawdown(equity: pd.Series) -> float:
    cumulative = equity.cummax()
    drawdowns = (equity - cumulative) / cumulative
    return float(drawdowns.min()) if not drawdowns.empty else 0.0


def summarize(ledger: pd.DataFrame, equity: pd.Series) -> Dict[str, float]:
    returns = ledger["net_ret"].fillna(0.0)
    sharpe = _sharpe(returns)
    sortino = _sortino(returns)
    cagr = _cagr(equity)
    max_dd = _max_drawdown(equity)
    calmar = abs(cagr / max_dd) if max_dd != 0 else 0.0
    exposure = ledger["weight"].abs().mean()
    turnover = ledger["weight"].diff().abs().sum()
    trades = (ledger["weight"].diff().abs() > 1e-3).sum()
    cost_share = ledger["cost"].sum() / returns.abs().sum() if returns.abs().sum() > 0 else 0.0

    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "cagr": float(cagr),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
        "exposure": float(exposure),
        "trade_count": int(trades),
        "turnover": float(turnover),
        "cost_share": float(cost_share),
    }
