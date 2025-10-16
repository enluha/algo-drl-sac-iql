"""
MVP - Risk/return metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


def sharpe(net_ret: np.ndarray, freq_per_year: int) -> float:
    """MVP - Annualized Sharpe ratio."""
    mean_ret = np.nanmean(net_ret) * freq_per_year
    std_ret = np.nanstd(net_ret) * np.sqrt(freq_per_year) + 1e-12
    return float(mean_ret / std_ret)


def max_drawdown(equity: pd.Series) -> float:
    """MVP - Maximum drawdown from equity curve."""
    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1.0
    return float(drawdown.min())


def summarize(ledger: pd.DataFrame, equity: pd.Series) -> dict:
    """MVP - Collect basic performance stats.

    Params:
        ledger: DataFrame from simulator.
        equity: cumulative equity curve.

    Returns:
        Dictionary with Sharpe, max drawdown, and final equity.
    """
    if "net" in ledger.columns:
        net_series = ledger["net"]
    else:
        net_cols = [col for col in ledger.columns if col.startswith("net_log_return")]
        if not net_cols:
            raise KeyError("Ledger missing net return column.")
        net_series = ledger[net_cols[0]]

    net = net_series.dropna().to_numpy()
    return {
        "sharpe_365d": sharpe(net, 365 * 24),  # assuming hourly bars
        "max_drawdown": max_drawdown(equity),
        "final_equity": float(equity.iloc[-1]),
    }


def calibration_table(y_true: pd.Series, p_hat: pd.Series, q: int = 10) -> list[dict]:
    cuts = np.quantile(p_hat, np.linspace(0.0, 1.0, q + 1))
    bins = pd.cut(p_hat, cuts, include_lowest=True, duplicates="drop")
    df = pd.DataFrame({"p": p_hat, "y": (y_true > 0).astype(int)})
    grouped = df.groupby(bins, observed=True).agg(
        p_mean=("p", "mean"),
        y_rate=("y", "mean"),
        count=("y", "size"),
    )
    grouped = grouped.replace({np.nan: 0.0})
    return grouped.reset_index(drop=True).to_dict(orient="records")


def classification_report(y_true: pd.Series, p_hat: pd.Series) -> dict:
    y_bin = (y_true > 0).astype(int)
    return {
        "auc": float(roc_auc_score(y_bin, p_hat)),
        "brier": float(brier_score_loss(y_bin, p_hat)),
        "calibration": calibration_table(y_true, p_hat),
    }
