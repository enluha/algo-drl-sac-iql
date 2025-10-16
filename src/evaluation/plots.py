"""
MVP - Interactive charts using Plotly + vectorbt + qf helper.

- Equity curve: vectorbt plot (application/vnd.plotly.v1+json)
- Candlestick with entries/exits: qf.candlestick(...) helper

Requirements:
  pip install plotly vectorbt
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:  # pragma: no cover - optional dependency
    import vectorbt as vbt  # type: ignore
except Exception:  # pragma: no cover
    vbt = None


class qf:
    """MVP - Minimal chart helper namespace."""

    @staticmethod
    def candlestick(df: pd.DataFrame, signals: pd.Series | np.ndarray, entry_trades: bool = True):
        """Return a Plotly candlestick with buy/sell markers."""
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df.index,
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                )
            ]
        )
        signal_series = pd.Series(signals, index=df.index)
        buys = signal_series[signal_series > 0].index
        sells = signal_series[signal_series < 0].index
        fig.add_trace(go.Scatter(x=buys, y=df.loc[buys, "close"], mode="markers", name="Buy"))
        fig.add_trace(go.Scatter(x=sells, y=df.loc[sells, "close"], mode="markers", name="Sell"))
        fig.update_layout(title="Candlestick with Signals", xaxis_rangeslider_visible=False)
        return fig


def plot_equity_with_vectorbt(
    equity: pd.Series,
    benchmarks: dict[str, pd.Series] | None = None,
):
    """Plot equity curve using vectorbt when available, else fall back to Plotly."""
    series = {equity.name or "equity": equity}
    if benchmarks:
        for name, bench in benchmarks.items():
            series[name] = bench.reindex(equity.index)

    df = pd.DataFrame(series)

    if vbt is not None:
        try:
            if hasattr(vbt, "Chart"):
                return vbt.Chart(df).fig
            accessor = vbt.GenericDFAccessor(df)
            return accessor.vbt.plot()
        except Exception:
            pass

    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))
    fig.update_layout(title="Equity Curve", xaxis_title="Time", yaxis_title="Normalized Value")
    return fig


def reliability_plot(calibration_rows: list[dict]):
    if not calibration_rows:
        return go.Figure()
    df = pd.DataFrame(calibration_rows)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["p_mean"],
            y=df["y_rate"],
            mode="lines+markers",
            name="Observed",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Ideal",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        title="Reliability Curve",
        xaxis_title="Predicted probability",
        yaxis_title="Observed frequency",
    )
    return fig

