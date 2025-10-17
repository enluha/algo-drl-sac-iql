from __future__ import annotations

from typing import Dict

import plotly.graph_objects as go
import vectorbt as vbt

_ = vbt.__version__

from src.utils.io_utils import save_html


def candlestick(df_ohlc, signals, path_html) -> None:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df_ohlc.index,
                open=df_ohlc["open"],
                high=df_ohlc["high"],
                low=df_ohlc["low"],
                close=df_ohlc["close"],
                name="Price",
            )
        ]
    )
    if signals is not None and len(signals) == len(df_ohlc):
        long_idx = signals[signals > 0].index
        short_idx = signals[signals < 0].index
        fig.add_trace(
            go.Scatter(
                x=long_idx,
                y=df_ohlc.loc[long_idx, "close"],
                mode="markers",
                marker=dict(color="green", size=6),
                name="Long",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=short_idx,
                y=df_ohlc.loc[short_idx, "close"],
                mode="markers",
                marker=dict(color="red", size=6),
                name="Short",
            )
        )
    fig.update_layout(title="Candlestick with signals", xaxis_title="Time", yaxis_title="Price")
    save_html(fig, path_html)


def equity_plot(equity, benchmarks: Dict[str, any], path_html) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Strategy"))
    for name, series in benchmarks.items():
        fig.add_trace(
            go.Scatter(x=series.index, y=series.values, mode="lines", name=name, line=dict(dash="dash"))
        )
    fig.update_layout(title="Equity Curves", xaxis_title="Time", yaxis_title="Equity")
    save_html(fig, path_html)
