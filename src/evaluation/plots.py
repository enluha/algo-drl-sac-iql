from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import plotly.graph_objects as go
import vectorbt as vbt

from src.utils.io_utils import save_html


def candlestick(df_ohlc: pd.DataFrame, signals: pd.Series | None, path_html: Path | str) -> None:
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
    if signals is not None:
        fig.add_trace(
            go.Scatter(
                x=signals.index,
                y=df_ohlc.loc[signals.index, "close"],
                mode="markers",
                marker=dict(color="#1f77b4", size=6),
                name="Signals",
            )
        )
    fig.update_layout(template="plotly_dark")
    save_html(fig, path_html)


def equity_plot(equity: pd.DataFrame, benchmarks: Dict[str, pd.Series], path_html: Path | str) -> None:
    data = equity.copy()
    for name, series in benchmarks.items():
        data[name] = series.reindex(data.index, method="ffill")
    fig = data.vbt.plot()
    save_html(fig, path_html)
