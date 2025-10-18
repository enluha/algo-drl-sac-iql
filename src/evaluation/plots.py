import plotly.graph_objects as go
import vectorbt as vbt
from src.utils.io_utils import save_html

def candlestick_html(df_ohlc, signals, path_html):
    fig = go.Figure(data=[go.Candlestick(
        x=df_ohlc.index, open=df_ohlc["open"], high=df_ohlc["high"],
        low=df_ohlc["low"], close=df_ohlc["close"]
    )])
    # overlay signals (green up, red down)
    sig = signals.reindex(df_ohlc.index).fillna(0)
    buys = df_ohlc.index[sig.values>0]; sells = df_ohlc.index[sig.values<0]
    fig.add_scatter(
        x=buys,
        y=df_ohlc["close"].reindex(buys),
        mode="markers",
        name="Long",
        marker=dict(symbol="triangle-up", size=14, color="#16a34a", line=dict(width=0)),
    )
    fig.add_scatter(
        x=sells,
        y=df_ohlc["close"].reindex(sells),
        mode="markers",
        name="Short",
        marker=dict(symbol="triangle-down", size=14, color="#dc2626", line=dict(width=0)),
    )
    save_html(fig, path_html)

def equity_html(equity, benchmarks: dict, path_html):
    _ = vbt.Portfolio.from_orders(close=equity.index.to_series().map(lambda _: 1.0),
                                  size=0, freq="H")  # dummy portfolio for plotting
    fig = go.Figure()
    fig.add_scatter(x=equity.index, y=equity.values, name="strategy")
    for k,v in (benchmarks or {}).items():
        fig.add_scatter(x=v.index, y=v.values, name=k)
    save_html(fig, path_html)
