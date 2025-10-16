"""
Enhanced feature assembly: trend, volatility, volume pressure, cyclical time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .technical import (
    rsi,
    log_returns,
    ewma_vol,
    ema,
    sma,
    ma_slope,
    macd_like,
    atr,
    parkinson_rv,
    donchian_distance,
    cyclical_time_features,
)


def assemble_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    low = df["low"]

    X = pd.DataFrame(index=df.index, dtype=float)

    # Returns / volatility horizon stack
    X["ret_1"] = log_returns(close, 1)
    X["ret_3"] = log_returns(close, 3)
    X["ret_6"] = log_returns(close, 6)
    X["ret_12"] = log_returns(close, 12)
    X["ewma_vol_20"] = ewma_vol(X["ret_1"], span=20)

    # RSI ladder
    X["rsi_6"] = rsi(close, 6)
    X["rsi_14"] = rsi(close, 14)
    X["rsi_21"] = rsi(close, 21)

    # Trend & momentum diagnostics
    X["macd_12_26"] = macd_like(close, 12, 26)
    X["ma_slope_24"] = ma_slope(close, 24)
    X["ma_slope_48"] = ma_slope(close, 48)
    X["ma_slope_96"] = ma_slope(close, 96)
    ma96 = sma(close, 96)
    X["pct_above_ma96"] = (close / (ma96 + 1e-12)) - 1.0

    # Range / vol / breakout structure
    X["atr_14"] = atr(high, low, close, 14)
    X["parkinson_20"] = parkinson_rv(high, low, 20)
    X["donchian_20"] = donchian_distance(close, 20)
    X["range_close"] = (high - low) / (close.replace(0, np.nan) + 1e-12)

    # Volume pressure (optional columns)
    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce")
        vol_mean = vol.rolling(96, min_periods=20).mean()
        vol_std = vol.rolling(96, min_periods=20).std()
        X["vol_z"] = (vol - vol_mean) / (vol_std + 1e-12)
        X["vol_chg"] = vol.pct_change()
        X["dollar_vol"] = (vol * close).pct_change()

        if "taker_buy_volume" in df.columns:
            tb = pd.to_numeric(df["taker_buy_volume"], errors="coerce")
            vol_safe = vol.replace(0, np.nan)
            X["taker_pressure"] = ((tb - (vol - tb)) / vol_safe).clip(-1, 1)

        if "quote_volume" in df.columns and "number_of_trades" in df.columns:
            qv = pd.to_numeric(df["quote_volume"], errors="coerce")
            ntr = pd.to_numeric(df["number_of_trades"], errors="coerce").replace(0, np.nan)
            avg_trade = qv / ntr
            X["dollars_per_trade"] = avg_trade / (avg_trade.rolling(96, min_periods=20).mean() + 1e-12) - 1.0

    if "number_of_trades" in df.columns and "volume" not in df.columns:
        trades = pd.to_numeric(df["number_of_trades"], errors="coerce")
        trades_mean = trades.rolling(96, min_periods=20).mean()
        trades_std = trades.rolling(96, min_periods=20).std()
        X["trades_z"] = (trades - trades_mean) / (trades_std + 1e-12)

    # Session / seasonality encodings
    X = X.join(cyclical_time_features(df.index))

    return X
