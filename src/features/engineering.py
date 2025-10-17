"""Candle-based feature engineering for DRL state construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


_EPS = 1e-8


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + _EPS)
    return 100 - (100 / (1 + rs))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_datetime_index(df)
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df.get("high", close), errors="coerce")
    low = pd.to_numeric(df.get("low", close), errors="coerce")
    volume = pd.to_numeric(df.get("volume"), errors="coerce") if "volume" in df else None

    features = pd.DataFrame(index=df.index, dtype=np.float32)

    log_close = np.log(close.replace(0, np.nan))
    for horizon in (1, 3, 6, 12, 24):
        features[f"ret_{horizon}"] = log_close.diff(horizon)

    features["ewma_vol_20"] = features["ret_1"].ewm(span=20, adjust=False).std()

    for period in (6, 14, 21):
        features[f"rsi_{period}"] = _rsi(close, period)

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    features["macd_12_26"] = ema_fast - ema_slow

    for window in (24, 48, 96):
        sma = close.rolling(window, min_periods=window).mean()
        slope = (sma - sma.shift(window)) / (window + _EPS)
        features[f"ma_slope_{window}"] = slope

    ma96 = close.rolling(96, min_periods=96).mean()
    features["pct_above_ma96"] = close / (ma96 + _EPS) - 1.0

    true_range = pd.concat(
        [
            (high - low),
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    features["atr_14"] = true_range.rolling(14, min_periods=14).mean()

    log_hl = np.log((high + _EPS) / (low + _EPS)).pow(2)
    coeff = 1.0 / (4.0 * np.log(2.0))
    features["parkinson_20"] = coeff * log_hl.rolling(20, min_periods=20).mean()

    highest = high.rolling(20, min_periods=20).max()
    lowest = low.rolling(20, min_periods=20).min()
    mid = (highest + lowest) / 2.0
    span = (highest - lowest) / 2.0
    features["donchian_20"] = (close - mid) / (span + _EPS)

    features["range_close"] = (high - low) / (close.abs() + _EPS)

    if volume is not None:
        vol_mean = volume.rolling(96, min_periods=24).mean()
        vol_std = volume.rolling(96, min_periods=24).std()
        features["volume_z"] = (volume - vol_mean) / (vol_std + _EPS)

    if volume is not None and "taker_buy_volume" in df:
        tb = pd.to_numeric(df["taker_buy_volume"], errors="coerce")
        denom = volume.replace(0, np.nan)
        features["taker_pressure"] = ((tb - (volume - tb)) / (denom + _EPS)).clip(-1, 1)

    # Cyclical time encodings
    hours = df.index.hour
    days = df.index.dayofweek
    features["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    features["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    features["day_sin"] = np.sin(2 * np.pi * days / 7.0)
    features["day_cos"] = np.cos(2 * np.pi * days / 7.0)

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.ffill().fillna(0.0)
    return features.astype(np.float32)
