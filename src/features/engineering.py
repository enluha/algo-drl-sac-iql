from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _parkinson_rv(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    hl = (np.log(high) - np.log(low)) ** 2
    return (hl.rolling(window).mean() / (4 * np.log(2))).pow(0.5)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct candle-based features for DRL state inputs."""
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df.get("volume", pd.Series(0.0, index=df.index)).astype(float)
    taker_buy_volume = df.get("taker_buy_base_volume")

    returns = {
        f"ret_{h}": close.pct_change(periods=h) for h in [1, 3, 6, 12, 24]
    }
    ewma_vol = close.pct_change().rolling(20).std().ewm(span=20, adjust=False).mean()

    rsi_feats = {f"rsi_{p}": _rsi(close, p) for p in [6, 14, 21]}

    ma24 = close.rolling(24).mean()
    ma48 = close.rolling(48).mean()
    ma96 = close.rolling(96).mean()
    ma_slopes = {
        "ma24_slope": ma24.pct_change(),
        "ma48_slope": ma48.pct_change(),
        "ma96_slope": ma96.pct_change(),
    }

    macd_fast = _ema(close, 12)
    macd_slow = _ema(close, 26)
    macd = macd_fast - macd_slow
    signal = _ema(macd, 9)

    atr = _atr(high, low, close, 14)
    parkinson = _parkinson_rv(high, low, 20)

    donchian_high = high.rolling(20).max()
    donchian_low = low.rolling(20).min()
    price_range = high - low

    features = pd.DataFrame(index=df.index)
    for name, series in returns.items():
        features[name] = series
    features["ewma_vol20"] = ewma_vol
    for name, series in rsi_feats.items():
        features[name] = series
    for name, series in ma_slopes.items():
        features[name] = series
    features["macd"] = macd
    features["macd_signal"] = signal
    features["macd_hist"] = macd - signal
    features["atr14"] = atr
    features["parkinson20"] = parkinson
    features["donchian_pos"] = (close - donchian_low) / (donchian_high - donchian_low + 1e-9)
    features["range_close_ratio"] = price_range / close.replace(0, np.nan)
    features["above_ma96"] = (close - ma96) / ma96.replace(0, np.nan)
    features["volume_z"] = (volume - volume.rolling(120).mean()) / (
        volume.rolling(120).std() + 1e-9
    )
    if taker_buy_volume is not None:
        taker_ratio = taker_buy_volume.astype(float) / volume.replace(0, np.nan)
        features["taker_pressure"] = taker_ratio.fillna(0.0)
    else:
        features["taker_pressure"] = 0.0

    # Cyclical time features
    hours = df.index.hour
    days = df.index.dayofweek
    features["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    features["dow_sin"] = np.sin(2 * np.pi * days / 7)
    features["dow_cos"] = np.cos(2 * np.pi * days / 7)

    features = features.replace([np.inf, -np.inf], np.nan).fillna(method="ffill")
    features = features.fillna(0.0)
    return features.astype(np.float32)
