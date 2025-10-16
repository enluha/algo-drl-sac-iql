"""
MVP - Core technical features (numpy-first).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int) -> pd.Series:
    """MVP - Vectorized RSI calculation.

    Params:
        close: price series indexed by datetime.
        period: lookback window for RSI.

    Returns:
        RSI series scaled 0-100 aligned with `close`.
    """
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).ewm(alpha=1 / period, adjust=False).mean()
    roll_down = pd.Series(down, index=close.index).ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))


def log_returns(close: pd.Series, lag: int = 1) -> pd.Series:
    """MVP - Log return over `lag` bars.

    Params:
        close: price series.
        lag: positive integer offset.

    Returns:
        Series of log returns.
    """
    return np.log(close / close.shift(lag))


def ewma_vol(returns: pd.Series, span: int) -> pd.Series:
    """MVP - EWMA volatility estimator (standard deviation).

    Params:
        returns: return series.
        span: EWMA span parameter.

    Returns:
        Series of EWMA volatility values.
    """
    return returns.ewm(span=span, adjust=False).std()


def ema(s: pd.Series, span: int) -> pd.Series:
    """Exponential moving average with numeric stability."""
    return s.ewm(span=span, adjust=False).mean()


def sma(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Simple moving average helper."""
    return s.rolling(window, min_periods=min_periods or window).mean()


def ma_slope(s: pd.Series, window: int) -> pd.Series:
    """Normalized slope of the moving average (momentum proxy)."""
    m = sma(s, window)
    return m.diff() / (s.shift(1).abs() + 1e-12)


def macd_like(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """MACD-style difference between fast and slow EMAs."""
    return ema(close, fast) - ema(close, slow)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range for range expansion."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def parkinson_rv(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """Parkinson range-based volatility estimate."""
    rs = (np.log(high / low)) ** 2
    return (rs.rolling(window, min_periods=window).mean() / (4 * np.log(2))).pow(0.5)


def donchian_distance(close: pd.Series, window: int = 20) -> pd.Series:
    """Normalized position of close within Donchian channel."""
    hh = close.rolling(window, min_periods=window).max()
    ll = close.rolling(window, min_periods=window).min()
    rng = (hh - ll).replace(0, np.nan)
    return (close - ll) / rng


def cyclical_time_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Encode hour-of-day / day-of-week cyclically (sin/cos)."""
    hod = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    phi_h = 2 * np.pi * hod / 24.0
    phi_d = 2 * np.pi * dow / 7.0
    return pd.DataFrame(
        {
            "hod_sin": np.sin(phi_h),
            "hod_cos": np.cos(phi_h),
            "dow_sin": np.sin(phi_d),
            "dow_cos": np.cos(phi_d),
        },
        index=idx,
    )


def trend_filter(close: pd.Series, span: int = 48, enable: bool = True) -> pd.Series:
    """Return boolean filter indicating long-only regime if enabled."""
    if not enable:
        return pd.Series(True, index=close.index)
    ma = close.ewm(span=span, adjust=False).mean()
    return close >= ma
