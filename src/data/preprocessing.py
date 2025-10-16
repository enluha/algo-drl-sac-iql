"""
MVP - Basic preprocessing for bar alignment and NA policy.

This module enforces a uniform time grid (matching the desired bar),
handles NA strategy, and provides simple helpers used by the training
and backtesting flows.

Functions:
    clean_and_resample(...)  # MVP - align index to bar and apply NA policy
    infer_pandas_freq(...)   # MVP - best-effort inference of frequency
    ensure_freq_index(...)   # MVP - reindex to a complete range for the bar
"""

from __future__ import annotations

from typing import Literal

import pandas as pd


# Map e.g. "1h" -> pandas frequency alias "1H"
_BAR_TO_FREQ = {
    "1min": "1T", "5min": "5T", "15min": "15T", "30min": "30T",
    "1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "8h": "8H", "12h": "12H",
    "1d": "1D",
}


def infer_pandas_freq(index: pd.DatetimeIndex) -> str | None:
    """
    MVP - Infer the most likely uniform frequency of an index.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Time index (ideally tz-aware UTC).

    Returns
    -------
    str | None
        Pandas frequency string (e.g., "1H") or None if unknown.
    """
    try:
        return pd.infer_freq(index)
    except Exception:
        return None


def ensure_freq_index(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    MVP - Reindex a DataFrame to a complete range at the specified frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame (UTC tz-aware recommended).
    freq : str
        Pandas frequency alias (e.g., "1H").

    Returns
    -------
    pd.DataFrame
        Reindexed DataFrame with a complete regular grid.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("MVP - ensure_freq_index: DataFrame index must be DatetimeIndex")
    start, end = df.index.min(), df.index.max()
    full_idx = pd.date_range(start=start, end=end, freq=freq, tz=df.index.tz)
    return df.reindex(full_idx)


def clean_and_resample(
    df: pd.DataFrame,
    bar: str,
    na_policy: Literal["ffill", "drop"] = "ffill",
) -> pd.DataFrame:
    """
    MVP - Enforce target bar frequency and apply a simple NA strategy.

    Behavior
    --------
    - If the inferred frequency equals the target, just apply NA policy.
    - Otherwise, reindex to the target frequency grid (no aggregation),
      then apply NA policy. This preserves price columns without resampling
      math (MVP). You can add true resampling logic later (NON MVP).

    Parameters
    ----------
    df : pd.DataFrame
        Input OHLC (columns: 'open','high','low','close'), DatetimeIndex.
    bar : str
        Target bar, e.g. "1h", "1d". See _BAR_TO_FREQ.
    na_policy : {"ffill", "drop"}, default "ffill"
        NA fill strategy:
            - "ffill": forward-fill then drop leading NAs
            - "drop": drop any rows with NA

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame on a regular grid.

    Raises
    ------
    KeyError
        If required columns are missing.
    ValueError
        If bar is unsupported.
    """
    required = {"open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"MVP - clean_and_resample missing columns: {sorted(missing)}")

    if bar not in _BAR_TO_FREQ:
        raise ValueError(f"MVP - Unsupported bar '{bar}'. Supported: {sorted(_BAR_TO_FREQ)}")

    target_freq = _BAR_TO_FREQ[bar]
    inferred = infer_pandas_freq(df.index)

    if inferred != target_freq:
        # Align to full grid for the target frequency.
        df = ensure_freq_index(df, target_freq)

    if na_policy == "ffill":
        df = df.ffill()
        # Drop any remaining NAs that couldn't be forward-filled (leading NAs).
        df = df.dropna()
    elif na_policy == "drop":
        df = df.dropna()
    else:
        raise ValueError("MVP - na_policy must be 'ffill' or 'drop'")

    # Ensure sorted & unique index after reindexing
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df
