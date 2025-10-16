"""
MVP - Data loaders: read OHLCV from CSV and slice to the requested window.

This module centralizes loading raw price data into a standard DataFrame
format that downstream feature engineering and modeling can rely on.

Functions:
    load_all(...)           # MVP - main entrypoint used by training/backtest
    _slice_time_window(...) # MVP - reusable time filtering helper
    _validate_columns(...)  # MVP - sanity check for required columns

Notes:
- For MVP we load from CSV produced by scripts/download_ohlcv_binance.py.
- NON MVP: Add REST loaders for funding, OI, orderbook, etc., and caching/parquet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from .fetchers import fetch_ohlcv_from_csv


def _validate_columns(df: pd.DataFrame) -> None:
    """
    MVP - Ensure the OHLCV DataFrame has the required price columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame expected to contain ['open', 'high', 'low', 'close'] columns.

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    required = {"open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"MVP - Missing required columns: {sorted(missing)}")


def _slice_time_window(
    df: pd.DataFrame,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
) -> pd.DataFrame:
    """
    MVP - Slice a DataFrame by inclusive start and exclusive end timestamps.

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame.
    start : str | pd.Timestamp | None
        Inclusive left bound (UTC). If None, start at the beginning.
    end : str | pd.Timestamp | None
        Exclusive right bound (UTC). If None, go to the end.

    Returns
    -------
    pd.DataFrame
        Sliced DataFrame.
    """
    if start is None and end is None:
        return df
    if start is None:
        return df.loc[:end]
    if end is None:
        return df.loc[start:]
    return df.loc[start:end]


def load_all(
    symbol: str,
    bar: str,
    start: str,
    end: str,
    cache: str,
    na_policy: Literal["ffill", "drop"],
    source: Literal["csv", "rest"],
    csv_path: str | Path,
) -> pd.DataFrame:
    """
    MVP - Load OHLCV from CSV and trim to [start, end].

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., "BTCUSDT"). Currently informational in MVP.
    bar : str
        Target bar size (e.g., "1h"). CSV should already match.
    start : str
        ISO-like timestamp or date (interpreted by pandas) - inclusive.
    end : str
        ISO-like timestamp or date - exclusive in downstream splits.
    cache : str
        Cache directory path. (NON MVP: parquet cache, not used here.)
    na_policy : {"ffill", "drop"}
        NA handling hint for later stages. (Not applied here.)
    source : {"csv", "rest"}
        Data source selection. MVP supports only "csv".
    csv_path : str | Path
        Path to CSV produced by the downloader.

    Returns
    -------
    pd.DataFrame
        Index: UTC (tz-aware) timestamps sorted ascending.
        Columns: ["open", "high", "low", "close"].

    Raises
    ------
    NotImplementedError
        If source != "csv" in MVP.
    ValueError
        If CSV does not contain required columns.
    """
    if source != "csv":
        raise NotImplementedError("MVP - load_all currently supports source='csv' only.")

    df = fetch_ohlcv_from_csv(csv_path)
    _validate_columns(df)
    df = _slice_time_window(df, start, end)

    # Ensure sorted & unique index
    df = df[~df.index.duplicated(keep="first")].sort_index()

    if df.empty:
        raise ValueError(
            f"MVP - No data in the requested window [{start}, {end}) from {csv_path}"
        )
    return df
