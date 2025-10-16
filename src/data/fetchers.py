# src/data/fetchers.py
"""
MVP - Binance OHLCV fetchers & CSV loader.

This module adapts the standalone downloader you provided into importable
functions for the MVP skeleton, while preserving:
- interval/file naming logic,
- REST pagination & retry behavior,
- CLI usage (so you can still run it as a script).

Public APIs (MVP):
    fetch_ohlcv_from_csv(csv_path) -> pd.DataFrame
        Load a CSV with columns: date, close, open, high, low.
    download_ohlcv_to_csv(symbol, interval_seconds, start, end, output_dir) -> Path
        Download OHLCV from Binance REST and save to CSV; returns the file path.
    fetch_ohlcv_rest(symbol, interval_seconds, start, end) -> pd.DataFrame
        Download OHLCV and return a DataFrame (without saving).

CLI usage (equivalent to your original script):
    python -m src.data.fetchers --symbol BTCUSDT --interval 3600 \
        --start "2024-06-10" --end "2025-10-12" --output-dir data/

Notes:
- Timestamps are UTC; CSV uses format "%d/%m/%Y %H:%M".
- Optional fields (volume, quote_volume, etc.) remain commented for MVP.

NON MVP:
- Funding/open interest/order book endpoints.
- Rate-limit/backoff sophistication beyond simple retry.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import requests

# ---------------------------- Defaults & Const -----------------------------

# Common symbols for future use: ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT, ADAUSDT.

DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL_SECONDS = 3600
DEFAULT_START_DATE = "2024-06-10"   # YYYY-MM-DD
DEFAULT_END_DATE = "2025-10-12"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data"  # repo_root/data

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
INTERVAL_MAP: Dict[int, str] = {
    60: "1m",
    180: "3m",
    300: "5m",
    900: "15m",
    1800: "30m",
    3600: "1h",
    7200: "2h",
    14400: "4h",
    21600: "6h",
    28800: "8h",
    43200: "12h",
    86400: "1d",
    259200: "3d",
    604800: "1w",
    2592000: "1M",
}

DATE_OUTPUT_FMT = "%d/%m/%Y %H:%M"


# ---------------------------- CSV Loader (MVP) -----------------------------

def fetch_ohlcv_from_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    MVP - Load OHLC-like CSV with columns: date, close, open, high, low. Extras optional.

    Parameters
    ----------
    csv_path : str | Path
        Path to a CSV created by this module (or the original downloader).

    Returns
    -------
    pd.DataFrame
        Includes at least ['open', 'high', 'low', 'close'] and any additional columns present.
        Index: tz-aware UTC DatetimeIndex sorted ascending.
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"MVP - CSV not found: {p}")
    df = pd.read_csv(p)
    if "date" not in df.columns:
        raise ValueError("MVP - CSV must contain a 'date' column")
    required = {"close", "open", "high", "low"}
    if not required.issubset(df.columns):
        missing = sorted(required.difference(df.columns))
        raise ValueError(f"MVP - CSV missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], format=DATE_OUTPUT_FMT, utc=True)
    df = df.set_index("date").sort_index()

    core = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "close_time",
        "number_of_trades",
        "taker_buy_volume",
        "taker_buy_quote_volume",
    ]
    available = [c for c in core if c in df.columns]
    return df[available]


# ----------------------------- REST Helpers -------------------------------

def ensure_interval(interval_seconds: int) -> str:
    """
    MVP - Validate interval (seconds) and return Binance interval code.

    Raises
    ------
    ValueError
        If the interval is unsupported.
    """
    if interval_seconds not in INTERVAL_MAP:
        available = ", ".join(str(k) for k in sorted(INTERVAL_MAP))
        raise ValueError(f"Unsupported interval {interval_seconds}. Choose from: {available}")
    return INTERVAL_MAP[interval_seconds]


def format_filename(symbol: str, interval_seconds: int, start: dt.datetime, end: dt.datetime) -> str:
    """
    MVP - Create a descriptive filename for the CSV (same as original logic)."""
    start_tag = start.strftime("%b%Y")
    end_tag = end.strftime("%b%Y")
    return f"{symbol.upper()}{interval_seconds}_{start_tag}_{end_tag}.csv"


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> List[List]:
    """
    MVP - Fetch up to 1000 klines from Binance with simple retry logic.

    Parameters
    ----------
    symbol : str
    interval : str
        Binance interval code (e.g., '1h', '1d').
    start_ms : int
    end_ms : int

    Returns
    -------
    List[List]
        Raw kline rows per Binance API spec. Empty list if none.

    Raises
    ------
    RuntimeError
        If all retries fail.
    """
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 1000,
    }
    retries = 3
    for attempt in range(1, retries + 1):
        response = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
        if response.status_code == 429:
            time.sleep(1.0 * attempt)
            continue
        response.raise_for_status()
        data = response.json()
        if not data:
            return []
        return data
    raise RuntimeError("Failed to fetch klines from Binance after retries")


def save_csv(rows: List[List], path: Path) -> None:
    """
    MVP - Save raw klines to CSV with OHLC plus volume/trade extras.
    """
    header = [
        "date",
        "close",
        "open",
        "high",
        "low",
        "volume",
        "quote_volume",
        "close_time",
        "number_of_trades",
        "taker_buy_volume",
        "taker_buy_quote_volume",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            open_time = dt.datetime.utcfromtimestamp(row[0] / 1000)
            close_time = dt.datetime.utcfromtimestamp(row[6] / 1000)
            writer.writerow(
                [
                    open_time.strftime(DATE_OUTPUT_FMT),
                    row[4],   # close
                    row[1],   # open
                    row[2],   # high
                    row[3],   # low
                    row[5],   # volume (base)
                    row[7],   # quote asset volume
                    close_time.strftime(DATE_OUTPUT_FMT),
                    row[8],   # number of trades
                    row[9],   # taker buy base volume
                    row[10],  # taker buy quote volume
                ]
            )


# --------------------------- High-level APIs (MVP) ------------------------

def download_ohlcv_to_csv(
    symbol: str,
    interval_seconds: int,
    start: str | dt.datetime,
    end: str | dt.datetime,
    output_dir: str | Path,
    sleep_s: float = 0.25,
) -> Path:
    """
    MVP - Download OHLCV from Binance and save to CSV (same behavior as original).

    Parameters
    ----------
    symbol : str
    interval_seconds : int
        Candle size in seconds (e.g., 3600 for 1h, 86400 for 1d).
    start : str | datetime
        Inclusive start (UTC). If str, must be 'YYYY-MM-DD'.
    end : str | datetime
        Exclusive end (UTC). If str, must be 'YYYY-MM-DD'.
    output_dir : str | Path
    sleep_s : float
        Throttle between REST calls.

    Returns
    -------
    Path
        Path to the written CSV file.

    Raises
    ------
    ValueError
        If start >= end or interval unsupported.
    RuntimeError
        If no data returned.
    """
    interval_code = ensure_interval(interval_seconds)

    if isinstance(start, str):
        start_dt = dt.datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    else:
        start_dt = start.astimezone(dt.timezone.utc)

    if isinstance(end, str):
        end_dt = dt.datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    else:
        end_dt = end.astimezone(dt.timezone.utc)

    if start_dt >= end_dt:
        raise ValueError("Start date must precede end date")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / format_filename(symbol, interval_seconds, start_dt, end_dt)

    # Iterate forward in time; Binance returns up to 1000 candles per request
    all_rows: List[List] = []
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    current = start_ms
    while current < end_ms:
        chunk = fetch_klines(symbol, interval_code, current, end_ms)
        if not chunk:
            break
        all_rows.extend(chunk)
        # Move to the next candlestick after the last open time
        current = int(chunk[-1][0]) + interval_seconds * 1000
        time.sleep(sleep_s)

    if not all_rows:
        raise RuntimeError("No data returned; check symbol, interval, or date range")

    # Write CSV (overwrites if exists)
    save_csv(all_rows, output_file)
    return output_file


def fetch_ohlcv_rest(
    symbol: str,
    interval_seconds: int,
    start: str | dt.datetime,
    end: str | dt.datetime,
    sleep_s: float = 0.25,
) -> pd.DataFrame:
    """
    MVP - Download OHLCV from Binance and return a DataFrame (no CSV).

    Parameters
    ----------
    symbol : str
    interval_seconds : int
    start : str | datetime
    end : str | datetime
    sleep_s : float

    Returns
    -------
    pd.DataFrame
        Columns: ['open', 'high', 'low', 'close']
        Index: tz-aware UTC DatetimeIndex sorted ascending.
    """
    interval_code = ensure_interval(interval_seconds)

    if isinstance(start, str):
        start_dt = dt.datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    else:
        start_dt = start.astimezone(dt.timezone.utc)

    if isinstance(end, str):
        end_dt = dt.datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    else:
        end_dt = end.astimezone(dt.timezone.utc)

    if start_dt >= end_dt:
        raise ValueError("Start date must precede end date")

    rows: List[List] = []
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    current = start_ms
    while current < end_ms:
        chunk = fetch_klines(symbol, interval_code, current, end_ms)
        if not chunk:
            break
        rows.extend(chunk)
        current = int(chunk[-1][0]) + interval_seconds * 1000
        time.sleep(sleep_s)

    if not rows:
        raise RuntimeError("No data returned; check symbol, interval, or date range")

    # Convert to DataFrame consistent with CSV loader
    records = []
    for row in rows:
        open_time = dt.datetime.utcfromtimestamp(row[0] / 1000).replace(tzinfo=dt.timezone.utc)
        records.append({
            "date": open_time,
            "close": float(row[4]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
        })
    df = pd.DataFrame.from_records(records).set_index("date").sort_index()
    return df[["open", "high", "low", "close"]]


# ------------------------------- CLI (MVP) --------------------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    """
    MVP - Parse CLI arguments to download and save OHLCV to CSV.
    """
    parser = argparse.ArgumentParser(description="Download OHLCV samples from Binance")
    parser.add_argument(
        "--symbol",
        default=DEFAULT_SYMBOL,
        help=f"Trading pair symbol (default: {DEFAULT_SYMBOL})",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL_SECONDS,
        help="Candle interval in seconds (e.g. 3600 for 1h, 86400 for 1d)",
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START_DATE,
        help="Start date (YYYY-MM-DD). Binance timestamps are UTC.",
    )
    parser.add_argument(
        "--end",
        default=DEFAULT_END_DATE,
        help="End date (YYYY-MM-DD). End is exclusive.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write the CSV (default: repo_root/data).",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    """
    MVP - Entrypoint for script-style usage.
    """
    args = parse_args(argv)

    # Download & save
    out_file = download_ohlcv_to_csv(
        symbol=args.symbol,
        interval_seconds=args.interval,
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
    )
    print(f"Saved candles to {out_file}")


if __name__ == "__main__":
    # Allow: python -m src.data.fetchers --symbol BTCUSDT --interval 3600 ...
    main(sys.argv[1:])
