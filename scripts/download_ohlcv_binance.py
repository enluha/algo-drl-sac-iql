#!/usr/bin/env python3
"""
MVP - READY: Download OHLCV samples from Binance and save to CSV.

Usage:
    python scripts/download_ohlcv_binance.py --symbol BTCUSDT --interval 3600 \
        --start "2024-06-10" --end "2025-10-12" --output-dir data/

Notes:
- Produces columns: date, close, open, high, low (UTC timestamps).
- Extend header to include volume/quote_volume/trades if needed later (NON MVP).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
import time
from pathlib import Path
from typing import List

import requests

DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL_SECONDS = 3600
DEFAULT_START_DATE = "2024-06-10"
DEFAULT_END_DATE = "2025-10-12"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
INTERVAL_MAP = {
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


def parse_args(argv: List[str]) -> argparse.Namespace:
    """MVP - Parse CLI arguments for Binance OHLCV download.

    Params:
        argv: list of CLI arguments (excluding script name).

    Returns:
        argparse.Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Download OHLCV from Binance")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL_SECONDS,
        help="Candle interval in seconds (e.g., 3600=1h, 86400=1d)",
    )
    parser.add_argument("--start", default=DEFAULT_START_DATE, help="YYYY-MM-DD (UTC)")
    parser.add_argument(
        "--end",
        default=DEFAULT_END_DATE,
        help="YYYY-MM-DD (UTC, exclusive)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def ensure_interval(interval_seconds: int) -> str:
    """MVP - Map seconds to Binance interval code; raise if unsupported.

    Params:
        interval_seconds: requested candle size in seconds.

    Returns:
        Binance interval code string (e.g., '1h').
    """
    if interval_seconds not in INTERVAL_MAP:
        avail = ", ".join(str(k) for k in sorted(INTERVAL_MAP))
        raise ValueError(f"Unsupported interval {interval_seconds}. Choose from: {avail}")
    return INTERVAL_MAP[interval_seconds]


def fetch_klines(symbol: str, interval_code: str, start_ms: int, end_ms: int) -> List[List]:
    """MVP - Fetch up to 1000 klines between start_ms and end_ms with basic retry.

    Params:
        symbol: trading pair symbol.
        interval_code: Binance interval code string.
        start_ms: start timestamp in milliseconds.
        end_ms: end timestamp in milliseconds (exclusive).

    Returns:
        List of kline rows (Binance API schema) or empty if no data.
    """
    params = {
        "symbol": symbol.upper(),
        "interval": interval_code,
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
        return data or []
    raise RuntimeError("Failed to fetch klines from Binance after retries")


def format_filename(symbol: str, interval_seconds: int, start: dt.datetime, end: dt.datetime) -> str:
    """MVP - Make a readable filename with month-year tags.

    Params:
        symbol: trading pair symbol.
        interval_seconds: interval size in seconds.
        start: start datetime.
        end: end datetime.

    Returns:
        Filename string like 'BTCUSDT3600_Jun2024_Oct2025.csv'.
    """
    start_tag = start.strftime("%b%Y")
    end_tag = end.strftime("%b%Y")
    return f"{symbol.upper()}{interval_seconds}_{start_tag}_{end_tag}.csv"


def save_csv(rows: List[List], path: Path) -> None:
    """MVP - Save klines to CSV with columns: date, close, open, high, low (UTC).

    Params:
        rows: list of kline rows returned by Binance.
        path: destination path for CSV.
    """
    header = ["date", "close", "open", "high", "low"]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            open_time = dt.datetime.fromtimestamp(row[0] / 1000, tz=dt.timezone.utc)
            writer.writerow(
                [open_time.strftime(DATE_OUTPUT_FMT), row[4], row[1], row[2], row[3]]
            )


def main(argv: List[str]) -> None:
    """MVP - Entrypoint: download klines from Binance and write CSV.

    Params:
        argv: list of CLI args, typically sys.argv[1:].
    """
    args = parse_args(argv)
    interval_code = ensure_interval(args.interval)
    start_dt = dt.datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    end_dt = dt.datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    if start_dt >= end_dt:
        raise ValueError("Start date must precede end date")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.output_dir / format_filename(args.symbol, args.interval, start_dt, end_dt)
    if out_file.exists():
        print(f"Overwriting {out_file}")

    all_rows: List[List] = []
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    current = start_ms
    while current < end_ms:
        chunk = fetch_klines(args.symbol, interval_code, current, end_ms)
        if not chunk:
            break
        all_rows.extend(chunk)
        current = int(chunk[-1][0]) + args.interval * 1000
        time.sleep(0.25)

    if not all_rows:
        raise RuntimeError("No data returned; check symbol/interval/date range")
    save_csv(all_rows, out_file)
    print(f"Saved {len(all_rows)} candles to {out_file}")


if __name__ == "__main__":
    main(sys.argv[1:])
