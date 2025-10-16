"""
MVP - Aggregate summaries (lightweight).
"""

from __future__ import annotations

import glob
from pathlib import Path

from .utils.io_utils import load_nested_config


def main(args) -> None:
    """MVP - Print the latest summary for the configured symbol."""
    cfg = load_nested_config(args.config)
    symbol = cfg.get("data", {}).get("symbol", "UNKNOWN")
    summary_path = Path(f"evaluation/reports/summary_{symbol}.json")

    if not summary_path.exists():
        summaries = sorted(glob.glob("evaluation/reports/summary_*.json"))
        if summaries:
            summary_path = Path(summaries[-1])
        else:
            print("No reports found. Run backtest first.")
            return

    print(f"Latest summary: {summary_path}")
    print(summary_path.read_text(encoding="utf-8"))
