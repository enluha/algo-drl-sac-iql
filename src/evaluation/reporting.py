"""
MVP - Write evaluation artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils.io_utils import save_csv, save_json


def write_reports(metrics_dict: dict, ledger: Any, equity: Any, outdir: str) -> None:
    """MVP - Save summary metrics and core outputs to evaluation directory.

    Params:
        metrics_dict: dictionary of scalar metrics.
        ledger: DataFrame or similar structure (optional).
        equity: Series or similar structure (optional).
        outdir: output directory path.
    """
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)

    save_json(metrics_dict, output / "summary.json")

    if ledger is not None:
        save_csv(ledger, output / "trades.csv")

    if equity is not None:
        save_csv(equity.to_frame("equity"), output / "equity_curve.csv")
