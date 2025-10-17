from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from src.utils.io_utils import atomic_write_text


def build_summary(
    config: Dict,
    metrics: Dict[str, float],
    equity: pd.Series,
    ledger: pd.DataFrame,
    path: Path | str,
) -> None:
    lines: list[str] = []
    lines.append("=== Configuration ===")
    lines.append(f"Symbol: {config['data']['symbol']}")
    lines.append(f"Date range: {config['data']['start']} -> {config['data']['end']}")
    lines.append("")
    lines.append("=== Performance ===")
    for key, value in metrics.items():
        lines.append(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    lines.append("")
    lines.append("=== Trade Stats ===")
    lines.append(f"Observations: {len(ledger)}")
    if not ledger.empty:
        win_rate = (ledger['net_ret'] > 0).mean() * 100
        profit_factor = ledger.loc[ledger['net_ret'] > 0, 'net_ret'].sum() / abs(
            ledger.loc[ledger['net_ret'] < 0, 'net_ret'].sum()
        ) if (ledger['net_ret'] < 0).any() else float('inf')
        lines.append(f"Win rate: {win_rate:.2f}%")
        lines.append(f"Profit factor: {profit_factor:.2f}")
    lines.append("")
    lines.append("=== Notes ===")
    lines.append("Offline IQL pretrain, SAC fine-tune, data-only run.")
    atomic_write_text(Path(path), "\n".join(lines))
