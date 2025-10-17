from __future__ import annotations

from pathlib import Path
from typing import Dict

from src.utils.io_utils import atomic_write_text


def build_report(symbol: str, config: Dict, metrics: Dict, trade_stats: Dict, path: Path) -> None:
    env_cfg = config.get("env", {})
    algo_iql = config.get("algo_iql", {})
    algo_sac = config.get("algo_sac", {})
    runtime = config.get("runtime", {})

    lines = [
        f"Summary Report — {symbol}",
        "=" * 72,
        "",
        "Runtime",
        f"  Seed: {runtime.get('seed', 'N/A')}",
        f"  Threads: {runtime.get('blas_threads', 'N/A')}",
        f"  Log level: {runtime.get('log_level', 'INFO')}",
        "",
        "Environment",
        f"  Window bars: {env_cfg.get('window_bars')}",
        f"  Latency bars: {env_cfg.get('latency_bars')}",
        f"  Deadband: {env_cfg.get('deadband')}",
        f"  Min step: {env_cfg.get('min_step')}",
        "",
        "Offline IQL",
        f"  Expectile beta: {algo_iql.get('expectile_beta')}",
        f"  Temperature: {algo_iql.get('temperature')}",
        f"  Grad steps: {algo_iql.get('grad_steps')}",
        "",
        "Online SAC",
        f"  Gamma: {algo_sac.get('gamma')}",
        f"  Tau: {algo_sac.get('tau')}",
        f"  Updates per step: {algo_sac.get('updates_per_step')}",
        "",
        "Performance",
        f"  Sharpe (hourly): {metrics.get('sharpe_hourly', 0):.3f}",
        f"  Sortino (hourly): {metrics.get('sortino_hourly', 0):.3f}",
        f"  CAGR: {metrics.get('cagr', 0):.2%}",
        f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}",
        f"  Calmar: {metrics.get('calmar', 0):.3f}",
        f"  Exposure: {metrics.get('exposure', 0):.2%}",
        f"  Trade count: {trade_stats.get('trades', 0)}",
        f"  Win rate: {trade_stats.get('win_rate', 0):.2%}",
        f"  Profit factor: {trade_stats.get('profit_factor', 0):.3f}",
        f"  SQN: {trade_stats.get('sqn', 0):.3f}",
        f"  Turnover: {metrics.get('turnover', 0):.3f}",
        f"  Cost share: {metrics.get('cost_share', 0):.2%}",
        "",
        "Notes",
        "  • Rewards penalize trading costs and drawdown.",
        "  • Latency=1 bar; actions reference desired next-position weights.",
    ]

    atomic_write_text(path, "\n".join(lines))
