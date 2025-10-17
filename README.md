# algo-drl-sac-iql — Offline IQL → Online SAC crypto trader (BTCUSDT-first)

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![d3rlpy](https://img.shields.io/badge/d3rlpy-2.6+-orange.svg)](https://github.com/takuseno/d3rlpy)

`algo-drl-sac-iql` is a data-only deep reinforcement learning (DRL) trading project for
crypto markets. The pipeline bootstraps an Offline RL agent with Implicit Q-Learning
(IQL) on historical heuristics, then fine-tunes the policy online via Soft Actor-Critic
(SAC) inside a realistic simulator with latency, slippage, and position costs.

## Why DRL for trading?

* **History-aware decisions** – the policy consumes a 96-hour window of engineered
  candle features with normalization and latency handling.
* **Cost-sensitive training** – rewards include trading costs, drawdown penalties, and
  deadbands so the policy learns to trade less and only when conviction is high.
* **Seamless offline→online bridge** – leverage `d3rlpy` to pretrain with IQL and warm
  start SAC on GPU, then evaluate via walk-forward analysis.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # optional, environment overrides
bash scripts/run_all.sh
```

The run script downloads BTCUSDT OHLCV data, builds an offline dataset, pretrains an
IQL policy, fine-tunes SAC online, and runs a walk-forward evaluation. Outputs land in
`evaluation/` and include candlestick HTML, vectorbt equity charts, CSV ledgers, and a
text summary report.

## Configuration overview

| File | Purpose |
| ---- | ------- |
| `config/data.yaml` | Symbol, data source and cache paths |
| `config/env.yaml` | Trading environment knobs (latency, leverage, reward) |
| `config/algo_iql.yaml` | Offline IQL hyper-parameters |
| `config/algo_sac.yaml` | SAC fine-tuning hyper-parameters |
| `config/walkforward.yaml` | Rolling window sizes for walk-forward analysis |
| `config/runtime.yaml` | Seeds, worker/thread counts, logging level |
| `config/costs.yaml` | Trading cost assumptions for metrics/reporting |

Override any setting via `CONFIG=config/custom.yaml` or environment variables consumed by
`scripts/run_all.sh`.

## Outputs

After `scripts/run_all.sh`, expect the following artifacts:

* `evaluation/reports/summary_report_BTCUSDT.txt` – human-readable summary
* `evaluation/charts/candlestick_BTCUSDT_*.html` – Plotly candlestick with signals
* `evaluation/charts/equity_BTCUSDT.html` – vectorbt equity curve vs. buy & hold
* `evaluation/reports/*.csv` – trades, equity curve, signals, predictions
* `evaluation/artifacts/` – offline dataset `.h5`, `iql_policy.d3`, `sac_policy.d3`,
  and the fitted normalizer

## GPU guidance

Training prefers CUDA. Set `device: "cuda"` in configs or export `DEVICE=cuda`. A mid-tier
GPU (e.g., RTX 2060 SUPER) comfortably handles the default 96×F observation tensor and
IQL/SAC workloads. CPU handles feature generation, dataset construction, and simulation.

## Roadmap

1. **Multi-asset expansion** – extend the environment to a universe of spot pairs.
2. **Live adapters** – route the SAC actor into paper/live trading infrastructure.
3. **Richer microstructure** – integrate order book features and execution simulators.

Contributions are welcome via pull requests.
