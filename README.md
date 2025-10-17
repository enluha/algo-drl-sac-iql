# algo-drl-sac-iql — Offline IQL → Online SAC crypto trader (BTCUSDT-first)

![Python](https://img.shields.io/badge/python-3.10+-blue.svg) ![PyTorch](https://img.shields.io/badge/pytorch-2.2+-ee4c2c.svg) ![d3rlpy](https://img.shields.io/badge/d3rlpy-2.6+-orange.svg)

Offline pretraining plus online fine-tuning for a BTCUSDT-focused crypto trading agent. We generate a data-only workflow that learns implicit Q-learning (IQL) behaviour from heuristics and then fine-tunes the policy online with soft actor-critic (SAC) inside a latency-aware simulator. Reporting mirrors the prior project: HTML candlesticks, vectorbt equity curves, and a text summary report.

## Why DRL here?

* Offline RL leverages cheap synthetic behaviour data to warm-start the policy.
* Online SAC fine-tuning adapts to the latest regime without catastrophic forgetting.
* Data-only design keeps everything reproducible and lightweight (no live exchange keys).

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # optional overrides
bash scripts/run_all.sh
```

`run_all.sh` downloads OHLCV, builds the offline dataset, pre-trains IQL, fine-tunes SAC, and runs walk-forward evaluation. Outputs land under `evaluation/`.

## Configuration

Key YAML files under `config/`:

* `data.yaml` – symbol, period, data source (CSV by default).
* `env.yaml` – environment knobs (window, leverage, deadband, reward costs/risk).
* `algo_iql.yaml` – offline IQL hyperparameters and encoder spec.
* `algo_sac.yaml` – SAC fine-tuning hyperparameters and vectorized env count.
* `walkforward.yaml` – rolling window sizes for train/valid/test.
* `runtime.yaml` – global seed, thread count, worker processes, log level.
* `costs.yaml` – reporting costs for continuity with past reports.

Override any value via environment variables consumed by `run_all.sh`, or pass `--config` to CLI entrypoints.

## Outputs

After `scripts/run_all.sh`, expect:

* `evaluation/reports/summary_report_BTCUSDT.txt`
* `evaluation/reports/{trades,equity_curve,signals,predictions}_BTCUSDT.csv`
* `evaluation/charts/{candlestick_BTCUSDT_*.html,equity_BTCUSDT.html}`
* `evaluation/artifacts/{offline_dataset.h5,iql_policy.d3,sac_policy.d3,normalizer.pkl}`

These replicate the original UX while replacing the classical ML pipeline with DRL.

## GPU note

Training prefers GPU for neural networks. A mid-range card (e.g., RTX 2060 SUPER) suffices. The CLI auto-detects CUDA; set `--device cpu` for CPU-only runs.

## Roadmap

* Multi-asset portfolio support.
* Live exchange adapters and real-time execution.
* Richer microstructure features (order book, funding, perp basis).
