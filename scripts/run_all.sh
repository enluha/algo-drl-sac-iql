#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

SYMBOL=${SYMBOL:-BTCUSDT}
START=${START:-2024-06-10}
END=${END:-2025-10-16}
CONFIG=${CONFIG:-config/config.yaml}
BLAS_THREADS=${BLAS_THREADS:-6}
N_WORKERS=${N_WORKERS:-1}

export OMP_NUM_THREADS=${BLAS_THREADS}
export MKL_NUM_THREADS=${BLAS_THREADS}
export NUMEXPR_NUM_THREADS=${BLAS_THREADS}
export BLAS_THREADS
export N_WORKERS

python - <<'PY'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    idx = torch.cuda.current_device()
    print(f"Using CUDA device {idx}: {torch.cuda.get_device_name(idx)}")
PY

CSV_PATH=$(python - <<PY
import datetime as dt
symbol = "${SYMBOL}"
start = dt.datetime.fromisoformat("${START}")
end = dt.datetime.fromisoformat("${END}")
print(f"data/{symbol.upper()}3600_{start.strftime('%b%Y')}_{end.strftime('%b%Y')}.csv")
PY
)

python scripts/download_ohlcv_binance.py \
  --symbol "$SYMBOL" \
  --interval 3600 \
  --start "$START" \
  --end "$END" \
  --output-dir data

python - <<PY
from pathlib import Path
import yaml
config_path = Path("config/data.yaml")
with config_path.open() as fh:
    cfg = yaml.safe_load(fh)
cfg["csv_path"] = "${CSV_PATH}"
with config_path.open("w") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False)
print(f"Updated config/data.yaml: csv_path -> {cfg['csv_path']}")
PY

python -m src.run_offline_pretrain --config "$CONFIG"
python -m src.run_sac_finetune --config "$CONFIG"
python -m src.run_walkforward --config "$CONFIG"

REPORT_PATH="evaluation/reports/summary_report_${SYMBOL}.txt"
if [[ -s "$REPORT_PATH" ]]; then
  echo "Summary report generated at $REPORT_PATH"
else
  echo "Warning: summary report missing or empty at $REPORT_PATH" >&2
fi
