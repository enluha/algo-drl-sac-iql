#!/usr/bin/env bash
set -euo pipefail

SYMBOL=${SYMBOL:-BTCUSDT}
START=${START:-2024-06-10}
END=${END:-2025-10-16}
BLAS_THREADS=${BLAS_THREADS:-6}
N_WORKERS=${N_WORKERS:-1}
CONFIG=${CONFIG:-config/config.yaml}

export OMP_NUM_THREADS=${BLAS_THREADS}
export MKL_NUM_THREADS=${BLAS_THREADS}
export NUMEXPR_NUM_THREADS=${BLAS_THREADS}

python - <<'PY'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
PY

python scripts/download_ohlcv_binance.py \
  --symbol "${SYMBOL}" \
  --interval 3600 \
  --start "${START}" \
  --end "${END}" \
  --output-dir data

CSV_PATH=$(python - <<PY
from datetime import datetime, timezone
start = datetime.strptime("${START}", "%Y-%m-%d").replace(tzinfo=timezone.utc)
end = datetime.strptime("${END}", "%Y-%m-%d").replace(tzinfo=timezone.utc)
start_tag = start.strftime("%b%Y")
end_tag = end.strftime("%b%Y")
print(f"data/${SYMBOL.upper()}3600_{start_tag}_{end_tag}.csv")
PY
)

python - <<PY
from pathlib import Path
import yaml
cfg_path = Path("${CONFIG}").parent / "data.yaml"
with cfg_path.open() as f:
    data_cfg = yaml.safe_load(f)
data_cfg["symbol"] = "${SYMBOL}"
data_cfg["start"] = "${START}"
data_cfg["end"] = "${END}"
data_cfg["csv_path"] = "${CSV_PATH}"
with cfg_path.open("w") as f:
    yaml.safe_dump(data_cfg, f)
print(f"Updated {cfg_path} -> csv_path={data_cfg['csv_path']}")
PY

python -m src.run_offline_pretrain --config "${CONFIG}" --n-workers "${N_WORKERS}"
python -m src.run_sac_finetune --config "${CONFIG}" --n-workers "${N_WORKERS}"
python -m src.run_walkforward --config "${CONFIG}" --n-workers "${N_WORKERS}"

echo "Summary report: evaluation/reports/summary_report_${SYMBOL}.txt"
