#!/usr/bin/env bash
set -euo pipefail

# --- knobs / overrides ---
SYMBOL="${SYMBOL:-BTCUSDT}"
INTERVAL_SEC="${INTERVAL_SEC:-3600}"
START="${START:-2024-06-10}"
END="${END:-2025-10-16}"
CONFIG="${CONFIG:-config/config.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [ -x ".venv/Scripts/python.exe" ]; then
  PYTHON_BIN="$(pwd)/.venv/Scripts/python.exe"
elif [ -x ".venv/bin/python" ]; then
  PYTHON_BIN="$(pwd)/.venv/bin/python"
fi

# threads & device
export OMP_NUM_THREADS="${BLAS_THREADS:-6}"
export MKL_NUM_THREADS="${BLAS_THREADS:-6}"
export NUMEXPR_NUM_THREADS="${BLAS_THREADS:-6}"

echo "Using Python interpreter: ${PYTHON_BIN}"
echo -n "CUDA available? "
CUDA_AVAILABLE=$("$PYTHON_BIN" - <<'PY'
import torch
print("True" if torch.cuda.is_available() else "False")
PY
)
CUDA_AVAILABLE=${CUDA_AVAILABLE//$'\r'/}
echo "${CUDA_AVAILABLE}"
echo "CUDA visible?: $("$PYTHON_BIN" - <<'PY'
import torch
print(torch.cuda.is_available(), 'torch_cuda', torch.version.cuda)
PY
)"

if [[ "${DEVICE:-}" != "cpu" && "${QA_DEVICE:-}" != "cpu" && "${CUDA_AVAILABLE}" != "True" ]]; then
  echo "CUDA not available; aborting run. Set DEVICE=cpu to override for CPU execution."
  exit 1
elif [[ "${DEVICE:-}" == "cpu" || "${QA_DEVICE:-}" == "cpu" ]]; then
  echo "CPU override detected; proceeding without CUDA acceleration."
fi

# --- 1) download OHLCV to ./data ---
echo "=== Downloading OHLCV: $SYMBOL ${INTERVAL_SEC}s $START -> $END ==="
"$PYTHON_BIN" scripts/download_ohlcv_binance.py \
  --symbol "$SYMBOL" --interval "$INTERVAL_SEC" \
  --start "$START" --end "$END" --output-dir data

CSV_PATH="$(ls -1 data/${SYMBOL}${INTERVAL_SEC}_*.csv | tail -n1)"
echo "Saved candles to $CSV_PATH"
echo "=== Active splits (from config/walkforward.yaml) ==="
"$PYTHON_BIN" - <<'PY'
import yaml
with open("config/walkforward.yaml","r", encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)
splits = cfg.get("splits")
if splits is None:
    print("No splits found in config/walkforward.yaml")
else:
    for name, span in splits.items():
        print(f"{name}: {span}")
print("warmup_bars:", cfg.get("warmup_bars"))
PY

# --- 2) patch config/data.yaml: csv_path ---
echo "=== Updating config/data.yaml with csv_path ==="
# GNU sed (Linux) / BSD sed (macOS) handling
if sed --version >/dev/null 2>&1; then
  sed -i "s#^csv_path:.*#csv_path: \"$CSV_PATH\"#g" config/data.yaml
else
  sed -i '' "s#^csv_path:.*#csv_path: \"$CSV_PATH\"#g" config/data.yaml
fi

# --- 3) offline pretrain (IQL) ---
echo "=== Offline pretrain (IQL) ==="
"$PYTHON_BIN" -m src.run_offline_pretrain --config "$CONFIG" ${DEVICE:+--device "$DEVICE"}

# --- 4) online fine-tune (SAC) ---
echo "=== Online fine-tune (SAC) ==="
"$PYTHON_BIN" -m src.run_sac_finetune --config "$CONFIG" ${DEVICE:+--device "$DEVICE"}

# --- 5) walk-forward evaluation ---
echo "=== Walk-forward evaluation (reports & charts) ==="
"$PYTHON_BIN" -m src.run_walkforward --config "$CONFIG" ${DEVICE:+--device "$DEVICE"}

echo "=== Latest summary ==="
ls -1 evaluation/reports/summary_report_*.txt || true
echo "=== Done ==="
