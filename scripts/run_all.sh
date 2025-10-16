#!/usr/bin/env bash
# Glue script: download OHLCV -> update config -> train -> backtest -> report
set -euo pipefail

# -------- Defaults (override by exporting env vars) --------
SYMBOL="${SYMBOL:-BTCUSDT}"
INTERVAL="${INTERVAL:-3600}"          # seconds (3600 = 1h)
START="${START:-2024-06-10}"
END="${END:-2025-10-16}"
BLAS_THREADS="${BLAS_THREADS:-6}"
N_WORKERS="${N_WORKERS:-1}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
CONFIG="${CONFIG:-config/config.yaml}"
DATA_DIR="${DATA_DIR:-data}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# -------- Resolve Python interpreter --------
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if [[ -x ".venv/Scripts/python.exe" ]]; then
    PYTHON_BIN=".venv/Scripts/python.exe"
  elif [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "ERROR: Unable to locate a Python interpreter. Set PYTHON_BIN." >&2
    exit 1
  fi
fi

echo "Using Python interpreter: $PYTHON_BIN"

export OMP_NUM_THREADS="$BLAS_THREADS"
export MKL_NUM_THREADS="$BLAS_THREADS"
export OPENBLAS_NUM_THREADS="$BLAS_THREADS"
export NUMEXPR_NUM_THREADS="$BLAS_THREADS"

echo "=== Downloading OHLCV: $SYMBOL ${INTERVAL}s $START -> $END ==="
mkdir -p "$DATA_DIR"
"$PYTHON_BIN" -m src.data.fetchers \
  --symbol "$SYMBOL" \
  --interval "$INTERVAL" \
  --start "$START" \
  --end "$END" \
  --output-dir "$DATA_DIR"

echo "=== Locating latest CSV in $DATA_DIR ==="
CSV_PATH="$(ls -t "$DATA_DIR"/"${SYMBOL}${INTERVAL}"_*.csv | head -n 1)"
if [[ -z "${CSV_PATH:-}" ]]; then
  echo "ERROR: Could not find downloaded CSV for ${SYMBOL}${INTERVAL}_*.csv"
  exit 1
fi
echo "Using CSV_PATH=$CSV_PATH"

echo "=== Updating config/data.yaml with csv_path ==="
if command -v sed >/dev/null 2>&1; then
  sed -i.bak -E "s|^csv_path:.*$|csv_path: \"${CSV_PATH//\//\\/}\"|" config/data.yaml
else
  echo "WARN: sed not available; please set csv_path manually in config/data.yaml"
fi

echo "=== Training (walk-forward artifacts) ==="
"$PYTHON_BIN" -m src.cli train \
  --config "$CONFIG" \
  --blas-threads "$BLAS_THREADS" \
  --n-workers "$N_WORKERS" \
  --log-level "$LOG_LEVEL"

echo "=== Backtesting (validation thresholds, simulation, plots) ==="
"$PYTHON_BIN" -m src.cli backtest \
  --config "$CONFIG" \
  --blas-threads "$BLAS_THREADS" \
  --n-workers "$N_WORKERS" \
  --log-level "$LOG_LEVEL"

echo "=== Latest summary ==="
"$PYTHON_BIN" -m src.cli report --config "$CONFIG" --log-level "$LOG_LEVEL" || true

echo "=== Done ==="
