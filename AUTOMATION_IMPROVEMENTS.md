# Configuration Cleanup & Automation Improvements

**Date**: October 19, 2025  
**Changes**: Eliminated config duplication, automated HTML generation

---

## ✅ Change 1: Eliminated Costs Duplication

### Problem
Transaction costs were defined in **two places**:
1. `config/costs.yaml` - Standalone file
2. `config/env.yaml` - Nested under `costs:` section

This was confusing and could lead to inconsistencies.

### Solution
**Deleted `config/costs.yaml`** and kept costs **only in `config/env.yaml`**.

### Why This Is Better
- ✅ **Single source of truth**: Costs are only in one place
- ✅ **Clearer structure**: Environment config includes everything needed for trading
- ✅ **Easier to maintain**: No risk of costs getting out of sync
- ✅ **More intuitive**: Costs are part of the environment configuration

### Final Config Structure

**`config/env.yaml`** (costs section):
```yaml
latency_bars: 1
leverage_max: 1.0
window_bars: 96
deadband: 0.05
min_step: 0.01
reward:
  kappa_cost: 10.0
  kappa_turnover: 0.01
  lambda_risk: 0.001
  risk_metric: "dd_velocity"
costs:
  slippage_bps: 5      # 5 basis points = 0.05%
  commission_bps: 10   # 10 basis points = 0.10%
  # Total: 15bps = 0.15% per trade (one-way)
```

**`config/config.yaml`** (updated includes):
```yaml
include:
  - data.yaml
  - env.yaml
  - algo_iql.yaml
  - algo_sac.yaml
  - walkforward.yaml
  - runtime.yaml
  # costs.yaml removed ✅
```

### Transaction Cost Details
- **Slippage**: 5 bps (0.05%)
- **Commission**: 10 bps (0.10%)
- **Total per trade**: 15 bps (0.15%) one-way
- **Round-trip cost**: 30 bps (0.30%)

These costs are **realistic for crypto exchanges**:
- Binance spot: ~10 bps commission (with BNB discount)
- Slippage: 3-8 bps typical for BTC on liquid markets
- Total: 13-18 bps realistic range

---

## ✅ Change 2: Automated feature_overview.html Generation

### Problem
After downloading/updating data, you had to manually run:
```bash
python scripts/plot_feature_overview.py
```

This was easy to forget and meant you couldn't visualize features until after the fact.

### Solution
**Modified `scripts/run_all.sh`** to automatically generate `feature_overview.html` right after data download.

### New Pipeline in run_all.sh

```bash
# 1) Download OHLCV data
python scripts/download_ohlcv_binance.py ...

# 2) Update config/data.yaml with csv_path
sed -i "s#^csv_path:.*#csv_path: \"$CSV_PATH\"#g" config/data.yaml

# 2.5) Generate feature_overview.html ✅ NEW
python scripts/plot_feature_overview.py
echo "Feature overview saved to data/feature_overview.html"

# 3) Offline pretrain (IQL)
python -m src.run_offline_pretrain ...
```

### Benefits
- ✅ **Always up-to-date**: Feature visualization generated with every data refresh
- ✅ **No manual steps**: Fully automated in the pipeline
- ✅ **Early validation**: See features before waiting 4+ hours for training
- ✅ **Consistent workflow**: One command (`bash scripts/run_all.sh`) does everything

### What Gets Visualized

`data/feature_overview.html` shows:
- **Panel 1**: Price & Moving Averages
- **Panel 2**: RSI indicators (6, 14, 21 periods)
- **Panel 3**: Oscillators (MACD, Donchian position)
- **Panel 4**: Volume metrics (z-score, change, taker pressure)
- **Panel 5**: Feature correlation heatmap

All features are now **smoothed (EMA span=3)** as per anti-overtrading fixes.

---

## ✅ Change 3: Evaluation Documents Already Automated

### Verified: No Changes Needed

The evaluation step (`src.run_walkforward`) **already automatically generates all documents**:

### CSVs Generated:
1. ✅ `evaluation/reports/trades_{SYMBOL}.csv` - Trade-by-trade log
2. ✅ `evaluation/reports/equity_curve_{SYMBOL}.csv` - Equity over time
3. ✅ `evaluation/reports/signals_{SYMBOL}.csv` - Position weights over time
4. ✅ `evaluation/reports/predictions_{SYMBOL}.csv` - Raw model predictions

### HTMLs Generated:
1. ✅ `evaluation/charts/equity_{SYMBOL}.html` - Interactive equity chart
2. ✅ `evaluation/charts/candlestick_{SYMBOL}_SepOct2025.html` - Price action with trades
3. ✅ `evaluation/charts/8.html` - Rebased equity comparison

### TXTs Generated:
1. ✅ `evaluation/reports/summary_report_{SYMBOL}.txt` - Performance metrics
2. ✅ `evaluation/reports/summary_report.txt` - Copy of summary

### JSONs Generated:
1. ✅ `evaluation/reports/walkforward_meta_{SYMBOL}.json` - Metadata

### Code Implementation (src/run_walkforward.py, lines 195-217):
```python
# Save all evaluation artifacts
save_csv(ledger_df, rep / f"trades_{symbol}.csv")
save_csv(equity_df, rep / f"equity_curve_{symbol}.csv")
save_csv(signals_df.to_frame(), rep / f"signals_{symbol}.csv")
save_csv(predictions_df, rep / f"predictions_{symbol}.csv")

candlestick_html(..., ch / f"candlestick_{symbol}_SepOct2025.html")
equity_html(..., ch / f"equity_{symbol}.html")
equity_html(..., ch / "8.html")

build_text_report(summary, rep / f"summary_report_{symbol}.txt", context=context)
build_text_report(summary, rep / "summary_report.txt", context=context)
save_json({...}, rep / f"walkforward_meta_{symbol}.json")
```

**Conclusion**: Evaluation automation was already implemented correctly. No changes needed!

---

## 📝 Complete Automated Workflow

Running `bash scripts/run_all.sh` now does **everything automatically**:

```
1. Download OHLCV data from Binance
   ↓
2. Update config/data.yaml with CSV path
   ↓
3. Generate feature_overview.html ✅ NEW
   ↓
4. IQL offline pretrain (1000 steps, ~5 seconds)
   ↓
5. SAC online finetune (100k steps, ~3-4 hours)
   ↓
6. Walk-forward evaluation
   ├── Generate all CSVs ✅
   ├── Generate all HTMLs ✅
   ├── Generate all TXTs ✅
   └── Generate all JSONs ✅
   ↓
7. Print summary report location
```

### What You Get Automatically

After running `bash scripts/run_all.sh`, these files will be ready:

**Visualization**:
- `data/feature_overview.html` - Feature inspection (NEW!)
- `evaluation/charts/equity_BTCUSDT.html` - Equity curve
- `evaluation/charts/candlestick_BTCUSDT_SepOct2025.html` - Trade visualization

**Analysis**:
- `evaluation/reports/summary_report_BTCUSDT.txt` - Performance metrics
- `evaluation/reports/equity_curve_BTCUSDT.csv` - Time series data
- `evaluation/reports/trades_BTCUSDT.csv` - Trade-by-trade analysis

**No manual steps required!** 🎉

---

## 🎯 Summary of Changes

| Change | Before | After | Benefit |
|--------|--------|-------|---------|
| **Costs config** | Duplicated in 2 files | Single location in `env.yaml` | No confusion, single source of truth |
| **Feature viz** | Manual: `python scripts/plot_feature_overview.py` | Auto-generated in `run_all.sh` | Always up-to-date, no extra steps |
| **Evaluation docs** | Already automated ✅ | Already automated ✅ | No changes needed |

---

## 🔧 Files Modified

1. ✅ **Deleted**: `config/costs.yaml`
2. ✅ **Modified**: `config/config.yaml` (removed costs.yaml from includes)
3. ✅ **Modified**: `scripts/run_all.sh` (added feature_overview.html generation)
4. ✅ **Verified**: `src/run_walkforward.py` (evaluation automation already works)

---

## ✨ Result

**One command does everything**:
```bash
bash scripts/run_all.sh
```

**Output**: Complete training pipeline with all visualizations and reports automatically generated!

No more manual steps. No more forgotten visualizations. Just run and analyze! 🚀
