# Auxiliary Task Activation & IQL-Only Evaluation

## Changes Implemented

### 1. ✅ Auxiliary Task Activated

**File**: `src/drl/offline/iql_pretrain.py`

The auxiliary task (price direction prediction) is now **FULLY INTEGRATED**:

#### What it does:
- **Multi-task learning**: Agent learns TWO objectives simultaneously:
  1. **Main task**: Maximize trading returns (IQL loss)
  2. **Auxiliary task**: Predict future price direction 24h ahead (classification loss)

- **Formula**: `total_loss = IQL_loss + 0.1 × auxiliary_loss`

- **Benefits**:
  - Forces encoder to learn predictive features about future prices
  - Improves feature representations for the policy
  - Better generalization through multi-task learning
  - Prevents overfitting to short-term patterns

#### How to enable:
Set in `config/algo_iql.yaml`:
```yaml
use_auxiliary_task: true  # Enable multi-task learning
aux_loss_weight: 0.1      # Weight for auxiliary loss (10% of total)
aux_future_bars: 24       # Predict 24 hours ahead
```

#### Architecture:
```
Observation (2496 dims)
    ↓
[Shared Trunk: 256→256]
    ↓
    ├─→ [Policy Head] → Actions (for trading)
    └─→ [Auxiliary Head: 128→3] → Price prediction (UP/NEUTRAL/DOWN)
```

---

### 2. ✅ IQL-Only Evaluation

**New Function**: `_evaluate_iql_only()` in `iql_pretrain.py`

Now generates **THREE** evaluation files after IQL training completes (BEFORE SAC):

1. **`evaluation/charts/equity_BTCUSDT_IQLonly.html`**
   - Equity curve for IQL policy in isolation
   - Compare this to final SAC equity to see fine-tuning improvement

2. **`evaluation/charts/candlestick_BTCUSDT_SepOct2025_IQLonly.html`**
   - Trading signals overlaid on price chart
   - Shows which trades IQL makes before online learning

3. **`evaluation/reports/summary_report_BTCUSDT_IQLonly.txt`**
   - Full performance metrics (Sharpe, drawdown, win rate, etc.)
   - Baseline to compare against SAC fine-tuned version

#### When it runs:
- Automatically after `agent.fit()` completes
- Uses test period (Sep-Oct 2025) with proper warmup
- Pure IQL policy evaluation (no exploration noise)

#### Purpose:
- **Isolate IQL effectiveness**: See how well offline pretraining works alone
- **Measure SAC contribution**: Compare IQL-only vs final SAC metrics
- **Diagnose issues**: If IQL-only is bad, problem is in expert policy or features
- **Celebrate wins**: If IQL-only is already good, SAC should make it even better!

---

### 3. ✅ Momentum Heuristics: KEPT (Not Removed)

**File**: `src/envs/dataset_builder.py`

**Decision**: Keep the simple momentum expert policy as-is.

#### Why keep it:
1. **Auxiliary task provides sophistication**: 
   - Agent learns from ALL 25 features through price prediction head
   - Momentum is just the LABEL generator, not the feature set
   
2. **Expert provides initialization**:
   - Simple momentum = reasonable starting point
   - IQL + SAC learn to IMPROVE upon it, not copy it
   
3. **Diversified dataset**:
   - No MA96 regime filter = includes bull, bear, and sideways conditions
   - Agent must learn when to trust momentum vs when to ignore it

#### Updated docstring:
```python
"""
Note: Momentum heuristics intentionally kept simple. With auxiliary task enabled,
the agent learns rich representations from all features and improves upon the 
basic momentum strategy. The expert provides reasonable initialization, not the 
final policy - that's what IQL + SAC learn!
"""
```

---

## Testing

### Quick Test (5k steps):
```bash
$env:QA_STEPS="5000"; python src/drl/offline/iql_pretrain.py
```

Expected output:
```
Auxiliary task ENABLED - using multi-task encoder with price direction prediction
Creating IQLWithAuxiliary (aux_loss_weight=0.1)
Auxiliary labels loaded: 10823 samples
  UP: 3621, NEUTRAL: 3601, DOWN: 3601
...
IQL-ONLY EVALUATION COMPLETE
  Sharpe: X.XX
  Total Return: X.XX%
  Max Drawdown: X.XX%
```

### Cloud Production Run (200k steps):
```bash
# In config/runtime.yaml
grad_steps_IQL: 200000

# Run
python src/drl/offline/iql_pretrain.py
```

---

## File Changes Summary

### Modified Files:
1. ✅ `src/drl/offline/iql_pretrain.py`
   - Imported `IQLWithAuxiliary` and `AuxiliaryEncoderFactory`
   - Added auxiliary task detection from config
   - Conditionally uses auxiliary encoder when `use_auxiliary_task: true`
   - Added `_evaluate_iql_only()` function
   - Calls evaluation after training completes

2. ✅ `src/envs/dataset_builder.py`
   - Added note about keeping momentum heuristics
   - Clarified auxiliary task benefits in docstring

### Existing Infrastructure (No Changes Needed):
- ✅ `src/drl/offline/iql_auxiliary.py` - Already complete
- ✅ `src/models/auxiliary_encoder.py` - Already complete
- ✅ `config/algo_iql.yaml` - Already has auxiliary flags

---

## What to Expect

### With Auxiliary Task Enabled:

**During Training:**
- You'll see auxiliary loss and accuracy metrics in logs
- Auxiliary accuracy ~35-40% is normal (random = 33% for 3-class problem)
- Don't expect high auxiliary accuracy - it's teaching feature representations, not making predictions

**IQL-Only Results:**
- Should beat random baseline (Sharpe > 0.5)
- Likely worse than final SAC (SAC adds online exploration)
- If IQL-only is already good (Sharpe > 1.5), excellent sign!

**After SAC Fine-tuning:**
- Compare `summary_report_BTCUSDT_IQLonly.txt` vs `summary_report_BTCUSDT.txt`
- SAC should improve Sharpe by 0.2-0.5+ points
- If not, may need to adjust SAC hyperparameters

---

## Network Size

**Current**: `[256, 256]` - KEPT as requested

**Future considerations** (if results plateau):
- Try `[512, 512]` for more capacity with 2496-dim input
- Auxiliary task helps, but larger networks may still help eventually

---

## Next Steps

1. Run quick test (5k steps) to verify auxiliary task works
2. Check logs for auxiliary metrics
3. Inspect `*_IQLonly.html` files to see baseline performance
4. Run full 200k step cloud training
5. Compare IQL-only vs final SAC results to quantify improvement

Happy training! 🚀
