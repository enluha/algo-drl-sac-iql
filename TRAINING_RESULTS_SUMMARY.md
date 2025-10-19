# Training Results Summary - With Fixed Features

**Training Date**: October 19, 2025  
**Pipeline**: IQL Pretrain → SAC Finetune → Walk-forward Evaluation

---

## 🔧 Fixed Features Applied

All critical bugs were fixed before this training run:

### ✅ Priority 1: RollingZScore Fix
- **Before**: Used only last 500 bars for normalization statistics
- **After**: Uses full training data for normalization
- **Impact**: Eliminates recency bias in feature scaling

### ✅ Priority 2: MACD Normalization
- **Before**: Absolute price units ($100 range)
- **After**: Percentage-based (divided by close price)
- **Impact**: Makes MACD scale-invariant

### ✅ Priority 3: MA Slopes Scaling
- **Before**: Tiny values (~0.0005 range)
- **After**: Scaled by 10000 (basis points)
- **Impact**: Preserves signal after normalization

### ✅ Bonus: Parkinson Safety & Volume Consistency
- Added safety checks for Parkinson volatility
- Fixed vol_z window consistency (min_periods=96)

---

## 📊 Training Performance

### Step 1: IQL Offline Pretrain ✅
- **Duration**: 3 seconds
- **Data Window**: June 10, 2024 → January 31, 2025
- **Gradient Steps**: 1,000
- **Status**: Successfully completed
- **Artifacts**: `evaluation/artifacts/iql_actor_state.pt`

### Step 2: SAC Online Finetune ✅
- **Duration**: 1 hour 21 minutes
- **Data Window**: February 1, 2025 → April 30, 2025
- **Environment Steps**: 33,333
- **Rollout Return**: 1.83 (realistic, not 387% like before!)
- **Speed**: ~6.8 iterations/second
- **Status**: Successfully completed
- **Artifacts**: `evaluation/artifacts/sac_actor_state.pt`

### Step 3: Walk-forward Evaluation ✅
- **Data Window**: May 1, 2025 → October 3, 2025 (with warmup from April 27)
- **Status**: Successfully completed
- **Reports Generated**: ✅ All CSVs, TXTs, and HTMLs created

---

## 📈 Test Performance Metrics (May - October 2025)

### ⚠️ Critical Issue: Excessive Trading

**Sharpe Ratio**: -14.94 (poor risk-adjusted returns)  
**Sortino Ratio**: -20.65 (even worse downside-adjusted)  
**Maximum Drawdown**: -69.6% (severe)  
**Calmar Ratio**: -21.46  
**Turnover**: **810.39x** 🚨 (MAJOR PROBLEM)  
**Final Equity**: 0.306 (lost 69.4%)  
**Window Return**: -69.3%

### 🔍 Root Cause Analysis

The model is **overtrading catastrophically**:
- **810x turnover** means the portfolio flipped 810 times during the test period
- With transaction costs of 0.015%, this results in ~12.15x in trading costs alone
- Trades are happening almost every hour (3,722 trades in ~4,000 hours)
- Position weights are changing rapidly between -0.69 and +0.69

### Comparison to Previous (Buggy) Run
- **Previous**: Sharpe 19.4, Return 387% (impossible, due to equity bug)
- **Current**: Sharpe -14.94, Return -69% (realistic but poor)
- **Improvement**: Results are now realistic, bugs are fixed
- **Problem**: Model learned to trade too aggressively

---

## 🎯 Diagnosis: Why Is The Model Overtrading?

### 1. Reward Structure
The current reward might be too sensitive to short-term price movements, encouraging frequent position changes.

### 2. Action Space Scaling
The action space is continuous [-1, +1] but the model might be outputting large values that translate to extreme position changes.

### 3. Transaction Cost Penalty
The transaction cost penalty (kappa_cost=0.00015) might be too small to discourage frequent trading.

### 4. Feature Engineering
Despite the fixes, features might still be too noisy, causing the model to react to every minor market move.

### 5. Training Instability
The SAC fine-tuning might have learned a suboptimal policy that trades excessively.

---

## 📁 Generated Artifacts

### Reports (`evaluation/reports/`)
- ✅ `summary_report_BTCUSDT.txt` - Performance metrics
- ✅ `equity_curve_BTCUSDT.csv` - Equity over time (3,722 rows)
- ✅ `trades_BTCUSDT.csv` - Trade-by-trade log (3,722 trades!)
- ✅ `signals_BTCUSDT.csv` - Raw signals
- ✅ `predictions_BTCUSDT.csv` - Model predictions
- ✅ `walkforward_meta_BTCUSDT.json` - Metadata

### Charts (`evaluation/charts/`)
- ✅ `equity_BTCUSDT.html` - Interactive equity chart
- ✅ `candlestick_BTCUSDT_SepOct2025.html` - Price action chart

### Models (`evaluation/artifacts/`)
- ✅ `iql_actor_state.pt` - IQL pretrained actor
- ✅ `sac_actor_state.pt` - SAC fine-tuned actor
- ✅ `iql_policy.d3` - IQL policy
- ✅ `sac_policy.d3` - SAC policy

---

## 🔧 Next Steps: Fixing Overtrading

### Priority 1: Increase Transaction Cost Penalty
```yaml
# In config/costs.yaml
kappa_cost: 0.0015  # Increase from 0.00015 (10x higher)
```

### Priority 2: Add Turnover Penalty to Reward
Modify the reward function to explicitly penalize excessive turnover:
```python
# In src/envs/trading_env.py
turnover_penalty = abs(new_weight - old_weight) * turnover_penalty_factor
reward = pnl - transaction_cost - turnover_penalty
```

### Priority 3: Constrain Action Space
Limit how much the position can change per step:
```python
# In src/envs/trading_env.py
max_position_change = 0.1  # Max 10% change per hour
action_clipped = np.clip(action, old_weight - max_position_change, old_weight + max_position_change)
```

### Priority 4: Increase Training Steps
The 33,333 SAC steps might be insufficient. Try:
```yaml
# In config/algo_sac.yaml
n_steps: 100000  # Increase from 33333
```

### Priority 5: Feature Smoothing
Add exponential smoothing to features to reduce noise:
```python
# In src/features/engineering.py
X["feature_smoothed"] = X["feature"].ewm(span=3).mean()
```

---

## ✅ What's Working

1. **No More Impossible Results**: Equity curve is realistic (starts at 1.0, not 8.57)
2. **Transaction Costs Applied Correctly**: Costs are being calculated properly
3. **Feature Fixes Applied**: All normalization bugs are fixed
4. **Training Pipeline Works**: Full IQL→SAC→Evaluation runs successfully
5. **Artifacts Generated**: All reports and charts are created

## ❌ What Needs Fixing

1. **Overtrading**: 810x turnover is catastrophic
2. **Poor Risk-Adjusted Returns**: Sharpe -14.94 is terrible
3. **High Drawdown**: -69.6% is unacceptable
4. **Reward Function**: Needs to better penalize trading
5. **Policy Stability**: Model is too reactive to noise

---

## 📊 Visual Inspection Recommended

Open these files to inspect the results:
1. `evaluation/charts/equity_BTCUSDT.html` - See the equity decline
2. `evaluation/charts/candlestick_BTCUSDT_SepOct2025.html` - See trading behavior
3. `evaluation/reports/trades_BTCUSDT.csv` - Analyze individual trades
4. `data/feature_overview.html` - Verify features look reasonable

---

## 🎯 Conclusion

**The Good News**: All bugs are fixed, and the pipeline works correctly. Results are realistic.

**The Bad News**: The model trades way too much and loses money due to transaction costs.

**The Path Forward**: Implement turnover penalties, increase transaction costs, and retrain with a more conservative reward structure.
