# EQUITY CURVE ANALYSIS - Unrealistic Profits Investigation
## Date: October 18, 2025

## EXECUTIVE SUMMARY
The trading algorithm shows **unrealistic profits** with ~387% return over 150 days (21 million % annualized), Sharpe ratio of 19.4, and other impossible metrics. Multiple critical issues have been identified.

---

## FINDINGS

### 1. **CRITICAL: EQUITY INITIALIZATION BUG**
**Status: CONFIRMED BUG**

**Problem:**
- The environment's equity column shows `8.571713` as the first value instead of `1.0`
- Equity from env: `8.571713` → `155.898615`
- Equity from net returns: `1.000876` → `4.874780`
- The equity is starting at 8.57x instead of 1.0

**Evidence:**
```
First trade equity: 8.571713 (should be ~1.0)
Manual reconstruction: 1.000876 (correct)
```

**Root Cause:**
Likely the `MarketEnv._equity` is not being reset properly between walk-forward folds, or there's initialization code that sets it to a non-1.0 value. The environment tracks equity internally and it appears to be carrying over between folds or getting multiplied incorrectly.

**Impact:** 
- Inflates all returns by ~8.5x from the start
- Makes the entire backtest meaningless
- Results in the reported 387% return being artificially inflated

---

### 2. **CRITICAL: COST PENALTY TOO SMALL**
**Status: CONFIRMED ISSUE**

**Problem:**
Transaction costs are being heavily discounted via `kappa_cost`:

```yaml
# From config/env.yaml:
kappa_cost: 0.00015  # This multiplies the actual costs

# Actual cost calculation:
cost = bps * turnover = 0.0015 * turnover
applied_cost = cost * kappa_cost = cost * 0.00015
# This makes costs 0.015% of what they should be!
```

**Evidence:**
- Config specifies 5 bps slippage + 10 bps commission = 15 bps (0.0015) total
- Total turnover: 1046.01 (meaning ~10.5 complete position flips)
- Expected cumulative costs: 1046.01 * 0.0015 = ~1.569
- Actual costs applied: ~1.569 * 0.00015 = ~0.000235
- **Costs are 99.985% smaller than they should be!**

**From trades data:**
- Cumulative costs recorded: 1.569017
- But these are reduced by `kappa_cost` in the equity calculation
- Result: Trading appears nearly cost-free

**Impact:**
- Allows extremely high-frequency trading without penalty
- Average turnover per trade: 0.29 (29% position change)
- With proper costs, this would be devastating to returns

---

### 3. **DESIGN ISSUE: COSTS IN REWARD VS EQUITY**
**Status: DESIGN INCONSISTENCY**

**Problem:**
Looking at `MarketEnv.step()`:

```python
raw = self._w_prev * ret_t
cost = self.bps * turn
eq = float(self._equity * (1.0 + raw - cost * self.kappa_cost))
reward = raw - cost * self.kappa_cost - self.lambda_risk * ddplus
```

**Issues:**
1. The cost is multiplied by `kappa_cost` (0.00015) in **both** equity and reward
2. This makes costs almost non-existent
3. The agent learned to trade frequently because costs don't matter
4. `kappa_cost` seems intended as a reward shaping parameter but affects actual equity

**Expected behavior:**
- Full costs should apply to equity: `eq = equity * (1.0 + raw - cost)`
- Reward can use scaled costs for training: `reward = raw - cost * kappa_cost - ...`
- These should be separate!

---

### 4. **UNREALISTIC PERFORMANCE METRICS**
**Status: RED FLAGS**

From the analysis:
- **Sharpe Ratio: 19.38** (anything > 3 is exceptional, > 10 is nearly impossible)
- **Sortino Ratio: 28.88** (even more unrealistic)
- **Calmar Ratio: 699.32** (absurdly high)
- **Max Drawdown: -2.77%** (impossibly low for crypto)
- **Total Return: 387%** in 150 days
- **Annualized Return: 21,306,563%** (not a typo)

**Reality Check:**
- Best hedge funds: Sharpe 1.5-2.5, annual returns 15-30%
- Renaissance Medallion (best quant fund ever): Sharpe ~3, ~40% annually
- This algo claims to be 10x better than the best fund in history

---

### 5. **POTENTIAL DATA LEAKAGE**
**Status: NEEDS INVESTIGATION**

With such extreme returns, consider:

1. **Look-ahead bias**: Are future prices leaking into features?
   - Check `latency_bars: 1` - is this applied correctly?
   - Check feature engineering for any t+1 data

2. **Target encoding issues**: 
   - Are targets calculated with information not available at decision time?

3. **Normalization leakage**:
   - `RollingZScore` - does it use future data?
   - Fitted on train, applied on test - this looks OK

4. **Walk-forward implementation**:
   - Model sees train+valid+test prices in `window_prices`
   - But `norm.fit()` only uses train data - this appears correct
   - Need to verify environment doesn't leak test data

---

## PROPOSED AMENDMENTS

### **CRITICAL FIX 1: Reset Equity Properly**
**Priority: P0**

**Issue:** Environment equity starts at 8.57 instead of 1.0

**Investigation needed:**
```python
# In src/envs/market_env.py, check:
def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    self.t = self.W + self.latency
    self._w_prev = 0.0
    self._equity, self._peak = 1.0, 1.0  # ← Is this being executed?
    self._ret_last = self._cost_last = self._net_last = 0.0
    return self._obs(), {}
```

**Hypothesis:**
- In `run_walkforward.py`, the environment is reused across folds
- `env.reset()` may not be called between folds
- Or there's state leakage in the environment

**Fix:**
Ensure `env.reset()` is called at the start of each walk-forward fold before any trading.

---

### **CRITICAL FIX 2: Remove kappa_cost from Equity Calculation**
**Priority: P0**

**Current (WRONG):**
```python
# In src/envs/market_env.py line ~63:
eq = float(self._equity * (1.0 + raw - cost * self.kappa_cost))
```

**Should be:**
```python
# Apply FULL costs to equity:
eq = float(self._equity * (1.0 + raw - cost))

# Keep kappa_cost only in reward for training:
reward = raw - cost * self.kappa_cost - self.lambda_risk * ddplus
```

**Rationale:**
- Real money equity must account for full transaction costs
- `kappa_cost` should only affect agent's learning signal
- Mixing reward shaping with actual P&L is fundamentally wrong

---

### **CRITICAL FIX 3: Increase kappa_cost or Remove It**
**Priority: P0**

**Current:** `kappa_cost: 0.00015` (makes costs 0.015% of actual)

**Option A - Use full costs:**
```yaml
# config/env.yaml:
kappa_cost: 1.0  # Apply full costs to reward
```

**Option B - More realistic discounting:**
```yaml
kappa_cost: 0.5  # Apply 50% of costs (still generous)
```

**Rationale:**
- Current value of 0.00015 is 6,666x smaller than 1.0
- Agent learned to overtrade because costs don't matter
- Real trading requires respecting transaction costs

---

### **HIGH PRIORITY FIX 4: Verify Environment Reset in Walk-Forward**
**Priority: P1**

**In `src/run_walkforward.py`**, add explicit reset:

```python
for idx, fold in enumerate(folds):
    # ... setup code ...
    
    env = MarketEnv(window_prices[["open","high","low","close"]], feats_norm, env_cfg)
    obs, _ = env.reset()  # ← Verify this happens
    
    # Add assertion to catch bugs:
    assert env.equity == 1.0, f"Equity should be 1.0, got {env.equity}"
    
    prev_weight = 0.0
    # ... trading loop ...
```

---

### **HIGH PRIORITY FIX 5: Add Reality Checks**
**Priority: P1**

Add validation after walk-forward:

```python
# After generating results:
if summary["sharpe_365d"] > 10.0:
    logger.warning("Sharpe > 10 is unrealistic - check for data leakage or bugs")

if summary["max_dd"] > -0.05:  # Less than 5% drawdown
    logger.warning("Max DD too small for crypto - suspicious")

ann_return = (equity.iloc[-1] / equity.iloc[0]) ** (365 / (len(equity) / 24)) - 1
if ann_return > 2.0:  # 200% annualized
    logger.error(f"Annualized return {ann_return*100:.1f}% is unrealistic")
```

---

### **MEDIUM PRIORITY: Check for Data Leakage**
**Priority: P2**

**Verify:**
1. Features don't use t+1 data
2. `latency_bars: 1` is correctly applied
3. Labels are properly aligned
4. Normalization doesn't leak

**Run existing test:**
```bash
python tests/test_label_exit_alignment.py
```

---

### **LOW PRIORITY: Increase Deadband**
**Priority: P3**

```yaml
# config/env.yaml:
deadband: 0.05  # Current
deadband: 0.10  # More realistic - reduces microtrading
```

**Rationale:**
- Prevents tiny position changes
- Reduces turnover
- Current 0.05 is reasonable but could be more conservative

---

## TESTING PROTOCOL

After fixes, expected reasonable metrics:
- **Sharpe ratio: 0.5 - 2.5** (1.0+ is good for crypto)
- **Max drawdown: -15% to -50%** (crypto is volatile)
- **Annual return: -20% to +100%** (break-even to exceptional)
- **Calmar ratio: 0.5 - 3.0** (return/drawdown)
- **Turnover: Should decrease** (with proper costs)

---

## ROOT CAUSE SUMMARY

1. ✅ **Confirmed:** Equity initialization bug (starts at 8.57 not 1.0)
2. ✅ **Confirmed:** Cost penalty far too small (kappa_cost = 0.00015)
3. ✅ **Confirmed:** Costs incorrectly applied to equity calculation
4. ⚠️ **Possible:** Environment state leakage between folds
5. ⚠️ **Possible:** Data leakage in features or labels

---

## RECOMMENDED ACTION PLAN

1. **Immediate:** Fix equity initialization - verify env.reset() works
2. **Immediate:** Remove kappa_cost from equity calculation
3. **Immediate:** Set kappa_cost to 1.0 or remove entirely
4. **Next:** Add assertion checks for equity == 1.0 at fold start
5. **Next:** Add reality check warnings for impossible metrics
6. **Later:** Investigate potential data leakage in features
7. **Later:** Retrain models with proper cost accounting

---

## DISCLAIMER

**DO NOT TRADE REAL MONEY** with this algorithm until:
- All critical fixes are implemented
- Results show realistic performance metrics
- Walk-forward validation shows consistent performance
- Independent code review confirms no data leakage

Current results are **100% unreliable** due to the bugs identified above.
