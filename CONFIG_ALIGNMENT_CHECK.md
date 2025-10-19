# Config Files Alignment Check

## ✅ All Config Files Are Aligned

### Summary
All YAML config files are properly structured for the anti-overtrading setup. There is a minor duplication (costs defined in both `env.yaml` and `costs.yaml`), but this doesn't cause conflicts because the code uses `env_cfg` which includes the costs section from `env.yaml`.

---

## 📁 Config File Status

### 1. ✅ `config/config.yaml` (Hub File)
```yaml
include:
  - data.yaml
  - env.yaml
  - algo_iql.yaml
  - algo_sac.yaml
  - walkforward.yaml
  - runtime.yaml
  - costs.yaml
```
**Status**: ✅ Correct - includes all necessary configs

---

### 2. ✅ `config/env.yaml` (MODIFIED - Anti-Overtrading)
```yaml
latency_bars: 1
leverage_max: 1.0
window_bars: 96
deadband: 0.05
min_step: 0.01
reward:
  kappa_cost: 10.0  # ✅ UPDATED from 1.0
  kappa_turnover: 0.01  # ✅ NEW parameter
  lambda_risk: 0.001
  risk_metric: "dd_velocity"
costs:
  slippage_bps: 5
  commission_bps: 10
```

**Changes Applied**:
- ✅ `kappa_cost: 10.0` (was 1.0) - 10x transaction cost penalty
- ✅ `kappa_turnover: 0.01` (new) - Direct turnover penalty
- ✅ All parameters properly formatted

**Status**: ✅ **Aligned with anti-overtrading strategy**

---

### 3. ✅ `config/costs.yaml` (Standalone - Not Used by MarketEnv)
```yaml
slippage_bps: 5
commission_bps: 10
```

**Note**: This file exists but the code uses costs from `env.yaml` instead. The duplication is harmless because `MarketEnv` receives `env_cfg` which includes its own `costs` section.

**Status**: ✅ Consistent with `env.yaml` costs (no conflict)

---

### 4. ✅ `config/algo_iql.yaml`
```yaml
expectile_beta: 0.8
temperature: 0.3
discount: 0.99
lr: 0.0003
batch_size: 256
grad_steps: 1000  # ✅ 1000 steps for quick pretrain
compile_graph: true
actor_encoder_factory:
  type: vector
  params: { hidden_units: [256, 256] }
# ... (optimizers with clip_grad_norm: 1.0)
device: "cuda"
```

**Status**: ✅ Correct - IQL pretrain config unchanged (no modifications needed)

---

### 5. ✅ `config/algo_sac.yaml`
```yaml
gamma: 0.99
tau: 0.005
lr_actor: 0.0003
lr_critic: 0.0003
alpha_learning_rate: 0.0003
initial_temperature: 0.1
target_entropy: -1.0
batch_size: 256
buffer_size: 10000
updates_per_step: 4
# ... (optimizers with clip_grad_norm: 1.0)
encoder:
  type: vector
  hidden_units: [256, 256]
device: "cuda"
n_envs: 1
```

**Note**: Training steps are controlled by `src/drl/online/sac_train.py` code (default 100k), not this config.

**Status**: ✅ Correct - SAC config parameters aligned

---

### 6. ✅ `config/data.yaml`
```yaml
symbol: BTCUSDT
bar: "1h"
start: "2024-06-10"
end: "2025-10-16"
source: "csv"
csv_path: "data/BTCUSDT3600_Jun2024_Oct2025.csv"
na_policy: "ffill"
cache: "evaluation/artifacts/cache"
```

**Status**: ✅ Correct - data config unchanged

---

### 7. ✅ `config/walkforward.yaml`
```yaml
splits:
  pretrain:
    start: "2024-06-10"
    end:   "2025-01-31"
  finetune:
    start: "2025-02-01"
    end:   "2025-04-30"
  test:
    start: "2025-05-01"
    end:   "2025-10-03"
warmup_bars: 96
```

**Status**: ✅ Correct - walk-forward splits aligned

---

### 8. ✅ `config/runtime.yaml`
```yaml
seed: 42
blas_threads: 6
n_workers: 1
log_level: INFO
normalizer: global
rolling_window: 500
normalizer_clip: 5.0
```

**Status**: ✅ Correct - runtime config unchanged

---

## 🔧 Code-Config Alignment

### How `env.yaml` is Used:
```python
# In sac_train.py, iql_pretrain.py, run_walkforward.py
cfg = load_yaml("config/config.yaml")
env_cfg = cfg["env"]  # Gets entire env.yaml
env = MarketEnv(prices, features, env_cfg)
```

### How MarketEnv Reads Parameters:
```python
# In src/envs/market_env.py
def __init__(self, prices, features, cfg):
    # Basic params
    self.W = int(cfg["window_bars"])                    # ✅ 96
    self.deadband = float(cfg["deadband"])              # ✅ 0.05
    self.min_step = float(cfg["min_step"])              # ✅ 0.01
    self.latency = int(cfg.get("latency_bars", 1))      # ✅ 1
    self.leverage = float(cfg.get("leverage_max", 1.0)) # ✅ 1.0
    
    # Costs (from cfg["costs"])
    costs = cfg.get("costs", {})
    slippage_bps = float(costs.get("slippage_bps", 0))  # ✅ 5
    commission_bps = float(costs.get("commission_bps", 0)) # ✅ 10
    self.bps = (slippage_bps + commission_bps) / 1e4     # ✅ 0.0015 (0.15%)
    
    # Reward parameters (from cfg["reward"])
    rw = cfg.get("reward", {})
    self.kappa_cost = float(rw.get("kappa_cost", 0.0))     # ✅ 10.0
    self.lambda_risk = float(rw.get("lambda_risk", 0.0))   # ✅ 0.001
    self.kappa_turnover = float(rw.get("kappa_turnover", 0.0)) # ✅ 0.01 (NEW)
    self.risk_metric = str(rw.get("risk_metric", "drawdown")).lower() # ✅ "dd_velocity"
```

**Status**: ✅ **All parameters correctly read from config**

---

## 🎯 Reward Function Implementation

```python
# In market_env.py step()
reward = raw - self.kappa_cost * cost - self.lambda_risk * risk_pen - self.kappa_turnover * turn
```

Where:
- `raw` = position weight × log return
- `cost` = transaction cost (0.15% × turnover)
- `risk_pen` = drawdown velocity penalty
- `turn` = |new_weight - old_weight|

**With Current Settings**:
- `raw`: unchanged (P&L from position)
- `- 10.0 * cost`: **10x penalty on transaction costs** 🔥
- `- 0.001 * risk_pen`: small drawdown velocity penalty
- `- 0.01 * turn`: **direct penalty on position changes** 🔥

**Example Calculation**:
- If agent changes position from 0.0 to 0.5 (turnover=0.5):
  - Transaction cost penalty: `10.0 × 0.0015 × 0.5 = 0.0075`
  - Direct turnover penalty: `0.01 × 0.5 = 0.005`
  - **Total penalty: 0.0125** (1.25% of equity!)
- Agent must expect >1.25% return to justify this trade

---

## ⚠️ Minor Issues (Non-Breaking)

### 1. Duplicate Costs Definition
- **Issue**: `costs` defined in both `env.yaml` and `costs.yaml`
- **Impact**: None - code uses `env.yaml` version
- **Recommendation**: Remove `costs.yaml` or remove `costs:` section from `env.yaml`
- **Priority**: Low (cosmetic only)

### 2. SAC Steps Not in Config
- **Issue**: Training steps hardcoded in `sac_train.py` (100k), not in `algo_sac.yaml`
- **Impact**: None - works correctly via environment variable `QA_STEPS`
- **Current**: `steps = int(os.getenv("QA_STEPS", 100000))`
- **Recommendation**: Consider adding `n_steps: 100000` to `algo_sac.yaml`
- **Priority**: Low (works as-is)

---

## ✅ Final Verdict

### All Critical Parameters Are Aligned:
1. ✅ **kappa_cost: 10.0** - 10x transaction cost penalty
2. ✅ **kappa_turnover: 0.01** - Direct turnover penalty  
3. ✅ **Training steps: 100,000** - Better convergence
4. ✅ **Feature smoothing: span=3** - Noise reduction
5. ✅ **Transaction costs: 15bps** - Realistic (5 slip + 10 commission)
6. ✅ **Window: 96 bars** - 4 days of hourly data
7. ✅ **Action deadband: 0.05** - Prevents micro-trades
8. ✅ **Min step: 0.01** - Position change threshold

### Ready to Train:
All config files are properly aligned with the anti-overtrading strategy. The pipeline is ready to run with:
- 10x higher transaction cost sensitivity
- Direct turnover penalties
- 3x more training steps
- Smoothed features

**No config changes needed** - proceed with training! 🚀
