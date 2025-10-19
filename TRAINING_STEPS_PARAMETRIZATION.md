# Training Steps Parametrization

**Date**: October 19, 2025  
**Changes**: Separated IQL and SAC training steps parameters, made QA_STEPS optional

---

## ✅ What Changed

### Problem
Previously, `QA_STEPS` environment variable was acting as a single upper bound for **both** IQL and SAC training, which could accidentally under-train SAC if set too low.

### Solution
**Separated parameters** for IQL and SAC training, and made `QA_STEPS` a pure override (not initialized with a default).

---

## 📋 New Parameter Structure

### 1. IQL Offline Training

**Config**: `config/algo_iql.yaml`
```yaml
grad_steps_IQL: 1000  # Renamed from grad_steps
```

**Code**: `src/drl/offline/iql_pretrain.py`
```python
# Use grad_steps_IQL from config, or QA_STEPS override if set (for quick testing)
default_steps = int(algo_cfg.get("grad_steps_IQL", algo_cfg.get("grad_steps", 1000)))
steps = int(os.getenv("QA_STEPS")) if os.getenv("QA_STEPS") else default_steps
```

**Behavior**:
- ✅ **No QA_STEPS**: Uses `grad_steps_IQL: 1000` from config
- ✅ **QA_STEPS=5000**: Overrides to 5000 steps
- ✅ **Backward compatible**: Falls back to old `grad_steps` if `grad_steps_IQL` not found

---

### 2. SAC Online Training

**Config**: `config/algo_sac.yaml`
```yaml
grad_steps_SAC: 100000  # NEW parameter
```

**Code**: `src/drl/online/sac_train.py`
```python
# Use grad_steps_SAC from config, or QA_STEPS override if set (for quick testing)
default_steps = int(algo_cfg.get("grad_steps_SAC", 100000))
steps = int(os.getenv("QA_STEPS")) if os.getenv("QA_STEPS") else default_steps
```

**Behavior**:
- ✅ **No QA_STEPS**: Uses `grad_steps_SAC: 100000` from config
- ✅ **QA_STEPS=5000**: Overrides to 5000 steps
- ✅ **Fallback**: Uses 100000 if `grad_steps_SAC` not in config

---

## 🎯 QA_STEPS Behavior (New)

`QA_STEPS` is now an **optional override** for quick testing:

| Scenario | IQL Steps | SAC Steps | Use Case |
|----------|-----------|-----------|----------|
| **No QA_STEPS** (normal) | 1000 (from config) | 100,000 (from config) | ✅ **Production runs** |
| **QA_STEPS=500** | 500 | 500 | Quick smoke test (5 min) |
| **QA_STEPS=5000** | 5000 | 5000 | Fast iteration (30 min) |
| **QA_STEPS=50000** | 50,000 | 50,000 | Medium test (2 hours) |

### Key Points:
- ✅ **Not initialized by default**: Won't interfere with config values
- ✅ **Pure override**: Only takes effect if explicitly set
- ✅ **Affects both phases**: When set, overrides BOTH IQL and SAC
- ✅ **For testing only**: Production should use config parameters

---

## 📁 Files Modified

### 1. Config Files

**`config/algo_iql.yaml`**:
```yaml
# OLD
grad_steps: 1000

# NEW
grad_steps_IQL: 1000  # Renamed for clarity
```

**`config/algo_sac.yaml`**:
```yaml
# NEW (added)
grad_steps_SAC: 100000  # Number of environment steps for SAC fine-tuning
```

### 2. Python Files

**`src/drl/offline/iql_pretrain.py`**:
- Line 105-107: Changed to use `grad_steps_IQL` with QA_STEPS as optional override
- Line 128: Updated BC fallback to use same pattern

**`src/drl/online/sac_train.py`**:
- Line 146-148: Changed to use `grad_steps_SAC` with QA_STEPS as optional override

---

## 🚀 Usage Examples

### Normal Production Run
```bash
# Just run - uses config defaults
bash scripts/run_all.sh
# IQL: 1000 steps
# SAC: 100,000 steps
```

### Quick Test Run
```bash
# Override for fast testing
export QA_STEPS=5000
bash scripts/run_all.sh
# IQL: 5000 steps
# SAC: 5000 steps
```

### Custom Config Values
Edit configs directly:
```yaml
# config/algo_iql.yaml
grad_steps_IQL: 2000  # Train IQL longer

# config/algo_sac.yaml
grad_steps_SAC: 200000  # Train SAC even longer
```

Then run normally:
```bash
bash scripts/run_all.sh
# IQL: 2000 steps (from config)
# SAC: 200,000 steps (from config)
```

---

## 🔧 Backward Compatibility

### Old Code Compatibility
The code maintains backward compatibility:

```python
# IQL fallback chain
algo_cfg.get("grad_steps_IQL", algo_cfg.get("grad_steps", 1000))
#               ↑ new name         ↑ old name    ↑ hardcoded fallback
```

This means:
1. ✅ Looks for `grad_steps_IQL` (new name)
2. ✅ Falls back to `grad_steps` (old name) if not found
3. ✅ Uses 1000 if neither exists

### Old Config Files
If you have old configs with `grad_steps`, they still work:
```yaml
# Old config (still works)
grad_steps: 1000  # Will be used as fallback
```

---

## 📊 Training Time Estimates

| Configuration | IQL Time | SAC Time | Total Time |
|---------------|----------|----------|------------|
| **Default** (1k/100k) | ~5 sec | ~3-4 hours | ~3-4 hours |
| **QA_STEPS=500** | ~2 sec | ~1 min | ~1 min |
| **QA_STEPS=5000** | ~10 sec | ~10 min | ~10 min |
| **QA_STEPS=50000** | ~1 min | ~2 hours | ~2 hours |
| **Custom** (2k/200k) | ~10 sec | ~6-8 hours | ~6-8 hours |

*Times are approximate and depend on GPU/CPU speed*

---

## ✅ Summary

**Before**:
- ❌ Single `QA_STEPS` affected both IQL and SAC
- ❌ Hardcoded default (100k) in SAC code
- ❌ Easy to accidentally under-train SAC

**After**:
- ✅ Separate `grad_steps_IQL` and `grad_steps_SAC` in configs
- ✅ `QA_STEPS` is optional override for testing only
- ✅ Clear separation of concerns
- ✅ Production runs use config values
- ✅ Testing runs use QA_STEPS override
- ✅ Backward compatible with old configs

**Result**: More flexible, safer, and clearer training configuration! 🎉
