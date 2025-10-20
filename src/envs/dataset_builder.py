import numpy as np, pandas as pd
from d3rlpy.dataset import MDPDataset

def build_offline_dataset(prices: pd.DataFrame, feats: pd.DataFrame, cfg: dict) -> MDPDataset:
    """
    Generate offline RL dataset with expert demonstrations for IQL pretraining.
    
    Expert policy: Simple momentum-based strategy (±2% threshold)
    Observation: All 25 technical features × 96h window + previous position = 2496 dims
    Action: Continuous position [-1, +1] scaled by volatility targeting
    Reward: PnL - transaction costs - drawdown penalty - turnover penalty
    
    Note: Momentum heuristics intentionally kept simple. With auxiliary task enabled,
    the agent learns rich representations from all features and improves upon the 
    basic momentum strategy. The expert provides reasonable initialization, not the 
    final policy - that's what IQL + SAC learn!
    """
    assert prices.index.min() >= feats.index.min() and prices.index.max() <= feats.index.max(), \
        "prices must lie within the feature index range"
    W = int(cfg["window_bars"])

    # ═══════════════════════════════════════════════════════════════════════════
    # INITIALIZATION & CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════
    # - Load window size, cost parameters, reward shaping coefficients
    # - Initialize position tracking, equity curve, and peak equity for drawdown
    
    # --- Position sizing: Volatility targeting ---
    # Scale positions inversely with realized volatility to maintain ~2% target vol
    ret1 = np.log(prices["close"]).diff()
    sigma = ret1.ewm(span=48, adjust=False).std().fillna(method="bfill")
    target_vol = 0.02
    base_size = (target_vol / (sigma + 1e-6)).clip(0.0, 1.0)

    # ═══════════════════════════════════════════════════════════════════════════
    # AUXILIARY TASK LABELS (Future Returns)
    # ═══════════════════════════════════════════════════════════════════════════
    # - Look-ahead labels for supervised auxiliary task (not used in main reward)
    # - Infrastructure in place but not yet integrated into IQL training
    
    # --- future return labels for auxiliary task (look-ahead for supervised learning) ---
    future_horizon = int(cfg.get("aux_future_bars", 24))  # Default: 24 hours ahead
    future_returns = np.log(prices["close"].shift(-future_horizon) / prices["close"])
    future_labels = np.sign(future_returns).fillna(0.0).astype(np.float32)  # -1, 0, +1

    # ═══════════════════════════════════════════════════════════════════════════
    # STATE VARIABLES & HYPERPARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    obs, act, rew, done, aux_labels = [], [], [], [], []
    w_prev, eq, peak = 0.0, 1.0, 1.0  # Previous position, equity, peak equity
    bps = (cfg["costs"]["slippage_bps"] + cfg["costs"]["commission_bps"]) / 1e4
    kappa = cfg["reward"]["kappa_cost"]      # Cost penalty multiplier
    lam = cfg["reward"]["lambda_risk"]       # Drawdown penalty coefficient
    kturn = float(cfg.get("reward", {}).get("kappa_turnover", 0.0))  # Turnover penalty

    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN TRAJECTORY GENERATION LOOP
    # ═══════════════════════════════════════════════════════════════════════════
    for t in range(W+1, len(prices)):
        # ─────────────────────────────────────────────────────────────────────
        # OBSERVATION CONSTRUCTION
        # ─────────────────────────────────────────────────────────────────────
        # Stack: [96 bars × 25 features] + [96 bars × previous position]
        # Result: Flattened 2496-dimensional observation vector
        window = np.concatenate(
            [feats.iloc[t-W:t].to_numpy(np.float32),
             np.full((W,1), w_prev, np.float32)],
            axis=1
        )
        if not np.isfinite(window).all():
            continue

        # ─────────────────────────────────────────────────────────────────────
        # EXPERT POLICY: Simple Momentum Strategy
        # ─────────────────────────────────────────────────────────────────────
        # No regime filter (MA96 removed) - agent must learn from raw features
        # Diversified dataset: includes bull, bear, and sideways conditions
        mom24 = (prices["close"].iloc[t] / prices["close"].iloc[t-24] - 1.0) if t >= 24 else 0.0
        sz = float(base_size.iloc[t]) if not np.isnan(base_size.iloc[t]) else 0.0
        
        # Momentum with ±2% threshold - agent must learn regime detection itself
        if mom24 > 0.02:  # Upward momentum → Long bias
            a = +0.5 * sz
        elif mom24 < -0.02:  # Downward momentum → Short bias
            a = -0.5 * sz
        else:  # Weak momentum → Neutral
            a = 0.0
        a = float(np.clip(a, -1.0, 1.0))
        
        # Future label for auxiliary task (not used in reward)
        aux_label = float(future_labels.iloc[t])

        # ─────────────────────────────────────────────────────────────────────
        # ENVIRONMENT DYNAMICS & REWARD COMPUTATION
        # ─────────────────────────────────────────────────────────────────────
        ret = float(np.log(prices["close"].iloc[t] / prices["close"].iloc[t-1]))
        raw = w_prev * ret                          # Raw PnL from previous position
        turn = abs(a - w_prev)                      # Position change (turnover)
        cost = bps * turn                           # Transaction costs (slippage + commission)
        eq = eq * (1 + raw - cost)                  # Update equity curve
        peak = max(peak, eq)                        # Track maximum equity
        ddplus = max(0.0, (peak-eq)/peak)          # Drawdown from peak (0 to 1)
        
        # Reward formula: PnL - cost penalty - drawdown penalty - turnover penalty
        r = raw - kappa*cost - lam*ddplus - kturn*turn
        if not np.isfinite(r):
            continue

        # ─────────────────────────────────────────────────────────────────────
        # STORE TRANSITION
        # ─────────────────────────────────────────────────────────────────────
        obs.append(window.reshape(-1).astype(np.float32))
        act.append([np.float32(a)])
        rew.append(np.float32(r))
        done.append(False)
        aux_labels.append(np.float32(aux_label))
        w_prev = a

    # ═══════════════════════════════════════════════════════════════════════════
    # FINALIZATION & DATASET CONSTRUCTION
    # ═══════════════════════════════════════════════════════════════════════════
    if len(obs) == 0:
        raise RuntimeError("Offline dataset empty after filtering; check features/normalizer.")

    done[-1] = True  # Mark final transition as terminal
    
    # Build d3rlpy MDPDataset for offline RL training
    dataset = MDPDataset(
        observations=np.stack(obs, 0).astype(np.float32),
        actions=np.array(act, dtype=np.float32),
        rewards=np.array(rew, dtype=np.float32),
        terminals=np.array(done, dtype=np.bool_)
    )
    
    # Attach auxiliary labels as metadata (infrastructure for future supervised auxiliary task)
    # Note: Not currently used in IQL training, but available for multi-task learning extensions
    dataset._aux_labels = np.array(aux_labels, dtype=np.float32)
    
    return dataset
