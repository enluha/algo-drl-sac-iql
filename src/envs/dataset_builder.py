import numpy as np, pandas as pd
from d3rlpy.dataset import MDPDataset

def build_offline_dataset(prices: pd.DataFrame, feats: pd.DataFrame, cfg: dict) -> MDPDataset:
    assert prices.index.min() >= feats.index.min() and prices.index.max() <= feats.index.max(), \
        "prices must lie within the feature index range"
    W = int(cfg["window_bars"])

    # --- sizing helpers (EWMA vol on 1-bar log returns) ---
    ret1 = np.log(prices["close"]).diff()
    sigma = ret1.ewm(span=48, adjust=False).std().fillna(method="bfill")
    target_vol = 0.02
    base_size = (target_vol / (sigma + 1e-6)).clip(0.0, 1.0)

    # --- future return labels for auxiliary task (look-ahead for supervised learning) ---
    future_horizon = int(cfg.get("aux_future_bars", 24))  # Default: 24 hours ahead
    future_returns = np.log(prices["close"].shift(-future_horizon) / prices["close"])
    future_labels = np.sign(future_returns).fillna(0.0).astype(np.float32)  # -1, 0, +1

    obs, act, rew, done, aux_labels = [], [], [], [], []
    w_prev, eq, peak = 0.0, 1.0, 1.0
    bps = (cfg["costs"]["slippage_bps"] + cfg["costs"]["commission_bps"]) / 1e4
    kappa = cfg["reward"]["kappa_cost"]; lam = cfg["reward"]["lambda_risk"]
    kturn = float(cfg.get("reward", {}).get("kappa_turnover", 0.0))

    for t in range(W+1, len(prices)):
        window = np.concatenate(
            [feats.iloc[t-W:t].to_numpy(np.float32),
             np.full((W,1), w_prev, np.float32)],
            axis=1
        )
        if not np.isfinite(window).all():
            continue

        # Expert policy: Simple momentum (diversified dataset, no regime filter)
        mom24 = (prices["close"].iloc[t] / prices["close"].iloc[t-24] - 1.0) if t >= 24 else 0.0
        sz = float(base_size.iloc[t]) if not np.isnan(base_size.iloc[t]) else 0.0
        
        # Momentum with threshold - agent must learn regime detection itself
        if mom24 > 0.02:  # Upward momentum
            a = +0.5 * sz
        elif mom24 < -0.02:  # Downward momentum
            a = -0.5 * sz
        else:
            a = 0.0  # Neutral
        a = float(np.clip(a, -1.0, 1.0))
        
        # Future label for auxiliary task
        aux_label = float(future_labels.iloc[t])

        ret = float(np.log(prices["close"].iloc[t] / prices["close"].iloc[t-1]))
        raw = w_prev * ret
        turn = abs(a - w_prev)
        cost = bps * turn
        eq = eq * (1 + raw - cost)
        peak = max(peak, eq); ddplus = max(0.0, (peak-eq)/peak)
        r = raw - kappa*cost - lam*ddplus - kturn*turn
        if not np.isfinite(r):
            continue

        obs.append(window.reshape(-1).astype(np.float32))
        act.append([np.float32(a)])
        rew.append(np.float32(r))
        done.append(False)
        aux_labels.append(np.float32(aux_label))
        w_prev = a

    if len(obs) == 0:
        raise RuntimeError("Offline dataset empty after filtering; check features/normalizer.")

    done[-1] = True
    dataset = MDPDataset(
        observations=np.stack(obs, 0).astype(np.float32),
        actions=np.array(act, dtype=np.float32),
        rewards=np.array(rew, dtype=np.float32),
        terminals=np.array(done, dtype=np.bool_)
    )
    
    # Attach auxiliary labels as metadata (d3rlpy will ignore, but we can use in custom training)
    dataset._aux_labels = np.array(aux_labels, dtype=np.float32)
    
    return dataset
