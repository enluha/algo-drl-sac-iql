import numpy as np, pandas as pd
from d3rlpy.dataset import MDPDataset

def build_offline_dataset(prices: pd.DataFrame, feats: pd.DataFrame, cfg: dict) -> MDPDataset:
    assert prices.index.min() >= feats.index.min() and prices.index.max() <= feats.index.max(), \
        "prices must lie within the feature index range"
    W = int(cfg["window_bars"]); ma = prices["close"].rolling(96, min_periods=96).mean()
    allow_long, allow_short = (prices["close"] > ma).fillna(False), (prices["close"] < ma).fillna(False)

    obs, act, rew, done = [], [], [], []
    w_prev, eq, peak = 0.0, 1.0, 1.0
    bps = (cfg["costs"]["slippage_bps"] + cfg["costs"]["commission_bps"]) / 1e4
    kappa = cfg["reward"]["kappa_cost"]; lam = cfg["reward"]["lambda_risk"]

    for t in range(W+1, len(prices)):
        window = np.concatenate(
            [feats.iloc[t-W:t].to_numpy(np.float32),
             np.full((W,1), w_prev, np.float32)],
            axis=1
        )
        if not np.isfinite(window).all():
            continue

        mom = (prices["close"].iloc[t] / prices["close"].iloc[t-24] - 1.0) if t>=24 else 0.0
        a = 0.0
        if mom>0 and allow_long.iloc[t]: a = 0.5
        if mom<0 and allow_short.iloc[t]: a = -0.5
        a = float(np.clip(a, -1.0, 1.0))

        ret = float(np.log(prices["close"].iloc[t] / prices["close"].iloc[t-1]))
        raw = w_prev * ret
        cost = bps * abs(a - w_prev)
        eq = eq * (1 + raw - kappa*cost)
        peak = max(peak, eq); ddplus = max(0.0, (peak-eq)/peak)
        r = raw - kappa*cost - lam*ddplus
        if not np.isfinite(r):
            continue

        obs.append(window.reshape(-1).astype(np.float32))
        act.append([np.float32(a)])
        rew.append(np.float32(r))
        done.append(False)
        w_prev = a

    if len(obs) == 0:
        raise RuntimeError("Offline dataset empty after filtering; check features/normalizer.")

    done[-1] = True
    return MDPDataset(
        observations=np.stack(obs, 0).astype(np.float32),
        actions=np.array(act, dtype=np.float32),
        rewards=np.array(rew, dtype=np.float32),
        terminals=np.array(done, dtype=np.bool_)
    )
