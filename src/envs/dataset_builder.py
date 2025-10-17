import numpy as np, pandas as pd
from d3rlpy.dataset import MDPDataset

def build_offline_dataset(prices: pd.DataFrame, feats: pd.DataFrame, cfg: dict) -> MDPDataset:
    W = int(cfg["window_bars"]); ma = prices["close"].rolling(96, min_periods=96).mean()
    allow_long, allow_short = (prices["close"] > ma).fillna(False), (prices["close"] < ma).fillna(False)

    obs, act, rew, done = [], [], [], []
    w_prev, eq, peak = 0.0, 1.0, 1.0
    bps = (cfg["costs"]["slippage_bps"] + cfg["costs"]["commission_bps"]) / 1e4
    kappa = cfg["reward"]["kappa_cost"]; lam = cfg["reward"]["lambda_risk"]

    for t in range(W+1, len(prices)):
        # make observation (flatten window)
        window = np.concatenate([feats.iloc[t-W:t].to_numpy(np.float32),
                                 np.full((W,1), w_prev, np.float32)], axis=1)
        obs.append(window.reshape(-1))

        # heuristic action
        mom = (prices["close"].iloc[t] / prices["close"].iloc[t-24] - 1.0) if t>=24 else 0.0
        a = 0.0
        if mom>0 and allow_long.iloc[t]: a = 0.5
        if mom<0 and allow_short.iloc[t]: a = -0.5
        # execute previous weight
        ret = np.log(prices["close"].iloc[t] / prices["close"].iloc[t-1])
        raw = w_prev * ret
        cost = bps * abs(a - w_prev)
        eq = eq * (1 + raw - kappa*cost)
        peak = max(peak, eq); ddplus = max(0.0, (peak-eq)/peak)
        r = raw - kappa*cost - lam*ddplus

        act.append([a]); rew.append(r); done.append(False)
        w_prev = a

    # final step done=True
    done[-1] = True
    return MDPDataset(
        observations=np.stack(obs, 0),
        actions=np.array(act, np.float32),
        rewards=np.array(rew, np.float32),
        terminals=np.array(done, np.bool_)
    )
