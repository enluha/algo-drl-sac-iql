import gymnasium as gym
import numpy as np, pandas as pd
from typing import Tuple, Dict, Any

class MarketEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, prices: pd.DataFrame, features: pd.DataFrame, cfg: dict):
        super().__init__()
        self.prices = prices  # DataFrame with 'close' (and open/high/low for charts)
        self.feats = features # DataFrame aligned to prices (float32)
        self.cfg = cfg
        self.W = int(cfg["window_bars"])
        self.deadband = float(cfg["deadband"])
        self.min_step = float(cfg["min_step"])
        self.latency = int(cfg.get("latency_bars", 1))
        self.leverage = float(cfg.get("leverage_max", 1.0))
        costs = cfg.get("costs", {})
        self.bps = (float(costs.get("slippage_bps", 0)) + float(costs.get("commission_bps", 0))) / 1e4
        rw = cfg.get("reward", {})
        self.kappa_cost = float(rw.get("kappa_cost", 0.0))
        self.lambda_risk = float(rw.get("lambda_risk", 0.0))
        self.kappa_turnover = float(rw.get("kappa_turnover", 0.0))  # Direct turnover penalty
        self.risk_metric = str(rw.get("risk_metric", "drawdown")).lower()

        self.t = 0
        self._w_prev = 0.0
        self._equity = 1.0
        self._ret_last = 0.0
        self._cost_last = 0.0
        self._net_last = 0.0
        self._peak = 1.0

        self.F = self.feats.shape[1] + 1  # + last action channel
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.W * self.F,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def _obs(self):
        sl = slice(self.t - self.W, self.t)
        x = self.feats.iloc[sl].to_numpy(dtype=np.float32)
        a = np.full((self.W,1), self._w_prev, dtype=np.float32)
        return np.concatenate([x,a], axis=1).reshape(-1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.W + self.latency
        self._w_prev = 0.0
        self._equity = 1.0  # MUST start at 1.0
        self._peak = 1.0
        self._ret_last = self._cost_last = self._net_last = 0.0
        obs = self._obs()
        # Guard: never start above 1.0
        assert abs(self._equity - 1.0) < 1e-9, f"equity initialized to {self._equity}, expected 1.0"
        return obs, {}

    def step(self, action: np.ndarray):
        # desired position for the NEXT bar
        a = float(np.tanh(action[0]))
        if abs(a) < self.deadband: a = 0.0
        # enforce min_step on change
        if abs(a - self._w_prev) < self.min_step: a = self._w_prev
        a = float(np.clip(a, -self.leverage, +self.leverage))

        ret_t = float(np.log(self.prices["close"].iloc[self.t] / self.prices["close"].iloc[self.t-1]))
        turn = abs(a - self._w_prev)
        cost = self.bps * turn  # full cost (slippage + commission) in P&L units
        raw = self._w_prev * ret_t
        eq_next = self._equity * (1.0 + raw - cost)

        prev_dd = max(0.0, (self._peak - self._equity) / (self._peak + 1e-12))
        if eq_next >= self._peak:
            new_dd = 0.0
            self._peak = eq_next
        else:
            new_dd = (self._peak - eq_next) / (self._peak + 1e-12)

        if self.risk_metric == "dd_velocity":
            risk_pen = max(0.0, new_dd - prev_dd)
        else:
            risk_pen = new_dd

        reward = raw - self.kappa_cost * cost - self.lambda_risk * risk_pen - self.kappa_turnover * turn

        if not np.isfinite([raw, cost, reward, eq_next]).all():
            raw = cost = reward = 0.0
            eq_next = max(1e-9, float(self._equity))
            new_dd = prev_dd = 0.0
            risk_pen = 0.0

        # advance time & state
        self._equity = float(eq_next)
        self._w_prev = a
        self._ret_last, self._cost_last, self._net_last = raw, cost, reward
        self.t += 1
        done = (self.t >= len(self.prices))
        return self._obs(), reward, done, False, {
            "raw_ret": raw, "cost": cost, "net_ret": reward,
            "weight": a, "turnover": turn,
            "equity": self._equity, "drawdown": new_dd, "dd_velocity": max(0.0, new_dd - prev_dd),
            "price": float(self.prices["close"].iloc[self.t-1]),
            "timestamp": self.prices.index[self.t-1]
        }

    # simple accessors for tests
    @property
    def last_weight(self): return float(self._w_prev)
    @property
    def last_return(self): return float(self._ret_last)
    @property
    def last_cost(self): return float(self._cost_last)
    @property
    def last_net_ret(self): return float(self._net_last)
    @property
    def equity(self): return float(self._equity)
