from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.features.normalizer import RollingZScore


@dataclass
class MarketEnvConfig:
    window_bars: int
    latency_bars: int
    leverage_max: float
    deadband: float
    min_step: float
    reward: Dict[str, float]
    costs: Dict[str, float]


class MarketEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(
        self,
        ohlcv: pd.DataFrame,
        features: pd.DataFrame,
        normalizer: RollingZScore,
        config: MarketEnvConfig,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config
        self.window = config.window_bars
        self.ohlcv = ohlcv.sort_index()
        self.features = normalizer.transform(features.loc[self.ohlcv.index])
        self.latency = max(1, config.latency_bars)
        self.device = "cpu"

        self._returns = np.log(self.ohlcv["close"].astype(float).replace(0, np.nan)).diff().fillna(0.0).to_numpy()
        self._prices = self.ohlcv["close"].astype(float).to_numpy()
        self._cost_rate = (config.costs["slippage_bps"] + config.costs["commission_bps"]) / 1e4

        obs_shape = (self.window, self.features.shape[1] + 1)
        self.observation_space = spaces.Box(low=-10, high=10, shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._t = self.window
        self._prev_weight = 0.0
        self._equity = 1.0
        self._peak = 1.0

    def seed(self, seed: int | None = None) -> None:
        self.np_random, _ = gym.utils.seeding.np_random(seed)

    def _get_observation(self) -> np.ndarray:
        window_slice = slice(self._t - self.window, self._t)
        feats = self.features.iloc[window_slice].to_numpy(dtype=np.float32, copy=True)
        action_channel = np.full((self.window, 1), self._prev_weight, dtype=np.float32)
        return np.concatenate([feats, action_channel], axis=1)

    def reset(self, *, seed: int | None = None, options: Dict | None = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.seed(seed)
        self._t = self.window
        self._prev_weight = 0.0
        self._equity = 1.0
        self._peak = 1.0
        obs = self._get_observation()
        info = {
            "weight": self._prev_weight,
            "equity": self._equity,
            "price": self._prices[self._t - 1],
            "timestamp": self.ohlcv.index[self._t - 1],
        }
        return obs, info

    def step(self, action: np.ndarray):
        desired = float(np.clip(action[0], -self.cfg.leverage_max, self.cfg.leverage_max))
        if abs(desired) < self.cfg.deadband:
            desired = 0.0
        delta = desired - self._prev_weight
        if abs(delta) < self.cfg.min_step:
            desired = self._prev_weight
            delta = 0.0

        raw_ret = self._prev_weight * self._returns[self._t - 1]
        trade_cost = self._cost_rate * abs(delta)
        net_ret = raw_ret - trade_cost
        self._equity *= float(1.0 + net_ret)
        self._peak = max(self._peak, self._equity)
        drawdown = max(0.0, 1.0 - self._equity / (self._peak + 1e-12))

        reward = (
            raw_ret
            - self.cfg.reward["kappa_cost"] * abs(delta)
            - self.cfg.reward["lambda_risk"] * drawdown
        )

        info = {
            "raw_ret": raw_ret,
            "cost": trade_cost,
            "net_ret": net_ret,
            "weight": self._prev_weight,
            "drawdown": drawdown,
            "price": self._prices[self._t - 1],
            "timestamp": self.ohlcv.index[self._t - 1],
            "equity": self._equity,
        }

        self._prev_weight = desired
        terminated = False
        truncated = self._t >= len(self.ohlcv) - 1
        self._t += 1
        obs = self._get_observation() if not truncated else np.zeros_like(self._get_observation())
        return obs, float(reward), terminated, truncated, info
