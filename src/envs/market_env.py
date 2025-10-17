from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import pandas as pd


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
    metadata = {"render_modes": []}

    def __init__(
        self,
        ohlcv: pd.DataFrame,
        norm_features: pd.DataFrame,
        config: MarketEnvConfig,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.ohlcv = ohlcv
        self.features = norm_features.to_numpy(dtype=np.float32)
        self.close = ohlcv["close"].astype(float).to_numpy()
        self.returns = ohlcv["close"].pct_change().fillna(0.0).to_numpy(dtype=np.float32)
        self.window = config.window_bars
        self.latency = config.latency_bars
        self.device_seed = seed

        feature_dim = self.features.shape[1] + 1
        obs_shape = (self.window, feature_dim)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self._index = 0
        self._position = 0.0
        self._last_action = 0.0
        self._equity = 1.0
        self._peak = 1.0

    def seed(self, seed: Optional[int] = None) -> None:  # type: ignore[override]
        self._rng = np.random.default_rng(seed)

    def _get_observation(self) -> np.ndarray:
        window_slice = self.features[self._index - self.window : self._index]
        last_action_channel = np.full((self.window, 1), self._last_action, dtype=np.float32)
        return np.concatenate([window_slice, last_action_channel], axis=1)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):  # type: ignore[override]
        if seed is not None:
            self.seed(seed)
        self._index = self.window
        self._position = 0.0
        self._last_action = 0.0
        self._equity = 1.0
        self._peak = 1.0
        return self._get_observation(), {}

    def step(self, action: np.ndarray):  # type: ignore[override]
        target = float(action[0])
        target = max(min(target, self.config.leverage_max), -self.config.leverage_max)
        if abs(target) < self.config.deadband:
            target = 0.0
        if abs(target - self._position) < self.config.min_step:
            target = self._position

        prev_position = self._position
        self._position = target

        ret = self.returns[self._index]
        raw_ret = prev_position * ret
        delta = target - prev_position
        cost_bps = (
            self.config.costs.get("slippage_bps", 0) + self.config.costs.get("commission_bps", 0)
        ) / 1e4
        execution_cost = cost_bps * abs(delta)
        net_ret = raw_ret - execution_cost

        self._equity *= 1.0 + net_ret
        self._peak = max(self._peak, self._equity)
        drawdown = (self._peak - self._equity) / self._peak if self._peak > 0 else 0.0

        reward = (
            raw_ret
            - self.config.reward["kappa_cost"] * abs(delta)
            - self.config.reward["lambda_risk"] * drawdown
        )

        self._index += 1
        terminated = self._index >= len(self.features)
        truncated = False
        self._last_action = self._position
        obs = self._get_observation() if not terminated else np.zeros_like(self.observation_space.low)

        info = {
            "raw_ret": raw_ret,
            "cost": execution_cost,
            "net_ret": net_ret,
            "weight": self._position,
            "drawdown": drawdown,
            "price": float(self.close[self._index - 1]) if self._index - 1 < len(self.close) else float("nan"),
            "timestamp": self.ohlcv.index[self._index - 1],
        }
        return obs, reward, terminated, truncated, info
