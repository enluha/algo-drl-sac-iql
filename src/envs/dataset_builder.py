from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from d3rlpy.dataset import MDPDataset

from src.utils.io_utils import cache_pickle


def _heuristic_policy(close: pd.Series, ma96: pd.Series, slope: pd.Series) -> pd.Series:
    bias_up = (close > ma96 * 1.001) & (slope > 0)
    bias_down = (close < ma96 * 0.999) & (slope < 0)
    position = pd.Series(0.0, index=close.index)
    hold = 0
    min_hold = 6
    last_pos = 0.0
    for idx in range(len(close)):
        target = last_pos
        if hold > 0:
            hold -= 1
        else:
            if bias_up.iloc[idx]:
                target = 1.0
                hold = min_hold
            elif bias_down.iloc[idx]:
                target = -1.0
                hold = min_hold
            elif abs(last_pos) < 1e-6:
                target = 0.0
        last_pos = target
        position.iloc[idx] = target
    return position


def _build_observations(
    norm_features: pd.DataFrame,
    window: int,
    actions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    feature_arr = norm_features.to_numpy(dtype=np.float32)
    total = feature_arr.shape[0]
    obs_list = []
    next_obs_list = []
    for idx in range(window, total - 1):
        prev_action = actions[idx - 1] if idx > 0 else 0.0
        next_action = actions[idx]
        window_slice = feature_arr[idx - window : idx]
        next_window = feature_arr[idx - window + 1 : idx + 1]
        obs = np.concatenate(
            [
                window_slice,
                np.full((window, 1), prev_action, dtype=np.float32),
            ],
            axis=1,
        )
        obs_list.append(obs)
        next_obs = np.concatenate(
            [
                next_window,
                np.full((window, 1), next_action, dtype=np.float32),
            ],
            axis=1,
        )
        next_obs_list.append(next_obs)
    return np.asarray(obs_list), np.asarray(next_obs_list)


def build_offline_dataset(
    ohlcv: pd.DataFrame,
    norm_features: pd.DataFrame,
    env_cfg: Dict,
    reward_cfg: Dict,
    costs_cfg: Dict,
    artifact_path: Path | str = Path("evaluation/artifacts/offline_dataset.h5"),
) -> MDPDataset:
    window = env_cfg["window_bars"]
    leverage_max = env_cfg["leverage_max"]
    deadband = env_cfg["deadband"]
    min_step = env_cfg["min_step"]

    close = ohlcv["close"].astype(float)
    ma96 = close.rolling(96).mean()
    slope = ma96.diff()
    heuristic_pos = _heuristic_policy(close, ma96, slope).to_numpy(dtype=np.float32)

    returns = close.pct_change().fillna(0.0).to_numpy(dtype=np.float32)
    slippage = costs_cfg.get("slippage_bps", 0) / 1e4
    commission = costs_cfg.get("commission_bps", 0) / 1e4
    trade_cost_bps = slippage + commission

    obs, next_obs = _build_observations(norm_features, window, heuristic_pos)

    rewards = []
    actions = []
    terminals = []
    weights = []
    equity = 1.0
    peak = 1.0
    lambda_risk = reward_cfg["lambda_risk"]
    kappa_cost = reward_cfg["kappa_cost"]

    for idx in range(window, len(returns) - 1):
        prev_weight = heuristic_pos[idx - 1] if idx > 0 else 0.0
        target = heuristic_pos[idx]
        target = np.clip(target, -leverage_max, leverage_max)
        if abs(target) < deadband:
            target = 0.0
        if abs(target - prev_weight) < min_step:
            target = prev_weight

        action = target
        weights.append(action)
        actions.append([action])

        ret = returns[idx]
        raw_ret = prev_weight * ret
        delta = action - prev_weight
        cost = trade_cost_bps * abs(delta)
        net_ret = raw_ret - cost
        equity *= 1.0 + net_ret
        peak = max(peak, equity)
        drawdown = (peak - equity) / peak if peak > 0 else 0.0
        reward = raw_ret - kappa_cost * abs(delta) - lambda_risk * drawdown
        rewards.append(reward)
        terminals.append(False)

    if terminals:
        terminals[-1] = True

    dataset = MDPDataset(
        observations=obs.astype(np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        terminals=np.asarray(terminals, dtype=np.float32),
        next_observations=next_obs.astype(np.float32),
    )

    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.dump(str(artifact_path))
    cache_pickle(weights, artifact_path.with_suffix("_weights.pkl"))
    return dataset
