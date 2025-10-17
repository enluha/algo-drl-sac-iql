from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from d3rlpy.dataset import MDPDataset

from src.features.engineering import build_features
from src.features.normalizer import RollingZScore


@dataclass
class DatasetArtifacts:
    dataset: MDPDataset
    normalizer: RollingZScore
    features: pd.DataFrame


def _heuristic_weights(close: pd.Series, ma_window: int, deadband: float) -> np.ndarray:
    ma = close.rolling(ma_window, min_periods=ma_window).mean()
    weights = np.zeros(len(close), dtype=np.float32)
    hold_counter = 0
    min_hold = 6
    for t in range(1, len(close)):
        price = close.iloc[t]
        prev = weights[t - 1]
        ma_t = ma.iloc[t]
        desired = prev
        if np.isnan(ma_t):
            desired = 0.0
            hold_counter = 0
        else:
            upper = ma_t * (1 + deadband * 0.5)
            lower = ma_t * (1 - deadband * 0.5)
            if hold_counter > 0:
                desired = prev
                hold_counter -= 1
            else:
                if price > upper:
                    desired = 1.0
                elif price < lower:
                    desired = -1.0
                else:
                    desired = 0.0
                if desired != prev:
                    hold_counter = min_hold
        weights[t] = desired
    return weights


def build_offline_dataset(ohlcv: pd.DataFrame, config: Dict) -> DatasetArtifacts:
    env_cfg = config["env"]
    wf_cfg = config["walkforward"]
    window = env_cfg["window_bars"]

    feats = build_features(ohlcv)
    normalizer = RollingZScore(window=500)

    train_end = ohlcv.index[0] + pd.Timedelta(days=wf_cfg["train_days"])
    train_slice = feats.loc[:train_end]
    normalizer.fit_partial(train_slice)
    feats_norm = normalizer.transform(feats)

    close = ohlcv["close"].astype(float)
    weights = _heuristic_weights(close, ma_window=96, deadband=env_cfg["deadband"])
    returns = np.log(close.replace(0, np.nan)).diff().fillna(0.0).to_numpy(dtype=np.float32)

    cost_rate = (env_cfg["costs"]["slippage_bps"] + env_cfg["costs"]["commission_bps"]) / 1e4
    kappa = env_cfg["reward"]["kappa_cost"]
    lambda_risk = env_cfg["reward"]["lambda_risk"]

    observations = []
    actions = []
    rewards = []
    terminals = []

    equity = 1.0
    peak = 1.0

    for t in range(window, len(feats_norm) - 1):
        prev_weight = weights[t - 1]
        action = weights[t]
        delta = action - prev_weight

        raw_ret = prev_weight * returns[t - 1]
        trade_cost = cost_rate * abs(delta)
        net_ret = raw_ret - trade_cost
        equity *= float(1.0 + net_ret)
        peak = max(peak, equity)
        drawdown = max(0.0, 1.0 - equity / (peak + 1e-12))
        reward = raw_ret - kappa * abs(delta) - lambda_risk * drawdown

        obs_slice = feats_norm.iloc[t - window : t].to_numpy(dtype=np.float32, copy=True)
        obs = np.concatenate([obs_slice, np.full((window, 1), prev_weight, dtype=np.float32)], axis=1)
        observations.append(obs)
        actions.append([action])
        rewards.append(reward)
        terminals.append(0.0)

    if terminals:
        terminals[-1] = 1.0

    dataset = MDPDataset(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        terminals=np.asarray(terminals, dtype=np.float32),
    )

    output_path = Path("evaluation/artifacts/offline_dataset.h5")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.dump(str(output_path))

    return DatasetArtifacts(dataset=dataset, normalizer=normalizer, features=feats_norm)
