import numpy as np
import pandas as pd

from src.envs.market_env import MarketEnv


def test_equity_starts_at_one_and_costs_full():
    idx = pd.date_range("2025-01-01", periods=5, freq="h")
    close = pd.Series([100, 101, 102, 101, 101], index=idx)
    ohlc = pd.DataFrame({"open": close, "high": close, "low": close, "close": close})
    feats = pd.DataFrame(np.zeros((5, 3), dtype=np.float32), index=idx, columns=list("abc"))
    cfg = {
        "window_bars": 2,
        "latency_bars": 1,
        "deadband": 0.0,
        "min_step": 0.0,
        "leverage_max": 1.0,
        "costs": {"slippage_bps": 5, "commission_bps": 10},
        "reward": {"kappa_cost": 1.0, "lambda_risk": 0.0},
    }
    env = MarketEnv(ohlc, feats, cfg)
    _, _ = env.reset()
    assert abs(env.equity - 1.0) < 1e-9

    action = np.array([1.0], dtype=np.float32)
    _, reward, *_ = env.step(action)
    turnover = abs(np.tanh(action[0]) - 0.0)
    cost = (cfg["costs"]["slippage_bps"] + cfg["costs"]["commission_bps"]) / 1e4 * turnover
    expected_equity = 1.0 - cost
    assert abs(env.equity - expected_equity) < 1e-6
    assert np.isclose(reward, -cost)
    assert reward <= env.equity  # reward should reflect cost deduction
