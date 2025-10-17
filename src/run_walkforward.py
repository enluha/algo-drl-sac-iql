from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from d3rlpy.algos import SAC
from d3rlpy.models.encoders import VectorEncoderFactory
from src.evaluation.metrics import summarize
from src.evaluation.plots import candlestick, equity_plot
from src.evaluation.reporter import build_report
from src.envs.market_env import MarketEnv, MarketEnvConfig
from src.features.engineering import build_features
from src.features.normalizer import RollingZScore
from src.utils import device as device_utils
from src.utils.io_utils import load_nested_config, load_pickle, save_csv
from src.utils.logging_utils import get_logger
from src.utils.seed import seed_everything


def _load_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward evaluation")
    parser.add_argument("--config", default="config/config.yaml", help="Path to master config")
    parser.add_argument("--device", default=None, help="Preferred torch device")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--blas-threads", type=int, default=None)
    parser.add_argument("--log-level", default=None)
    parser.add_argument("--n-workers", type=int, default=None)
    return parser.parse_args(argv)


def _make_env_config(env_cfg: Dict) -> MarketEnvConfig:
    return MarketEnvConfig(
        window_bars=env_cfg["window_bars"],
        latency_bars=env_cfg["latency_bars"],
        leverage_max=env_cfg["leverage_max"],
        deadband=env_cfg["deadband"],
        min_step=env_cfg["min_step"],
        reward=env_cfg["reward"],
        costs=env_cfg["costs"],
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_nested_config(args.config)
    runtime_cfg = cfg["runtime"]
    overrides = {
        "seed": args.seed,
        "blas_threads": args.blas_threads,
        "log_level": args.log_level,
        "n_workers": args.n_workers,
    }
    runtime_cfg.update({k: v for k, v in overrides.items() if v is not None})

    seed_everything(runtime_cfg.get("seed", 42))
    device_utils.set_num_threads(runtime_cfg.get("blas_threads", 6))

    logger = get_logger(__name__, runtime_cfg.get("log_level", "INFO"))
    torch_device = device_utils.get_torch_device(args.device or cfg["algo_sac"].get("device"))
    device_utils.log_device(logger)

    data_cfg = cfg["data"]
    prices = _load_prices(Path(data_cfg["csv_path"]))
    features = build_features(prices)

    artifacts_dir = Path("evaluation/artifacts")
    normalizer_path = artifacts_dir / "normalizer.pkl"
    if normalizer_path.exists():
        normalizer_template: RollingZScore = load_pickle(normalizer_path)
    else:
        normalizer_template = RollingZScore()

    sac_policy_path = artifacts_dir / "sac_policy.d3"
    if not sac_policy_path.exists():
        raise FileNotFoundError("SAC policy missing; run fine-tune first")

    algo_cfg = cfg["algo_sac"]
    encoder_factory = VectorEncoderFactory(hidden_units=algo_cfg["encoder"]["mlp_hidden"])
    sac = SAC(
        gamma=algo_cfg["gamma"],
        tau=algo_cfg["tau"],
        actor_learning_rate=algo_cfg["lr_actor"],
        critic_learning_rate=algo_cfg["lr_critic"],
        batch_size=algo_cfg["batch_size"],
        encoder_factory=encoder_factory,
        use_gpu=torch_device.type == "cuda",
        device=torch_device,
    )

    wf_cfg = cfg["walkforward"]
    env_cfg = _make_env_config(cfg["env"])

    train_days = pd.Timedelta(days=wf_cfg["train_days"])
    valid_days = pd.Timedelta(days=wf_cfg["valid_days"])
    test_days = pd.Timedelta(days=wf_cfg["test_days"])
    step_days = pd.Timedelta(days=wf_cfg["step_days"])

    start = prices.index[0]
    end = prices.index[-1]

    ledger_frames: List[pd.DataFrame] = []
    signals_frames: List[pd.Series] = []
    predictions_frames: List[pd.Series] = []

    fold = 0
    current_start = start
    last_test_prices = None
    policy_loaded = False
    last_signals = None

    while current_start + train_days + valid_days + test_days <= end:
        train_end = current_start + train_days
        valid_end = train_end + valid_days
        test_end = valid_end + test_days

        train_slice = slice(current_start, train_end)
        test_slice = slice(valid_end, test_end)

        train_feats = features.loc[train_slice]
        test_feats = features.loc[test_slice]
        test_prices = prices.loc[test_slice]

        if len(test_feats) < env_cfg.window_bars + 2:
            current_start += step_days
            continue

        normalizer = RollingZScore(window=normalizer_template.window)
        normalizer.fit_partial(train_feats)

        env = MarketEnv(test_prices, test_feats, normalizer, env_cfg)
        sac.build_with_env(env)
        if not policy_loaded:
            sac.load_policy(str(sac_policy_path))
            policy_loaded = True

        obs, info = env.reset()
        ledger_rows = []
        signals_rows = []
        predictions_rows = []
        done = False
        step_idx = 0
        while not done:
            obs_array = np.expand_dims(obs, axis=0)
            action = sac.predict(obs_array, deterministic=True)[0]
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            timestamp = step_info["timestamp"]
            ledger_rows.append(
                {
                    "timestamp": timestamp,
                    "raw_ret": step_info["raw_ret"],
                    "cost": step_info["cost"],
                    "net_ret": step_info["net_ret"],
                    "weight": step_info["weight"],
                    "equity": step_info["equity"],
                }
            )
            signals_rows.append((timestamp, float(np.sign(action[0] if isinstance(action, np.ndarray) else action))))
            predictions_rows.append((timestamp, float(action[0] if isinstance(action, np.ndarray) else action)))
            obs = next_obs
            done = terminated or truncated
            step_idx += 1

        fold_ledger = pd.DataFrame(ledger_rows).set_index("timestamp")
        ledger_frames.append(fold_ledger)
        signals_frames.append(pd.Series(dict(signals_rows), name=f"fold_{fold}"))
        predictions_frames.append(pd.Series(dict(predictions_rows), name=f"fold_{fold}"))
        last_test_prices = test_prices
        last_signals = pd.Series(dict(signals_rows))

        fold += 1
        current_start += step_days

    if not ledger_frames:
        raise RuntimeError("No walk-forward folds produced outputs")

    ledger = pd.concat(ledger_frames).sort_index()
    strategy_equity = ledger["equity"].sort_index()

    close_prices = prices["close"].reindex(strategy_equity.index, method="pad")
    log_returns = np.log(close_prices).diff().fillna(0.0)
    buy_hold = (1.0 + log_returns).cumprod()

    metrics = summarize(ledger, strategy_equity)
    trades = len(ledger)
    positive = ledger["net_ret"][ledger["net_ret"] > 0]
    negative = ledger["net_ret"][ledger["net_ret"] < 0]
    win_rate = len(positive) / trades if trades else 0.0
    profit_factor = positive.sum() / abs(negative.sum()) if abs(negative.sum()) > 0 else math.inf
    sqn = (ledger["net_ret"].mean() / (ledger["net_ret"].std() + 1e-8)) * np.sqrt(trades)
    trade_stats = {
        "trades": trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "sqn": sqn,
    }

    reports_dir = Path("evaluation/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    symbol = data_cfg["symbol"]

    trades_path = reports_dir / f"trades_{symbol}.csv"
    save_csv(ledger, trades_path)

    equity_curve = pd.DataFrame({"strategy": strategy_equity, "buy_and_hold": buy_hold})
    equity_path = reports_dir / f"equity_curve_{symbol}.csv"
    save_csv(equity_curve, equity_path)

    signals_series = pd.concat(signals_frames).sort_index()
    signals_path = reports_dir / f"signals_{symbol}.csv"
    save_csv(signals_series.to_frame(name="signal"), signals_path)

    predictions_series = pd.concat(predictions_frames).sort_index()
    predictions_path = reports_dir / f"predictions_{symbol}.csv"
    save_csv(predictions_series.to_frame(name="action"), predictions_path)

    charts_dir = Path("evaluation/charts")
    charts_dir.mkdir(parents=True, exist_ok=True)
    if last_test_prices is not None:
        month_tag = (
            f"{last_test_prices.index[0]:%b}{last_test_prices.index[-1]:%b%Y}"
            if len(last_test_prices)
            else "period"
        )
        candle_path = charts_dir / f"candlestick_{symbol}_{month_tag}.html"
        signal_slice = last_signals.reindex(last_test_prices.index).fillna(0) if last_signals is not None else None
        candlestick(last_test_prices, signal_slice, candle_path)
    equity_chart_path = charts_dir / f"equity_{symbol}.html"
    equity_plot(strategy_equity, {"Buy&Hold": buy_hold}, equity_chart_path)

    summary_path = reports_dir / f"summary_report_{symbol}.txt"
    build_report(symbol, cfg, metrics, trade_stats, summary_path)


if __name__ == "__main__":
    main()
