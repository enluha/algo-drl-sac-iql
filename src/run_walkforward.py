from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from d3rlpy.algos import SAC
from d3rlpy.models.encoders import VectorEncoderFactory

from src.drl.online.sac_train import _load_price_data
from src.evaluation import metrics as metrics_mod
from src.evaluation import plots as plots_mod
from src.evaluation import reporter
from src.envs.market_env import MarketEnv, MarketEnvConfig
from src.features.engineering import build_features
from src.utils import device as device_utils
from src.utils import io_utils
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward evaluation")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--log-level", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = io_utils.load_nested_config(args.config)

    runtime_cfg = config.get("runtime", {})
    log_level = args.log_level or runtime_cfg.get("log_level", "INFO")
    logging.basicConfig(level=getattr(logging, log_level))
    logger = logging.getLogger("walkforward")

    seed = args.seed or runtime_cfg.get("seed", 42)
    seed_everything(seed)
    device_utils.set_num_threads(runtime_cfg.get("blas_threads", 6))
    torch_device = device_utils.get_torch_device(args.device or config["algo_sac"].get("device"))
    device_utils.log_device(logger)

    df = _load_price_data(Path(config["data"]["csv_path"]))
    features = build_features(df)
    normalizer = io_utils.load_pickle(Path("evaluation/artifacts/normalizer.pkl"))
    norm_features = normalizer.transform(features)

    env_cfg_dict = config["env"]
    env_cfg = MarketEnvConfig(
        window_bars=env_cfg_dict["window_bars"],
        latency_bars=env_cfg_dict["latency_bars"],
        leverage_max=env_cfg_dict["leverage_max"],
        deadband=env_cfg_dict["deadband"],
        min_step=env_cfg_dict["min_step"],
        reward=env_cfg_dict["reward"],
        costs=env_cfg_dict["costs"],
    )

    env = MarketEnv(df, norm_features, env_cfg, seed)

    algo_sac = config["algo_sac"]
    encoder_factory = VectorEncoderFactory(hidden_units=algo_sac["encoder"]["mlp_hidden"])
    sac = SAC(
        gamma=algo_sac["gamma"],
        tau=algo_sac["tau"],
        batch_size=algo_sac["batch_size"],
        actor_learning_rate=algo_sac["lr_actor"],
        critic_learning_rate=algo_sac["lr_critic"],
        target_entropy=algo_sac["target_entropy"],
        encoder_factory=encoder_factory,
        device=torch_device.type,
    )
    sac.build_with_env(env)
    sac.load_model("evaluation/artifacts/sac_policy.d3")

    obs, _ = env.reset()
    done = False
    ledger_records = []
    weights = []
    predictions = []
    equity = [1.0]
    timestamps = []
    while not done:
        try:
            action = sac.predict(np.expand_dims(obs, axis=0))[0]
        except Exception:
            action = np.array([0.0], dtype=np.float32)
        predictions.append(action[0])
        obs, reward, terminated, truncated, info = env.step(action)
        timestamps.append(info.get("timestamp"))
        ledger_records.append({
            "timestamp": info.get("timestamp"),
            "raw_ret": info["raw_ret"],
            "cost": info["cost"],
            "net_ret": info["net_ret"],
            "weight": info["weight"],
            "drawdown": info["drawdown"],
            "price": info["price"],
        })
        weights.append(info["weight"])
        equity.append(equity[-1] * (1.0 + info["net_ret"]))
        done = terminated or truncated

    ledger_df = pd.DataFrame(ledger_records).set_index("timestamp")
    equity_series = pd.Series(equity[1:], index=ledger_df.index, name="strategy")
    buy_and_hold = df["close"].loc[ledger_df.index] / df["close"].iloc[0]
    equity_df = pd.DataFrame({"strategy": equity_series, "buy_and_hold": buy_and_hold})

    reports_dir = Path("evaluation/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    io_utils.save_csv(ledger_df, reports_dir / f"trades_{config['data']['symbol']}.csv")
    io_utils.save_csv(equity_df, reports_dir / f"equity_curve_{config['data']['symbol']}.csv")
    signals = pd.Series(np.sign(weights), index=ledger_df.index, name="signal")
    io_utils.save_csv(signals, reports_dir / f"signals_{config['data']['symbol']}.csv")
    preds_series = pd.Series(predictions, index=ledger_df.index, name="prediction")
    io_utils.save_csv(preds_series, reports_dir / f"predictions_{config['data']['symbol']}.csv")

    metrics = metrics_mod.summarize(ledger_df, equity_series)
    summary_path = reports_dir / f"summary_report_{config['data']['symbol']}.txt"
    reporter.build_summary(config, metrics, equity_series, ledger_df, summary_path)

    charts_dir = Path("evaluation/charts")
    charts_dir.mkdir(parents=True, exist_ok=True)
    plots_mod.candlestick(df.loc[ledger_df.index], signals, charts_dir / f"candlestick_{config['data']['symbol']}_SepOct2025.html")
    plots_mod.equity_plot(equity_df, {}, charts_dir / f"equity_{config['data']['symbol']}.html")

    logger.info("Walk-forward evaluation completed")


if __name__ == "__main__":
    main()
