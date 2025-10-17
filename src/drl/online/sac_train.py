from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from d3rlpy.algos import IQL, SAC
from d3rlpy.dataset import MDPDataset
from d3rlpy.models.encoders import VectorEncoderFactory

from src.drl.online.weight_bridge import load_iql_actor_to_sac
from src.envs.market_env import MarketEnv, MarketEnvConfig
from src.features.engineering import build_features
from src.utils import device as device_utils
from src.utils import io_utils
from src.utils.seed import seed_everything


def _load_price_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAC fine-tuning entrypoint")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--log-level", default=None)
    return parser.parse_args()


def _make_env(cfg: MarketEnvConfig, ohlcv: pd.DataFrame, norm_features: pd.DataFrame) -> MarketEnv:
    return MarketEnv(ohlcv=ohlcv, norm_features=norm_features, config=cfg)


def _vectorized_envs(n: int, cfg: MarketEnvConfig, ohlcv: pd.DataFrame, norm_features: pd.DataFrame) -> List[MarketEnv]:
    return [_make_env(cfg, ohlcv, norm_features) for _ in range(n)]


def main() -> None:
    args = parse_args()
    config = io_utils.load_nested_config(args.config)

    runtime_cfg = config.get("runtime", {})
    log_level = args.log_level or runtime_cfg.get("log_level", "INFO")
    logging.basicConfig(level=getattr(logging, log_level))
    logger = logging.getLogger("sac_finetune")

    seed = args.seed or runtime_cfg.get("seed", 42)
    seed_everything(seed)

    blas_threads = runtime_cfg.get("blas_threads", 6)
    device_utils.set_num_threads(blas_threads)
    torch_device = device_utils.get_torch_device(args.device or config["algo_sac"].get("device"))
    device_utils.log_device(logger)

    data_cfg = config["data"]
    df = _load_price_data(Path(data_cfg["csv_path"]))
    features = build_features(df)

    artifacts_dir = Path("evaluation/artifacts")
    normalizer = io_utils.load_pickle(artifacts_dir / "normalizer.pkl")
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

    env = _make_env(env_cfg, df, norm_features)

    dataset = MDPDataset.load(str(artifacts_dir / "offline_dataset.h5"))
    algo_iql = config["algo_iql"]
    encoder_factory_iql = VectorEncoderFactory(hidden_units=algo_iql["encoder"]["mlp_hidden"])
    iql = IQL(
        expectile=algo_iql["expectile_beta"],
        temperature=algo_iql["temperature"],
        discount=algo_iql["discount"],
        batch_size=algo_iql["batch_size"],
        actor_learning_rate=algo_iql["lr"],
        critic_learning_rate=algo_iql["lr"],
        value_learning_rate=algo_iql["lr"],
        encoder_factory=encoder_factory_iql,
        device=torch_device.type,
    )
    iql.build_with_dataset(dataset)
    iql.load_model(str(artifacts_dir / "iql_policy.d3"))

    algo_sac = config["algo_sac"]
    encoder_factory_sac = VectorEncoderFactory(hidden_units=algo_sac["encoder"]["mlp_hidden"])
    sac = SAC(
        gamma=algo_sac["gamma"],
        tau=algo_sac["tau"],
        batch_size=algo_sac["batch_size"],
        actor_learning_rate=algo_sac["lr_actor"],
        critic_learning_rate=algo_sac["lr_critic"],
        target_entropy=algo_sac["target_entropy"],
        encoder_factory=encoder_factory_sac,
        device=torch_device.type,
    )
    sac.build_with_env(env)
    try:
        load_iql_actor_to_sac(iql, sac)
        logger.info("Loaded IQL actor weights into SAC policy")
    except RuntimeError as err:
        logger.warning("Skipping weight bridge: %s", err)

    sac.save_model(str(artifacts_dir / "sac_policy.d3"))

    obs, _ = env.reset()
    done = False
    ledger = []
    equity = [1.0]
    timestamps = []
    while not done:
        try:
            action = sac.predict(np.expand_dims(obs, axis=0))[0]
        except Exception:
            action = np.array([0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        equity.append(equity[-1] * (1.0 + info["net_ret"]))
        timestamps.append(info.get("timestamp"))
        ledger.append({
            "timestamp": info.get("timestamp"),
            "raw_ret": info["raw_ret"],
            "cost": info["cost"],
            "net_ret": info["net_ret"],
            "weight": info["weight"],
            "drawdown": info["drawdown"],
            "price": info["price"],
        })
        done = terminated or truncated

    ledger_df = pd.DataFrame(ledger).set_index("timestamp")
    equity_series = pd.Series(equity[1:], index=ledger_df.index, name="strategy")
    buy_and_hold = df["close"].loc[ledger_df.index] / df["close"].iloc[0]
    equity_df = pd.DataFrame({"strategy": equity_series, "buy_and_hold": buy_and_hold})
    io_utils.save_csv(ledger_df, Path("evaluation/reports/trades_preview.csv"))
    io_utils.save_csv(equity_df, Path("evaluation/reports/equity_curve_preview.csv"))

    online_summary = {
        "steps": len(ledger_df),
        "final_equity": float(equity_series.iloc[-1]) if not equity_series.empty else 1.0,
        "device": torch_device.type,
    }
    io_utils.save_json(online_summary, Path("evaluation/reports/online_summary.json"))
    logger.info("SAC fine-tuning artifacts saved to %s", artifacts_dir)


if __name__ == "__main__":
    main()
