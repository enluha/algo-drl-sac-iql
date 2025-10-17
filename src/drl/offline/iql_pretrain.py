from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from d3rlpy.algos import IQL
from d3rlpy.models.encoders import VectorEncoderFactory

from src.envs.dataset_builder import build_offline_dataset
from src.features.engineering import build_features
from src.features.normalizer import RollingZScore
from src.utils import device as device_utils
from src.utils import io_utils
from src.utils.seed import seed_everything


def _load_price_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline IQL pretraining entrypoint")
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
    logger = logging.getLogger("offline_pretrain")

    seed = args.seed or runtime_cfg.get("seed", 42)
    seed_everything(seed)

    blas_threads = runtime_cfg.get("blas_threads", 6)
    device_utils.set_num_threads(blas_threads)
    torch_device = device_utils.get_torch_device(args.device or config["algo_iql"].get("device"))
    device_utils.log_device(logger)

    data_cfg = config["data"]
    data_path = Path(data_cfg["csv_path"])
    if not data_path.exists():
        raise FileNotFoundError(f"Data CSV not found at {data_path}")
    df = _load_price_data(data_path)

    features = build_features(df)
    normalizer = RollingZScore()
    normalizer.fit_partial(features)
    norm_features = normalizer.transform(features)

    env_cfg = config["env"]
    algo_cfg = config["algo_iql"]
    costs_cfg = config["costs"]

    dataset = build_offline_dataset(
        df,
        norm_features,
        env_cfg,
        env_cfg["reward"],
        costs_cfg,
        artifact_path=Path("evaluation/artifacts/offline_dataset.h5"),
    )

    encoder_factory = VectorEncoderFactory(hidden_units=algo_cfg["encoder"]["mlp_hidden"])
    iql = IQL(
        expectile=algo_cfg["expectile_beta"],
        temperature=algo_cfg["temperature"],
        discount=algo_cfg["discount"],
        batch_size=algo_cfg["batch_size"],
        actor_learning_rate=algo_cfg["lr"],
        critic_learning_rate=algo_cfg["lr"],
        value_learning_rate=algo_cfg["lr"],
        encoder_factory=encoder_factory,
        device=torch_device.type,
    )
    iql.build_with_dataset(dataset)

    artifacts_dir = Path("evaluation/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    models_path = artifacts_dir / "iql_policy.d3"
    iql.save_model(str(models_path))

    io_utils.cache_pickle(normalizer, artifacts_dir / "normalizer.pkl")

    summary = {
        "device": torch_device.type,
        "dataset_size": int(dataset.observations.shape[0]),
        "grad_steps": algo_cfg["grad_steps"],
        "batch_size": algo_cfg["batch_size"],
    }
    io_utils.save_json(summary, Path("evaluation/reports/offline_summary.json"))
    logger.info("Offline pretraining artifacts saved to %s", artifacts_dir)


if __name__ == "__main__":
    main()
