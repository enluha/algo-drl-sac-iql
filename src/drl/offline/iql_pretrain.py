from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd
from d3rlpy.algos import IQL
from d3rlpy.models.encoders import VectorEncoderFactory

from src.envs.dataset_builder import build_offline_dataset
from src.utils import device as device_utils
from src.utils.io_utils import cache_pickle, load_nested_config, save_json
from src.utils.logging_utils import get_logger
from src.utils.seed import seed_everything


def _load_prices(data_cfg: Dict) -> pd.DataFrame:
    path = Path(data_cfg["csv_path"])
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV at {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def run(
    config_path: str,
    prefer_device: str | None = None,
    overrides: Dict | None = None,
) -> Dict:
    cfg = load_nested_config(config_path)
    runtime_cfg = cfg["runtime"]
    if overrides:
        runtime_cfg.update({k: v for k, v in overrides.items() if v is not None})
    seed = runtime_cfg.get("seed", 42)
    seed_everything(seed)

    blas_threads = runtime_cfg.get("blas_threads", 6)
    device_utils.set_num_threads(blas_threads)

    logger = get_logger(__name__, runtime_cfg.get("log_level", "INFO"))
    torch_device = device_utils.get_torch_device(prefer_device or cfg["algo_iql"].get("device"))
    device_utils.log_device(logger)

    data_cfg = cfg["data"]
    logger.info("Loading OHLCV data from %s", data_cfg["csv_path"])
    prices = _load_prices(data_cfg)

    logger.info("Building offline dataset via heuristic policy")
    artifacts = build_offline_dataset(prices, cfg)
    dataset = artifacts.dataset
    normalizer = artifacts.normalizer

    algo_cfg = cfg["algo_iql"]
    encoder_hidden = algo_cfg["encoder"]["mlp_hidden"]
    encoder_factory = VectorEncoderFactory(hidden_units=encoder_hidden)

    iql = IQL(
        expectile=algo_cfg["expectile_beta"],
        temperature=algo_cfg["temperature"],
        gamma=algo_cfg["discount"],
        actor_learning_rate=algo_cfg["lr"],
        critic_learning_rate=algo_cfg["lr"],
        batch_size=algo_cfg["batch_size"],
        encoder_factory=encoder_factory,
        use_gpu=torch_device.type == "cuda",
        device=torch_device,
    )

    grad_steps = int(algo_cfg["grad_steps"])
    logger.info("Starting IQL pretraining for %s gradient steps", grad_steps)
    iql.fit(dataset, n_steps=grad_steps, progress_bar=True, logger=logger)

    artifacts_dir = Path("evaluation/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    policy_path = artifacts_dir / "iql_policy.d3"
    logger.info("Saving IQL policy to %s", policy_path)
    iql.save_policy(str(policy_path))

    normalizer_path = artifacts_dir / "normalizer.pkl"
    cache_pickle(normalizer, normalizer_path)

    summary = {
        "grad_steps": grad_steps,
        "expectile": algo_cfg["expectile_beta"],
        "temperature": algo_cfg["temperature"],
        "batch_size": algo_cfg["batch_size"],
        "device": str(torch_device),
        "dataset_size": len(dataset),
    }
    reports_dir = Path("evaluation/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "offline_summary.json"
    save_json(summary, summary_path)
    logger.info("Offline pretraining complete")
    return summary
