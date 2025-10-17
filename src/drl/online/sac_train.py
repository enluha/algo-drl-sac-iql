from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from d3rlpy.algos import IQL, SAC
from d3rlpy.dataset import MDPDataset
from d3rlpy.models.encoders import VectorEncoderFactory

from src.drl.online.weight_bridge import load_iql_actor_to_sac
from src.utils import device as device_utils
from src.utils.io_utils import load_nested_config, save_csv, save_json
from src.utils.logging_utils import get_logger
from src.utils.seed import seed_everything

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

    log = get_logger(__name__, runtime_cfg.get("log_level", "INFO"))
    torch_device = device_utils.get_torch_device(prefer_device or cfg["algo_sac"].get("device"))
    device_utils.log_device(log)

    artifacts_dir = Path("evaluation/artifacts")
    dataset_path = artifacts_dir / "offline_dataset.h5"
    if not dataset_path.exists():
        raise FileNotFoundError("Offline dataset missing; run offline pretrain first")

    log.info("Loading offline dataset from %s", dataset_path)
    dataset = MDPDataset.load(str(dataset_path))

    iql_policy_path = artifacts_dir / "iql_policy.d3"
    if not iql_policy_path.exists():
        raise FileNotFoundError("IQL policy missing; cannot warm start SAC")

    iql = IQL()
    iql.build_with_dataset(dataset)
    iql.load_policy(str(iql_policy_path))

    algo_cfg = cfg["algo_sac"]
    encoder_factory = VectorEncoderFactory(hidden_units=algo_cfg["encoder"]["mlp_hidden"])
    sac = SAC(
        gamma=algo_cfg["gamma"],
        tau=algo_cfg["tau"],
        actor_learning_rate=algo_cfg["lr_actor"],
        critic_learning_rate=algo_cfg["lr_critic"],
        batch_size=algo_cfg["batch_size"],
        use_gpu=torch_device.type == "cuda",
        device=torch_device,
        encoder_factory=encoder_factory,
    )
    sac.build_with_dataset(dataset)
    sac = load_iql_actor_to_sac(iql, sac)

    total_steps = int(len(dataset) * algo_cfg.get("updates_per_step", 1))
    log.info("Starting SAC fine-tuning for %s steps", total_steps)
    sac.fit(dataset, n_steps=total_steps, progress_bar=True, logger=log)

    sac_policy_path = artifacts_dir / "sac_policy.d3"
    sac.save_policy(str(sac_policy_path))

    rewards = np.asarray(dataset.rewards, dtype=np.float32)
    equity = np.cumprod(1.0 + rewards)
    equity_path = Path("evaluation/reports/equity_curve_training.csv")
    save_csv(
        {
            "step": np.arange(len(equity)),
            "equity": equity,
        },
        equity_path,
    )

    summary = {
        "steps": total_steps,
        "batch_size": algo_cfg["batch_size"],
        "gamma": algo_cfg["gamma"],
        "device": str(torch_device),
    }
    reports_dir = Path("evaluation/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "online_summary.json"
    save_json(summary, summary_path)
    log.info("SAC fine-tuning complete")
    return summary
