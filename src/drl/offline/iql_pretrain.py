import os, logging
from pathlib import Path
import numpy as np
import pandas as pd
from d3rlpy.algos import IQL, IQLConfig, BC
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.preprocessing import MinMaxActionScaler
from src.utils.io_utils import load_yaml
from src.utils.logging_utils import get_logger
from src.utils.device import get_torch_device, log_device, set_num_threads
from src.features.engineering import build_features
from src.features.normalizer import RollingZScore
from src.envs.dataset_builder import build_offline_dataset

def _load_config() -> dict:
    cfg_path = Path(os.getenv("CONFIG","config/config.yaml"))
    cfg = load_yaml(cfg_path)
    if "data" not in cfg:
        raise KeyError(f"Config hub {cfg_path} missing required sections")
    return cfg

def main():
    cfg = _load_config()
    data_cfg, env_cfg, algo_cfg, rt_cfg = cfg["data"], cfg["env"], cfg["algo_iql"], cfg["runtime"]
    logger = get_logger("iql_pretrain", level=rt_cfg.get("log_level","INFO"))
    set_num_threads(rt_cfg.get("blas_threads",6))
    device = get_torch_device(None); log_device(logger)

    df = pd.read_csv(data_cfg["csv_path"], parse_dates=["date"]).set_index("date")
    df = df.rename(columns=str.lower)
    feats = build_features(df)
    norm = RollingZScore(window=500).fit(feats)
    feats_n = norm.transform(feats)
    arr = feats_n.to_numpy()
    if not np.isfinite(arr).all():
        bad = arr.size - np.isfinite(arr).sum()
        raise RuntimeError(f"Non-finite features after normalization: {bad} cells")

    dset = build_offline_dataset(df[["open","high","low","close"]], feats_n, env_cfg)
    hidden = tuple(algo_cfg["encoder"]["mlp_hidden"])
    encoder = VectorEncoderFactory(hidden_units=hidden)
    config_iql = IQLConfig(
        batch_size=algo_cfg["batch_size"],
        gamma=algo_cfg["discount"],
        actor_learning_rate=algo_cfg["lr"],
        critic_learning_rate=algo_cfg["lr"],
        expectile=algo_cfg["expectile_beta"],
        weight_temp=algo_cfg["temperature"],
        actor_encoder_factory=encoder,
        critic_encoder_factory=encoder,
        value_encoder_factory=encoder,
        action_scaler=MinMaxActionScaler(minimum=-1.0, maximum=1.0),
    )
    agent = IQL(config=config_iql, device=device.type, enable_ddp=False)
    out = Path("evaluation/artifacts"); out.mkdir(parents=True, exist_ok=True)
    steps = int(os.getenv("QA_STEPS", algo_cfg["grad_steps"]))
    try:
        agent.fit(dset, n_steps=steps, save_interval=int(1e9))
        agent.save_model(str(out / "iql_policy.d3"))
    except Exception as err:
        logger.error("IQL fit failed (%s). Falling back to Behavior Cloning pretrain.", err)
        bc = BC(
            learning_rate=algo_cfg["lr"],
            batch_size=algo_cfg["batch_size"],
            use_gpu=(device.type == "cuda")
        )
        bc.fit(dset, n_steps=int(os.getenv("QA_STEPS", 300000)), save_interval=int(1e9))
        bc.save_model(str(out / "iql_policy.d3"))
        logger.info("BC pretrain saved; continuing.")
    norm.save(out / "normalizer.pkl")
    logger.info("Offline IQL pretrain complete; artifacts saved.")

if __name__ == "__main__":
    main()
