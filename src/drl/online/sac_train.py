import os, logging
from pathlib import Path
import numpy as np
import pandas as pd
from d3rlpy.algos import SAC, SACConfig
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.dataset import FIFOBuffer
from d3rlpy.base import load_learnable
from src.utils.io_utils import load_yaml
from src.utils.logging_utils import get_logger
from src.utils.device import get_torch_device, log_device, set_num_threads
from src.features.engineering import build_features
from src.features.normalizer import RollingZScore
from src.envs.market_env import MarketEnv
from src.drl.online.weight_bridge import load_iql_actor_to_sac

def _load_config() -> dict:
    cfg_path = Path(os.getenv("CONFIG","config/config.yaml"))
    cfg = load_yaml(cfg_path)
    if "data" not in cfg:
        raise KeyError(f"Config hub {cfg_path} missing required sections")
    return cfg

def main():
    cfg = _load_config()
    data_cfg, env_cfg, algo_cfg, rt_cfg = cfg["data"], cfg["env"], cfg["algo_sac"], cfg["runtime"]
    logger = get_logger("sac_finetune", level=rt_cfg.get("log_level","INFO"))
    set_num_threads(rt_cfg.get("blas_threads",6))
    device = get_torch_device(None); log_device(logger)

    df = pd.read_csv(data_cfg["csv_path"], parse_dates=["date"]).set_index("date").rename(columns=str.lower)
    feats = build_features(df)
    norm = RollingZScore.load(Path("evaluation/artifacts/normalizer.pkl"))
    feats_n = norm.transform(feats)

    env = MarketEnv(df[["open","high","low","close"]], feats_n, env_cfg)

    hidden = tuple(algo_cfg["encoder"]["mlp_hidden"])
    encoder = VectorEncoderFactory(hidden_units=hidden)
    config_sac = SACConfig(
        batch_size=algo_cfg["batch_size"],
        gamma=algo_cfg["gamma"],
        actor_learning_rate=algo_cfg["lr_actor"],
        critic_learning_rate=algo_cfg["lr_critic"],
        temp_learning_rate=algo_cfg.get("lr_alpha", algo_cfg["lr_actor"]),
        actor_encoder_factory=encoder,
        critic_encoder_factory=encoder,
        q_func_factory="mean",
        tau=algo_cfg["tau"],
        initial_temperature=float(abs(algo_cfg.get("target_entropy", 1.0))),
    )
    agent = SAC(config=config_sac, device=device.type, enable_ddp=False)
    # warm start from IQL
    try:
        iql = load_learnable("evaluation/artifacts/iql_policy.d3")
        load_iql_actor_to_sac(iql, agent, logger)
    except Exception as e:
        logger.warning(f"Could not warm-start from IQL: {e}")

    buf = FIFOBuffer(algo_cfg["buffer_size"])
    obs, _ = env.reset()
    steps = int(os.getenv("QA_STEPS", 100000))
    for _ in range(steps):
        action = agent.predict(obs[np.newaxis,...])[0]
        nobs, reward, terminated, truncated, info = env.step(action)
        buf.append(obs, action, reward, terminated)
        obs = nobs
        if len(buf) >= agent.batch_size:
            agent.update(buf, n_steps=1)
        if terminated or truncated:
            obs, _ = env.reset()

    artifacts = Path("evaluation/artifacts"); artifacts.mkdir(parents=True, exist_ok=True)
    agent.save_model(str(artifacts / "sac_policy.d3"))
    logger.info("SAC online fine-tune done; model saved.")

if __name__ == "__main__":
    main()
