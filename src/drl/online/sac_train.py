import os, logging
from pathlib import Path
import numpy as np
import pandas as pd
from d3rlpy.algos import SAC, SACConfig
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory
from d3rlpy.preprocessing import MinMaxActionScaler
from d3rlpy.dataset import ReplayBuffer
from d3rlpy.dataset.buffers import FIFOBuffer
from src.utils.io_utils import load_yaml
from src.utils.logging_utils import get_logger
from src.utils.device import get_torch_device, log_device, set_num_threads, resolve_compile_flag
from src.features.engineering import build_features
from src.features.normalizer import RollingZScore
from src.envs.market_env import MarketEnv


def _build_vector_factory(cfg: dict | None, fallback: tuple[int, ...]) -> VectorEncoderFactory:
    params_source: dict = {}
    if isinstance(cfg, dict):
        params_source = cfg.get("params") if "params" in cfg else cfg
    params = dict(params_source)
    params.pop("type", None)
    hidden = params.pop("hidden_units", None) or params.pop("mlp_hidden", None) or fallback
    hidden_units = tuple(int(x) for x in hidden)
    return VectorEncoderFactory(hidden_units=hidden_units, **params)


def _resolve_q_func_factory(value) -> MeanQFunctionFactory:
    if isinstance(value, dict):
        factory_type = value.get("type", "mean").lower()
    else:
        factory_type = str(value or "mean").lower()
    if factory_type != "mean":
        raise ValueError(f"Unsupported q_func_factory '{factory_type}' for quick run.")
    return MeanQFunctionFactory()

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
    arr = feats_n.to_numpy(dtype=np.float32)
    if not np.isfinite(arr).all():
        bad = int(arr.size - np.isfinite(arr).sum())
        raise RuntimeError(f"Non-finite features after normalization: {bad} cells")

    env = MarketEnv(df[["open","high","low","close"]], feats_n, env_cfg)

    encoder_cfg = algo_cfg.get("encoder", {})
    fallback_hidden = tuple(
        int(x) for x in encoder_cfg.get("mlp_hidden", encoder_cfg.get("hidden_units", [256, 256]))
    )
    encoder = _build_vector_factory(encoder_cfg, fallback_hidden)
    compile_requested = bool(algo_cfg.get("compile_graph", False))
    compile_graph = resolve_compile_flag(compile_requested, device, logger)
    if compile_requested and not compile_graph:
        logger.warning("compile_graph requested but Triton is missing; running without torch.compile.")
    action_scaler = MinMaxActionScaler(minimum=-1.0, maximum=1.0)
    alpha_lr = float(algo_cfg.get("alpha_learning_rate", algo_cfg.get("lr_alpha", algo_cfg["lr_actor"])))
    init_temperature = float(algo_cfg.get("initial_temperature", 0.1))
    target_entropy = float(algo_cfg.get("target_entropy", -1.0))
    config_sac = SACConfig(
        batch_size=algo_cfg["batch_size"],
        gamma=algo_cfg["gamma"],
        action_scaler=action_scaler,
        actor_learning_rate=algo_cfg["lr_actor"],
        critic_learning_rate=algo_cfg["lr_critic"],
        temp_learning_rate=alpha_lr,
        actor_encoder_factory=encoder,
        critic_encoder_factory=encoder,
        q_func_factory=_resolve_q_func_factory(algo_cfg.get("q_func_factory", "mean")),
        tau=algo_cfg["tau"],
        initial_temperature=init_temperature,
        compile_graph=compile_graph,
    )
    agent = SAC(config=config_sac, device=device.type, enable_ddp=False)
    agent.build_with_env(env)
    if hasattr(agent.impl, "target_entropy"):
        agent.impl.target_entropy = target_entropy
    elif hasattr(agent.impl, "_target_entropy"):
        agent.impl._target_entropy = target_entropy

    # Warm-start SAC actor from IQL actor if present (raw state_dict)
    actor_sd_path = Path("evaluation/artifacts/iql_actor_state.pt")
    if actor_sd_path.exists():
        try:
            import torch
            state_dict = torch.load(actor_sd_path, map_location="cpu")
            agent.impl.policy.load_state_dict(state_dict, strict=False)
            logger.info("Warm-started SAC actor from IQL actor_state.pt")
            try:
                obs_preview, _ = env.reset()
                if not np.isfinite(obs_preview).all():
                    obs_preview = np.nan_to_num(obs_preview, nan=0.0, posinf=0.0, neginf=0.0)
                sample_action = agent.predict(obs_preview.reshape(1, -1))[0]
                logger.debug("Sample action post warm-start: %.4f", float(sample_action))
            except Exception:
                pass
        except Exception as e:
            logger.warning("Warm-start failed (actor_state.pt): %s", e)
    else:
        logger.warning("No iql_actor_state.pt found; SAC starts from scratch.")

    env.reset()

    steps = int(os.getenv("QA_STEPS", 100000))
    buffer_limit = int(algo_cfg.get("buffer_size", 200000))
    cache_size = max(buffer_limit, len(df))
    replay_buffer = ReplayBuffer(
        FIFOBuffer(buffer_limit),
        env=env,
        cache_size=cache_size,
        write_at_termination=True,
    )

    agent.fit_online(
        env,
        buffer=replay_buffer,
        n_steps=steps,
        n_steps_per_epoch=max(steps, 1),
        n_updates=int(algo_cfg.get("updates_per_step", 1)),
        update_interval=1,
        save_interval=int(1e9),
        logging_steps=max(steps // 5, 1),
        show_progress=False,
    )

    artifacts = Path("evaluation/artifacts"); artifacts.mkdir(parents=True, exist_ok=True)
    agent.save_model(str(artifacts / "sac_policy.d3"))
    logger.info("SAC online fine-tune done; model saved.")

if __name__ == "__main__":
    main()
