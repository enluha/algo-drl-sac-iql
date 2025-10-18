import os, logging
from pathlib import Path
import pandas as pd
from d3rlpy.algos import SAC, SACConfig
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory
from d3rlpy.base import load_learnable
from d3rlpy.dataset import ReplayBuffer
from d3rlpy.dataset.buffers import FIFOBuffer
from src.utils.io_utils import load_yaml
from src.utils.logging_utils import get_logger
from src.utils.device import get_torch_device, log_device, set_num_threads, resolve_compile_flag
from src.features.engineering import build_features
from src.features.normalizer import RollingZScore
from src.envs.market_env import MarketEnv
from src.drl.online.weight_bridge import load_iql_actor_to_sac


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
    config_sac = SACConfig(
        batch_size=algo_cfg["batch_size"],
        gamma=algo_cfg["gamma"],
        actor_learning_rate=algo_cfg["lr_actor"],
        critic_learning_rate=algo_cfg["lr_critic"],
        temp_learning_rate=algo_cfg.get("lr_alpha", algo_cfg["lr_actor"]),
        actor_encoder_factory=encoder,
        critic_encoder_factory=encoder,
        q_func_factory=_resolve_q_func_factory(algo_cfg.get("q_func_factory", "mean")),
        tau=algo_cfg["tau"],
        initial_temperature=float(abs(algo_cfg.get("target_entropy", 1.0))),
        compile_graph=compile_graph,
    )
    agent = SAC(config=config_sac, device=device.type, enable_ddp=False)
    agent.build_with_env(env)
    # warm start from IQL
    try:
        iql = load_learnable("evaluation/artifacts/iql_policy.d3")
        load_iql_actor_to_sac(iql, agent, logger)
    except Exception as e:
        logger.warning(f"Could not warm-start from IQL: {e}")

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
