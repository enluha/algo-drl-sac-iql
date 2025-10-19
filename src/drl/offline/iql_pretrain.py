import os, logging
from pathlib import Path
import numpy as np
import pandas as pd
from d3rlpy.algos import IQL, IQLConfig, BC, BCConfig
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.preprocessing import MinMaxActionScaler
from src.utils.io_utils import load_yaml
from src.utils.logging_utils import get_logger
from src.utils.device import get_torch_device, log_device, set_num_threads, resolve_compile_flag
from src.features.engineering import build_features
from src.features.normalizer import RollingZScore, GlobalZScore
from src.envs.dataset_builder import build_offline_dataset
from src.utils.splits import load_splits

def _load_config() -> dict:
    cfg_path = Path(os.getenv("CONFIG","config/config.yaml"))
    cfg = load_yaml(cfg_path)
    if "data" not in cfg:
        raise KeyError(f"Config hub {cfg_path} missing required sections")
    return cfg

def _build_vector_factory(cfg: dict | None, fallback: tuple[int, ...]) -> VectorEncoderFactory:
    params_source: dict = {}
    if isinstance(cfg, dict):
        params_source = cfg.get("params") if "params" in cfg else cfg
    params = dict(params_source)
    params.pop("type", None)
    hidden = params.pop("hidden_units", None) or params.pop("mlp_hidden", None) or fallback
    hidden_units = tuple(int(x) for x in hidden)
    return VectorEncoderFactory(hidden_units=hidden_units, **params)


def main():
    cfg = _load_config()
    data_cfg, env_cfg, algo_cfg, rt_cfg = cfg["data"], cfg["env"], cfg["algo_iql"], cfg["runtime"]
    logger = get_logger("iql_pretrain", level=rt_cfg.get("log_level","INFO"))
    set_num_threads(rt_cfg.get("blas_threads",6))
    device = get_torch_device(None); log_device(logger)

    df = pd.read_csv(
        data_cfg["csv_path"],
        parse_dates=["date"],
        dayfirst=True,
    ).set_index("date").rename(columns=str.lower).sort_index()
    splits = load_splits(cfg, df.index, data_cfg.get("bar"))
    pre = splits.pretrain

    df_pre = df.loc[pre.start:pre.end]
    if df_pre.empty:
        raise RuntimeError("Pretraining split produced no rows; check walkforward splits.")
    logger.info("Offline pretrain window: %s → %s", df_pre.index.min(), df_pre.index.max())

    feats = build_features(df_pre)
    normalizer_mode = str(rt_cfg.get("normalizer", "global")).lower()
    clip = float(rt_cfg.get("normalizer_clip", 5.0))
    window = int(rt_cfg.get("rolling_window", 500))
    if normalizer_mode == "global":
        norm = GlobalZScore(clip=clip).fit(feats)
    elif normalizer_mode == "rolling":
        norm = RollingZScore(window=window, clip=clip).fit(feats)
    else:
        raise ValueError(f"Unsupported normalizer mode '{normalizer_mode}'")
    feats_n = norm.transform(feats)
    arr = feats_n.to_numpy()
    if not np.isfinite(arr).all():
        bad = arr.size - np.isfinite(arr).sum()
        raise RuntimeError(f"Non-finite features after normalization: {bad} cells")
    assert df_pre.index.max() <= pre.end, "Pretrain slice exceeded pretrain end."

    dset = build_offline_dataset(df_pre[["open","high","low","close"]], feats_n, env_cfg)
    encoder_cfg = algo_cfg.get("encoder", {})
    fallback_hidden = encoder_cfg.get(
        "mlp_hidden", encoder_cfg.get("hidden_units", [256, 256])
    )
    default_hidden = tuple(int(x) for x in fallback_hidden)
    actor_factory = _build_vector_factory(
        algo_cfg.get("actor_encoder_factory"), default_hidden
    )
    critic_factory = _build_vector_factory(
        algo_cfg.get("critic_encoder_factory"), default_hidden
    )
    value_factory = _build_vector_factory(
        algo_cfg.get("value_encoder_factory"), default_hidden
    )
    compile_requested = bool(algo_cfg.get("compile_graph", False))
    compile_graph = resolve_compile_flag(compile_requested, device, logger)
    if compile_requested and not compile_graph:
        logger.warning("compile_graph requested but Triton is missing; running without torch.compile.")
    config_iql = IQLConfig(
        batch_size=algo_cfg["batch_size"],
        gamma=algo_cfg["discount"],
        actor_learning_rate=algo_cfg["lr"],
        critic_learning_rate=algo_cfg["lr"],
        expectile=algo_cfg["expectile_beta"],
        weight_temp=algo_cfg["temperature"],
        actor_encoder_factory=actor_factory,
        critic_encoder_factory=critic_factory,
        value_encoder_factory=value_factory,
        action_scaler=MinMaxActionScaler(minimum=-1.0, maximum=1.0),
        compile_graph=compile_graph,
    )
    agent = IQL(config=config_iql, device=device.type, enable_ddp=False)
    out = Path("evaluation/artifacts"); out.mkdir(parents=True, exist_ok=True)
    steps = int(os.getenv("QA_STEPS", algo_cfg["grad_steps"]))
    logger.info("Starting IQL offline pretrain for %s gradient steps (show_progress enabled).", steps)
    try:
        agent.fit(dset, n_steps=steps, save_interval=int(1e9), show_progress=True)
        agent.save_model(str(out / "iql_policy.d3"))
        # --- save both d3rlpy artifact (for inspection) and raw actor state_dict (for warm-start) ---
        try:
            import torch
            actor_sd = agent.impl.policy.state_dict()
            torch.save(actor_sd, out / "iql_actor_state.pt")
            logger.info("Saved IQL actor state to evaluation/artifacts/iql_actor_state.pt")
        except Exception as e:
            logger.warning("Could not export IQL actor state_dict: %s", e)
    except Exception as err:
        logger.exception("IQL fit failed; falling back to Behavior Cloning pretrain.")
        bc_cfg = BCConfig(
            learning_rate=algo_cfg["lr"],
            batch_size=algo_cfg["batch_size"],
            encoder_factory=actor_factory,
            compile_graph=compile_graph,
        )
        bc = BC(config=bc_cfg, device=device.type, enable_ddp=False)
        bc_steps = int(os.getenv("QA_STEPS", 300000))
        logger.info("Starting BC fallback pretrain for %s steps (show_progress enabled).", bc_steps)
        bc.fit(dset, n_steps=bc_steps, save_interval=int(1e9), show_progress=True)
        bc.save_model(str(out / "iql_policy.d3"))
        logger.info("BC pretrain saved; continuing.")
    norm.save(out / "pretrain_normalizer.pkl")
    logger.info("Offline IQL pretrain complete; artifacts saved.")

if __name__ == "__main__":
    main()
