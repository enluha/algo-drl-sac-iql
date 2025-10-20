import os, logging
from pathlib import Path
import numpy as np
import pandas as pd
from d3rlpy.algos import IQL, IQLConfig, BC, BCConfig
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.preprocessing import MinMaxActionScaler
from src.utils.io_utils import load_yaml, save_csv, save_json
from src.utils.logging_utils import get_logger
from src.utils.device import get_torch_device, log_device, set_num_threads, resolve_compile_flag
from src.features.engineering import build_features
from src.drl.offline.iql_auxiliary import IQLWithAuxiliary
from src.models.auxiliary_encoder import AuxiliaryEncoderFactory
from src.envs.market_env import MarketEnv
from src.evaluation.metrics import summarize
from src.evaluation.plots import candlestick_html, equity_html
from src.evaluation.reporter import build_text_report
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


def _evaluate_iql_only(agent, cfg: dict, df: pd.DataFrame, norm, logger, symbol: str = "BTCUSDT"):
    """
    Evaluate IQL policy in isolation (before SAC fine-tuning).
    Generates: equity chart, candlestick chart, and summary report with '_IQLonly' suffix.
    """
    logger.info("=" * 80)
    logger.info("EVALUATING IQL POLICY IN ISOLATION (before SAC fine-tuning)")
    logger.info("=" * 80)
    
    from src.utils.splits import load_splits, window_for_test_with_warmup
    
    env_cfg = cfg["env"]
    data_cfg = cfg["data"]
    splits = load_splits(cfg, df.index, data_cfg.get("bar"))
    
    # Get test period with warmup
    test_start, test_end = window_for_test_with_warmup(splits, df.index)
    test_df = df.loc[test_start:test_end]
    logger.info(f"IQL evaluation window: {test_df.index.min()} → {test_df.index.max()}")
    
    # Build features and normalize
    feats = build_features(test_df)
    feats_n = norm.transform(feats)
    
    # Create environment (MarketEnv takes cfg dict with window_bars inside)
    env = MarketEnv(
        prices=test_df[["open", "high", "low", "close"]],
        features=feats_n,
        cfg=env_cfg
    )
    
    # Run episode with IQL policy
    logger.info("Running IQL policy on test period...")
    obs, _ = env.reset()
    done = False
    ledger = []
    
    while not done:
        # Use same prediction format as run_walkforward.py
        action_value = agent.predict(obs.reshape(1, -1))[0]
        action_arr = np.array([float(action_value)], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action_arr)
        done = terminated or truncated
        
        # Store ledger info with proper timestamp handling
        ledger_record = {
            "date": pd.Timestamp(info["timestamp"]),
            "raw": info["raw_ret"],
            "cost": info["cost"],
            "net": info["net_ret"],
            "weight": info["weight"],
            "turnover": info.get("turnover", abs(info["weight"] - action_arr[0])),
            "equity": info["equity"],
            "price": info["price"],
            "drawdown": info["drawdown"],
        }
        ledger.append(ledger_record)
    
    ledger_df = pd.DataFrame(ledger).set_index("date")
    logger.info(f"IQL episode complete: {len(ledger_df)} steps")
    
    # Compute metrics
    equity_series = ledger_df["equity"]
    summary = summarize(ledger_df, equity_series)
    
    # Save outputs with '_IQLonly' suffix
    out_dir = Path("evaluation")
    reports_dir = out_dir / "reports"
    charts_dir = out_dir / "charts"
    reports_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    # Save equity curve
    equity_df = ledger_df[["equity"]].copy()
    save_csv(equity_df, reports_dir / f"equity_curve_{symbol}_IQLonly.csv")
    
    # Generate charts
    equity_html(
        equity_series,
        benchmarks={},  # No benchmarks for IQL-only evaluation
        path_html=charts_dir / f"equity_{symbol}_IQLonly.html"
    )
    logger.info(f"Saved: evaluation/charts/equity_{symbol}_IQLonly.html")
    
    candlestick_html(
        test_df,
        ledger_df["weight"],  # Use weight as signals
        path_html=charts_dir / f"candlestick_{symbol}_SepOct2025_IQLonly.html"
    )
    logger.info(f"Saved: evaluation/charts/candlestick_{symbol}_SepOct2025_IQLonly.html")
    
    # Generate summary report
    context = {
        "model": "IQL (offline pretrain only)",
        "symbol": symbol,
        "test_period": f"{splits.test.start.date()} to {splits.test.end.date()}",
        "note": "Evaluation BEFORE SAC fine-tuning - pure offline RL performance"
    }
    build_text_report(summary, reports_dir / f"summary_report_{symbol}_IQLonly.txt", context=context)
    logger.info(f"Saved: evaluation/reports/summary_report_{symbol}_IQLonly.txt")
    
    logger.info("=" * 80)
    logger.info("IQL-ONLY EVALUATION COMPLETE")
    logger.info(f"  Sharpe: {summary.get('sharpe', 0):.2f}")
    logger.info(f"  Total Return: {summary.get('total_return_pct', 0):.2f}%")
    logger.info(f"  Max Drawdown: {summary.get('max_drawdown_pct', 0):.2f}%")
    logger.info("=" * 80)


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

    # Auto-generate feature overview visualization (after features are ready)
    try:
        from scripts.plot_feature_overview import generate_feature_overview
        output_html = Path("data/feature_overview.html")
        csv_path = Path(data_cfg["csv_path"])
        generate_feature_overview(output_html, csv_path)
        logger.info(f"Feature overview saved to {output_html}")
    except Exception as e:
        logger.warning(f"Failed to generate feature_overview.html: {e}")

    dset = build_offline_dataset(df_pre[["open","high","low","close"]], feats_n, env_cfg)
    encoder_cfg = algo_cfg.get("encoder", {})
    fallback_hidden = encoder_cfg.get(
        "mlp_hidden", encoder_cfg.get("hidden_units", [256, 256])
    )
    default_hidden = tuple(int(x) for x in fallback_hidden)
    
    # Check if auxiliary task is enabled
    use_aux = algo_cfg.get("use_auxiliary_task", False)
    
    if use_aux:
        # Use auxiliary encoder with price prediction head
        logger.info("Auxiliary task ENABLED - using multi-task encoder with price direction prediction")
        actor_factory = AuxiliaryEncoderFactory(
            feature_size=256,  # Keep same as hidden_units[-1]
            hidden_units=default_hidden,
            dropout_rate=0.1
        )
        critic_factory = AuxiliaryEncoderFactory(
            feature_size=256,
            hidden_units=default_hidden,
            dropout_rate=0.1
        )
        value_factory = AuxiliaryEncoderFactory(
            feature_size=256,
            hidden_units=default_hidden,
            dropout_rate=0.1
        )
    else:
        # Standard vector encoder (no auxiliary task)
        logger.info("Auxiliary task DISABLED - using standard vector encoder")
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
    
    # Create IQL agent with optional auxiliary task
    if use_aux:
        aux_weight = float(algo_cfg.get("aux_loss_weight", 0.1))
        logger.info(f"Creating IQLWithAuxiliary (aux_loss_weight={aux_weight})")
        agent = IQLWithAuxiliary(config=config_iql, device=device.type, enable_ddp=False, aux_loss_weight=aux_weight)
    else:
        logger.info("Creating standard IQL")
        agent = IQL(config=config_iql, device=device.type, enable_ddp=False)
    out = Path("evaluation/artifacts"); out.mkdir(parents=True, exist_ok=True)
    
    # Use grad_steps_IQL from config, or QA_STEPS override if set (for quick testing)
    default_steps = int(algo_cfg.get("grad_steps_IQL", algo_cfg.get("grad_steps", 1000)))
    steps = int(os.getenv("QA_STEPS")) if os.getenv("QA_STEPS") else default_steps
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
        
        # ═══════════════════════════════════════════════════════════════════════
        # EVALUATE IQL IN ISOLATION (before SAC fine-tuning)
        # ═══════════════════════════════════════════════════════════════════════
        try:
            symbol = data_cfg.get("symbol", "BTCUSDT")
            _evaluate_iql_only(agent, cfg, df, norm, logger, symbol)
        except Exception as eval_err:
            logger.error(f"IQL-only evaluation failed: {eval_err}")
            import traceback
            traceback.print_exc()
            logger.info("Continuing to save normalizer...")
        
    except Exception as err:
        logger.exception("IQL fit failed; falling back to Behavior Cloning pretrain.")
        bc_cfg = BCConfig(
            learning_rate=algo_cfg["lr"],
            batch_size=algo_cfg["batch_size"],
            encoder_factory=actor_factory,
            compile_graph=compile_graph,
        )
        bc = BC(config=bc_cfg, device=device.type, enable_ddp=False)
        # Use QA_STEPS if set, otherwise use 300000 for BC fallback
        bc_steps = int(os.getenv("QA_STEPS")) if os.getenv("QA_STEPS") else 300000
        logger.info("Starting BC fallback pretrain for %s steps (show_progress enabled).", bc_steps)
        bc.fit(dset, n_steps=bc_steps, save_interval=int(1e9), show_progress=True)
        bc.save_model(str(out / "iql_policy.d3"))
        logger.info("BC pretrain saved; continuing.")
    norm.save(out / "pretrain_normalizer.pkl")
    logger.info("Offline IQL pretrain complete; artifacts saved.")

if __name__ == "__main__":
    main()
