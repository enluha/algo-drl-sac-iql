import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
from d3rlpy.algos import SAC, SACConfig
from d3rlpy.preprocessing import MinMaxActionScaler

from src.utils.io_utils import load_yaml, save_csv, save_json
from src.utils.logging_utils import get_logger
from src.utils.device import get_torch_device, log_device, set_num_threads, resolve_compile_flag
from src.features.engineering import build_features
from src.features.normalizer import RollingZScore
from src.envs.market_env import MarketEnv
from src.evaluation.metrics import summarize
from src.evaluation.plots import candlestick_html, equity_html
from src.evaluation.reporter import build_text_report
from src.drl.online.sac_train import _build_vector_factory, _resolve_q_func_factory
from src.utils.splits import load_splits, window_for_test_with_warmup


def _load_config() -> dict:
    cfg_path = Path(os.getenv("CONFIG", "config/config.yaml"))
    cfg = load_yaml(cfg_path)
    if "data" not in cfg:
        raise KeyError(f"Config hub {cfg_path} missing required sections")
    return cfg


def _build_agent(cfg: dict, env, device, logger):
    algo_cfg = cfg["algo_sac"]
    encoder_cfg = algo_cfg.get("encoder", {})
    fallback_hidden = tuple(int(x) for x in encoder_cfg.get("mlp_hidden", encoder_cfg.get("hidden_units", [256, 256])))
    encoder = _build_vector_factory(encoder_cfg, fallback_hidden)
    action_scaler = MinMaxActionScaler(minimum=-1.0, maximum=1.0)

    compile_requested = bool(algo_cfg.get("compile_graph", False))
    compile_graph = resolve_compile_flag(compile_requested, device, logger)
    if compile_requested and not compile_graph:
        logger.warning("compile_graph requested but Triton is missing; running without torch.compile.")

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
    return agent


def run():
    cfg = _load_config()
    data_cfg, env_cfg, rt_cfg = cfg["data"], cfg["env"], cfg["runtime"]
    logger = get_logger("walkforward", level=rt_cfg.get("log_level", "INFO"))
    set_num_threads(rt_cfg.get("blas_threads", 6))
    device = get_torch_device(None)
    log_device(logger)

    df = pd.read_csv(
        data_cfg["csv_path"],
        parse_dates=["date"],
        dayfirst=True,
    ).set_index("date").rename(columns=str.lower).sort_index()

    splits = load_splits(cfg, df.index, data_cfg.get("bar"))
    tw_start, tw_end = window_for_test_with_warmup(splits, df.index)

    df_test_ctx = df.loc[tw_start:tw_end]
    if df_test_ctx.empty:
        raise RuntimeError("Test window produced no data; check walkforward splits.")
    logger.info("Evaluation window (with warm-up): %s -> %s", df_test_ctx.index.min(), df_test_ctx.index.max())

    feats_ctx = build_features(df_test_ctx)
    norm_path = Path("evaluation/artifacts/finetune_normalizer.pkl")
    if not norm_path.exists():
        raise FileNotFoundError("Missing finetune_normalizer.pkl. Run SAC fine-tune before evaluation.")
    norm_ft = RollingZScore.load(norm_path)
    feats_ctx_n = norm_ft.transform(feats_ctx)

    env = MarketEnv(df_test_ctx[["open", "high", "low", "close"]], feats_ctx_n, env_cfg)

    agent = _build_agent(cfg, env, device, logger)
    assert hasattr(agent, "predict"), "Evaluation agent must support inference."
    if hasattr(agent, "fit"):
        def _forbid_fit(*args, **kwargs):
            raise RuntimeError("fit() must not be called during evaluation.")
        agent.fit = _forbid_fit  # type: ignore[attr-defined]
    if hasattr(agent, "fit_online"):
        def _forbid_fit_online(*args, **kwargs):
            raise RuntimeError("fit_online() must not be called during evaluation.")
        agent.fit_online = _forbid_fit_online  # type: ignore[attr-defined]

    actor_sd_path = Path("evaluation/artifacts/sac_actor_state.pt")
    if not actor_sd_path.exists():
        raise FileNotFoundError("Missing sac_actor_state.pt. Run SAC fine-tune before evaluation.")
    import torch

    state_dict = torch.load(actor_sd_path, map_location="cpu")
    agent.impl.policy.load_state_dict(state_dict, strict=False)
    logger.info("Loaded SAC actor state for evaluation.")

    obs, _ = env.reset()

    # Warm-up with flat position until the next bar to trade is in the test window
    zero_action = np.zeros(1, dtype=np.float32)
    while env.t < len(env.prices) and env.prices.index[env.t] < splits.test.start:
        obs, _, done, _, _ = env.step(zero_action)
        if done:
            raise RuntimeError("Ran out of data during warm-up before reaching test start.")

    ledger_records: list[dict] = []
    signals_map: dict[pd.Timestamp, float] = {}
    preds_map: dict[pd.Timestamp, float] = {}

    while env.t < len(env.prices):
        action_value = agent.predict(obs.reshape(1, -1))[0]
        action_arr = np.array([float(action_value)], dtype=np.float32)
        obs, reward, done, truncated, info = env.step(action_arr)
        ts = pd.Timestamp(info["timestamp"])

        if ts >= splits.test.start:
            ledger_records.append(
                {
                    "timestamp": ts,
                    "raw": info["raw_ret"],
                    "cost": info["cost"],
                    "net": info["net_ret"],
                    "weight": info["weight"],
                    "turnover": info.get("turnover", abs(info["weight"] - action_arr[0])),
                    "equity": info["equity"],
                    "price": info["price"],
                    "drawdown": info["drawdown"],
                }
            )
            signals_map[ts] = info["weight"]
            preds_map[ts] = float(action_arr[0])

        if done or truncated or ts >= splits.test.end:
            break

    if not ledger_records:
        raise RuntimeError("No trades captured during evaluation window.")

    ledger_df = pd.DataFrame(ledger_records).set_index("timestamp").sort_index()
    signals_df = pd.Series(signals_map, name="weight").sort_index()
    predictions_df = pd.Series(preds_map, name="prediction").sort_index().to_frame()

    equity = (1.0 + ledger_df["net"].fillna(0.0)).cumprod().rename("equity")
    price = df_test_ctx["close"].reindex(equity.index).ffill()
    buy_hold = (price / price.iloc[0]).rename(f"buy_and_hold_{data_cfg['symbol']}")
    equity_df = equity.to_frame().join(buy_hold, how="left")

    # Rebased window for zoomed charts
    plot_start, plot_end = splits.test.start, splits.test.end
    equity_window = equity.loc[plot_start:plot_end]
    if not equity_window.empty:
        equity_rebased = (equity_window / equity_window.iloc[0]).rename("equity_rebased")
    else:
        equity_rebased = pd.Series(dtype="float64", name="equity_rebased")
    buy_hold_window = buy_hold.loc[plot_start:plot_end]
    if not buy_hold_window.empty:
        buy_hold_rebased = (buy_hold_window / buy_hold_window.iloc[0]).rename(buy_hold.name)
    else:
        buy_hold_rebased = buy_hold_window

    rep = Path("evaluation/reports")
    ch = Path("evaluation/charts")
    rep.mkdir(parents=True, exist_ok=True)
    ch.mkdir(parents=True, exist_ok=True)

    symbol = data_cfg["symbol"]
    save_csv(ledger_df, rep / f"trades_{symbol}.csv")
    save_csv(equity_df, rep / f"equity_curve_{symbol}.csv")
    save_csv(signals_df.to_frame(), rep / f"signals_{symbol}.csv")
    save_csv(predictions_df, rep / f"predictions_{symbol}.csv")

    candlestick_html(df_test_ctx.loc[plot_start:plot_end][["open", "high", "low", "close"]], signals_df, ch / f"candlestick_{symbol}_SepOct2025.html")
    equity_html(equity_df["equity"], {buy_hold.name: buy_hold}, ch / f"equity_{symbol}.html")
    if not equity_rebased.empty:
        equity_html(equity_rebased, {buy_hold_rebased.name: buy_hold_rebased}, ch / "8.html")
    else:
        equity_html(equity_df["equity"], {buy_hold.name: buy_hold}, ch / "8.html")

    summary = summarize(ledger_df, equity_df["equity"])
    summary["equity_end_total"] = float(equity.iloc[-1]) if not equity.empty else float("nan")
    summary["window_return_rebased"] = float(equity_rebased.iloc[-1]) if not equity_rebased.empty else float("nan")

    context = {
        "overview": f"SAC evaluation | {symbol}",
        "params": {
            "pretrain": f"{splits.pretrain.start.date()} -> {splits.pretrain.end.date()}",
            "finetune": f"{splits.finetune.start.date()} -> {splits.finetune.end.date()}",
            "test": f"{splits.test.start.date()} -> {splits.test.end.date()}",
            "warmup_bars": splits.warmup_bars,
        },
    }
    build_text_report(summary, rep / f"summary_report_{symbol}.txt", context=context)
    build_text_report(summary, rep / "summary_report.txt", context=context)
    save_json({"records": len(ledger_df), "test_start": str(splits.test.start), "test_end": str(splits.test.end)}, rep / f"walkforward_meta_{symbol}.json")


def main():
    run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.environ["CONFIG"] = args.config
    if args.device:
        os.environ["QA_DEVICE"] = args.device
    from src.utils.seed import seed_everything
    seed_everything(args.seed)
    run()
