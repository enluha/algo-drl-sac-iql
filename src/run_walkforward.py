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

def _load_config() -> dict:
    cfg_path = Path(os.getenv("CONFIG","config/config.yaml"))
    cfg = load_yaml(cfg_path)
    if "data" not in cfg:
        raise KeyError(f"Config hub {cfg_path} missing required sections")
    return cfg

def _build_slices(timestamps: pd.Index, wf_cfg: dict) -> list[dict]:
    td = pd.to_timedelta
    train = td(f"{wf_cfg['train_days']}D")
    valid = td(f"{wf_cfg['valid_days']}D")
    test = td(f"{wf_cfg['test_days']}D")
    step = td(f"{wf_cfg['step_days']}D")
    start = timestamps.min()
    end = timestamps.max()
    folds = []
    cursor = start
    while True:
        train_start = cursor
        train_end = train_start + train
        valid_end = train_end + valid
        test_end = valid_end + test
        if test_end > end:
            break
        folds.append({
            "train": (train_start, train_end),
            "valid": (train_end, valid_end),
            "test": (valid_end, test_end)
        })
        cursor += step
    return folds

def run():
    cfg = _load_config()
    data_cfg, env_cfg, algo_cfg, wf_cfg, rt_cfg = (
        cfg["data"],
        cfg["env"],
        cfg["algo_sac"],
        cfg["walkforward"],
        cfg["runtime"],
    )
    logger = get_logger("walkforward", level=rt_cfg.get("log_level","INFO"))
    set_num_threads(rt_cfg.get("blas_threads",6))
    device = get_torch_device(None); log_device(logger)

    df = pd.read_csv(
        data_cfg["csv_path"],
        parse_dates=["date"],
        dayfirst=True,
    ).set_index("date").rename(columns=str.lower)
    feats_all = build_features(df)

    encoder_cfg = algo_cfg.get("encoder", {})
    fallback_hidden = tuple(
        int(x) for x in encoder_cfg.get("mlp_hidden", encoder_cfg.get("hidden_units", [256, 256]))
    )
    encoder = _build_vector_factory(encoder_cfg, fallback_hidden)
    action_scaler = MinMaxActionScaler(minimum=-1.0, maximum=1.0)
    alpha_lr = float(algo_cfg.get("alpha_learning_rate", algo_cfg.get("lr_alpha", algo_cfg["lr_actor"])))
    init_temperature = float(algo_cfg.get("initial_temperature", 0.1))
    target_entropy = float(algo_cfg.get("target_entropy", -1.0))
    compile_requested = bool(algo_cfg.get("compile_graph", False))
    compile_graph = resolve_compile_flag(compile_requested, device, logger)
    if compile_requested and not compile_graph:
        logger.warning("compile_graph requested but Triton is missing; running without torch.compile.")
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
    actor_sd_path = Path("evaluation/artifacts/sac_actor_state.pt")
    try:
        import torch
        actor_state = torch.load(actor_sd_path, map_location="cpu")
    except FileNotFoundError as err:
        raise FileNotFoundError(
            "Missing SAC actor weights. Expected evaluation/artifacts/sac_actor_state.pt. "
            "Run SAC fine-tune before walk-forward evaluation."
        ) from err
    except Exception as err:
        raise RuntimeError(f"Failed to load SAC actor state_dict: {err}") from err
    weights_loaded = False

    folds = _build_slices(df.index, wf_cfg)
    if not folds:
        raise RuntimeError("Walk-forward config produced no folds; check date range and durations.")

    ledger_records: list[dict] = []
    signals_map: dict[pd.Timestamp, float] = {}
    preds_map: dict[pd.Timestamp, float] = {}

    for idx, fold in enumerate(folds):
        train_start, train_end = fold["train"]
        test_start, test_end = fold["test"]
        logger.info("Fold %s train %s→%s | test %s→%s", idx, train_start, train_end, test_start, test_end)

        window_prices = df.loc[train_start:test_end].copy()
        window_feats = feats_all.loc[window_prices.index]

        norm = RollingZScore(window=wf_cfg.get("normalizer_window", env_cfg["window_bars"]))
        norm.fit(window_feats.loc[train_start:train_end])
        feats_norm = norm.transform(window_feats)

        env = MarketEnv(window_prices[["open","high","low","close"]], feats_norm, env_cfg)
        obs, _ = env.reset()
        prev_weight = 0.0
        if not weights_loaded:
            agent.build_with_env(env)
            if hasattr(agent.impl, "target_entropy"):
                agent.impl.target_entropy = target_entropy
            elif hasattr(agent.impl, "_target_entropy"):
                agent.impl._target_entropy = target_entropy
            agent.impl.policy.load_state_dict(actor_state, strict=False)
            weights_loaded = True
        while True:
            action = agent.predict(obs[np.newaxis,...])[0]
            obs, reward, done, truncated, info = env.step(action)
            ts = pd.Timestamp(info["timestamp"])
            weight = float(info["weight"])
            turn = abs(weight - prev_weight)
            prev_weight = weight

            if ts >= test_start:
                ledger_records.append({
                    "timestamp": ts,
                    "fold": idx,
                    "raw": info["raw_ret"],
                    "cost": info["cost"],
                    "net": info["net_ret"],
                    "weight": weight,
                    "turnover": turn,
                    "equity": info["equity"],
                    "price": info["price"],
                })
                signals_map[ts] = weight
                preds_map[ts] = float(action[0])

            if done or truncated or ts >= test_end:
                break

    if not ledger_records:
        raise RuntimeError("No walk-forward test trades recorded.")

    ledger_all = pd.DataFrame(ledger_records).set_index("timestamp").sort_index()
    signals_df = pd.Series(signals_map, name="weight").sort_index()
    predictions_df = pd.Series(preds_map, name="prediction").sort_index().to_frame()

    ohlc = df[["open","high","low","close"]]
    symbol = data_cfg["symbol"]

    equity = (1.0 + ledger_all["net"].fillna(0.0)).cumprod().rename("equity")
    price = ohlc["close"].reindex(equity.index).ffill()
    buy_hold = (price/price.iloc[0]).rename(f"buy_and_hold_{symbol.split(':')[-1]}")
    equity_df = equity.to_frame().join(buy_hold, how="left")

    rep = Path("evaluation/reports"); ch = Path("evaluation/charts")
    rep.mkdir(parents=True, exist_ok=True); ch.mkdir(parents=True, exist_ok=True)

    save_csv(ledger_all, rep / f"trades_{symbol}.csv")
    save_csv(equity_df, rep / f"equity_curve_{symbol}.csv")
    save_csv(signals_df.to_frame(), rep / f"signals_{symbol}.csv")
    save_csv(predictions_df, rep / f"predictions_{symbol}.csv")

    candlestick_html(ohlc.loc["2025-09-01":"2025-10-16"], signals_df, ch / f"candlestick_{symbol}_SepOct2025.html")
    equity_html(equity_df["equity"], {buy_hold.name: buy_hold}, ch / f"equity_{symbol}.html")
    equity_html(equity_df["equity"], {buy_hold.name: buy_hold}, ch / "8.html")

    summary = summarize(ledger_all, equity_df["equity"])
    build_text_report(summary, rep / f"summary_report_{symbol}.txt", context={
        "overview": f"SAC walk-forward | {symbol}",
        "params": {
            "train_days": wf_cfg["train_days"],
            "valid_days": wf_cfg["valid_days"],
            "test_days": wf_cfg["test_days"],
            "step_days": wf_cfg["step_days"],
        },
    })
    build_text_report(summary, rep / "summary_report.txt", context={
        "overview": f"SAC walk-forward | {symbol}",
        "params": {
            "train_days": wf_cfg["train_days"],
            "valid_days": wf_cfg["valid_days"],
            "test_days": wf_cfg["test_days"],
            "step_days": wf_cfg["step_days"],
        },
    })

    save_json({"folds": len(folds), "records": len(ledger_all)}, rep / f"walkforward_meta_{symbol}.json")

def main():
    run()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--device", choices=["cpu","cuda"], default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.environ["CONFIG"] = args.config
    if args.device: os.environ["QA_DEVICE"] = args.device
    from src.utils.seed import seed_everything
    seed_everything(args.seed)
    run()
