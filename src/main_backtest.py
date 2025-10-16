"""
Enhanced backtest runner with summary report generation.

Outputs
-------
evaluation/reports/
  trades_{symbol}.csv
  equity_curve_{symbol}.csv
  signals_{symbol}.csv
  predictions_{symbol}.csv
  summary_report_{symbol}.txt
evaluation/artifacts/
  summary_{symbol}.json
  threshold_diagnostics_{symbol}.json
evaluation/charts/
  candlestick_{symbol}_SepOct2025.html
  equity_{symbol}.html
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .utils.io_utils import load_yaml, load_pickle, save_csv, save_json
from .utils.logging_utils import get_logger
from .utils.parallel import run_parallel
from .utils.symbols import parse_symbol

from .data.loaders import load_all
from .data.preprocessing import clean_and_resample
from .features.engineering import assemble_features
from .labels.targets import k_ahead_log_return, make_labels, ewma_vol
from .labels.triple_barrier import triple_barrier_labels

from .backtest.thresholds import pick_threshold
from .backtest.simulator import simulate
from .backtest.metrics import summarize, classification_report

from .evaluation.plots import qf, plot_equity_with_vectorbt, reliability_plot


# --------------------------------------------------------------------------- #
# Configuration helpers
# --------------------------------------------------------------------------- #

def _load_config_with_includes(config_path: str | Path) -> Dict[str, Any]:
    cfg = load_yaml(config_path)
    base_dir = Path(config_path).parent
    if isinstance(cfg, dict) and "include" in cfg:
        merged: Dict[str, Any] = {}
        for rel in cfg["include"]:
            part = load_yaml(base_dir / rel)
            key = Path(rel).stem
            merged[key] = part
        return merged
    return cfg


def _parse_ts(ts: str | None) -> pd.Timestamp | None:
    return None if ts is None else pd.to_datetime(ts)


def _prepare_window_xy(
    X_df: pd.DataFrame,
    y: pd.Series,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    if start is None and end is None:
        Xw = X_df
        yw = y
    elif start is None:
        Xw = X_df.loc[:end]
        yw = y.loc[:end]
    elif end is None:
        Xw = X_df.loc[start:]
        yw = y.loc[start:]
    else:
        Xw = X_df.loc[start:end]
        yw = y.loc[start:end]

    aligned = pd.concat([Xw, yw.rename("y")], axis=1).dropna()
    if X_df.index.tz is not None and aligned.index.tz is None:
        aligned.index = aligned.index.tz_localize(X_df.index.tz)
    elif X_df.index.tz is not None and aligned.index.tz is not None and str(aligned.index.tz) != str(X_df.index.tz):
        aligned.index = aligned.index.tz_convert(X_df.index.tz)

    Xw = aligned[X_df.columns]
    yw = aligned["y"]
    return Xw.to_numpy(dtype=np.float64), yw.to_numpy(), aligned.index


def _pipe_predict(pipe, X: np.ndarray) -> np.ndarray:
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)
    return pipe.predict(X)


def _score_series(yhat: np.ndarray) -> np.ndarray:
    if yhat.ndim == 2 and yhat.shape[1] >= 2:
        return yhat[:, 1]
    return yhat


def _ewma_vol_numpy(ret: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(ret).ewm(span=span, adjust=False).std().to_numpy()


def _collect_artifacts(art_dir: Path) -> List[Tuple[int, Path, Dict[str, Any]]]:
    items: List[Tuple[int, Path, Dict[str, Any]]] = []
    for p in sorted(art_dir.glob("pipe_fold_*.pkl")):
        fold_id = int(p.stem.split("_")[-1])
        meta_path = art_dir / f"meta_fold_{fold_id}.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        items.append((fold_id, p, meta))
    items.sort(key=lambda t: t[0])
    return items


# --------------------------------------------------------------------------- #
# Calibration, trend gating, hysteresis, and threshold selection
# --------------------------------------------------------------------------- #

def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _make_platt_calibrator(p_raw: np.ndarray, y_bin: np.ndarray):
    """Fit Platt scaling on validation; returns p_calibrated(p_raw)."""
    from sklearn.linear_model import LogisticRegression

    yb = y_bin.astype(int).ravel()
    if yb.sum() == 0 or yb.sum() == len(yb):
        return lambda p: np.clip(p, 1e-6, 1 - 1e-6)

    X = _logit(np.clip(p_raw, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(X, yb)

    def transform(p_new: np.ndarray) -> np.ndarray:
        Xn = _logit(np.clip(p_new, 1e-6, 1 - 1e-6)).reshape(-1, 1)
        return lr.predict_proba(Xn)[:, 1]

    return transform


def _make_trend_gates(close: pd.Series, span: int) -> Tuple[pd.Series, pd.Series]:
    sma = close.rolling(span, min_periods=span).mean()
    allow_long = (close > sma).fillna(False)
    allow_short = (close < sma).fillna(False)
    return allow_long, allow_short


def _generate_positions_with_hysteresis(
    scores: np.ndarray,
    allow_long: np.ndarray,
    allow_short: np.ndarray,
    long_in: float,
    long_out: float,
    short_in: float,
    short_out: float,
    min_hold: int = 0,
    cooldown: int = 0,
) -> np.ndarray:
    """
    Hysteresis + operational constraints:
      - min_hold: minimum bars to remain in a position after entry.
      - cooldown: bars to wait after flatting before a new entry.
    """
    pos, hold, cd = 0, 0, 0
    out = np.zeros_like(scores, dtype=int)

    for i, s in enumerate(scores):
        can_long = bool(allow_long[i])
        can_short = bool(allow_short[i])
        if hold > 0:
            hold -= 1
        if cd > 0:
            cd -= 1

        if pos == 0:
            if cd == 0:
                if can_long and s >= long_in:
                    pos = 1
                    hold = max(hold, min_hold)
                elif can_short and s <= short_in:
                    pos = -1
                    hold = max(hold, min_hold)
        elif pos == 1:
            if hold == 0 and ((not can_long) or (s <= long_out)):
                if cd == 0 and can_short and s <= short_in:
                    pos = -1
                    hold = max(hold, min_hold)
                else:
                    pos = 0
                    cd = max(cd, cooldown)
        else:  # pos == -1
            if hold == 0 and ((not can_short) or (s >= short_out)):
                if cd == 0 and can_long and s >= long_in:
                    pos = 1
                    hold = max(hold, min_hold)
                else:
                    pos = 0
                    cd = max(cd, cooldown)
        out[i] = pos
    return out


def _select_thresholds_hysteresis_trend(
    scores_val: np.ndarray,
    ret_val: np.ndarray,
    close_val: pd.Series,
    vol_span: int,
    slippage_bps: float,
    commission_bps: float,
    entry_pct_grid: List[int],
    gap_grid: List[int],
    trend_ma_span: int,
) -> Tuple[float, float, float, float, Dict[str, Any]]:
    """Grid search entry percentile + hysteresis gap under trend gates; maximize Sharpe (cost-aware)."""
    scores_val = np.asarray(scores_val, dtype=np.float64)
    ret_val = np.nan_to_num(np.asarray(ret_val, dtype=np.float64), nan=0.0)

    allow_long_s, allow_short_s = _make_trend_gates(close_val, trend_ma_span)
    allow_long, allow_short = allow_long_s.to_numpy(), allow_short_s.to_numpy()
    vol = _ewma_vol_numpy(ret_val, span=vol_span)
    vol = np.clip(vol, 1e-6, None)
    per_side = (slippage_bps + commission_bps) / 1e4

    best = None
    diag: Dict[str, Any] = {"grid": []}

    for pin in entry_pct_grid:
        if pin <= 0 or pin >= 100:
            continue
        hi_in = np.percentile(scores_val, pin)
        lo_in = np.percentile(scores_val, 100 - pin)
        for gap in gap_grid:
            if gap <= 0 or gap >= pin:
                continue
            hi_out = np.percentile(scores_val, pin - gap)
            lo_out = np.percentile(scores_val, 100 - (pin - gap))
            pos = _generate_positions_with_hysteresis(
                scores_val,
                allow_long,
                allow_short,
                long_in=hi_in,
                long_out=hi_out,
                short_in=lo_in,
                short_out=lo_out,
                min_hold=0,
                cooldown=0,
            )
            w = vol_targeted_size(pos, vol, risk_aversion=1.0, w_max=1.0)
            w = pd.Series(w)
            raw = w.shift(1).fillna(0.0).to_numpy() * ret_val
            dw = w.diff().abs().fillna(abs(w.iloc[0])).to_numpy()
            net = raw - dw * per_side
            sharpe = float(net.mean() / (net.std() + 1e-12))
            diag["grid"].append(
                {
                    "entry_pct": pin,
                    "gap": gap,
                    "long_in": float(hi_in),
                    "long_out": float(hi_out),
                    "short_in": float(lo_in),
                    "short_out": float(lo_out),
                    "sharpe": sharpe,
                }
            )
            if (best is None) or (sharpe > best[0]):
                best = (sharpe, hi_in, hi_out, lo_in, lo_out)

    if best is None:
        median = float(np.median(scores_val))
        return median, median, median, median, {"best": "median_fallback", "grid": []}

    _, hi_in, hi_out, lo_in, lo_out = best
    return (
        float(hi_in),
        float(hi_out),
        float(lo_in),
        float(lo_out),
        {
            "best": {
                "long_in": float(hi_in),
                "long_out": float(hi_out),
                "short_in": float(lo_in),
                "short_out": float(lo_out),
            },
            "grid": diag["grid"],
        },
    )


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #

def _summarize_trades(weights: pd.Series, net_returns: pd.Series) -> Dict[str, float]:
    trades = []
    durations = []
    in_trade = False
    running_ret = 0.0
    running_dur = 0

    for r, w in zip(net_returns.fillna(0.0), weights.fillna(0.0)):
        if not in_trade and w != 0:
            in_trade = True
            running_ret = 0.0
            running_dur = 0
        if in_trade:
            running_ret += r
            running_dur += 1
        if in_trade and w == 0:
            trades.append(running_ret)
            durations.append(running_dur)
            in_trade = False

    if in_trade:
        trades.append(running_ret)
        durations.append(running_dur)

    if not trades:
        return {
            "trade_count": 0,
            "win_rate_pct": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "avg_trade": 0.0,
            "max_trade_duration": 0.0,
            "avg_trade_duration": 0.0,
            "profit_factor": 0.0,
            "sqn": 0.0,
        }

    trades = np.array(trades)
    durations = np.array(durations)
    wins = trades[trades > 0]
    losses = trades[trades < 0]
    profit_factor = float(wins.sum() / abs(losses.sum())) if losses.size else float("inf")
    win_rate = float((trades > 0).mean() * 100.0)
    sqn = float(np.sqrt(len(trades)) * trades.mean() / (trades.std() + 1e-12))

    return {
        "trade_count": len(trades),
        "win_rate_pct": win_rate,
        "best_trade": float(trades.max()),
        "worst_trade": float(trades.min()),
        "avg_trade": float(trades.mean()),
        "max_trade_duration": float(durations.max()),
        "avg_trade_duration": float(durations.mean()),
        "profit_factor": profit_factor,
        "sqn": sqn,
    }


def _drawdown_stats(equity: pd.Series) -> Tuple[float, float, float, float]:
    dd = equity / equity.cummax() - 1.0
    max_dd = float(dd.min())
    avg_dd = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0
    durations = []
    current = 0
    for val in dd:
        if val < 0:
            current += 1
        elif current > 0:
            durations.append(current)
            current = 0
    if current > 0:
        durations.append(current)
    max_duration = float(max(durations)) if durations else 0.0
    avg_duration = float(sum(durations) / len(durations)) if durations else 0.0
    return max_dd, avg_dd, max_duration, avg_duration


def _render_window_string(
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    global_start: pd.Timestamp,
    global_end: pd.Timestamp,
) -> str:
    tz = global_start.tz

    def _align(ts: pd.Timestamp | None) -> pd.Timestamp | None:
        if ts is None:
            return None
        out = pd.Timestamp(ts)
        if tz is not None:
            if out.tzinfo is None:
                out = out.tz_localize(tz)
            elif str(out.tzinfo) != str(tz):
                out = out.tz_convert(tz)
        return out

    train_start = _align(train_start)
    train_end = _align(train_end)
    test_start = _align(test_start)
    test_end = _align(test_end)

    months = pd.period_range(
        _align(global_start).normalize().replace(day=1),
        _align(global_end).normalize().replace(day=1),
        freq="M",
    )
    codes = []
    for period in months:
        start = _align(period.to_timestamp())
        end = _align((period + 1).to_timestamp() - pd.Timedelta(seconds=1))
        if None in (start, end, train_start, train_end, test_start, test_end):
            codes.append("0")
            continue
        if (start <= train_end) and (end >= train_start):
            codes.append("T")
        elif (start <= test_end) and (end >= test_start):
            codes.append("W")
        else:
            codes.append("0")
    return "".join(codes)


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #

def _eval_one_fold(task: dict) -> dict:
    pipe = load_pickle(Path(task["pipe_path"]))

    idx_va = pd.DatetimeIndex(task["idx_va"])
    idx_te = pd.DatetimeIndex(task["idx_te"])
    X_va = np.asarray(task["X_va"])
    X_te = np.asarray(task["X_te"])
    y_va = np.asarray(task["y_va"])
    y_te = np.asarray(task["y_te"])
    close_va = pd.Series(task["close_va"], index=idx_va)
    close_te = pd.Series(task["close_te"], index=idx_te)

    scores_va = _score_series(_pipe_predict(pipe, X_va))
    scores_te = _score_series(_pipe_predict(pipe, X_te))

    score_span = int(task.get("score_ema_span", 1))
    if score_span > 1:
        scores_va = (
            pd.Series(scores_va, index=idx_va)
            .ewm(span=score_span, adjust=False)
            .mean()
            .to_numpy()
        )
        scores_te = (
            pd.Series(scores_te, index=idx_te)
            .ewm(span=score_span, adjust=False)
            .mean()
            .to_numpy()
        )

    pred_va = pd.Series(scores_va, index=idx_va)
    pred_te = pd.Series(scores_te, index=idx_te)
    ret_va = pd.Series(np.asarray(task["ret_va"], dtype=np.float64), index=idx_va)

    threshold_info = pick_threshold(
        pred=pred_va,
        ret_fwd=ret_va,
        costs_bps=task["costs_cfg"],
        grid=task["threshold_grid"],
        min_trades=int(task["min_trades"]),
        long_only=bool(task["long_only"]),
    )
    tau = threshold_info["tau"]

    ledger, equity = simulate(
        prices=close_te,
        predictions=pred_te,
        tau=tau,
        t_max=int(task["t_max"]),
        costs_cfg=task["costs_cfg"],
        long_only=bool(task["long_only"]),
    )

    class_diag = None
    if str(task.get("task_type", "regression")).lower() in {"classification", "hgb_classifier"}:
        y_true = pd.Series(y_te, index=idx_te)
        class_diag = classification_report(y_true, pred_te)

    return {
        "fold_id": task["fold_id"],
        "ledger": ledger,
        "equity": equity,
        "predictions": pred_te,
        "threshold": threshold_info,
        "tau": tau,
        "classification": class_diag,
    }


# --------------------------------------------------------------------------- #
# Main entrypoint
# --------------------------------------------------------------------------- #

def main(args) -> None:
    log = get_logger("main_backtest", level=args.log_level)
    cfg = _load_config_with_includes(args.config)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    costs_cfg = cfg.get("costs", {})
    runtime_cfg = cfg.get("runtime", {})

    symbol = data_cfg.get("symbol", "UNKNOWN")
    base_asset, quote_asset = parse_symbol(symbol)

    task_type = model_cfg.get("task", "regression").lower()
    horizon = int(model_cfg.get("horizon_bars", model_cfg.get("horizon", 6)))
    vol_span = int(model_cfg.get("vol_span", 20))
    score_ema_span = int(model_cfg.get("score_ema_span", 6))
    threshold_cfg = model_cfg.get("thresholds", {})
    threshold_grid = [float(x) for x in threshold_cfg.get("grid", [0.5, 0.6, 0.7])]
    min_trades = int(threshold_cfg.get("min_trades_valid", 30))
    trade_policy = model_cfg.get("trade_policy", {})
    long_only = bool(trade_policy.get("long_only", True))
    allow_shorts = bool(trade_policy.get("allow_shorts", not long_only))
    long_only_flag = long_only and not allow_shorts
    calibration_cfg = model_cfg.get("calibration", {})
    tb_cfg = model_cfg.get("triple_barrier", {})
    t_max = int(tb_cfg.get("t_max", horizon))
    n_workers = int(args.n_workers or runtime_cfg.get("n_workers", 1))

    df = load_all(
        symbol=data_cfg.get("symbol"),
        bar=data_cfg.get("bar"),
        start=data_cfg.get("start"),
        end=data_cfg.get("end"),
        cache=data_cfg.get("cache"),
        na_policy=data_cfg.get("na_policy"),
        source=data_cfg.get("source", "csv"),
        csv_path=data_cfg.get("csv_path"),
    )
    df = clean_and_resample(df, data_cfg.get("bar"), data_cfg.get("na_policy", "ffill"))
    X_df = assemble_features(df)
    close = df["close"]
    y_reg = k_ahead_log_return(close, horizon)

    labeling_cfg = model_cfg.get("labeling", {})
    lab_mode = labeling_cfg.get("mode", "direction")
    vol_k_sigma = float(labeling_cfg.get("vol_k_sigma", 0.5))
    tb_cfg = model_cfg.get("triple_barrier", {})

    returns1 = close.pct_change().fillna(0.0)
    vol_series = ewma_vol(returns1, span=vol_span).fillna(method="bfill").fillna(0.0)

    if tb_cfg.get("enable", False):
        y_tb, _ = triple_barrier_labels(
            close,
            vol_series,
            tb_cfg.get("pt_mult", 1.5),
            tb_cfg.get("sl_mult", 1.0),
            t_max,
        )
        if task_type in {"classification", "hgb_classifier"}:
            y_labeled = y_tb.replace(-1, 0)
        else:
            y_labeled = y_tb.astype(float)
    else:
        if task_type in {"classification", "hgb_classifier"}:
            y_labeled = make_labels(
                y_reg,
                vol_span=vol_span,
                mode=lab_mode,
                vol_k_sigma=vol_k_sigma,
            )
        else:
            y_labeled = y_reg

    aligned = pd.concat([X_df, y_labeled.rename("y")], axis=1).dropna()
    X_df = aligned[X_df.columns]
    y_series = aligned["y"]

    artifacts_dir = Path("evaluation/artifacts")
    items = _collect_artifacts(artifacts_dir)
    if not items:
        raise RuntimeError("No artifacts found. Run training first.")

    global_start = df.index.min()
    global_end = df.index.max()
    window_strings: List[str] = []
    tasks: List[Dict[str, Any]] = []
    for fold_id, pipe_path, meta in items:
        tr_start = _parse_ts(meta.get("train_start"))
        tr_end = _parse_ts(meta.get("train_end"))
        va_start = _parse_ts(meta.get("valid_start"))
        va_end = _parse_ts(meta.get("valid_end"))
        te_start = _parse_ts(meta.get("test_start"))
        te_end = _parse_ts(meta.get("test_end"))
        if None in (va_start, va_end, te_start, te_end, tr_start, tr_end):
            continue

        X_va, y_va, idx_va = _prepare_window_xy(X_df, y_series, va_start, va_end)
        X_te, y_te, idx_te = _prepare_window_xy(X_df, y_series, te_start, te_end)
        if len(X_va) == 0 or len(X_te) == 0:
            continue

        ret_series = k_ahead_log_return(df["close"], horizon)
        ret_va = ret_series.reindex(idx_va).to_numpy()
        ret_te = ret_series.reindex(idx_te).to_numpy()
        close_va = df["close"].reindex(idx_va).to_numpy()
        close_te = df["close"].reindex(idx_te).to_numpy()

        tasks.append({
            "fold_id": fold_id,
            "pipe_path": pipe_path.as_posix(),
            "X_va": X_va,
            "y_va": y_va,
            "idx_va": idx_va.astype(str).tolist(),
            "X_te": X_te,
            "y_te": y_te,
            "idx_te": idx_te.astype(str).tolist(),
            "ret_va": ret_va,
            "ret_te": ret_te,
            "close_va": close_va,
            "close_te": close_te,
            "vol_span": vol_span,
            "horizon": horizon,
            "task_type": task_type,
            "costs_cfg": costs_cfg,
            "threshold_grid": threshold_grid,
            "min_trades": min_trades,
            "score_ema_span": score_ema_span,
            "long_only": long_only_flag,
            "t_max": t_max,
        })

        window_strings.append(
            _render_window_string(tr_start, tr_end, te_start, te_end, global_start, global_end)
        )

    if not tasks:
        raise RuntimeError("No valid folds to evaluate.")

    results = run_parallel(tasks, n_workers=n_workers, func=_eval_one_fold)
    if not results:
        raise RuntimeError("Evaluation returned no folds.")

    ledgers = []
    threshold_rows: List[Dict[str, Any]] = []
    classification_rows: List[Dict[str, Any]] = []
    calibration_tables: List[List[Dict[str, Any]]] = []
    prediction_frames: List[pd.Series] = []
    position_frames: List[pd.Series] = []

    for res in results:
        ledger = res["ledger"].copy()
        ledger.index = pd.DatetimeIndex(ledger.index)
        ledgers.append(ledger)

        threshold_row = {
            "fold_id": res["fold_id"],
            "tau": res["tau"],
            **res["threshold"],
        }
        threshold_rows.append(threshold_row)

        prediction_frames.append(res["predictions"].rename(str(res["fold_id"])))
        position_frames.append(ledger["position"].rename(str(res["fold_id"])))

        if res.get("classification"):
            classification_rows.append(
                {
                    "fold_id": res["fold_id"],
                    "auc": res["classification"]["auc"],
                    "brier": res["classification"]["brier"],
                }
            )
            calibration_tables.append(res["classification"]["calibration"])

    ledger_all = pd.concat(ledgers).sort_index()
    equity_all = (1.0 + ledger_all["net"].fillna(0.0)).cumprod()
    equity_all.name = f"equity_curve_{quote_asset}"

    price_series = df["close"].reindex(ledger_all.index).ffill()
    buy_hold = (price_series / price_series.iloc[0]).rename(f"buy_and_hold_{quote_asset}")
    equity_df = pd.concat([equity_all, buy_hold], axis=1)
    metrics = summarize(ledger_all, equity_all)

    reports_dir = Path("evaluation/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = Path("evaluation/charts")
    charts_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "metrics": metrics,
        "thresholds": threshold_rows,
        "classification": classification_rows,
    }
    save_json(summary_payload, artifacts_dir / f"summary_{symbol}.json")
    save_json(
        {"folds": threshold_rows, "calibration": calibration_tables},
        artifacts_dir / f"threshold_diagnostics_{symbol}.json",
    )

    ledger_all.index.name = "timestamp"
    equity_df.index.name = "timestamp"

    if prediction_frames:
        prediction_series = (
            pd.concat(prediction_frames, axis=1)
            .mean(axis=1)
            .rename("probability_up")
        )
    else:
        prediction_series = pd.Series(dtype=float, name="probability_up")

    if position_frames:
        position_series = (
            pd.concat(position_frames, axis=1)
            .mean(axis=1)
            .rename("position")
        )
    else:
        position_series = pd.Series(dtype=float, name="position")

    start_time = ledger_all.index.min()
    end_time = ledger_all.index.max()

    save_csv(ledger_all, reports_dir / f"trades_{symbol}.csv")
    save_csv(equity_df, reports_dir / f"equity_curve_{symbol}.csv")
    save_csv(position_series.to_frame(), reports_dir / f"signals_{symbol}.csv")
    save_csv(prediction_series.to_frame(), reports_dir / f"predictions_{symbol}.csv")

    cs_start = start_time or df.index.min()
    cs_end = end_time or df.index.max()
    df_cs = df.loc[cs_start:cs_end, ["open", "high", "low", "close"]]
    fig_c = qf.candlestick(
        df_cs,
        signals=position_series.reindex(df_cs.index).fillna(0.0),
        entry_trades=True,
    )
    try:
        fig_c.write_html(charts_dir / f"candlestick_{symbol}.html")
    except Exception:
        pass

    fig_e = plot_equity_with_vectorbt(equity_all, benchmarks={buy_hold.name: buy_hold})
    try:
        fig_e.write_html(charts_dir / f"equity_{symbol}.html")
    except Exception:
        pass

    if calibration_tables:
        try:
            rel_fig = reliability_plot(calibration_tables[0])
            rel_fig.write_html(charts_dir / f"reliability_{symbol}.html")
        except Exception:
            pass

    slippage_pct = float(costs_cfg.get("slippage_bps", 0)) / 100.0
    commission_pct = float(costs_cfg.get("commission_bps", 0)) / 100.0
    currency_symbol = "$" if quote_asset.upper() in {"USDT", "USD"} else quote_asset.upper()

    net_returns = ledger_all["net"].fillna(0.0)
    weights = ledger_all["position"].fillna(0.0)
    trade_stats = _summarize_trades(weights, net_returns)

    duration_bars = float(len(net_returns))
    duration_hours = (
        float(((end_time - start_time) / pd.Timedelta(hours=1)) + 1)
        if start_time is not None and end_time is not None
        else 0.0
    )

    equity_final = float(equity_all.iloc[-1]) if not equity_all.empty else float("nan")
    equity_peak = float(equity_all.max()) if not equity_all.empty else float("nan")
    return_pct = float((equity_final - 1.0) * 100.0) if not np.isnan(equity_final) else float("nan")
    buy_hold_pct = float((buy_hold.iloc[-1] - 1.0) * 100.0) if not buy_hold.empty else float("nan")

    annual_factor = 24 * 365
    if len(net_returns) > 0:
        ann_return = (
            (equity_final ** (annual_factor / len(net_returns))) - 1.0
            if equity_final > 0
            else float("nan")
        )
        ann_return_pct = float(ann_return * 100.0) if not np.isnan(ann_return) else float("nan")
        ann_vol_pct = float(net_returns.std(ddof=0) * np.sqrt(annual_factor) * 100.0)
    else:
        ann_return_pct = float("nan")
        ann_vol_pct = float("nan")

    mean_ret = net_returns.mean()
    downside = net_returns[net_returns < 0]
    downside_std = downside.std(ddof=0)
    sortino = (
        (mean_ret * annual_factor) / ((downside_std * np.sqrt(annual_factor)) + 1e-12)
        if len(net_returns) > 0
        else float("nan")
    )

    max_dd, avg_dd, max_dd_dur, avg_dd_dur = _drawdown_stats(equity_all)
    calmar = (ann_return_pct / 100.0) / abs(max_dd) if max_dd < 0 else float("nan")
    exposure_pct = float(weights.ne(0.0).mean() * 100.0)

    best_trade_pct = trade_stats["best_trade"] * 100.0
    worst_trade_pct = trade_stats["worst_trade"] * 100.0
    avg_trade_pct = trade_stats["avg_trade"] * 100.0

    reducer_name = str(model_cfg.get("reducer", "pca")).upper()
    reducer_desc = "Sign-stable PLS" if reducer_name == "PLS" else "Incremental PCA"
    if task_type in {"classification", "hgb_classifier"}:
        model_desc = "calibrated logistic classifier"
    elif task_type == "hgb_regressor":
        model_desc = "hist-gradient boosting regressor"
    else:
        model_desc = "elastic-net regressor"

    def fmt_currency(val: float) -> str:
        if np.isnan(val):
            return "n/a"
        if currency_symbol == "$":
            return f"${val:,.4f}"
        if len(currency_symbol) == 1:
            return f"{currency_symbol}{val:,.4f}"
        return f"{val:,.4f} {currency_symbol}"

    def fmt_pct(val: float) -> str:
        if np.isnan(val):
            return "n/a"
        return f"{val:.2f}%"

    def fmt_float(val: float, digits: int = 4) -> str:
        if np.isnan(val):
            return "n/a"
        return f"{val:.{digits}f}"

    overview_lines = [
        f"We trained a {reducer_desc} reducer with a {model_desc} on a {horizon}-bar horizon.",
        f"Labels use {'triple-barrier' if tb_cfg.get('enable', False) else lab_mode} logic with timeout {t_max} bars.",
        f"Probabilities are smoothed with EMA span {score_ema_span} and the trading threshold maximises expected net P&L after costs of {fmt_pct(slippage_pct)} slippage and {fmt_pct(commission_pct)} commission.",
    ]

    param_lines = [
        "Parameters:",
        "  General:",
        f"    Start: {start_time}",
        f"    End: {end_time}",
        f"    Duration [bars]: {fmt_float(duration_bars, digits=0)}",
        f"    Duration [hours]: {fmt_float(duration_hours, digits=1)}",
        f"    Exposure Time [%]: {fmt_pct(exposure_pct)}",
        "  Model & Signals:",
        f"    Task: {task_type}",
        f"    Horizon [bars]: {horizon}",
        f"    Reducer: {reducer_name}",
        f"    Head: {model_desc}",
        f"    Threshold grid: {threshold_grid}",
        f"    Min trades (valid): {min_trades}",
        f"    Score EMA span: {score_ema_span}",
        f"    Long only: {long_only}",
        f"    Allow shorts: {allow_shorts}",
        f"    Calibration: {calibration_cfg.get('method', 'isotonic') if calibration_cfg.get('enable', False) else 'disabled'}",
    ]
    if tb_cfg.get("enable", False):
        param_lines.append(
            f"    Triple barrier: pt={tb_cfg.get('pt_mult', 1.5)}, sl={tb_cfg.get('sl_mult', 1.0)}, t_max={t_max}"
        )
    else:
        param_lines.append(f"    Label mode: {lab_mode} (vol_k_sigma={vol_k_sigma})")
    param_lines.extend(
        [
            "  Costs:",
            f"    Slippage per side: {fmt_pct(slippage_pct)}",
            f"    Commission per side: {fmt_pct(commission_pct)}",
        ]
    )

    equity_lines = [
        "Equity & Returns:",
        f"  Equity Final [{currency_symbol}]: {fmt_currency(equity_final)}",
        f"  Equity Peak [{currency_symbol}]: {fmt_currency(equity_peak)}",
        f"  Return [%]: {fmt_pct(return_pct)}",
        f"  Return (Ann.) [%]: {fmt_pct(ann_return_pct)}",
        f"  Buy & Hold Return [%]: {fmt_pct(buy_hold_pct)}",
        f"  Volatility (Ann.) [%]: {fmt_pct(ann_vol_pct)}",
        f"  Sharpe Ratio: {fmt_float(metrics.get('sharpe_365d', float('nan')))}",
        f"  Sortino Ratio: {fmt_float(sortino)}",
        f"  Calmar Ratio: {fmt_float(calmar)}",
    ]

    risk_lines = [
        "Risk:",
        f"  Max. Drawdown [%]: {fmt_pct(max_dd * 100.0)}",
        f"  Avg. Drawdown [%]: {fmt_pct(avg_dd * 100.0)}",
        f"  Max. Drawdown Duration [bars]: {fmt_float(max_dd_dur, digits=1)}",
        f"  Avg. Drawdown Duration [bars]: {fmt_float(avg_dd_dur, digits=1)}",
    ]

    trade_lines = [
        "Trades:",
        f"  # Trades: {trade_stats['trade_count']}",
        f"  Win Rate [%]: {fmt_pct(trade_stats['win_rate_pct'])}",
        f"  Best Trade [%]: {fmt_pct(best_trade_pct)}",
        f"  Worst Trade [%]: {fmt_pct(worst_trade_pct)}",
        f"  Avg. Trade [%]: {fmt_pct(avg_trade_pct)}",
        f"  Max. Trade Duration [bars]: {fmt_float(trade_stats['max_trade_duration'], digits=1)}",
        f"  Avg. Trade Duration [bars]: {fmt_float(trade_stats['avg_trade_duration'], digits=1)}",
        f"  Profit Factor: {fmt_float(trade_stats['profit_factor'])}",
        f"  SQN: {fmt_float(trade_stats['sqn'])}",
    ]

    data_sample_name = Path(data_cfg.get("csv_path", "")).name or "N/A"
    range_label = f"{start_time} to {end_time}"

    report_lines = ["Strategy Overview:"]
    report_lines.extend(f"  {line}" for line in overview_lines)
    report_lines.append("")
    report_lines.append(f"Data sample used: {data_sample_name}")
    report_lines.append(f"Reported results shown for: {range_label}")
    report_lines.append("")
    report_lines.extend(param_lines)
    report_lines.append("")
    report_lines.extend(equity_lines)
    report_lines.append("")
    report_lines.extend(risk_lines)
    report_lines.append("")
    report_lines.extend(trade_lines)
    if classification_rows:
        avg_auc = fmt_float(np.mean([row["auc"] for row in classification_rows]))
        avg_brier = fmt_float(np.mean([row["brier"] for row in classification_rows]))
        report_lines.append("")
        report_lines.append("Classification:")
        report_lines.append(f"  Mean AUC: {avg_auc}")
        report_lines.append(f"  Mean Brier: {avg_brier}")

    report_lines.append("")
    report_lines.append("Thresholds (per fold):")
    for row in threshold_rows:
        report_lines.append(
            f"  Fold {row['fold_id']}: "
            f"tau={fmt_float(row['tau'], 4)}, "
            f"ev={fmt_float(row.get('ev', float('nan')), 6)}, "
            f"turns={fmt_float(row.get('turns', 0.0), 4)}"
        )
    report_lines.append("")
    report_lines.append("Window Overview:")
    for s in window_strings:
        report_lines.append(f"  {s}")

    summary_path = reports_dir / f"summary_report_{symbol}.txt"
    summary_path.write_text("\n".join(report_lines), encoding="utf-8")

    log.info(
        "Backtest complete (n_workers=%d). Summary at %s",
        n_workers,
        summary_path,
    )