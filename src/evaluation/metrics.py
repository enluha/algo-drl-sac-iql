import numpy as np, pandas as pd

def summarize(ledger: pd.DataFrame, equity: pd.Series, turn_thresh: float | None = None) -> dict:
    # Prefer true PnL returns if available; otherwise fall back.
    if "pnl" in ledger.columns:
        r = ledger["pnl"].fillna(0.0)
    elif "net" in ledger.columns:
        r = ledger["net"].fillna(0.0)
    elif {"raw", "cost"}.issubset(ledger.columns):
        r = (ledger["raw"].fillna(0.0) - ledger["cost"].fillna(0.0))
    else:
        r = pd.Series(dtype="float64")
    ann = 24*365
    sharpe = float(r.mean() / (r.std()+1e-12) * np.sqrt(ann)) if len(r)>0 else float("nan")
    downside = r[r<0]; sortino = float(r.mean()/(downside.std()+1e-12)*np.sqrt(ann)) if len(r)>0 else float("nan")
    dd = equity/equity.cummax()-1.0; maxdd = float(dd.min()) if not equity.empty else float("nan")
    calmar = (sharpe/abs(maxdd)) if (maxdd<0 and not np.isnan(sharpe)) else float("nan")
    out = {"sharpe_365d": sharpe, "sortino_365d": sortino, "max_dd": maxdd, "calmar": calmar}
    if "turnover" in ledger:
        turn = ledger["turnover"].fillna(0.0)
        out["turnover"] = float(turn.sum())
        out["turnover_nonzero_frac"] = float((turn > 0).mean())
        out["turnover_mean"] = float(turn.mean())
        out["turnover_median"] = float(turn.median())
        if isinstance(turn_thresh, (int, float)) and turn_thresh is not None:
            out["trades_ge_min_step"] = int((turn >= float(turn_thresh)).sum())
    if "weight" in ledger:
        w = ledger["weight"].fillna(0.0)
        out["avg_abs_weight"] = float(w.abs().mean())
        sign = (w > 1e-9).astype(int) - (w < -1e-9).astype(int)
        flips = int((sign.shift(1).fillna(0) != sign).sum())
        out["sign_flips"] = flips
        runs = (sign != sign.shift()).cumsum()
        run_len = runs.groupby(runs).size()
        if len(run_len) > 0:
            out["median_hold_bars_by_sign"] = int(run_len.median())
            out["p95_hold_bars_by_sign"] = int(run_len.quantile(0.95))
    return out
