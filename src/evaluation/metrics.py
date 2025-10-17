import numpy as np, pandas as pd

def summarize(ledger: pd.DataFrame, equity: pd.Series) -> dict:
    r = ledger["net"].fillna(0.0)
    ann = 24*365
    sharpe = float(r.mean() / (r.std()+1e-12) * np.sqrt(ann)) if len(r)>0 else float("nan")
    downside = r[r<0]; sortino = float(r.mean()/(downside.std()+1e-12)*np.sqrt(ann)) if len(r)>0 else float("nan")
    dd = equity/equity.cummax()-1.0; maxdd = float(dd.min()) if not equity.empty else float("nan")
    calmar = (sharpe/abs(maxdd)) if (maxdd<0 and not np.isnan(sharpe)) else float("nan")
    turnover = float(ledger["turnover"].sum()) if "turnover" in ledger else float("nan")
    return {"sharpe_365d": sharpe, "sortino_365d": sortino, "max_dd": maxdd, "calmar": calmar, "turnover": turnover}
