from __future__ import annotations
import pandas as pd, numpy as np, pickle
from pathlib import Path

class RollingZScore:
    def __init__(self, window: int = 500):
        self.window = int(window)
        self.stats_: dict[str, tuple[float,float]] | None = None

    def fit(self, X_train: pd.DataFrame) -> "RollingZScore":
        mu = X_train.rolling(self.window, min_periods=self.window).mean().iloc[-1]
        sd = X_train.rolling(self.window, min_periods=self.window).std().iloc[-1].replace(0, np.nan)
        self.stats_ = {col: (float(mu[col]), float(sd[col])) for col in X_train.columns}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.stats_ is not None, "call fit first"
        out = X.copy()
        for col,(m,s) in self.stats_.items():
            if col not in out.columns: continue
            out[col] = (out[col] - m) / (s if s and not np.isnan(s) else 1.0)
        return out.clip(-5,5).astype("float32")

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path,"wb") as f: pickle.dump({"window":self.window,"stats":self.stats_}, f)

    @classmethod
    def load(cls, path: str | Path) -> "RollingZScore":
        with open(path,"rb") as f: obj = pickle.load(f)
        inst = cls(obj["window"]); inst.stats_ = obj["stats"]; return inst
