from __future__ import annotations
import pandas as pd, numpy as np, pickle
from pathlib import Path

_EPS = 1e-8

class RollingZScore:
    """Train-only rolling z-score; robust to short windows and flat cols."""
    def __init__(self, window: int = 500, clip: float = 5.0):
        self.window = int(window)
        self.clip = float(clip)
        self.stats_: dict[str, tuple[float,float]] | None = None

    def fit(self, X_train: pd.DataFrame) -> "RollingZScore":
        # If the window is too big, fallback to full-sample stats per feature
        if len(X_train) < self.window:
            mu = X_train.mean()
            sd = X_train.std(ddof=0)
        else:
            min_periods = max(20, self.window // 5)
            rolling = X_train.rolling(self.window, min_periods=min_periods)
            mu = rolling.mean().iloc[-1]
            sd = rolling.std(ddof=0).iloc[-1]
        mu = mu.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        sd = sd.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=_EPS)
        self.stats_ = {c: (float(mu[c]), float(sd[c])) for c in X_train.columns}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.stats_ is not None, "call fit first"
        out = X.copy()
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for c,(m,s) in self.stats_.items():
            if c in out.columns:
                out[c] = (out[c] - m) / s
        return out.clip(-self.clip, self.clip).astype("float32")

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path,"wb") as f:
            pickle.dump({"window": self.window, "clip": self.clip, "stats": self.stats_}, f)

    @classmethod
    def load(cls, path: str | Path) -> "RollingZScore":
        with open(path,"rb") as f:
            obj = pickle.load(f)
        inst = cls(obj["window"], obj.get("clip", 5.0))
        inst.stats_ = obj["stats"]
        return inst
