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
        # Use full training data statistics to avoid recency bias
        # Previous implementation only used last window, creating regime-specific normalization
        mu = X_train.mean()
        sd = X_train.std(ddof=0)
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
        data = {
            "kind": "rolling",
            "window": self.window,
            "clip": self.clip,
            "stats": self.stats_,
        }
        _dump_normalizer(path, data)

    @classmethod
    def load(cls, path: str | Path) -> "RollingZScore":
        obj = _load_pickle(path)
        inst = cls(obj.get("window", 500), obj.get("clip", 5.0))
        inst.stats_ = obj["stats"]
        return inst


class GlobalZScore:
    """Fit mean/std on the entire train slice (global statistics)."""

    def __init__(self, clip: float = 5.0):
        self.clip = float(clip)
        self.stats_: dict[str, tuple[float, float]] | None = None

    def fit(self, X_train: pd.DataFrame) -> "GlobalZScore":
        mu = X_train.mean().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        sd = X_train.std(ddof=0).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=_EPS)
        self.stats_ = {c: (float(mu[c]), float(sd[c])) for c in X_train.columns}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.stats_ is not None, "call fit first"
        out = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).copy()
        for c, (m, s) in self.stats_.items():
            if c in out.columns:
                out[c] = (out[c] - m) / s
        return out.clip(-self.clip, self.clip).astype("float32")

    def save(self, path: str | Path):
        data = {
            "kind": "global",
            "clip": self.clip,
            "stats": self.stats_,
        }
        _dump_normalizer(path, data)

    @classmethod
    def load(cls, path: str | Path) -> "GlobalZScore":
        obj = _load_pickle(path)
        inst = cls(obj.get("clip", 5.0))
        inst.stats_ = obj["stats"]
        return inst


def _dump_normalizer(path: str | Path, payload: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def _load_pickle(path: str | Path) -> dict:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def load_normalizer(path: str | Path):
    obj = _load_pickle(path)
    kind = obj.get("kind", "rolling")
    if kind == "global":
        inst = GlobalZScore(obj.get("clip", 5.0))
    else:
        inst = RollingZScore(obj.get("window", 500), obj.get("clip", 5.0))
    inst.stats_ = obj["stats"]
    return inst
