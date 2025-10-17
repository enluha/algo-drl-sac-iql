from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class RollingZScore:
    window: int = 500
    eps: float = 1e-8
    means_: Dict[str, float] = field(default_factory=dict)
    stds_: Dict[str, float] = field(default_factory=dict)

    def fit_partial(self, data: pd.DataFrame) -> None:
        if data.empty:
            return
        rolling = data.rolling(self.window, min_periods=1)
        means = rolling.mean().iloc[-1]
        stds = rolling.std(ddof=0).iloc[-1]
        for col, mean in means.items():
            self.means_[col] = float(mean)
            self.stds_[col] = float(stds[col]) if stds[col] > 0 else self.eps

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.means_:
            raise RuntimeError("Normalizer must be fit before transform.")
        aligned = data.copy()
        for col in aligned.columns:
            mean = self.means_.get(col, 0.0)
            std = self.stds_.get(col, 1.0)
            aligned[col] = ((aligned[col] - mean) / (std + self.eps)).clip(-5.0, 5.0)
        return aligned.astype(np.float32)
