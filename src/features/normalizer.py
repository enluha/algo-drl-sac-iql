from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class RollingZScore:
    window: int = 500
    means_: Dict[str, float] = field(default_factory=dict)
    stds_: Dict[str, float] = field(default_factory=dict)

    def fit_partial(self, df: pd.DataFrame) -> None:
        """Update rolling statistics using training data only."""
        if df.empty:
            return
        rolling = df.rolling(self.window, min_periods=1)
        means = rolling.mean().iloc[-1]
        stds = rolling.std(ddof=0).iloc[-1].replace(0.0, np.nan)
        for col, mean in means.items():
            self.means_[col] = float(mean)
        for col, std in stds.items():
            self.stds_[col] = float(std if not np.isnan(std) else 1.0)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply z-score normalization with clipping."""
        if not self.means_:
            raise ValueError("Normalizer must be fit before calling transform().")
        out = df.copy()
        for col in out.columns:
            mean = self.means_.get(col, 0.0)
            std = self.stds_.get(col, 1.0)
            if std == 0:
                std = 1.0
            out[col] = ((out[col] - mean) / std).clip(-5.0, 5.0)
        return out.astype(np.float32)
