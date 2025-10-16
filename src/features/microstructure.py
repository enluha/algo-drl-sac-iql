"""
NON MVP - Placeholder for microstructure feature engineering.
"""

from __future__ import annotations

import pandas as pd


def micro_features(df: pd.DataFrame) -> pd.DataFrame:
    """NON MVP - Derive features from funding, open interest, etc.

    Params:
        df: DataFrame with supplemental derivatives data.

    Returns:
        DataFrame of microstructure features.
    """
    raise NotImplementedError("NON MVP - implement microstructure features later")
