import numpy as np
import pandas as pd

from src.features.normalizer import GlobalZScore, RollingZScore, load_normalizer


def test_global_normalizer_round_trip(tmp_path):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 2.5, 3.0, 3.5]})
    norm = GlobalZScore(clip=5.0).fit(df)
    transformed = norm.transform(df)
    assert np.all(np.isfinite(transformed.values))
    path = tmp_path / "global.pkl"
    norm.save(path)
    loaded = load_normalizer(path)
    transformed_loaded = loaded.transform(df)
    np.testing.assert_allclose(transformed.values, transformed_loaded.values)


def test_rolling_normalizer_round_trip(tmp_path):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 11.0, 12.0, 13.0]})
    norm = RollingZScore(window=3, clip=5.0).fit(df)
    transformed = norm.transform(df)
    assert np.all(np.isfinite(transformed.values))
    path = tmp_path / "rolling.pkl"
    norm.save(path)
    loaded = load_normalizer(path)
    transformed_loaded = loaded.transform(df)
    np.testing.assert_allclose(transformed.values, transformed_loaded.values)
