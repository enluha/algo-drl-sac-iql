import numpy as np
import pandas as pd

from src.cv.walk_forward import walk_forward_splits
from src.features.engineering import assemble_features
from src.models.pipeline import ForecastPipeline
from src.models.regressors import ElasticNetRegressor
from src.transforms.reduce import RollingPLS
from src.transforms.scaling import RollingStandardScaler
from src.utils.parallel import run_parallel
from src.utils.symbols import parse_symbol


def _make_price_frame():
    idx = pd.date_range("2024-01-01", periods=48, freq="H", tz="UTC")
    base_price = 40000 + np.linspace(0, 1000, len(idx))
    df = pd.DataFrame(
        {
            "open": base_price * 0.999,
            "high": base_price * 1.001,
            "low": base_price * 0.998,
            "close": base_price,
            "volume": np.random.default_rng(42).normal(loc=100, scale=5, size=len(idx)).clip(1),
        },
        index=idx,
    )
    df["number_of_trades"] = np.arange(len(idx))
    df["taker_buy_volume"] = df["volume"] * 0.5
    return df


def test_parse_symbol_handles_known_quotes():
    assert parse_symbol("BTCUSDT") == ("BTC", "USDT")
    assert parse_symbol("ethbtc") == ("ETH", "BTC")
    base, quote = parse_symbol("SOLTRY")
    assert base == "SOL" and quote == "TRY"


def test_walk_forward_preserves_timezone():
    idx = pd.date_range("2024-01-01", periods=24, freq="H", tz="UTC")
    splits = walk_forward_splits(idx, train="12h", valid="6h", test="6h", step="6h", embargo="0h")
    assert len(splits) > 0
    train_idx, valid_idx, test_idx = splits[0]
    assert isinstance(train_idx, pd.DatetimeIndex)
    assert train_idx.tz == idx.tz == valid_idx.tz == test_idx.tz


def test_assemble_features_returns_expected_columns():
    df = _make_price_frame()
    features = assemble_features(df)
    expected = {"ret_1", "ewma_vol_20", "rsi_14", "vol_z", "taker_buy_ratio"}
    assert expected.issubset(set(features.columns))
    assert features.index.equals(df.index)


def test_pipeline_end_to_end_with_pls():
    df = _make_price_frame()
    features = assemble_features(df).dropna()
    X = features.to_numpy(dtype=np.float64)
    y = np.random.default_rng(123).normal(size=len(features))

    scaler = RollingStandardScaler()
    reducer = RollingPLS(n_components=2)
    model = ElasticNetRegressor(alpha=0.001, l1_ratio=0.1, max_iter=10_000)
    pipe = ForecastPipeline(scaler, reducer, model).fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (len(features),)


def test_run_parallel_sequential_vs_parallel():
    data = list(range(5))

    def square(x):
        return x * x

    sequential = run_parallel(data, n_workers=1, func=square)
    parallel = run_parallel(data, n_workers=2, func=square)
    assert sequential == [x * x for x in data]
    assert parallel == sequential
