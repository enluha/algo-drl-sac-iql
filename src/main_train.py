"""
MVP - Training entrypoint: build features, prepare targets, fit scaler+reducer+model per
walk-forward split, and persist artifacts.

Key points:
- Supports reducer: PLS (supervised) or PCA (IncrementalPCA).
- Supports model: ElasticNet (regression) or Logistic (classification).
- Fits on TRAIN window only (no leakage).
- Saves each fold's pipeline into evaluation/artifacts/pipe_fold_{i}.pkl
- Respects BLAS threads set by CLI (see src/cli.py).

NON MVP areas marked in comments (e.g., hyperparam search, multiprocessing).
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .utils.io_utils import load_yaml
from .utils.logging_utils import get_logger
from .utils.parallel import run_parallel
from .utils.seed import fix_seeds

from .data.loaders import load_all
from .data.preprocessing import clean_and_resample
from .features.engineering import assemble_features
from .labels.targets import k_ahead_log_return, make_labels, ewma_vol
from .labels.triple_barrier import triple_barrier_labels
from .cv.purged import CombinatorialPurgedCV, PurgedKFold
from .transforms.reducers import make_reducer
from .models.linear import make_model

from .cv.walk_forward import walk_forward_splits


# --------------------------- helpers ---------------------------

def _load_config_with_includes(config_path: str | Path) -> Dict[str, Any]:
    """
    MVP - Load top-level YAML and merge includes if present.

    The root YAML may contain:
        include:
          - data.yaml
          - model.yaml
          - cv.yaml
          - costs.yaml
          - runtime.yaml

    Returns a single dict: {"data":..., "model":..., "cv":..., "costs":..., "runtime":...}
    """
    import os
    cfg = load_yaml(config_path)
    base_dir = Path(config_path).parent

    def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(a)
        for k, v in b.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    if isinstance(cfg, dict) and "include" in cfg:
        merged: Dict[str, Any] = {}
        for rel in cfg["include"]:
            inc_path = base_dir / rel
            part = load_yaml(inc_path)
            # filename (e.g., data.yaml) becomes a top-level key by its stem
            key = Path(rel).stem
            merged[key] = part
        return merged

    # Already a merged dict (single-file config); keep as-is
    return cfg


def _prepare_xy(
    X_df: pd.DataFrame,
    y: pd.Series,
    index_slice,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    MVP - Align features and target on a window and drop NaNs (tz-robust).
    """
    idx = pd.DatetimeIndex(index_slice)
    if X_df.index.tz is not None and idx.tz is None:
        idx = idx.tz_localize(X_df.index.tz)
    elif X_df.index.tz is not None and idx.tz is not None and str(idx.tz) != str(X_df.index.tz):
        idx = idx.tz_convert(X_df.index.tz)

    Xw = X_df.reindex(idx)
    yw = y.reindex(idx)

    mask = ~(Xw.isna().any(axis=1) | yw.isna())
    Xw = Xw.loc[mask]
    yw = yw.loc[mask]

    return Xw.to_numpy(dtype=np.float64), yw.to_numpy()


def _train_one_fold(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    MVP - Train a single walk-forward fold and persist artifacts.

    Parameters
    ----------
    task : dict
        Serialized payload containing numpy arrays and configuration.

    Returns
    -------
    dict
        Metadata describing saved artifact paths.
    """
    from pathlib import Path

    import numpy as np

    from .models.pipeline import ForecastPipeline
    from .transforms.scaling import RollingStandardScaler
    from .utils.io_utils import cache_pickle, save_json

    fold_id = task["fold_id"]
    X_tr = np.asarray(task["X_tr"])
    y_tr = np.asarray(task["y_tr"])

    reducer_name = task["reducer"]
    n_components = task["n_components"]
    model_task = task["model_task"]
    model_cfg = task["model_cfg"]

    scaler = RollingStandardScaler()
    reducer = make_reducer(reducer_name, n_components)

    if model_task in {"classification", "regression"}:
        model = make_model(model_task, model_cfg)
    elif model_task == "hgb_classifier":
        from .models.gbm import HGBClassifier

        model = HGBClassifier(**model_cfg.get("hgb_classifier", {}))
    elif model_task == "hgb_regressor":
        from .models.gbm import HGBRegressor

        model = HGBRegressor(**model_cfg.get("hgb_regressor", {}))
    else:
        raise ValueError(
            "model.task must be one of: 'regression', 'classification', "
            "'hgb_classifier', 'hgb_regressor'"
        )

    pipeline = ForecastPipeline(scaler, reducer, model).fit(X_tr, y_tr)

    outdir = Path("evaluation/artifacts")
    outdir.mkdir(parents=True, exist_ok=True)
    pipe_path = outdir / f"pipe_fold_{fold_id}.pkl"
    meta_path = outdir / f"meta_fold_{fold_id}.json"

    cache_pickle(pipeline, pipe_path)
    save_json(task["meta"], meta_path)

    return {
        "fold_id": fold_id,
        "pipe": pipe_path.as_posix(),
        "meta": meta_path.as_posix(),
    }


# --------------------------- main ---------------------------

def main(args) -> None:
    """
    MVP - Parallel training harness for walk-forward folds.
    """
    log = get_logger("main_train", level=str(getattr(args, "log_level", "INFO")).upper())
    cfg = _load_config_with_includes(args.config)

    runtime = cfg.get("runtime", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    cv_cfg = cfg.get("cv", {})

    seed = int(runtime.get("seed", 42))
    fix_seeds(seed)

    n_workers = int(getattr(args, "n_workers", runtime.get("n_workers", 1)))
    log.info(f"MVP - Using seed={seed}, n_workers={n_workers}")

    df = load_all(
        symbol=data_cfg.get("symbol"),
        bar=data_cfg.get("bar"),
        start=data_cfg.get("start"),
        end=data_cfg.get("end"),
        cache=data_cfg.get("cache"),
        na_policy=data_cfg.get("na_policy"),
        source=data_cfg.get("source", "csv"),
        csv_path=data_cfg.get("csv_path"),
    )
    df = clean_and_resample(df, data_cfg.get("bar"), data_cfg.get("na_policy", "ffill"))
    log.info(
        "Data loaded & cleaned: %s .. %s rows=%d",
        df.index.min(),
        df.index.max(),
        len(df),
    )

    X_df = assemble_features(df)
    horizon = int(model_cfg.get("horizon_bars", model_cfg.get("horizon", 6)))
    task_name = model_cfg.get("task", "regression").lower()
    labeling_cfg = model_cfg.get("labeling", {})
    lab_mode = labeling_cfg.get("mode", "direction")
    vol_k_sigma = float(labeling_cfg.get("vol_k_sigma", 0.5))
    vol_span = int(model_cfg.get("vol_span", 20))
    tb_cfg = model_cfg.get("triple_barrier", {})

    close = df["close"]
    y_reg = k_ahead_log_return(close, horizon)
    vol = ewma_vol(close.pct_change().fillna(0.0), span=vol_span).fillna(method="bfill").fillna(0.0)

    if tb_cfg.get("enable", False):
        y_tb, t1_series = triple_barrier_labels(
            close,
            vol,
            tb_cfg.get("pt_mult", 1.5),
            tb_cfg.get("sl_mult", 1.0),
            int(tb_cfg.get("t_max", horizon)),
        )
        if task_name in {"classification", "hgb_classifier"}:
            y_raw = y_tb.replace(-1, 0)
        else:
            y_raw = y_tb.astype(float)
    else:
        if task_name in {"classification", "hgb_classifier"}:
            y_raw = make_labels(
                y_reg,
                vol_span=vol_span,
                mode=lab_mode,
                vol_k_sigma=vol_k_sigma,
            )
        else:
            y_raw = y_reg
        t1_series = close.index.to_series().shift(-horizon)

    aligned = pd.concat(
        [
            X_df,
            y_raw.rename("y"),
            t1_series.rename("t1"),
        ],
        axis=1,
    ).dropna()
    X_df = aligned[X_df.columns]
    y_series = aligned["y"]
    t1_series = aligned["t1"]
    log.info(
        "After alignment: X shape=%s, y len=%d (task=%s, label_mode=%s)",
        X_df.shape,
        len(y_series),
        task_name,
        lab_mode,
    )

    idx = X_df.index
    scheme = str(cv_cfg.get("scheme", "time_split")).lower()
    splits = []

    if scheme == "cpcv":
        splitter = CombinatorialPurgedCV(
            n_splits=int(cv_cfg.get("n_folds", 8)),
            n_test_folds=int(cv_cfg.get("n_test_folds", 2)),
            purge_bars=int(cv_cfg.get("purge_bars", 0)),
            embargo_bars=int(cv_cfg.get("embargo_bars", 0)),
        )
        for train_idx, test_idx in splitter.split(idx, t1_series):
            splits.append((idx[train_idx], idx[test_idx], idx[test_idx]))
    elif scheme == "purged":
        splitter = PurgedKFold(
            n_splits=int(cv_cfg.get("n_folds", 5)),
            purge_bars=int(cv_cfg.get("purge_bars", 0)),
            embargo_bars=int(cv_cfg.get("embargo_bars", 0)),
        )
        for train_idx, test_idx in splitter.split(idx, t1_series):
            splits.append((idx[train_idx], idx[test_idx], idx[test_idx]))
    else:
        splits = walk_forward_splits(
            idx,
            train=cv_cfg.get("train", "365d"),
            valid=cv_cfg.get("valid", "90d"),
            test=cv_cfg.get("test", "30d"),
            step=cv_cfg.get("step", "30d"),
            embargo=cv_cfg.get("embargo", "0d"),
        )

    if not splits:
        raise RuntimeError("No CV splits generated; adjust cv configuration.")

    tasks: list[Dict[str, Any]] = []
    for fold_id, (tr_idx, va_idx, te_idx) in enumerate(splits):
        X_tr, y_tr = _prepare_xy(X_df, y_series, tr_idx)
        if len(X_tr) == 0:
            log.warning("MVP - Empty TRAIN for fold %s; skipping", fold_id)
            continue

        meta = {
            "fold_id": fold_id,
            "train_start": str(pd.Timestamp(tr_idx.min())),
            "train_end": str(pd.Timestamp(tr_idx.max())),
            "valid_start": str(pd.Timestamp(va_idx.min())) if len(va_idx) else None,
            "valid_end": str(pd.Timestamp(va_idx.max())) if len(va_idx) else None,
            "test_start": str(pd.Timestamp(te_idx.min())) if len(te_idx) else None,
            "test_end": str(pd.Timestamp(te_idx.max())) if len(te_idx) else None,
            "X_tr_shape": list(X_tr.shape),
            "task": task_name,
            "reducer": model_cfg.get("reducer", "pls"),
            "n_components": int(model_cfg.get("n_components", 6)),
            "horizon": horizon,
            "seed": seed,
        }

        tasks.append(
            {
                "fold_id": fold_id,
                "X_tr": X_tr,
                "y_tr": y_tr,
                "reducer": model_cfg.get("reducer", "pls"),
                "n_components": int(model_cfg.get("n_components", 6)),
                "model_task": task_name,
                "model_cfg": model_cfg,
                "meta": meta,
            }
        )

    if not tasks:
        raise RuntimeError("MVP - No trainable folds assembled.")

    results = run_parallel(tasks, n_workers=n_workers, func=_train_one_fold)
    log.info("MVP - Trained %d folds (n_workers=%d)", len(results), n_workers)
