"""
MVP - IO utilities for YAML/CSV/JSON and pickles.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_yaml(path: str | Path) -> dict:
    """MVP - Load a YAML file into a dict.

    Params:
        path: pathlike pointing to YAML file.

    Returns:
        Parsed YAML content as dictionary.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_json(obj: Any, path: str | Path) -> None:
    """MVP - Serialize object to JSON with indentation.

    Params:
        obj: JSON-serializable object.
        path: destination path for JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)


def save_csv(df_or_arr: Any, path: str | Path) -> None:
    """MVP - Save pandas DataFrame or ndarray-like to CSV.

    Params:
        df_or_arr: object with to_csv or array-like convertible to DataFrame.
        path: destination path for CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(df_or_arr, "to_csv"):
        index_label = getattr(getattr(df_or_arr, "index", None), "name", None)
        df_or_arr.to_csv(path, index=True, index_label=index_label)
    else:
        pd.DataFrame(df_or_arr).to_csv(path, index=False)


def cache_pickle(obj: Any, path: str | Path) -> None:
    """MVP - Serialize object to pickle.

    Params:
        obj: Python object to persist.
        path: destination path for pickle file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def load_pickle(path: str | Path) -> Any:
    """MVP - Load object from pickle.

    Params:
        path: path to pickle file.

    Returns:
        Deserialized Python object.
    """
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_nested_config(path: str | Path) -> dict:
    """MVP - Load master config supporting include directives.

    Params:
        path: path to master YAML file.

    Returns:
        Aggregated configuration dictionary keyed by included file stem.
    """
    path = Path(path)
    master = load_yaml(path)
    if "include" not in master:
        return master

    config: dict[str, Any] = {}
    for rel in master["include"]:
        sub_path = path.parent / rel
        config[sub_path.stem] = load_yaml(sub_path)
    return config
