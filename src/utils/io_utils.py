"""IO utilities for YAML, JSON, CSV, pickle, and HTML outputs."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)


def save_csv(df_or_arr: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(df_or_arr, "to_csv"):
        index_label = getattr(getattr(df_or_arr, "index", None), "name", None)
        df_or_arr.to_csv(path, index=True, index_label=index_label)
    else:
        pd.DataFrame(df_or_arr).to_csv(path, index=False)


def save_html(fig: Any, path: str | Path) -> None:
    """Persist Plotly/matplotlib HTML representations."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(fig, "to_html"):
        html = fig.to_html(full_html=True, include_plotlyjs="cdn")
        path.write_text(html, encoding="utf-8")
    else:
        raise TypeError("Object does not support to_html().")


def cache_pickle(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write text atomically by staging to a tmp file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def load_nested_config(path: str | Path) -> dict:
    path = Path(path)
    master = load_yaml(path)
    if "include" not in master:
        return master

    config: dict[str, Any] = {}
    for rel in master["include"]:
        sub_path = path.parent / rel
        config[sub_path.stem] = load_yaml(sub_path)
    return config
