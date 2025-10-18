from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd


@dataclass(frozen=True)
class Phase:
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class Splits:
    pretrain: Phase
    finetune: Phase
    test: Phase
    warmup_bars: int


def _ts(value) -> pd.Timestamp:
    ts = pd.to_datetime(value)
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp: {value}")
    return ts


def load_splits(cfg: Dict, df_index: pd.DatetimeIndex, bar: str | None, warmup_bars_default: int = 96) -> Splits:
    wf = cfg.get("walkforward", cfg)
    if "splits" not in wf:
        raise KeyError("walkforward config must include 'splits'.")
    sp = wf["splits"]

    pre = Phase(_ts(sp["pretrain"]["start"]), _ts(sp["pretrain"]["end"]))
    fin = Phase(_ts(sp["finetune"]["start"]), _ts(sp["finetune"]["end"]))
    tes = Phase(_ts(sp["test"]["start"]), _ts(sp["test"]["end"]))
    warm = int(wf.get("warmup_bars", warmup_bars_default))

    assert pre.start < pre.end < fin.start <= fin.end < tes.start <= tes.end, (
        f"Non-monotonic splits: {pre} {fin} {tes}"
    )

    dmin, dmax = df_index.min(), df_index.max()
    for name, ph in (("pretrain", pre), ("finetune", fin), ("test", tes)):
        assert dmin <= ph.start <= ph.end <= dmax, (
            f"{name} outside data range: {ph} not in [{dmin}, {dmax}]"
        )

    return Splits(pretrain=pre, finetune=fin, test=tes, warmup_bars=warm)


def window_for_test_with_warmup(splits: Splits, df_index: pd.DatetimeIndex) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if len(df_index) < 2:
        raise ValueError("Need at least two timestamps to determine frequency.")
    freq = df_index[1] - df_index[0]
    warmup_delta = splits.warmup_bars * freq
    start = splits.test.start - warmup_delta
    if start < df_index.min():
        start = df_index.min()
    end = splits.test.end
    return start, end
