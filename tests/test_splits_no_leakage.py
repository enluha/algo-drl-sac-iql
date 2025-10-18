import pandas as pd
import yaml

from src.utils.splits import load_splits, window_for_test_with_warmup


def test_splits_chronology():
    cfg = {"walkforward": yaml.safe_load(open("config/walkforward.yaml", "r", encoding="utf-8"))}
    idx = pd.date_range("2024-06-10", "2025-10-16", freq='h')
    sp = load_splits(cfg, idx, "1h")
    assert sp.pretrain.end < sp.finetune.start <= sp.finetune.end < sp.test.start


def test_test_window_has_warmup_only():
    cfg = {"walkforward": yaml.safe_load(open("config/walkforward.yaml", "r", encoding="utf-8"))}
    idx = pd.date_range("2024-06-10", "2025-10-16", freq='h')
    sp = load_splits(cfg, idx, "1h")
    w0, w1 = window_for_test_with_warmup(sp, idx)
    assert w0 <= sp.test.start
    assert w1 == sp.test.end
