import yaml
from pathlib import Path

from src.backtest.simulator import decide_side


def _load_model_cfg():
    cfg_path = Path("config/model.yaml")
    return yaml.safe_load(cfg_path.read_text())


def test_timeouts_match_horizon():
    cfg = _load_model_cfg()
    horizon = int(cfg.get("horizon_bars", cfg.get("horizon", 0)))
    tb = cfg.get("triple_barrier", {})
    assert int(tb.get("t_max", horizon)) == horizon


def test_sign_mapping_is_explicit():
    assert decide_side(0.8, 0.6, long_only=False) == 1
    assert decide_side(0.1, 0.6, long_only=False) == -1
