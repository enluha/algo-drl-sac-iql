import numpy as np, pandas as pd

def _logret(s: pd.Series, k: int) -> pd.Series:
    return np.log(s / s.shift(k))

def _ema(s, span): return s.ewm(span=span, adjust=False).mean()
def _sma(s, w):   return s.rolling(w, min_periods=w).mean()
def _rsi(close, n=14):
    d = close.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    rs = _ema(up, n) / (_ema(dn, n) + 1e-12)
    return 100 - 100/(1+rs)

def _atr(h, l, c, w=14):
    pc = c.shift(1); tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(w, min_periods=w).mean()

def _parkinson(h,l,w=20):
    rs = (np.log(h/l))**2
    return ((rs.rolling(w, min_periods=w).mean()) / (4*np.log(2))).pow(0.5)

def _donchian_pos(c,w=20):
    hh = c.rolling(w, min_periods=w).max(); ll = c.rolling(w, min_periods=w).min(); rng = (hh-ll).replace(0,np.nan)
    return (c-ll)/rng

def _cyc(idx):
    hod, dow = idx.hour.values, idx.dayofweek.values
    return pd.DataFrame({
        "hod_sin": np.sin(2*np.pi*hod/24), "hod_cos": np.cos(2*np.pi*hod/24),
        "dow_sin": np.sin(2*np.pi*dow/7),  "dow_cos": np.cos(2*np.pi*dow/7),
    }, index=idx)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, dayfirst=True, utc=True, errors="coerce")
    else:
        df = df.copy()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(None)
    df = df[df.index.notna()]
    c,h,l,o = df["close"], df["high"], df["low"], df["open"]
    X = pd.DataFrame(index=df.index)
    for k in (1,3,6,12,24): X[f"ret_{k}"] = _logret(c,k)
    X["ewma_vol_20"] = X["ret_1"].ewm(span=20, adjust=False).std()
    # trend/momentum
    X["rsi_6"], X["rsi_14"], X["rsi_21"] = _rsi(c,6), _rsi(c,14), _rsi(c,21)
    X["macd_12_26"] = _ema(c,12) - _ema(c,26)
    for w in (24,48,96): X[f"ma_slope_{w}"] = _sma(c,w).diff()/(c.shift(1).abs()+1e-12)
    X["pct_above_ma96"] = c/_sma(c,96) - 1.0
    # range/vol
    X["atr_14"] = _atr(h,l,c,14); X["parkinson_20"] = _parkinson(h,l,20)
    X["donchian_20"] = _donchian_pos(c,20)
    X["range_close"] = (h-l)/(c+1e-12)
    # volume
    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce")
        X["vol_z"] = (vol - vol.rolling(96, min_periods=20).mean()) / (vol.rolling(96, min_periods=20).std() + 1e-12)
        X["vol_chg"] = vol.pct_change()
        if "taker_buy_volume" in df.columns:
            tb = pd.to_numeric(df["taker_buy_volume"], errors="coerce")
            X["taker_pressure"] = ((tb - (vol - tb)) / (vol.replace(0,np.nan))).clip(-1,1)
    # session
    X = X.join(_cyc(df.index))
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = X.clip(lower=-1e6, upper=1e6)
    return X.astype("float32")
