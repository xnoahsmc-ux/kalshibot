from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(2, n // 2)).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def bollinger(s: pd.Series, n: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    m = sma(s, n)
    sd = s.rolling(n, min_periods=max(2, n // 2)).std()
    return m, m + k * sd, m - k * sd


def donchian(s: pd.Series, n: int = 20) -> tuple[pd.Series, pd.Series]:
    hi = s.rolling(n, min_periods=max(2, n // 2)).max()
    lo = s.rolling(n, min_periods=max(2, n // 2)).min()
    return hi, lo


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    line = ema(s, fast) - ema(s, slow)
    sig = ema(line, signal)
    return line, sig


def zscore(s: pd.Series, n: int = 20) -> pd.Series:
    m = s.rolling(n, min_periods=max(2, n // 2)).mean()
    sd = s.rolling(n, min_periods=max(2, n // 2)).std().replace(0, np.nan)
    return ((s - m) / sd).fillna(0)


def realized_vol(s: pd.Series, n: int = 20) -> pd.Series:
    return s.pct_change().rolling(n, min_periods=max(2, n // 2)).std().fillna(0)


def momentum(s: pd.Series, n: int = 10) -> pd.Series:
    return (s - s.shift(n)).fillna(0)


def logit(p: pd.Series | float) -> pd.Series | float:
    if isinstance(p, pd.Series):
        p = p.clip(1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))
    p = min(max(p, 1e-4), 1 - 1e-4)
    return float(np.log(p / (1 - p)))


def sigmoid(x: pd.Series | float) -> pd.Series | float:
    if isinstance(x, pd.Series):
        return 1 / (1 + np.exp(-x))
    return 1 / (1 + float(np.exp(-x)))
