"""Regime detection. Classifies each bar as trending or mean-reverting so
strategies can be gated appropriately.

Uses two cheap signals:
  * Rolling autocorrelation of returns — positive → trending, negative → MR.
  * Variance ratio (Lo-MacKinlay): VR > 1 ~ trending, VR < 1 ~ mean-reverting.

Regime in {-1, 0, +1}: mean-reverting, mixed, trending.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def autocorr(returns: pd.Series, n: int = 30, lag: int = 1) -> pd.Series:
    def _ac(x: np.ndarray) -> float:
        if len(x) < lag + 2 or x.std() == 0:
            return 0.0
        a = x[:-lag]
        b = x[lag:]
        sa, sb = a.std(), b.std()
        if sa == 0 or sb == 0:
            return 0.0
        return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))
    return returns.rolling(n, min_periods=lag + 2).apply(_ac, raw=True).fillna(0.0)


def variance_ratio(prices: pd.Series, n: int = 30, k: int = 5) -> pd.Series:
    r1 = prices.pct_change().fillna(0.0)
    rk = prices.pct_change(k).fillna(0.0)
    v1 = r1.rolling(n, min_periods=max(3, n // 2)).var()
    vk = rk.rolling(n, min_periods=max(3, n // 2)).var()
    vr = (vk / (k * v1.replace(0, np.nan))).fillna(1.0)
    return vr


def regime_score(prices: pd.Series, n: int = 30) -> pd.Series:
    """Continuous score in [-1, 1]: +1 trending, -1 mean-reverting."""
    r = prices.pct_change().fillna(0.0)
    ac = autocorr(r, n=n).clip(-1, 1)
    vr = variance_ratio(prices, n=n) - 1.0
    vr_n = (vr / (vr.abs().rolling(n, min_periods=5).mean().replace(0, np.nan))).fillna(0).clip(-1, 1)
    return (0.5 * ac + 0.5 * vr_n).clip(-1, 1)


def regime_label(prices: pd.Series, n: int = 30, threshold: float = 0.1) -> pd.Series:
    s = regime_score(prices, n=n)
    out = pd.Series(0, index=s.index, dtype=int)
    out[s > threshold] = 1
    out[s < -threshold] = -1
    return out
