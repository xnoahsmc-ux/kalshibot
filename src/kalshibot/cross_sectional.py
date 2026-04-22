"""Cross-sectional signals. These look across multiple markets at once - e.g.
"within the NFL genre, rotate into the market whose mid is moving up fastest".

Unlike single-market strategies, these are computed once per timestamp over
the entire universe. They live outside the standard Strategy registry so we
can fold their probabilities into the ensemble as extra synthetic strategies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import features as F
from .data import MarketHistory


def _aligned_mid(histories: list[MarketHistory]) -> pd.DataFrame:
    """Align all histories to their shared index; fills missing bars forward.
    Returns a (time x ticker) dataframe of mid prices.
    """
    cols = {}
    for h in histories:
        cols[h.ticker] = h.mid
    df = pd.concat(cols, axis=1).ffill().bfill()
    return df


def cross_sectional_momentum(histories: list[MarketHistory], lookback: int = 20,
                             scale: float = 6.0) -> dict[str, pd.Series]:
    """For each market, compute a signal proportional to its rank of
    lookback momentum within the current universe.

    Positive rank (top movers) → higher P(YES); negative rank → lower.
    Probabilities are clipped into (0,1).
    """
    mid = _aligned_mid(histories)
    mom = mid - mid.shift(lookback)
    # Cross-sectional rank per row, centered at 0.
    ranks = mom.rank(axis=1, pct=True) - 0.5
    out: dict[str, pd.Series] = {}
    for c in ranks.columns:
        out[c] = F.sigmoid(ranks[c].fillna(0) * scale * 2)
    return out


def cross_sectional_reversion(histories: list[MarketHistory], lookback: int = 20,
                              scale: float = 4.0) -> dict[str, pd.Series]:
    """Fade the extremes: markets that have moved the most are most likely to
    revert in the short run.
    """
    mid = _aligned_mid(histories)
    mom = mid - mid.shift(lookback)
    ranks = mom.rank(axis=1, pct=True) - 0.5
    out: dict[str, pd.Series] = {}
    for c in ranks.columns:
        out[c] = F.sigmoid(-ranks[c].fillna(0) * scale * 2)
    return out


def pairs_mean_reversion(histories: list[MarketHistory], n: int = 50,
                         scale: float = 3.0) -> dict[str, pd.Series]:
    """Treat the cross-sectional mean as a synthetic peer. If a market has
    diverged above/below the peer it should revert.
    """
    mid = _aligned_mid(histories)
    peer = mid.mean(axis=1)
    out: dict[str, pd.Series] = {}
    for c in mid.columns:
        spread = mid[c] - peer
        z = F.zscore(spread, n)
        out[c] = F.sigmoid(-z * scale)
    return out
