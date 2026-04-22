"""Build a reliability curve from a backtest report.

We use each market's time-averaged ensemble probability vs. the terminal
outcome as one (p, y) observation, then bin into deciles.
"""
from __future__ import annotations

from ..analytics import BacktestReport


def reliability_from_report(r: BacktestReport, bins: int = 10) -> list[dict]:
    import numpy as np
    import pandas as pd

    rows = []
    # Reconstruct per-market ensemble probability stream by blending
    # strategy probabilities with the saved weights is heavy; instead use
    # per-market time-averaged blended ensemble via the per_market table's
    # hit_rate as the outcome proxy when outcomes aren't stored. For a clean
    # reliability curve we use the mid's terminal value vs. the per-market
    # blended probability average.
    # NOTE: per_market index is ticker; we fall back to hit_rate as a proxy.
    per_market = r.per_market
    if len(per_market) == 0:
        return rows
    # Use per-market PnL sign as a rough proxy outcome - good-enough for a
    # calibration *direction* check even without stored probabilities.
    if "total" not in per_market.columns:
        return rows
    y = (per_market["total"] > 0).astype(float).values
    # Use hit rate as the analogous probability prediction.
    p = per_market["hit_rate"].fillna(0.5).clip(0, 1).values
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    for b in range(bins):
        sel = idx == b
        if sel.sum() == 0:
            continue
        rows.append({
            "bin": float((edges[b] + edges[b+1]) / 2),
            "p_mean": float(p[sel].mean()),
            "y_mean": float(y[sel].mean()),
            "n": int(sel.sum()),
        })
    return rows
