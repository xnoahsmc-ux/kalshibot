from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cross_sectional import (
    cross_sectional_momentum,
    cross_sectional_reversion,
    pairs_mean_reversion,
)
from .data import MarketHistory
from .strategies.registry import Strat


@dataclass
class StrategyEval:
    name: str
    probs: pd.Series       # probability-of-YES per bar
    pnl: pd.Series         # per-bar realized PnL (in price units)
    sharpe: float
    hit_rate: float
    log_loss: float
    edge_mean: float


def _signed_position(probs: pd.Series, mid: pd.Series, threshold: float) -> pd.Series:
    """Convert prob-of-YES into a signed position vs. market mid.

    Position is in [-1,1]. +1 = full long YES, -1 = full long NO.
    Below `threshold` of edge we don't take a position.
    """
    edge = (probs - mid)
    pos = edge.where(edge.abs() >= threshold, 0.0)
    return pos.clip(-1.0, 1.0)


def _bar_pnl(pos: pd.Series, mid: pd.Series, fee_bps: float) -> pd.Series:
    # Use prior-bar position so we don't peek; PnL = pos_{t-1} * (mid_t - mid_{t-1}).
    rets = mid.diff().fillna(0.0)
    raw = pos.shift(1).fillna(0.0) * rets
    turn = pos.diff().abs().fillna(0.0)
    cost = turn * (fee_bps / 1e4)
    return raw - cost


def evaluate_strategy(
    strat: Strat,
    h: MarketHistory,
    threshold: float = 0.02,
    fee_bps: float = 5.0,
) -> StrategyEval:
    probs = strat.predict_series(h)
    mid = h.mid
    pos = _signed_position(probs, mid, threshold)
    pnl = _bar_pnl(pos, mid, fee_bps)
    # Resolution: terminal mid as a 0/1 proxy for YES outcome
    terminal = float(mid.iloc[-1])
    y = 1.0 if terminal >= 0.5 else 0.0
    p_clip = probs.clip(1e-3, 1 - 1e-3)
    ll = float(-(y * np.log(p_clip) + (1 - y) * np.log(1 - p_clip)).mean())
    sd = float(pnl.std())
    sharpe = float(pnl.mean() / sd * np.sqrt(252 * 390)) if sd > 0 else 0.0
    hit = float((np.sign(pnl) > 0).mean())
    edge = float((probs - mid).abs().mean())
    return StrategyEval(strat.name, probs, pnl, sharpe, hit, ll, edge)


def _eval_from_probs(name: str, probs_per_ticker: dict[str, pd.Series],
                     histories: list[MarketHistory], threshold: float,
                     fee_bps: float) -> list[StrategyEval]:
    """Evaluate a synthetic strategy whose probabilities come from a
    cross-sectional signal (one series per market)."""
    evals: list[StrategyEval] = []
    for h in histories:
        probs = probs_per_ticker.get(h.ticker)
        if probs is None:
            probs = pd.Series(0.5, index=h.df.index)
        probs = probs.reindex(h.df.index).ffill().fillna(0.5).clip(0.01, 0.99)
        mid = h.mid
        pos = _signed_position(probs, mid, threshold)
        pnl = _bar_pnl(pos, mid, fee_bps)
        terminal = float(mid.iloc[-1])
        y = 1.0 if terminal >= 0.5 else 0.0
        p_clip = probs.clip(1e-3, 1 - 1e-3)
        ll = float(-(y * np.log(p_clip) + (1 - y) * np.log(1 - p_clip)).mean())
        sd = float(pnl.std())
        sharpe = float(pnl.mean() / sd * np.sqrt(252 * 390)) if sd > 0 else 0.0
        hit = float((np.sign(pnl) > 0).mean())
        edge = float((probs - mid).abs().mean())
        evals.append(StrategyEval(name, probs, pnl, sharpe, hit, ll, edge))
    return evals


def evaluate_all(
    strats: list[Strat],
    histories: list[MarketHistory],
    threshold: float = 0.02,
    fee_bps: float = 5.0,
    include_cross_sectional: bool = True,
) -> dict[str, list[StrategyEval]]:
    """Evaluate each strategy across every market. Indexed by strategy name.

    Cross-sectional signals (momentum, reversion, pairs) are appended as
    synthetic strategies so the combiner can weight them alongside the
    per-market ones.
    """
    out: dict[str, list[StrategyEval]] = {s.name: [] for s in strats}
    for h in histories:
        for s in strats:
            out[s.name].append(evaluate_strategy(s, h, threshold, fee_bps))

    if include_cross_sectional and len(histories) >= 2:
        xs_families = [
            ("xs_mom_10", cross_sectional_momentum(histories, lookback=10)),
            ("xs_mom_30", cross_sectional_momentum(histories, lookback=30)),
            ("xs_mom_60", cross_sectional_momentum(histories, lookback=60)),
            ("xs_rev_10", cross_sectional_reversion(histories, lookback=10)),
            ("xs_rev_30", cross_sectional_reversion(histories, lookback=30)),
            ("xs_pairs_30", pairs_mean_reversion(histories, n=30)),
            ("xs_pairs_60", pairs_mean_reversion(histories, n=60)),
        ]
        for name, probs in xs_families:
            out[name] = _eval_from_probs(name, probs, histories, threshold, fee_bps)

    return out


def stack_pnls(evals: dict[str, list[StrategyEval]]) -> pd.DataFrame:
    """Stack per-strategy PnL into a single matrix (rows=time across markets)."""
    cols = {}
    for name, runs in evals.items():
        if not runs:
            continue
        pnl = pd.concat([r.pnl.reset_index(drop=True) for r in runs], ignore_index=True)
        cols[name] = pnl
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).fillna(0.0)


def stack_probs(evals: dict[str, list[StrategyEval]]) -> pd.DataFrame:
    cols = {}
    for name, runs in evals.items():
        if not runs:
            continue
        p = pd.concat([r.probs.reset_index(drop=True) for r in runs], ignore_index=True)
        cols[name] = p
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).fillna(0.5)
