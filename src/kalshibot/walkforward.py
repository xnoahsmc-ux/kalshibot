"""Walk-forward out-of-sample backtesting.

The in-sample fit on *all* data we used before is optimistic - the combiner
will always pick the strategies that worked best on that exact slice. This
module splits each market's bars into rolling train/test windows, fits the
ensemble on the train slice, and evaluates on the held-out test slice, then
stitches the OOS results together. The reported Sharpe/PnL is much closer to
what the bot would have earned in live trading.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtester import evaluate_strategy, stack_pnls
from .combiner import best_combination
from .data import MarketHistory
from .strategies.registry import Strat


@dataclass
class OOSFold:
    train_end: int
    test_end: int
    weights: pd.Series
    test_pnl: pd.Series


@dataclass
class WalkForwardResult:
    folds: list[OOSFold]
    oos_pnl: pd.Series
    oos_sharpe: float
    avg_weights: pd.Series


def _fit_combo(strats: list[Strat], histories_slice: list[MarketHistory],
               top_k: int, threshold: float, fee_bps: float) -> pd.Series:
    evals = {s.name: [evaluate_strategy(s, h, threshold, fee_bps) for h in histories_slice]
             for s in strats}
    pnl = stack_pnls(evals)
    combo = best_combination(pnl, top_k=top_k)
    return combo.weights


def _pnl_with_weights(strats: list[Strat], histories_slice: list[MarketHistory],
                      weights: pd.Series, threshold: float, fee_bps: float) -> pd.Series:
    running: pd.Series | None = None
    for s in strats:
        w = float(weights.get(s.name, 0.0))
        if w == 0:
            continue
        parts = [evaluate_strategy(s, h, threshold, fee_bps).pnl.reset_index(drop=True)
                 for h in histories_slice]
        full = pd.concat(parts, ignore_index=True) * w
        running = full if running is None else running.add(full, fill_value=0.0)
    return running if running is not None else pd.Series(dtype=float)


def walk_forward(
    strats: list[Strat],
    histories: list[MarketHistory],
    train_bars: int = 300,
    test_bars: int = 100,
    top_k: int = 30,
    threshold: float = 0.02,
    fee_bps: float = 5.0,
) -> WalkForwardResult:
    """Split each market on the same bar indices and fold them together."""
    min_len = min(len(h.df) for h in histories)
    folds: list[OOSFold] = []
    all_oos: list[pd.Series] = []
    all_weights: list[pd.Series] = []
    start = 0
    while start + train_bars + test_bars <= min_len:
        train_slice = [MarketHistory(h.ticker, h.df.iloc[start:start + train_bars].copy())
                       for h in histories]
        test_slice = [MarketHistory(
            h.ticker, h.df.iloc[start + train_bars:start + train_bars + test_bars].copy()
        ) for h in histories]
        w = _fit_combo(strats, train_slice, top_k, threshold, fee_bps)
        test_pnl = _pnl_with_weights(strats, test_slice, w, threshold, fee_bps)
        folds.append(OOSFold(train_end=start + train_bars,
                             test_end=start + train_bars + test_bars,
                             weights=w, test_pnl=test_pnl))
        all_oos.append(test_pnl)
        all_weights.append(w)
        start += test_bars

    oos_pnl = (pd.concat(all_oos, ignore_index=True) if all_oos
               else pd.Series(dtype=float))
    sharpe = float(oos_pnl.mean() / oos_pnl.std() * np.sqrt(252 * 390)) if len(oos_pnl) and oos_pnl.std() > 0 else 0.0
    if all_weights:
        avg = pd.concat(all_weights, axis=1).fillna(0.0).mean(axis=1)
    else:
        avg = pd.Series(dtype=float)
    return WalkForwardResult(folds=folds, oos_pnl=oos_pnl, oos_sharpe=sharpe, avg_weights=avg)
