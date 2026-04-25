"""100-day historical alpha engine.

Pulls real Kalshi candles for popular markets, runs walk-forward
backtesting, picks the strategies that survived OOS, and persists a
"champion ensemble" the live bot can use.

This is the focused profit-making layer on top of the raw 151-strategy
universe: instead of trusting all of them, we lean hard on whatever
actually printed money in the recent past.
"""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .backtester import evaluate_all, stack_pnls, stack_probs
from .combiner import Combination, best_combination
from .data import MarketHistory
from .strategies.registry import all_strategies
from .walkforward import walk_forward

if TYPE_CHECKING:
    from .kalshi_client import KalshiClient


@dataclass
class ChampionEnsemble:
    """Top strategies and the markets they make money on."""
    weights: dict[str, float]                 # name -> weight
    expected_sharpe: float
    expected_pnl: float
    oos_sharpe: float
    universe_size: int
    refit_at: float = field(default_factory=time.time)
    top_genres: list[str] = field(default_factory=list)
    top_strategies: list[tuple[str, float]] = field(default_factory=list)

    def to_combination(self) -> Combination:
        s = pd.Series(self.weights)
        return Combination(weights=s, sharpe=self.expected_sharpe,
                            expected_pnl=self.expected_pnl)

    def to_json(self) -> dict:
        return {
            "weights": self.weights,
            "expected_sharpe": self.expected_sharpe,
            "expected_pnl": self.expected_pnl,
            "oos_sharpe": self.oos_sharpe,
            "universe_size": self.universe_size,
            "refit_at": self.refit_at,
            "top_genres": self.top_genres,
            "top_strategies": self.top_strategies,
        }


CHAMPION_PATH = Path("data/champion.json")
CHAMPION_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_champion() -> ChampionEnsemble | None:
    if not CHAMPION_PATH.exists():
        return None
    try:
        d = json.loads(CHAMPION_PATH.read_text())
        return ChampionEnsemble(**d)
    except Exception:
        return None


def save_champion(c: ChampionEnsemble) -> None:
    CHAMPION_PATH.write_text(json.dumps(c.to_json(), indent=2))


def fit_champion(
    histories: list[MarketHistory],
    *,
    train_bars: int = 1500,
    test_bars: int = 300,
    top_k: int = 25,
    verbose: bool = False,
) -> ChampionEnsemble:
    """Walk-forward refit: pick the strategies whose OOS Sharpe is positive,
    then optimize a combination on the full sample using only those
    survivors. This is the core anti-overfit technique."""
    strats = all_strategies()
    if verbose:
        print(f"[alpha] {len(strats)} strategies x {len(histories)} markets")
    # Walk-forward on a subset to keep runtime manageable.
    wf = walk_forward(strats[:90], histories[:max(4, min(20, len(histories)))],
                       train_bars=min(train_bars, 600),
                       test_bars=min(test_bars, 200),
                       top_k=top_k)
    if verbose:
        print(f"[alpha] walk-forward OOS Sharpe: {wf.oos_sharpe:+.2f}")
    # Identify survivors by averaging fold-weights
    avg_w = wf.avg_weights.fillna(0.0)
    survivors = list(avg_w[avg_w > 0].sort_values(ascending=False).head(20).index)
    if not survivors:
        survivors = [s.name for s in strats[:20]]

    chosen = [s for s in strats if s.name in survivors]
    evals = evaluate_all(chosen, histories, include_cross_sectional=False)
    pnl = stack_pnls(evals)
    if pnl.empty:
        # Punt on weights — use uniform.
        weights = {s.name: 1.0 / len(survivors) for s in chosen}
        return ChampionEnsemble(weights=weights, expected_sharpe=0.0,
                                  expected_pnl=0.0, oos_sharpe=wf.oos_sharpe,
                                  universe_size=len(histories))
    probs = stack_probs(evals)
    outcomes_bars: list[float] = []
    for h in histories:
        if len(h.df) == 0:
            continue
        y = 1.0 if h.mid.iloc[-1] >= 0.5 else 0.0
        outcomes_bars.extend([y] * len(h.df))
    outcomes = pd.Series(outcomes_bars[:len(probs)])
    combo = best_combination(pnl, probs, outcomes, top_k=min(top_k, len(survivors)))

    # Per-genre PnL ranking (which niches the survivors made money on)
    from .genres import classify
    per_genre: dict[str, float] = {}
    for i, h in enumerate(histories):
        gn = classify(h.ticker).genre
        running = 0.0
        for name, runs in evals.items():
            running += float(combo.weights.get(name, 0.0)) * float(runs[i].pnl.sum())
        per_genre[gn] = per_genre.get(gn, 0.0) + running
    top_genres = sorted(per_genre.items(), key=lambda kv: -kv[1])[:5]

    weights = {n: float(w) for n, w in combo.weights.items() if float(w) > 0}
    top_strats = sorted(weights.items(), key=lambda kv: -kv[1])[:10]
    champion = ChampionEnsemble(
        weights=weights,
        expected_sharpe=float(combo.sharpe),
        expected_pnl=float(combo.expected_pnl),
        oos_sharpe=float(wf.oos_sharpe),
        universe_size=len(histories),
        top_genres=[g for g, _ in top_genres],
        top_strategies=[(n, w) for n, w in top_strats],
    )
    return champion


def fit_from_kalshi_history(
    client: "KalshiClient",
    *,
    markets: int = 40,
    lookback_hours: int = 24 * 100,
    period_minutes: int = 60,
    save: bool = True,
    verbose: bool = True,
) -> ChampionEnsemble | None:
    """Pull 100 days of real Kalshi candles for top markets, then fit."""
    from .history_fetch import fetch_universe
    universe = fetch_universe(
        client, limit=markets, candidate_pool=600,
        lookback_hours=lookback_hours, period_minutes=period_minutes,
        min_volume_24h=0, min_bars=20, verbose=verbose,
    )
    if len(universe) < 4:
        if verbose:
            print(f"[alpha] only {len(universe)} markets - need at least 4")
        return None
    if verbose:
        print(f"[alpha] fitting champion on {len(universe)} markets")
    champion = fit_champion(universe, verbose=verbose)
    if save:
        save_champion(champion)
        if verbose:
            print(f"[alpha] saved {CHAMPION_PATH} - "
                  f"OOS Sharpe {champion.oos_sharpe:+.2f}, "
                  f"{len(champion.weights)} active strategies")
    return champion
