"""Aggregations on top of the backtester results.

Produces the data tables the dashboard renders: per-strategy stats, per-genre
stats, equity curves, drawdowns, and the optimal weighted ensemble.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtester import StrategyEval
from .combiner import Combination
from .genres import classify


def _stats_from_pnl(pnl: pd.Series) -> dict:
    if len(pnl) == 0:
        return {"total": 0.0, "mean": 0.0, "sharpe": 0.0, "sortino": 0.0,
                "calmar": 0.0, "max_dd": 0.0, "hit_rate": 0.0, "n_bars": 0}
    eq = pnl.cumsum()
    roll_max = eq.cummax()
    dd_series = (eq - roll_max)
    dd = float(dd_series.min())
    sd = float(pnl.std())
    mean = float(pnl.mean())
    # Annualization factor matching the rest of the codebase.
    ann = np.sqrt(252 * 390)
    sharpe = mean / sd * ann if sd > 0 else 0.0
    # Sortino uses only downside deviation.
    downside = pnl[pnl < 0]
    dsd = float(downside.std()) if len(downside) else 0.0
    sortino = mean / dsd * ann if dsd > 0 else 0.0
    calmar = (eq.iloc[-1] / abs(dd)) if dd < 0 else 0.0
    return {
        "total": float(eq.iloc[-1]),
        "mean": mean,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "max_dd": dd,
        "hit_rate": float((pnl > 0).mean()),
        "n_bars": int(len(pnl)),
    }


@dataclass
class BacktestReport:
    per_strategy: pd.DataFrame
    per_genre: pd.DataFrame
    per_market: pd.DataFrame
    equity_curve: pd.Series          # for the chosen ensemble
    strategy_equity: pd.DataFrame    # per-strategy cumulative PnL
    genre_equity: pd.DataFrame       # per-genre cumulative PnL of best ensemble
    combo: Combination
    best_strategy_per_genre: dict[str, str]


def build_report(
    evals: dict[str, list[StrategyEval]],
    market_tickers: list[str],
    combo: Combination,
) -> BacktestReport:
    rows = []
    for name, runs in evals.items():
        all_pnl = pd.concat([r.pnl.reset_index(drop=True) for r in runs], ignore_index=True)
        s = _stats_from_pnl(all_pnl)
        s["strategy"] = name
        rows.append(s)
    per_strategy = pd.DataFrame(rows).set_index("strategy").sort_values("sharpe", ascending=False)

    market_genre = {t: classify(t).genre for t in market_tickers}

    market_rows = []
    for i, ticker in enumerate(market_tickers):
        # Use the chosen ensemble PnL for this market
        ens_pnl = None
        for name, runs in evals.items():
            w = float(combo.weights.get(name, 0.0))
            if w == 0:
                continue
            p = runs[i].pnl.reset_index(drop=True) * w
            ens_pnl = p if ens_pnl is None else ens_pnl.add(p, fill_value=0.0)
        if ens_pnl is None:
            ens_pnl = pd.Series(dtype=float)
        s = _stats_from_pnl(ens_pnl)
        s["market"] = ticker
        s["genre"] = market_genre[ticker]
        market_rows.append(s)
    per_market = pd.DataFrame(market_rows).set_index("market")

    per_genre = (
        per_market.groupby("genre")
        .agg(total=("total", "sum"),
             mean=("mean", "mean"),
             sharpe=("sharpe", "mean"),
             max_dd=("max_dd", "min"),
             hit_rate=("hit_rate", "mean"),
             n_markets=("total", "count"))
        .sort_values("sharpe", ascending=False)
    )

    # Best single strategy per genre
    best_per_genre: dict[str, str] = {}
    for genre in per_genre.index:
        gtickers = [t for t, g in market_genre.items() if g == genre]
        gidx = [market_tickers.index(t) for t in gtickers]
        scores = {}
        for name, runs in evals.items():
            pnl = pd.concat([runs[i].pnl.reset_index(drop=True) for i in gidx], ignore_index=True)
            sd = pnl.std()
            scores[name] = (pnl.mean() / sd * np.sqrt(252 * 390)) if sd > 0 else 0.0
        best_per_genre[genre] = max(scores, key=scores.get) if scores else ""

    # Equity curves
    ens_global = None
    strategy_equity_cols = {}
    for name, runs in evals.items():
        full = pd.concat([r.pnl.reset_index(drop=True) for r in runs], ignore_index=True)
        strategy_equity_cols[name] = full.cumsum()
        w = float(combo.weights.get(name, 0.0))
        if w == 0:
            continue
        contrib = full * w
        ens_global = contrib if ens_global is None else ens_global.add(contrib, fill_value=0.0)
    equity_curve = (ens_global if ens_global is not None else pd.Series(dtype=float)).cumsum()
    strategy_equity = pd.DataFrame(strategy_equity_cols)

    # Per-genre equity for the ensemble
    genre_equity_cols = {}
    for genre in per_genre.index:
        gtickers = [t for t, g in market_genre.items() if g == genre]
        gidx = [market_tickers.index(t) for t in gtickers]
        running = None
        for name, runs in evals.items():
            w = float(combo.weights.get(name, 0.0))
            if w == 0:
                continue
            pnl = pd.concat([runs[i].pnl.reset_index(drop=True) for i in gidx], ignore_index=True) * w
            running = pnl if running is None else running.add(pnl, fill_value=0.0)
        genre_equity_cols[genre] = (running if running is not None else pd.Series(dtype=float)).cumsum()
    genre_equity = pd.DataFrame(genre_equity_cols)

    return BacktestReport(
        per_strategy=per_strategy,
        per_genre=per_genre,
        per_market=per_market,
        equity_curve=equity_curve,
        strategy_equity=strategy_equity,
        genre_equity=genre_equity,
        combo=combo,
        best_strategy_per_genre=best_per_genre,
    )
