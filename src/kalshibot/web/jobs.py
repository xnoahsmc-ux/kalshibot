"""Background jobs run from the web UI (full backtest, walk-forward)."""
from __future__ import annotations

import threading

import pandas as pd

from ..analytics import build_report
from ..backtester import evaluate_all, stack_pnls, stack_probs
from ..combiner import best_combination
from ..data import synthesize_history
from ..walkforward import walk_forward
from ..strategies.registry import all_strategies
from .state import AppState


def _synth_universe(n_markets: int, seed: int = 0):
    archetypes = [
        ("KXNFL-DEMO", 0.0, 0.025, 0.55),
        ("KXNBA-DEMO", 0.001, 0.020, 0.48),
        ("KXWEATHERHIGH-DEMO", -0.0005, 0.012, 0.40),
        ("KXWEATHERSNOW-DEMO", 0.0, 0.018, 0.35),
        ("KXPOLPRES-DEMO", 0.0008, 0.030, 0.52),
        ("KXSENATE-DEMO", -0.0003, 0.022, 0.45),
        ("KXCPI-DEMO", 0.0, 0.015, 0.50),
        ("KXFEDRATE-DEMO", 0.0002, 0.018, 0.42),
        ("KXBTC-DEMO", 0.001, 0.035, 0.60),
        ("KXETH-DEMO", -0.001, 0.030, 0.55),
        ("KXSPX-DEMO", 0.0005, 0.014, 0.55),
        ("KXOSCAR-DEMO", 0.0, 0.020, 0.30),
        ("KXAI-DEMO", 0.0008, 0.022, 0.65),
        ("KXSPACE-DEMO", 0.0, 0.025, 0.40),
        ("KXOIL-DEMO", -0.0006, 0.020, 0.50),
        ("KXMUSK-DEMO", 0.0, 0.030, 0.55),
    ]
    out = []
    for i in range(n_markets):
        ticker, drift, vol, start = archetypes[i % len(archetypes)]
        out.append(synthesize_history(
            ticker=f"{ticker}-{i:03d}", n=600, seed=seed + i,
            drift=drift, vol=vol, start_p=start,
        ))
    return out


def run_backtest(state: AppState, markets: int = 24, top_k: int = 30,
                 method: str = "hybrid", walk_forward_on: bool = True) -> str:
    job = state.register_job("backtest")

    def target() -> None:
        try:
            strats = all_strategies()
            state.update_job(job.id, progress=0.05, message=f"loaded {len(strats)} strategies")
            universe = _synth_universe(markets)
            state.update_job(job.id, progress=0.15, message=f"synthesized {len(universe)} markets")

            evals = evaluate_all(strats, universe)
            state.update_job(job.id, progress=0.6, message="evaluating strategies")

            pnl = stack_pnls(evals)
            probs = stack_probs(evals)
            outcomes_bars = []
            for h in universe:
                y = 1.0 if h.mid.iloc[-1] >= 0.5 else 0.0
                outcomes_bars.extend([y] * len(h.df))
            outcomes = pd.Series(outcomes_bars[:len(probs)])
            combo = best_combination(pnl, probs, outcomes, top_k=top_k, method=method)
            state.update_job(job.id, progress=0.8, message="optimizing combination")

            oos_sharpe = None
            if walk_forward_on:
                wf = walk_forward(strats[:60], universe[:min(8, len(universe))],
                                  train_bars=300, test_bars=100, top_k=20)
                oos_sharpe = wf.oos_sharpe

            report = build_report(evals, [h.ticker for h in universe], combo)
            if oos_sharpe is not None:
                report.per_genre.attrs["oos_sharpe"] = oos_sharpe
            state.set_report(report)
            state.update_job(job.id, progress=1.0, message="done", finished=job.started + 1)
        except Exception as e:
            state.update_job(job.id, error=str(e), message=f"error: {e}", finished=job.started + 1)

    threading.Thread(target=target, daemon=True).start()
    return job.id
