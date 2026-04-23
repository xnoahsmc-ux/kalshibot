"""Online learner that updates per-strategy realized hit rates from
settled trades, then adjusts ensemble weights in response.

Each settled trade contributes (p_strategy_said_YES, outcome) -> the
strategies that "voted" closest to the actual outcome get a small weight
boost; the ones that voted opposite get a small penalty.

The update is deliberately conservative (small learning rate) so the
ensemble doesn't overfit to the last few trades and so strategy weights
stay near their backtested base.
"""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .combiner import Combination


@dataclass
class StrategyStats:
    name: str
    trades_counted: int = 0
    correct_calls: float = 0.0         # weighted by |p-0.5| (confidence)
    wrong_calls: float = 0.0
    realized_edge: float = 0.0         # running log-loss-like score


class OnlineLearner:
    """Keeps per-strategy realized hit stats on disk and returns an adjusted
    weight vector."""

    def __init__(self, path: str | Path = "data/learner.json",
                  learning_rate: float = 0.02) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lr = learning_rate
        self._lock = threading.Lock()
        self.stats: dict[str, StrategyStats] = {}
        self.updates = 0
        self.last_updated = 0.0
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text())
            self.updates = payload.get("updates", 0)
            self.last_updated = payload.get("last_updated", 0.0)
            for name, st in payload.get("stats", {}).items():
                self.stats[name] = StrategyStats(name=name, **st)
        except Exception:
            pass

    def _save(self) -> None:
        data = {
            "updates": self.updates,
            "last_updated": self.last_updated,
            "stats": {k: {"trades_counted": v.trades_counted,
                           "correct_calls": v.correct_calls,
                           "wrong_calls": v.wrong_calls,
                           "realized_edge": v.realized_edge}
                       for k, v in self.stats.items()},
        }
        self.path.write_text(json.dumps(data, indent=2))

    def update_from_trade(
        self,
        *,
        strategy_probs: dict[str, float],       # strategy name -> P(YES) at entry
        outcome: int,                             # 1 for YES, 0 for NO
        weight: float = 1.0,                      # importance of this trade
    ) -> None:
        with self._lock:
            for name, p in strategy_probs.items():
                st = self.stats.setdefault(name, StrategyStats(name=name))
                st.trades_counted += 1
                # Reward: (p aligned with outcome) scaled by conviction (|p-0.5|)
                conviction = abs(p - 0.5) * 2
                if (p > 0.5 and outcome == 1) or (p < 0.5 and outcome == 0):
                    st.correct_calls += conviction * weight
                else:
                    st.wrong_calls += conviction * weight
                # Log-score-like running measure
                p_clip = min(max(p, 1e-3), 1 - 1e-3)
                st.realized_edge += (
                    math.log(p_clip) if outcome == 1 else math.log(1 - p_clip)
                ) * weight
            self.updates += 1
            self.last_updated = time.time()
            self._save()

    def score_for(self, name: str) -> float:
        st = self.stats.get(name)
        if not st or st.trades_counted == 0:
            return 0.5
        c, w = st.correct_calls, st.wrong_calls
        total = c + w
        if total <= 0:
            return 0.5
        # Bayesian shrinkage toward 0.5 with a 10-trade prior.
        prior_n = 10.0
        prior_p = 0.5
        return (c + prior_p * prior_n) / (total + prior_n)

    def adjust_combination(self, base: Combination) -> Combination:
        """Multiplicatively tilt `base.weights` by each strategy's realized
        score, then renormalize. With few trades this barely changes weights."""
        weights = base.weights.copy()
        if weights.sum() == 0:
            return base
        multipliers = {name: (0.5 + 2 * self.lr * (self.score_for(name) - 0.5))
                        for name in weights.index}
        for name, m in multipliers.items():
            weights[name] = max(0.0, float(weights[name]) * m)
        total = weights.sum()
        if total > 0:
            weights = weights / total
        return Combination(weights=weights, sharpe=base.sharpe,
                           expected_pnl=base.expected_pnl)

    def top_strategies(self, k: int = 10) -> list[tuple[str, float, int]]:
        rows = []
        for name, st in self.stats.items():
            if st.trades_counted <= 0:
                continue
            rows.append((name, self.score_for(name), st.trades_counted))
        rows.sort(key=lambda r: -r[1])
        return rows[:k]

    def summary(self) -> dict:
        return {
            "total_updates": self.updates,
            "strategies_tracked": len([s for s in self.stats.values()
                                         if s.trades_counted > 0]),
            "last_updated": self.last_updated,
            "top": [{"name": n, "score": s, "trades": t}
                     for n, s, t in self.top_strategies(10)],
        }
