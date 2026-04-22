"""Find the best weighted combination of strategies.

Two complementary methods:
  * `optimize_sharpe`  — picks weights that maximize backtested risk-adjusted PnL
  * `optimize_logloss` — picks weights that minimize calibration error of the
                        blended probability against realized outcomes

We constrain weights to the simplex (non-negative, sum to 1) plus a sparsity
penalty so the final ensemble is interpretable and not overfit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class Combination:
    weights: pd.Series       # indexed by strategy name
    sharpe: float
    expected_pnl: float

    def top(self, k: int = 10) -> pd.Series:
        return self.weights.sort_values(ascending=False).head(k)

    def blended_prob(self, prob_matrix: pd.DataFrame) -> pd.Series:
        w = self.weights.reindex(prob_matrix.columns).fillna(0.0).values
        return pd.Series(prob_matrix.values @ w, index=prob_matrix.index)


def _simplex_project(w: np.ndarray) -> np.ndarray:
    # Project arbitrary vector onto the probability simplex (Duchi et al. 2008).
    n = len(w)
    u = np.sort(w)[::-1]
    cssv = np.cumsum(u) - 1
    rho = np.where(u - cssv / (np.arange(n) + 1) > 0)[0][-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(w - theta, 0.0)


def optimize_sharpe(
    pnl_matrix: pd.DataFrame,
    l1: float = 0.001,
    seed: int = 0,
    n_restarts: int = 6,
) -> Combination:
    """Maximize Sharpe(w'P) subject to w >= 0, sum(w) = 1, with an L1 penalty
    that encourages sparsity. Solved by projected-gradient with random restarts.
    """
    P = pnl_matrix.values
    names = pnl_matrix.columns
    rng = np.random.default_rng(seed)

    def neg_sharpe(w: np.ndarray) -> float:
        r = P @ w
        sd = r.std()
        if sd <= 1e-12:
            return 1e6
        return -(r.mean() / sd) + l1 * np.abs(w).sum()

    best: tuple[float, np.ndarray] | None = None
    n = P.shape[1]
    for i in range(n_restarts):
        if i == 0:
            w0 = np.full(n, 1.0 / n)
        else:
            w0 = rng.dirichlet(np.full(n, 0.5))
        # SLSQP with simplex constraints.
        cons = [
            {"type": "eq", "fun": lambda w: w.sum() - 1.0},
        ]
        bounds = [(0.0, 1.0)] * n
        res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 200, "ftol": 1e-7})
        w = _simplex_project(res.x)
        v = neg_sharpe(w)
        if best is None or v < best[0]:
            best = (v, w)

    assert best is not None
    w = best[1]
    r = P @ w
    sharpe = float(r.mean() / r.std() * np.sqrt(252 * 390)) if r.std() > 0 else 0.0
    return Combination(
        weights=pd.Series(w, index=names),
        sharpe=sharpe,
        expected_pnl=float(r.mean()),
    )


def optimize_logloss(
    prob_matrix: pd.DataFrame,
    outcomes: pd.Series,
    l2: float = 1e-3,
) -> Combination:
    """Find weights on the simplex that minimize blended cross-entropy."""
    P = prob_matrix.values
    y = outcomes.values
    names = prob_matrix.columns
    n = P.shape[1]

    def loss(w: np.ndarray) -> float:
        p = np.clip(P @ w, 1e-4, 1 - 1e-4)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean() + l2 * (w * w).sum())

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * n
    res = minimize(loss, np.full(n, 1.0 / n), method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-8})
    w = _simplex_project(res.x)
    return Combination(
        weights=pd.Series(w, index=names),
        sharpe=float("nan"),
        expected_pnl=float("nan"),
    )


def select_top_k(
    pnl_matrix: pd.DataFrame,
    k: int = 20,
    metric: str = "sharpe",
) -> list[str]:
    """Pre-filter strategies before the heavier weight optimization."""
    if metric == "sharpe":
        means = pnl_matrix.mean()
        sds = pnl_matrix.std().replace(0, np.nan)
        score = (means / sds).fillna(-np.inf)
    elif metric == "mean":
        score = pnl_matrix.mean()
    else:
        raise ValueError(metric)
    return list(score.sort_values(ascending=False).head(k).index)


def best_combination(
    pnl_matrix: pd.DataFrame,
    prob_matrix: pd.DataFrame | None = None,
    outcomes: pd.Series | None = None,
    top_k: int = 30,
    blend: float = 0.5,
) -> Combination:
    """End-to-end: pick top-k by Sharpe, then optimize. Optionally blend with a
    log-loss-optimal vector if outcomes are provided.
    """
    keep = select_top_k(pnl_matrix, k=top_k)
    Pp = pnl_matrix[keep]
    sharpe_combo = optimize_sharpe(Pp)
    if prob_matrix is not None and outcomes is not None:
        Pq = prob_matrix[keep]
        ll_combo = optimize_logloss(Pq, outcomes)
        w = blend * sharpe_combo.weights + (1 - blend) * ll_combo.weights.reindex(
            sharpe_combo.weights.index
        ).fillna(0)
        w = w / w.sum()
        r = Pp.values @ w.values
        sharpe = float(r.mean() / r.std() * np.sqrt(252 * 390)) if r.std() > 0 else 0.0
        return Combination(weights=w, sharpe=sharpe, expected_pnl=float(r.mean()))
    # Pad weights back to full universe with zeros for dropped strategies.
    full = pd.Series(0.0, index=pnl_matrix.columns)
    full.loc[sharpe_combo.weights.index] = sharpe_combo.weights.values
    return Combination(weights=full, sharpe=sharpe_combo.sharpe,
                       expected_pnl=sharpe_combo.expected_pnl)
