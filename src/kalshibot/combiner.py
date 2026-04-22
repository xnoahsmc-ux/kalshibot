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


def optimize_risk_parity(pnl_matrix: pd.DataFrame) -> Combination:
    """Equal risk-contribution weighting: each chosen strategy contributes the
    same share of portfolio variance. Much more robust than Sharpe
    maximization because it ignores point estimates of returns, which are
    notoriously noisy.
    """
    P = pnl_matrix.values
    names = pnl_matrix.columns
    cov = np.cov(P.T) + 1e-8 * np.eye(P.shape[1])
    n = P.shape[1]

    def loss(w: np.ndarray) -> float:
        w = np.maximum(w, 1e-8)
        port_var = w @ cov @ w
        mrc = cov @ w
        rc = w * mrc
        target = port_var / n
        return float(((rc - target) ** 2).sum())

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * n
    res = minimize(loss, np.full(n, 1.0 / n), method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 400, "ftol": 1e-10})
    w = _simplex_project(res.x)
    r = P @ w
    sharpe = float(r.mean() / r.std() * np.sqrt(252 * 390)) if r.std() > 0 else 0.0
    return Combination(weights=pd.Series(w, index=names), sharpe=sharpe,
                       expected_pnl=float(r.mean()))


def shrink_by_correlation(pnl_matrix: pd.DataFrame, weights: pd.Series,
                          corr_cap: float = 0.85) -> pd.Series:
    """De-duplicate highly correlated strategies: when two are >corr_cap the
    one with the lower standalone Sharpe gets its weight folded into the
    other. Stops the ensemble from loading up on 10 near-identical SMA
    crossovers.
    """
    keep = [c for c in weights.index if weights[c] > 0]
    if len(keep) < 2:
        return weights
    sub = pnl_matrix[keep]
    corr = sub.corr().fillna(0.0)
    sharpes = (sub.mean() / sub.std().replace(0, np.nan)).fillna(-np.inf)
    w = weights.copy()
    for a in keep:
        if w[a] == 0:
            continue
        for b in keep:
            if a == b or w[b] == 0:
                continue
            if corr.loc[a, b] > corr_cap:
                loser = a if sharpes[a] < sharpes[b] else b
                winner = b if loser == a else a
                w[winner] += w[loser]
                w[loser] = 0.0
    total = w.sum()
    return w / total if total > 0 else w


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
    method: str = "hybrid",            # "sharpe" | "risk_parity" | "hybrid"
    dedupe_corr_cap: float = 0.9,
) -> Combination:
    """End-to-end: pick top-k by Sharpe, optimize, dedupe by correlation, and
    optionally blend with a log-loss-optimal vector if outcomes are provided.

    `method="hybrid"` averages the Sharpe-optimal and risk-parity weight
    vectors. Risk parity is much more robust to noise in mean estimates, so
    blending the two tends to outperform pure Sharpe maximization in OOS.
    """
    keep = select_top_k(pnl_matrix, k=top_k)
    Pp = pnl_matrix[keep]

    sharpe_combo = optimize_sharpe(Pp)
    if method == "risk_parity":
        core = optimize_risk_parity(Pp)
    elif method == "hybrid":
        rp = optimize_risk_parity(Pp)
        w = 0.5 * sharpe_combo.weights + 0.5 * rp.weights
        w = w / w.sum()
        r = Pp.values @ w.values
        sharpe = float(r.mean() / r.std() * np.sqrt(252 * 390)) if r.std() > 0 else 0.0
        core = Combination(weights=w, sharpe=sharpe, expected_pnl=float(r.mean()))
    else:
        core = sharpe_combo

    if prob_matrix is not None and outcomes is not None:
        Pq = prob_matrix[keep]
        ll_combo = optimize_logloss(Pq, outcomes)
        w = blend * core.weights + (1 - blend) * ll_combo.weights.reindex(
            core.weights.index
        ).fillna(0)
        w = w / w.sum()
        core = Combination(weights=w, sharpe=core.sharpe, expected_pnl=core.expected_pnl)

    deduped = shrink_by_correlation(Pp, core.weights, corr_cap=dedupe_corr_cap)
    r = Pp.values @ deduped.values
    sharpe = float(r.mean() / r.std() * np.sqrt(252 * 390)) if r.std() > 0 else 0.0

    full = pd.Series(0.0, index=pnl_matrix.columns)
    full.loc[deduped.index] = deduped.values
    return Combination(weights=full, sharpe=sharpe, expected_pnl=float(r.mean()))
