from kalshibot.backtester import evaluate_all, stack_pnls, stack_probs
from kalshibot.combiner import best_combination
from kalshibot.data import synthesize_history
from kalshibot.strategies.registry import all_strategies


def _universe():
    return [
        synthesize_history(ticker=f"DEMO-{i}", n=300, seed=i, drift=(-1) ** i * 0.001)
        for i in range(4)
    ]


def test_backtest_produces_pnl_and_probs():
    strats = all_strategies()[:25]
    universe = _universe()
    evals = evaluate_all(strats, universe)
    p = stack_pnls(evals)
    q = stack_probs(evals)
    assert p.shape[1] == len(strats)
    assert q.shape == p.shape
    assert q.min().min() >= 0.0 and q.max().max() <= 1.0


def test_combiner_returns_simplex_weights():
    strats = all_strategies()[:25]
    evals = evaluate_all(strats, _universe())
    p = stack_pnls(evals)
    combo = best_combination(p, top_k=10)
    assert abs(combo.weights.sum() - 1.0) < 1e-6
    assert (combo.weights >= -1e-9).all()
    # Sparsity: at least some weights are zero (top-k filter dropped most).
    assert (combo.weights == 0).sum() > 0
