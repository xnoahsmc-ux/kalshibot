from kalshibot.backtester import evaluate_all
from kalshibot.cross_sectional import (
    cross_sectional_momentum,
    cross_sectional_reversion,
    pairs_mean_reversion,
)
from kalshibot.data import synthesize_history
from kalshibot.strategies.registry import all_strategies


def test_cross_sectional_signals_cover_all_markets():
    hs = [synthesize_history(ticker=f"T-{i}", n=200, seed=i) for i in range(5)]
    for fn in (cross_sectional_momentum, cross_sectional_reversion,
               pairs_mean_reversion):
        probs = fn(hs)
        assert set(probs.keys()) == {h.ticker for h in hs}
        for p in probs.values():
            assert p.between(0, 1).all()


def test_evaluate_all_includes_cross_sectional():
    strats = all_strategies()[:10]
    hs = [synthesize_history(ticker=f"T-{i}", n=200, seed=i) for i in range(4)]
    evals = evaluate_all(strats, hs, include_cross_sectional=True)
    names = set(evals.keys())
    assert "xs_mom_10" in names
    assert "xs_pairs_30" in names
    # Each synthetic has one eval per market
    assert all(len(evals[n]) == len(hs) for n in ["xs_mom_10", "xs_pairs_30"])
