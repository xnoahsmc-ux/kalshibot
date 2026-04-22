from kalshibot.data import synthesize_history
from kalshibot.strategies.registry import all_strategies


def test_at_least_100_strategies_load():
    strats = all_strategies()
    assert len(strats) >= 100
    assert len({s.name for s in strats}) == len(strats)


def test_every_strategy_emits_valid_probabilities():
    h = synthesize_history(n=300, seed=42)
    for s in all_strategies():
        sig = s.predict(h)
        assert 0.0 <= sig.prob_yes <= 1.0, s.name
        assert 0.0 <= sig.confidence <= 1.0, s.name
        series = s.predict_series(h)
        assert series.between(0.0, 1.0).all(), s.name
        assert series.notna().all(), s.name
