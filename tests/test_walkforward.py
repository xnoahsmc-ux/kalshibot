from kalshibot.data import synthesize_history
from kalshibot.strategies.registry import all_strategies
from kalshibot.walkforward import walk_forward


def test_walk_forward_produces_oos_pnl():
    strats = all_strategies()[:12]
    hs = [synthesize_history(ticker=f"DEMO-{i}", n=500, seed=i) for i in range(3)]
    wf = walk_forward(strats, hs, train_bars=200, test_bars=100, top_k=6)
    assert len(wf.folds) >= 2
    assert len(wf.oos_pnl) > 0
    # OOS series should cover the full test bars across folds
    assert len(wf.oos_pnl) == len(wf.folds) * 100 * len(hs)
