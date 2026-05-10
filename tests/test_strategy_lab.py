from pathlib import Path

import pandas as pd

from kalshibot.data import MarketHistory
from kalshibot.v2.config import load
from kalshibot.v2.strategy_lab import (
    SignalResult,
    autotune_config,
    evaluate_signal_on_history,
    write_report,
)


def _make_history(ticker: str, mids: list[float]) -> MarketHistory:
    df = pd.DataFrame({
        "yes_bid": [m - 0.01 for m in mids],
        "yes_ask": [m + 0.01 for m in mids],
        "last": mids,
        "volume": [5000] * len(mids),
        "open_interest": [1500] * len(mids),
        "minutes_to_close": [200 - i * 30 for i in range(len(mids))],
    }, index=pd.date_range("2025-01-01", periods=len(mids), freq="h"))
    return MarketHistory(ticker=ticker, df=df)


def test_evaluate_signal_runs_without_error():
    cfg = load("config.yaml")
    cfg.signals.min_edge_cents = 2
    hs = [
        _make_history("KXBTC-1", [0.40, 0.45, 0.55, 0.65, 0.75]),
        _make_history("KXHIGHNY-1", [0.30, 0.30, 0.32, 0.30, 0.28]),
    ]
    res = evaluate_signal_on_history("mispricing", hs, cfg)
    assert isinstance(res, SignalResult)
    assert res.trades >= 0


def test_signal_result_metrics_are_consistent():
    r = SignalResult(name="x", trades=4, wins=3, losses=1,
                       total_pnl_usd=10.0, pnls=[3, 4, -2, 5])
    assert r.hit_rate == 0.75
    assert abs(r.avg_pnl - 2.5) < 1e-9
    assert r.sharpe > 0   # positive mean, positive sharpe


def test_autotune_zeros_losers(tmp_path):
    cfg_text = """signals:
  weights:
    mispricing: 1.0
    microstructure: 0.6
    external_anchor: 1.4
"""
    p = tmp_path / "config.yaml"
    p.write_text(cfg_text)
    lab_out = {"results": {
        "mispricing":     SignalResult(name="mispricing",
                                        trades=20, wins=12, total_pnl_usd=15.0,
                                        pnls=[1, 2, -1, 3, 2] * 4),
        "microstructure": SignalResult(name="microstructure",
                                        trades=20, wins=5, total_pnl_usd=-5.0,
                                        pnls=[-1, -1, 1, -2, 1] * 4),
        "external_anchor": SignalResult(name="external_anchor",
                                          trades=20, wins=14, total_pnl_usd=20.0,
                                          pnls=[2, 3, -1, 4, 1] * 4),
    }}
    autotune_config(lab_out, config_path=str(p))
    new_text = p.read_text()
    # Microstructure should have been zeroed; the others kept.
    assert "microstructure: 0.0" in new_text
    assert "mispricing: 1.0" in new_text
    assert "external_anchor: 1.4" in new_text


def test_write_report_includes_recommendations(tmp_path):
    out = tmp_path / "STRATEGY_REPORT.md"
    lab_out = {"n_markets": 5, "results": {
        "mispricing": SignalResult(name="mispricing",
                                     trades=10, wins=7, total_pnl_usd=5.0,
                                     pnls=[1, 2, -1, 1, 2, -1, 1, 0, 0, 0]),
        "external_anchor": SignalResult(name="external_anchor",
                                          trades=10, wins=3, total_pnl_usd=-3.0,
                                          pnls=[-1, 1, -1, 0, 0, -1, 0, -1, 0, 0]),
    }}
    write_report(lab_out, str(out))
    text = out.read_text()
    assert "Per-Signal Backtest Report" in text
    assert "Recommendation" in text
