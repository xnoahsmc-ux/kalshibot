from kalshibot.bot import kelly_fraction, snapshot_from_market
import pandas as pd


def test_kelly_sign_matches_edge():
    assert kelly_fraction(0.7, 0.5) > 0
    assert kelly_fraction(0.3, 0.5) < 0
    assert kelly_fraction(0.5, 0.5) == 0


def test_snapshot_from_market_normalizes_cents():
    m = {
        "ticker": "DEMO",
        "yes_bid": 45,
        "yes_ask": 55,
        "last_price": 50,
        "volume": 10,
        "open_interest": 1000,
    }
    snap = snapshot_from_market(m, pd.Timestamp("2025-01-01"))
    assert snap.ticker == "DEMO"
    assert 0.4 < snap.mid < 0.6
    assert snap.spread > 0
