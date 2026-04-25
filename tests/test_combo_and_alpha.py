from kalshibot.combo_engine import build_betting_slip
from kalshibot.live import LiveSignal


def _sig(ticker, side, prob, ask, bid, conf="high", genre="crypto"):
    return LiveSignal(
        ticker=ticker, genre=genre, title=f"Test {ticker}",
        yes_bid=bid, yes_ask=ask, mid=(bid + ask) / 2, last=(bid + ask) / 2,
        volume=100, ensemble_prob=prob, dispersion=0.05,
        edge=prob - (bid + ask) / 2,
        kelly=0.05, suggested_side=side,
        suggested_notional=10.0, minutes_to_close=120,
        ts="2026-01-01T00:00:00Z", confidence=conf,
        win_prob=prob if side == "YES" else 1 - prob,
        expected_profit=1.0, rationale="ok",
    )


def test_build_betting_slip_picks_and_combos():
    sigs = [
        _sig("KXBTC-1",  "YES", 0.70, 0.55, 0.53, "high", "crypto"),
        _sig("KXNFL-1",  "YES", 0.65, 0.50, 0.48, "high", "sports"),
        _sig("KXHIGHNY-1", "NO",  0.60, 0.40, 0.38, "med",  "weather"),
        _sig("KXFEDDECISION-1", "YES", 0.62, 0.50, 0.48, "med", "economy"),
        _sig("KXBTC-2",  "YES", 0.30, 0.55, 0.53, "low", "crypto"),  # neg EV, dropped
    ]
    slip = build_betting_slip(sigs, bankroll=100.0)
    assert len(slip["picks"]) >= 3
    # Combos require >= 2 picks across genres
    assert len(slip["combos"]) >= 1
    for combo in slip["combos"]:
        genres = {leg["genre"] for leg in combo["legs"]}
        assert len(genres) == len(combo["legs"])  # all unique genres


def test_combo_drops_when_ev_negative():
    # All long-shots so combo EV is dominated by losses.
    sigs = [
        _sig("KXBTC-1", "YES", 0.20, 0.15, 0.13, "high", "crypto"),
        _sig("KXNFL-1", "YES", 0.20, 0.15, 0.13, "high", "sports"),
    ]
    slip = build_betting_slip(sigs, bankroll=100.0)
    # Singles may pass (high payout), combos must be EV>0 to be included
    for combo in slip["combos"]:
        assert combo["ev_per_dollar"] > 0
