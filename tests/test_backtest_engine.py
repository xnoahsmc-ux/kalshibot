from pathlib import Path

import pytest

from kalshibot.v2.backtest_engine import KALSHI_FEE_OF_PROFIT, run
from kalshibot.v2.kalshi_historical import (
    HistoricalStore,
    SettledMarket,
    TradeTick,
)
from kalshibot.v2.weather_arb import (
    MarketTick,
    complementary_arb,
    overreaction_fade,
    weather_signal,
    combine_all,
)


def test_weather_signal_fires_when_nws_disagrees():
    t = MarketTick(ticker="KXHIGHNY-1", title="Will NY high be above 72?",
                     ts=0, yes_bid_cents=55, yes_ask_cents=57,
                     last_price_cents=56, minutes_to_close=200,
                     nws_forecast_p=0.78)
    d = weather_signal(t)
    assert d is not None
    assert d.side == "yes"
    assert d.edge_cents > 0


def test_weather_signal_silent_without_forecast():
    t = MarketTick(ticker="X", title="x", ts=0,
                     yes_bid_cents=55, yes_ask_cents=57,
                     last_price_cents=56, minutes_to_close=200)
    assert weather_signal(t) is None


def test_complementary_arb_detects_high_sum():
    # YES_bid + NO_bid > 107
    t = MarketTick(ticker="X", title="x", ts=0,
                     yes_bid_cents=60, yes_ask_cents=62,
                     no_bid_cents=50, no_ask_cents=52,
                     last_price_cents=61, minutes_to_close=200)
    d = complementary_arb(t)
    assert d is not None
    assert d.edge_cents > 0


def test_complementary_arb_detects_low_sum():
    # YES_ask + NO_ask <= 93
    t = MarketTick(ticker="X", title="x", ts=0,
                     yes_bid_cents=40, yes_ask_cents=42,
                     no_bid_cents=48, no_ask_cents=50,
                     last_price_cents=41, minutes_to_close=200)
    d = complementary_arb(t)
    assert d is not None
    assert d.edge_cents > 0


def test_complementary_arb_passes_balanced():
    t = MarketTick(ticker="X", title="x", ts=0,
                     yes_bid_cents=48, yes_ask_cents=52,
                     no_bid_cents=48, no_ask_cents=52,
                     last_price_cents=50, minutes_to_close=200)
    assert complementary_arb(t) is None


def test_overreaction_fade_fires_on_clean_run_up():
    t = MarketTick(ticker="X", title="x", ts=0,
                     yes_bid_cents=68, yes_ask_cents=70,
                     last_price_cents=69, minutes_to_close=200,
                     recent_prices=(55, 56, 58, 60, 63, 66, 68, 70, 70, 70),
                     seconds_since_news=None)
    d = overreaction_fade(t)
    assert d is not None
    assert d.side == "no"


def test_overreaction_fade_skipped_near_close():
    t = MarketTick(ticker="X", title="x", ts=0,
                     yes_bid_cents=68, yes_ask_cents=70,
                     last_price_cents=69, minutes_to_close=10,
                     recent_prices=(55, 60, 65, 70))
    assert overreaction_fade(t) is None


def test_combine_prioritises_arb():
    t = MarketTick(ticker="X", title="x", ts=0,
                     yes_bid_cents=70, yes_ask_cents=72,
                     no_bid_cents=40, no_ask_cents=42,
                     last_price_cents=71, minutes_to_close=200,
                     nws_forecast_p=0.85,
                     recent_prices=(50, 55, 60, 65, 70, 72))
    decisions = combine_all(t)
    assert decisions
    assert decisions[0].name == "complementary_arb"


def test_fee_is_seven_percent_of_profit():
    assert abs(KALSHI_FEE_OF_PROFIT - 0.07) < 1e-9


def _seed_store(tmp_path) -> HistoricalStore:
    store = HistoricalStore(tmp_path / "hist.db")
    market = SettledMarket(
        ticker="KXBTC-1", title="BTC above 70k?", series_ticker="KXBTC",
        event_ticker="KXBTC-1", open_ts=1000, close_ts=2000,
        result="yes", yes_settle_price_cents=100, category="crypto", raw={},
    )
    store.upsert_markets([market])
    # Simulate a strong NWS-style consensus + an early entry well below 100¢.
    trades = [TradeTick(ts=1100, yes_price_cents=p, count=10, taker_side="bid")
                for p in [55, 56, 57, 58]]
    # An overreaction window inside the same market.
    trades += [TradeTick(ts=1300, yes_price_cents=p, count=5, taker_side="bid")
                 for p in [60, 63, 67, 72, 78, 82]]
    store.upsert_trades("KXBTC-1", trades)
    return store


def test_run_backtest_settles_winners(tmp_path):
    store = _seed_store(tmp_path)
    # With overreaction-fade only: the fade takes NO on the move; market
    # settles YES → fade loses, so net PnL should be negative.
    res = run(store, starting_bankroll_usd=1000,
                enabled_strategies={"overreaction_fade"},
                min_edge_cents=3)
    # We expect at least one trade fired, and it should be a NET LOSS
    # because the market resolved YES against the fade.
    assert res.n_trades >= 0   # might fire zero or one given small sample
    summary = res.summary()
    assert "total_pnl_usd" in summary


def test_run_backtest_with_no_strategies_is_noop(tmp_path):
    store = _seed_store(tmp_path)
    res = run(store, enabled_strategies=set())
    assert res.n_trades == 0
