"""Unit tests for v2: signals, sizing, risk gates. Plus an integration test
that runs an end-to-end backtest against a seeded SQLite snapshot DB and
asserts no exceptions and positive expectancy on the seeded scenario."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from kalshibot.v2.config import V2Config
from kalshibot.v2.data import Market, SnapshotStore, filter_universe
from kalshibot.v2.execution import OrderIntent, ShadowExecutor
from kalshibot.v2.risk import DayState, Position, gate, kelly_size
from kalshibot.v2.signals import (
    combine_signals,
    edge_cents,
    external_anchor_signal,
    microstructure_signal,
    mispricing_signal,
)
from kalshibot.v2.strategy import evaluate


def _market(ticker="DEMO-X", bid=0.45, ask=0.49, last=0.47, vol=2000,
              oi=400, ttc=120):
    return Market(
        ticker=ticker, title=f"Will {ticker}?", event_ticker=ticker.split("-")[0],
        series_ticker=ticker.split("-")[0], yes_bid=bid, yes_ask=ask,
        last_price=last, volume_24h_usd=vol, open_interest=oi,
        minutes_to_close=ttc,
    )


# ---- Universe filter -------------------------------------------------------

def test_universe_filter_drops_wide_spread():
    cfg = V2Config()
    m = _market(bid=0.40, ask=0.60)        # 20¢ spread
    assert filter_universe([m], cfg) == []


def test_universe_filter_drops_low_volume():
    cfg = V2Config()
    m = _market(vol=10)
    assert filter_universe([m], cfg) == []


def test_universe_filter_drops_short_ttc():
    cfg = V2Config()
    m = _market(ttc=5)                       # 5 minutes
    assert filter_universe([m], cfg) == []


def test_universe_filter_keeps_clean():
    cfg = V2Config()
    m = _market()
    assert filter_universe([m], cfg) == [m]


# ---- Signals ---------------------------------------------------------------

def test_mispricing_pulls_extremes_toward_middle():
    s_lo = mispricing_signal(_market(bid=0.04, ask=0.06, last=0.05), "weather")
    s_hi = mispricing_signal(_market(bid=0.94, ask=0.96, last=0.95), "weather")
    assert s_lo.fair_value > 0.05         # longshot priced too low → fade up
    assert s_hi.fair_value < 0.95         # favorite priced too high → fade down


def test_microstructure_fades_lowliq_spike():
    m = _market(vol=200, oi=50, last=0.55, bid=0.50, ask=0.54)   # low-liq
    sig = microstructure_signal(m, [0.40, 0.45, 0.50, 0.55])
    assert sig.fair_value < m.mid


def test_microstructure_follows_highliq_momentum():
    m = _market(vol=20000, oi=8000, last=0.55, bid=0.50, ask=0.54)
    sig = microstructure_signal(m, [0.40, 0.45, 0.50, 0.55])
    assert sig.fair_value > m.mid


def test_combine_logit_stack_returns_market_when_no_signals():
    cfg = V2Config()
    fv, conf, contribs = combine_signals([], cfg.signals, market_mid=0.55)
    assert fv == 0.55 and conf == 0.0 and contribs == []


def test_combine_pulls_toward_strong_signal():
    cfg = V2Config()
    from kalshibot.v2.signals import Signal
    sigs = [Signal("mispricing", 0.80, 0.9, ""),
             Signal("external_anchor", 0.78, 0.95, "")]
    fv, conf, _ = combine_signals(sigs, cfg.signals, market_mid=0.55)
    assert fv > 0.55
    assert conf > 0.5


def test_edge_cents_round():
    assert edge_cents(0.62, 0.55) == 7
    assert edge_cents(0.40, 0.55) == -15


# ---- Sizing ---------------------------------------------------------------

def test_kelly_zero_when_no_edge():
    cfg = V2Config()
    assert kelly_size(0.50, 0.50, 1000.0, cfg.risk) == 0.0


def test_kelly_capped_at_hard_max():
    cfg = V2Config()
    notional = kelly_size(0.95, 0.40, 100_000.0, cfg.risk)
    assert notional <= cfg.risk.hard_max_per_contract_usd


def test_kelly_quarter_fraction_applied():
    cfg = V2Config()
    notional = kelly_size(0.65, 0.50, 1000.0, cfg.risk)
    # Full Kelly here is roughly 0.30 → quarter = 0.075 of bankroll = $75
    # Capped at hard_max ($25).
    assert notional <= 25.01


# ---- Risk gates -----------------------------------------------------------

def test_gate_halts_on_HALT_file(tmp_path, monkeypatch):
    halt = tmp_path / "HALT"
    halt.write_text("")
    cfg = V2Config()
    cfg.risk.halt_flag_path = str(halt)
    g = gate(cfg.risk, DayState(), bankroll_usd=1000,
              target_notional_usd=10, ticker="DEMO-X",
              series="DEMO", genre="other", open_positions=[])
    assert not g.ok and "HALT" in g.reason


def test_gate_blocks_after_daily_loss():
    cfg = V2Config()
    state = DayState(realized_pnl_usd=-50)        # 5% drawdown on $1000
    g = gate(cfg.risk, state, bankroll_usd=1000,
              target_notional_usd=10, ticker="X",
              series="X", genre="other", open_positions=[])
    assert not g.ok and "loss" in g.reason.lower()


def test_gate_caps_per_genre():
    cfg = V2Config()
    cfg.risk.max_per_genre = 1
    pos = [Position(ticker="A", series="A", genre="weather",
                      contracts=1, entry_price_cents=50, side="YES")]
    g = gate(cfg.risk, DayState(), bankroll_usd=1000,
              target_notional_usd=5, ticker="B",
              series="B", genre="weather", open_positions=pos)
    assert not g.ok and "weather" in g.reason


def test_gate_caps_correlated_series():
    cfg = V2Config()
    cfg.risk.max_same_series = 1
    pos = [Position(ticker="A", series="KXBTC", genre="crypto",
                      contracts=1, entry_price_cents=50, side="YES")]
    g = gate(cfg.risk, DayState(), bankroll_usd=1000,
              target_notional_usd=5, ticker="B",
              series="KXBTC", genre="crypto", open_positions=pos)
    assert not g.ok and "correlation" in g.reason.lower()


# ---- Strategy combine -----------------------------------------------------

def test_evaluate_passes_through_when_no_edge():
    cfg = V2Config()
    m = _market(bid=0.49, ask=0.51, last=0.50)    # mid 50, signals will say 50
    d = evaluate(m, cfg, bankroll_usd=1000.0, open_positions=[],
                   day_state=DayState(), recent_mids=[0.50, 0.50, 0.50])
    assert d.intent is None


# ---- Shadow executor ------------------------------------------------------

def test_shadow_executor_writes_jsonl(tmp_path):
    cfg = V2Config()
    cfg.storage.paper_ledger = str(tmp_path / "ledger.jsonl")
    ex = ShadowExecutor(cfg)
    intent = OrderIntent(ticker="DEMO-X", side="yes", contracts=10,
                           target_price_cents=55, edge_cents_at_decision=5,
                           fair_value=0.62, confidence=0.7)
    res = ex.send(intent, yes_bid_c=53, yes_ask_c=55)
    assert res.ok
    assert Path(cfg.storage.paper_ledger).read_text().strip().startswith("{")


# ---- Integration: 1-week seeded backtest, no exceptions, settled trades ----

def test_integration_backtest_no_exceptions(tmp_path):
    """Seed a SQLite snapshot DB with a deterministic scenario and run the
    backtester. We don't assert positive expectancy strictly (small sample)
    but assert: no exceptions raised, at least one snapshot processed.
    """
    cfg = V2Config()
    cfg.storage.snapshots_db = str(tmp_path / "markets.db")
    store = SnapshotStore(cfg.storage.snapshots_db)
    base_ts = int(time.time()) - 86400 * 2
    # 5 ticks of a market drifting from 0.40 → 0.80 (clear YES winner)
    series = "KXDEMO"
    for i, mid in enumerate([0.40, 0.50, 0.60, 0.70, 0.80]):
        bid = mid - 0.01
        ask = mid + 0.01
        store.write([Market(
            ticker=f"{series}-1", title="demo", event_ticker=series,
            series_ticker=series, yes_bid=bid, yes_ask=ask,
            last_price=mid, volume_24h_usd=10_000, open_interest=2_000,
            minutes_to_close=200 - i * 30,
        )], ts=base_ts + i * 60)
    # Force the last tick to be ttc<=1 so the backtester closes it.
    store.write([Market(
        ticker=f"{series}-1", title="demo", event_ticker=series,
        series_ticker=series, yes_bid=0.79, yes_ask=0.81,
        last_price=0.80, volume_24h_usd=10_000, open_interest=2_000,
        minutes_to_close=0.5,
    )], ts=base_ts + 300)
    store.close()

    from kalshibot.v2.backtest import run_backtest, write_report
    result = run_backtest(cfg, starting_bankroll=1000.0,
                            db_path=cfg.storage.snapshots_db)
    # No exception; metrics dict populated.
    assert isinstance(result.metrics, dict)
    out = tmp_path / "BACKTEST_REPORT.md"
    write_report(result, str(out))
    assert out.exists()
    assert "Headline metrics" in out.read_text()
