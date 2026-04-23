import pytest

from kalshibot.risk_limits import RiskLimits, RiskState, check_order
from kalshibot.trade_journal import TradeJournal
from kalshibot.learner import OnlineLearner
from kalshibot.combiner import Combination
import pandas as pd


def test_risk_check_kill_switch(tmp_path):
    L = RiskLimits(kill_switch=True)
    S = RiskState()
    ok, why = check_order(L, S, ticker="KXBTC-X", genre="crypto",
                           notional_usd=5, cash_available=100,
                           open_positions=[])
    assert not ok and "Kill switch" in why


def test_risk_check_daily_loss_limit():
    L = RiskLimits(daily_loss_limit_usd=20)
    S = RiskState(today_realized_pnl=-25)
    ok, why = check_order(L, S, ticker="KXBTC-X", genre="crypto",
                           notional_usd=5, cash_available=100,
                           open_positions=[])
    assert not ok and "Daily loss" in why


def test_risk_check_max_per_genre():
    L = RiskLimits(max_open_per_genre=2)
    S = RiskState()
    positions = [{"ticker": "KXBTC-1", "genre": "crypto", "notional": 10},
                  {"ticker": "KXBTC-2", "genre": "crypto", "notional": 10}]
    ok, why = check_order(L, S, ticker="KXBTC-3", genre="crypto",
                           notional_usd=5, cash_available=100,
                           open_positions=positions)
    assert not ok and "crypto" in why


def test_risk_check_series_correlation():
    L = RiskLimits(max_same_series=2)
    S = RiskState()
    positions = [{"ticker": "KXHIGHNY-1", "genre": "weather", "notional": 5},
                  {"ticker": "KXHIGHNY-2", "genre": "weather", "notional": 5}]
    ok, why = check_order(L, S, ticker="KXHIGHNY-3", genre="weather",
                           notional_usd=5, cash_available=100,
                           open_positions=positions)
    assert not ok and "correlation" in why


def test_journal_add_and_settle(tmp_path):
    j = TradeJournal(path=tmp_path / "j.jsonl")
    e = j.add(
        ticker="KXBTC-X", genre="crypto", title="Will BTC close above 100k?",
        side="YES", contracts=10, entry_price_cents=55,
        notional_usd=5.50, source="manual", confidence="high",
        ensemble_prob=0.70, fair_value_prob=None,
        reason="test", contributing_strategies=["momentum_20"],
    )
    assert j.stats()["total_trades"] == 1
    assert j.stats()["open_trades"] == 1
    j.update_settlement(e.id, settlement_price_cents=100)
    s = j.stats()
    assert s["settled_trades"] == 1
    assert s["wins"] == 1
    assert s["realized_pnl_usd"] > 0


def test_learner_adjusts_weights(tmp_path):
    lr = OnlineLearner(path=tmp_path / "l.json", learning_rate=0.1)
    lr.update_from_trade(strategy_probs={"A": 0.8, "B": 0.2}, outcome=1)
    s_a = lr.score_for("A")
    s_b = lr.score_for("B")
    assert s_a > s_b   # A said YES (p=0.8) and outcome was YES, so score higher
    base = Combination(weights=pd.Series({"A": 0.5, "B": 0.5}), sharpe=0.0, expected_pnl=0.0)
    adj = lr.adjust_combination(base)
    assert adj.weights["A"] > adj.weights["B"]
