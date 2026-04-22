from kalshibot.paper import PRESETS, PaperManager
from kalshibot.live import LiveSignal


def _make_sig(ticker="KXHIGHLAX-X-T72", side="YES", edge=0.1, conf="high",
               kelly=0.05):
    return LiveSignal(
        ticker=ticker, genre="weather", title="Will LA high be above 72?",
        yes_bid=0.45, yes_ask=0.47, mid=0.46, last=0.46, volume=100,
        ensemble_prob=0.56, dispersion=0.05, edge=edge, kelly=kelly,
        suggested_side=side, suggested_notional=5.0, minutes_to_close=180,
        ts="2026-01-01T00:00:00Z", confidence=conf, win_prob=0.56,
        expected_profit=0.4, rationale="test",
    )


def test_presets_exist_and_spawn():
    pm = PaperManager()
    for key in PRESETS:
        bot = pm.add(name=key, style=key, bankroll=100)
        assert bot.equity == 100
        assert bot.cash == 100
    assert len(pm.bots()) == len(PRESETS)


def test_process_signal_opens_trade_for_matching_bot():
    pm = PaperManager()
    pm.add(name="conservative", style="conservative", bankroll=100)
    # High-confidence signal should meet the conservative filter
    sig = _make_sig(conf="high", edge=0.10, kelly=0.1)
    pm.process([sig])
    bot = pm.bots()[0]
    assert len(bot.open_trades) == 1
    assert bot.cash < 100


def test_low_conf_signal_ignored_by_conservative_bot():
    pm = PaperManager()
    pm.add(name="cons", style="conservative", bankroll=100)
    sig = _make_sig(conf="low", edge=0.02, kelly=0.02)
    pm.process([sig])
    assert len(pm.bots()[0].open_trades) == 0


def test_equity_mark_to_market():
    pm = PaperManager()
    pm.add(name="c", style="conservative", bankroll=100)
    sig = _make_sig(conf="high", edge=0.10, kelly=0.1)
    pm.process([sig])
    # Price moves in our favor
    sig.mid = 0.8
    sig.yes_bid = 0.79; sig.yes_ask = 0.81
    pm.process([sig])
    bot = pm.bots()[0]
    assert bot.equity > 100   # unrealized gain
