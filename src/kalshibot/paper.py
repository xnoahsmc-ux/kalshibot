"""Paper-trading bots that race on live Kalshi signals.

Each bot has a config (min confidence, bet sizing, max position), a
starting bankroll, and a ledger of simulated trades. The PaperManager
iterates all bots on each live poll, marks open positions to market, and
opens new positions when the bot's criteria are met.

PnL is mark-to-market (unrealized) + realized from closed trades. Closed
trades happen when a signal for that ticker disappears (market closed) or
when the bot chooses to exit.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .live import LiveSignal


@dataclass
class BotConfig:
    min_confidence: str = "med"    # "high" | "med" | "low"
    min_edge: float = 0.05
    bet_sizer: str = "kelly"       # "kelly" | "fixed" | "proportional"
    fixed_bet_usd: float = 5.0
    bet_fraction: float = 0.05     # for "proportional"
    max_bet_usd: float = 25.0
    allow_reopen: bool = False     # re-enter after closing the position


@dataclass
class PaperTrade:
    ticker: str
    side: str                      # "YES" | "NO"
    contracts: int
    entry_price: float             # 0-1
    opened_at: float
    current_price: float
    closed_at: float | None = None
    close_price: float | None = None

    @property
    def is_open(self) -> bool:
        return self.closed_at is None

    def _value(self, price: float) -> float:
        if self.side == "YES":
            return self.contracts * price
        return self.contracts * (1 - price)

    @property
    def cost_basis(self) -> float:
        return self._value(self.entry_price)

    @property
    def unrealized_pnl(self) -> float:
        if not self.is_open:
            return 0.0
        return self._value(self.current_price) - self.cost_basis

    @property
    def realized_pnl(self) -> float:
        if self.is_open or self.close_price is None:
            return 0.0
        return self._value(self.close_price) - self.cost_basis


@dataclass
class PaperBot:
    id: str
    name: str
    style: str                      # preset identifier
    config: BotConfig
    starting_cash: float
    cash: float                      # cash not tied up in open trades
    open_trades: dict[str, PaperTrade] = field(default_factory=dict)
    closed_trades: list[PaperTrade] = field(default_factory=list)
    equity_history: list[tuple[float, float]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def open_position_value(self) -> float:
        return sum(t._value(t.current_price) for t in self.open_trades.values())

    @property
    def equity(self) -> float:
        return self.cash + self.open_position_value()

    @property
    def total_pnl(self) -> float:
        return self.equity - self.starting_cash

    @property
    def pnl_pct(self) -> float:
        if self.starting_cash <= 0:
            return 0.0
        return self.total_pnl / self.starting_cash

    @property
    def wins(self) -> int:
        return sum(1 for t in self.closed_trades if t.realized_pnl > 0)

    @property
    def losses(self) -> int:
        return sum(1 for t in self.closed_trades if t.realized_pnl <= 0)

    @property
    def hit_rate(self) -> float:
        n = len(self.closed_trades)
        return (self.wins / n) if n else 0.0


# ---- Preset bot personalities -----------------------------------------------

PRESETS: dict[str, dict] = {
    "conservative": {
        "label": "The Accountant",
        "blurb": "Only high-confidence bets, small fixed $5 stakes.",
        "emoji": "📘",
        "config": BotConfig(min_confidence="high", min_edge=0.08,
                             bet_sizer="fixed", fixed_bet_usd=5.0,
                             max_bet_usd=10.0),
    },
    "balanced": {
        "label": "Mr. Balanced",
        "blurb": "Medium+ confidence, half-Kelly sizing capped at $25.",
        "emoji": "⚖️",
        "config": BotConfig(min_confidence="med", min_edge=0.05,
                             bet_sizer="kelly", max_bet_usd=25.0),
    },
    "aggressive": {
        "label": "Diamond Hands",
        "blurb": "Takes every signal, 10% of bankroll per bet.",
        "emoji": "💎",
        "config": BotConfig(min_confidence="low", min_edge=0.02,
                             bet_sizer="proportional", bet_fraction=0.10,
                             max_bet_usd=80.0),
    },
    "scalper": {
        "label": "The Scalper",
        "blurb": "High-conf only, $2 stakes, many small bets.",
        "emoji": "🎯",
        "config": BotConfig(min_confidence="high", min_edge=0.06,
                             bet_sizer="fixed", fixed_bet_usd=2.0,
                             max_bet_usd=5.0, allow_reopen=True),
    },
    "whale": {
        "label": "The Whale",
        "blurb": "Medium conf, goes big - up to $80 per bet.",
        "emoji": "🐋",
        "config": BotConfig(min_confidence="med", min_edge=0.04,
                             bet_sizer="proportional", bet_fraction=0.15,
                             max_bet_usd=80.0),
    },
    "yolo": {
        "label": "YOLO",
        "blurb": "Any edge, full-Kelly sizing. Wins big or blows up.",
        "emoji": "🚀",
        "config": BotConfig(min_confidence="low", min_edge=0.01,
                             bet_sizer="kelly", max_bet_usd=100.0),
    },
}


# ---- Manager ----------------------------------------------------------------

class PaperManager:
    MAX_EQUITY_POINTS = 500

    def __init__(self) -> None:
        self._bots: dict[str, PaperBot] = {}
        self._lock = threading.Lock()

    def bots(self) -> list[PaperBot]:
        with self._lock:
            return list(self._bots.values())

    def get(self, bot_id: str) -> PaperBot | None:
        return self._bots.get(bot_id)

    def add(self, name: str, style: str = "balanced",
             bankroll: float = 100.0) -> PaperBot:
        preset = PRESETS.get(style, PRESETS["balanced"])
        bot = PaperBot(
            id=uuid.uuid4().hex[:10],
            name=name or preset["label"],
            style=style,
            config=preset["config"],
            starting_cash=bankroll,
            cash=bankroll,
        )
        with self._lock:
            self._bots[bot.id] = bot
        return bot

    def remove(self, bot_id: str) -> None:
        with self._lock:
            self._bots.pop(bot_id, None)

    def reset_all(self) -> None:
        with self._lock:
            self._bots.clear()

    def seed_defaults(self) -> None:
        """Populate the leaderboard with one bot per preset so the user sees
        interesting data immediately."""
        if self._bots:
            return
        for key, preset in PRESETS.items():
            self.add(name=preset["label"], style=key, bankroll=100.0)

    # ------------------------------------------------------------------ core
    def process(self, signals: list["LiveSignal"]) -> None:
        if not signals:
            return
        signals_by_ticker = {s.ticker: s for s in signals}
        now = time.time()
        with self._lock:
            for bot in self._bots.values():
                self._mark_to_market(bot, signals_by_ticker)
                self._consider_new_entries(bot, signals)
                bot.equity_history.append((now, bot.equity))
                if len(bot.equity_history) > self.MAX_EQUITY_POINTS:
                    bot.equity_history = bot.equity_history[-self.MAX_EQUITY_POINTS:]

    def _mark_to_market(self, bot: PaperBot,
                         signals: dict[str, "LiveSignal"]) -> None:
        dropped: list[str] = []
        for ticker, trade in bot.open_trades.items():
            sig = signals.get(ticker)
            if sig is None:
                # Market disappeared - close at last known price.
                trade.closed_at = time.time()
                trade.close_price = trade.current_price
                bot.cash += trade._value(trade.close_price)
                bot.closed_trades.append(trade)
                dropped.append(ticker)
                continue
            trade.current_price = sig.mid
            # Close if very close to resolution and edge has faded.
            if isfinite_none(sig.minutes_to_close) and sig.minutes_to_close < 5:
                trade.closed_at = time.time()
                trade.close_price = trade.current_price
                bot.cash += trade._value(trade.close_price)
                bot.closed_trades.append(trade)
                dropped.append(ticker)
        for t in dropped:
            bot.open_trades.pop(t, None)

    def _consider_new_entries(self, bot: PaperBot,
                               signals: list["LiveSignal"]) -> None:
        for sig in signals:
            if not self._passes_filter(sig, bot.config):
                continue
            if sig.ticker in bot.open_trades and not bot.config.allow_reopen:
                continue
            stake = self._size_bet(bot, sig)
            if stake <= 0.50 or stake > bot.cash:
                continue
            entry_price = sig.yes_ask if sig.suggested_side == "YES" \
                          else (1 - sig.yes_bid)
            if entry_price <= 0:
                continue
            contracts = int(stake / max(0.01, entry_price))
            if contracts <= 0:
                continue
            trade = PaperTrade(
                ticker=sig.ticker,
                side=sig.suggested_side,
                contracts=contracts,
                entry_price=entry_price,
                opened_at=time.time(),
                current_price=sig.mid,
            )
            bot.cash -= contracts * entry_price
            bot.open_trades[sig.ticker] = trade

    def _passes_filter(self, sig: "LiveSignal", cfg: BotConfig) -> bool:
        if sig.suggested_side == "flat":
            return False
        rank = {"low": 0, "med": 1, "high": 2}
        if rank.get(sig.confidence, 0) < rank.get(cfg.min_confidence, 1):
            return False
        if abs(sig.edge) < cfg.min_edge:
            return False
        return True

    def _size_bet(self, bot: PaperBot, sig: "LiveSignal") -> float:
        cfg = bot.config
        if cfg.bet_sizer == "fixed":
            return min(cfg.fixed_bet_usd, cfg.max_bet_usd, bot.cash)
        if cfg.bet_sizer == "proportional":
            return min(bot.equity * cfg.bet_fraction, cfg.max_bet_usd, bot.cash)
        # kelly
        return min(abs(sig.kelly) * bot.equity, cfg.max_bet_usd, bot.cash)


def isfinite_none(v: float) -> bool:
    if v is None:
        return False
    try:
        return v == v and v != float("inf") and v != float("-inf")
    except Exception:
        return False
