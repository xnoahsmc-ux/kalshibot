"""Risk guard rails. Every order goes through check_order() which can veto
the trade for any of these reasons:

* kill switch is on
* daily realized loss exceeds the cap
* open-position count above the limit overall or per genre
* correlation cap exceeded (too many positions in same cluster)
* bankroll too low to cover the bet
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class RiskLimits:
    kill_switch: bool = False                    # master emergency stop
    daily_loss_limit_usd: float = 50.0           # stop if today's realized loss > this
    max_open_positions: int = 15                 # across all markets
    max_open_per_genre: int = 6                  # per genre cap
    max_same_series: int = 3                     # per series (e.g. 3 KXHIGHNY bets max)
    min_cash_buffer_usd: float = 5.0             # keep at least this much liquid
    max_notional_per_trade_usd: float = 100.0    # hard ceiling on any single bet
    auto_trade_enabled: bool = False             # if true, execute at min_auto_conf
    auto_trade_min_conf: str = "high"            # "high" | "med" | "low"
    auto_trade_min_edge: float = 0.07            # tighter than live-feed MIN_EDGE


@dataclass
class RiskState:
    started_at: float = field(default_factory=time.time)
    today_realized_pnl: float = 0.0              # settled today, starts 0
    today_orders_sent: int = 0

    def record_pnl(self, delta: float) -> None:
        self.today_realized_pnl += delta

    def rolling_reset_if_needed(self) -> None:
        # Reset at UTC midnight boundary
        import datetime as dt
        now = dt.datetime.utcnow()
        start = dt.datetime.utcfromtimestamp(self.started_at)
        if now.date() != start.date():
            self.started_at = time.time()
            self.today_realized_pnl = 0.0
            self.today_orders_sent = 0


def check_order(
    limits: RiskLimits,
    state: RiskState,
    *,
    ticker: str,
    genre: str,
    notional_usd: float,
    cash_available: float,
    open_positions: list[dict],      # [{ticker, genre, series, notional}]
) -> tuple[bool, str]:
    """Returns (ok, reason). If ok is False, reason explains why."""
    state.rolling_reset_if_needed()
    if limits.kill_switch:
        return False, "Kill switch engaged."
    if state.today_realized_pnl <= -abs(limits.daily_loss_limit_usd):
        return False, (f"Daily loss limit hit "
                        f"(${state.today_realized_pnl:+.2f} ≤ "
                        f"-${limits.daily_loss_limit_usd:.2f}). Paused for today.")
    if notional_usd > limits.max_notional_per_trade_usd:
        return False, (f"Requested ${notional_usd:.2f} exceeds per-trade cap "
                        f"${limits.max_notional_per_trade_usd:.2f}.")
    if notional_usd + limits.min_cash_buffer_usd > cash_available:
        return False, "Not enough cash for this bet (keeps a small buffer)."
    if len(open_positions) >= limits.max_open_positions:
        return False, (f"At max open positions "
                        f"({limits.max_open_positions}). Close some first.")
    same_genre = sum(1 for p in open_positions if p.get("genre") == genre)
    if same_genre >= limits.max_open_per_genre:
        return False, f"At max {limits.max_open_per_genre} open positions in {genre}."
    series = ticker.split("-", 1)[0] if "-" in ticker else ticker
    same_series = sum(1 for p in open_positions
                       if p.get("ticker", "").startswith(series + "-"))
    if same_series >= limits.max_same_series:
        return False, (f"At max {limits.max_same_series} positions in {series} "
                        "(correlation cap).")
    return True, "ok"
