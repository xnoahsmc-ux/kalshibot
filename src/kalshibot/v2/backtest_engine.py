"""Event-driven backtester over resolved Kalshi markets.

Walks each settled market's trade history forward in time. At each tick,
runs the selected strategies; when one fires, simulates a fill at the
next available price with slippage, deducts 7% Kalshi fee on wins, and
settles at the market's resolved outcome.

Self-contained: only depends on HistoricalStore (resolved-market cache),
weather_arb (the 3 strategies), and a callable that returns the historical
NWS forecast for (city, ts). For backtests where we don't have historical
NWS data, the weather signal silently no-ops.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from .kalshi_historical import HistoricalStore, SettledMarket, TradeTick
from .weather_arb import MarketTick, StrategyDecision, combine_all


# Kalshi fee schedule: 7% of profit on a winning trade (no fee on losses).
KALSHI_FEE_OF_PROFIT = 0.07


@dataclass
class BacktestTrade:
    ticker: str
    strategy: str
    side: str
    contracts: int
    entry_price_cents: int
    exit_price_cents: int
    gross_pnl_cents: float
    fee_cents: float
    net_pnl_cents: float
    edge_at_entry_cents: int
    confidence: float
    entry_ts: int


@dataclass
class BacktestResult:
    trades: list[BacktestTrade] = field(default_factory=list)
    bankroll_curve: list[tuple[int, float]] = field(default_factory=list)
    config: dict = field(default_factory=dict)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.net_pnl_cents > 0)

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if t.net_pnl_cents <= 0)

    @property
    def total_pnl_usd(self) -> float:
        return sum(t.net_pnl_cents for t in self.trades) / 100.0

    @property
    def hit_rate(self) -> float:
        return self.wins / self.n_trades if self.n_trades else 0.0

    @property
    def avg_edge_cents(self) -> float:
        return (sum(t.edge_at_entry_cents for t in self.trades) / self.n_trades
                 if self.n_trades else 0.0)

    @property
    def sharpe(self) -> float:
        if self.n_trades < 2:
            return 0.0
        pnls = [t.net_pnl_cents / 100 for t in self.trades]
        sd = statistics.pstdev(pnls)
        if sd <= 0:
            return 0.0
        return (statistics.mean(pnls) / sd) * math.sqrt(252)

    @property
    def max_drawdown_pct(self) -> float:
        if not self.bankroll_curve:
            return 0.0
        peak = self.bankroll_curve[0][1]
        max_dd = 0.0
        for _, b in self.bankroll_curve:
            peak = max(peak, b)
            if peak <= 0:
                continue
            max_dd = max(max_dd, (peak - b) / peak)
        return max_dd * 100

    def summary(self) -> dict:
        per_strategy: dict[str, float] = defaultdict(float)
        per_strategy_count: dict[str, int] = defaultdict(int)
        for t in self.trades:
            per_strategy[t.strategy] += t.net_pnl_cents / 100.0
            per_strategy_count[t.strategy] += 1
        return {
            "trades": self.n_trades, "wins": self.wins, "losses": self.losses,
            "hit_rate": round(self.hit_rate, 3),
            "total_pnl_usd": round(self.total_pnl_usd, 2),
            "avg_edge_cents": round(self.avg_edge_cents, 2),
            "sharpe": round(self.sharpe, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "pnl_by_strategy": {k: round(v, 2) for k, v in per_strategy.items()},
            "trade_count_by_strategy": dict(per_strategy_count),
        }


# Provided by caller; e.g. for weather backtests we look up the NWS forecast
# that was available at `ts` for the city referenced in `market.title`.
NWSForecastFn = Callable[[SettledMarket, int], float | None]


def _no_op_nws(_market: SettledMarket, _ts: int) -> float | None:
    return None


def _apply_fee(gross_pnl_cents: float) -> float:
    if gross_pnl_cents <= 0:
        return 0.0
    return gross_pnl_cents * KALSHI_FEE_OF_PROFIT


def _slippage(side: str, target_cents: int, last_cents: int,
                slippage_cents: int) -> int:
    """How much worse than `target_cents` we actually fill at."""
    s = max(0, slippage_cents)
    if side == "yes":
        return min(99, last_cents + s)
    return min(99, (100 - last_cents) + s)


def _build_tick(market: SettledMarket, t: TradeTick,
                 recent: list[int], minutes_to_close: float,
                 nws_p: float | None,
                 seconds_since_news: int | None) -> MarketTick:
    """Convert a TradeTick into a MarketTick. Bid/ask aren't in the trade
    feed so we approximate them with a 2-cent spread around the trade
    price — realistic for the tight Kalshi orderbooks we filter for."""
    mid = t.yes_price_cents
    bid = max(1, mid - 1)
    ask = min(99, mid + 1)
    return MarketTick(
        ticker=market.ticker, title=market.title, ts=t.ts,
        yes_bid_cents=bid, yes_ask_cents=ask,
        last_price_cents=mid,
        minutes_to_close=minutes_to_close,
        nws_forecast_p=nws_p,
        recent_prices=tuple(recent[-10:]),
        seconds_since_news=seconds_since_news,
    )


def run(
    store: HistoricalStore,
    *,
    starting_bankroll_usd: float = 10_000.0,
    enabled_strategies: set[str] | None = None,
    min_edge_cents: int = 3,
    max_position_usd: float = 250.0,
    slippage_cents: int = 1,
    series_prefix: str | None = None,
    since_ts: int = 0,
    market_limit: int = 500,
    nws_fn: NWSForecastFn | None = None,
) -> BacktestResult:
    nws_fn = nws_fn or _no_op_nws
    enabled = enabled_strategies or {"weather_signal", "complementary_arb",
                                       "overreaction_fade"}
    result = BacktestResult(config={
        "starting_bankroll": starting_bankroll_usd,
        "enabled": sorted(enabled),
        "min_edge_cents": min_edge_cents,
        "max_position_usd": max_position_usd,
        "slippage_cents": slippage_cents,
        "fee_pct_of_profit": KALSHI_FEE_OF_PROFIT * 100,
    })
    bankroll = starting_bankroll_usd
    markets = store.list_markets(series_prefix=series_prefix,
                                   since_ts=since_ts, limit=market_limit)
    for market in markets:
        if market.result not in ("yes", "no"):
            continue
        trades = store.trades_for(market.ticker)
        if not trades:
            continue
        recent: list[int] = []
        open_position: BacktestTrade | None = None
        close_ts = market.close_ts or trades[-1].ts
        for t in trades:
            recent.append(t.yes_price_cents)
            if open_position is not None:
                continue
            mtc = max(1.0, (close_ts - t.ts) / 60.0)
            nws_p = nws_fn(market, t.ts)
            seconds_since_news = None
            tick = _build_tick(market, t, recent, mtc, nws_p,
                                  seconds_since_news)
            decisions = combine_all(tick, enabled=enabled)
            decisions = [d for d in decisions if abs(d.edge_cents) >= min_edge_cents]
            if not decisions:
                continue
            d = decisions[0]
            entry_price = _slippage(d.side, t.yes_price_cents,
                                      t.yes_price_cents, slippage_cents)
            contracts = max(1, int(min(max_position_usd,
                                          bankroll * 0.05) /
                                      max(0.01, entry_price / 100.0)))
            cost = contracts * entry_price / 100.0
            if cost > bankroll * 0.5:
                continue
            # Settle at the market's resolved outcome.
            settled_yes = 100 if market.result == "yes" else 0
            if d.side == "yes":
                gross = contracts * (settled_yes - entry_price)
            else:
                gross = contracts * ((100 - settled_yes) - entry_price)
            fee = _apply_fee(gross)
            net = gross - fee
            bankroll += net / 100.0
            tr = BacktestTrade(
                ticker=market.ticker, strategy=d.name, side=d.side,
                contracts=contracts, entry_price_cents=entry_price,
                exit_price_cents=settled_yes if d.side == "yes" else (100 - settled_yes),
                gross_pnl_cents=gross, fee_cents=fee, net_pnl_cents=net,
                edge_at_entry_cents=d.edge_cents, confidence=d.confidence,
                entry_ts=t.ts,
            )
            result.trades.append(tr)
            result.bankroll_curve.append((t.ts, bankroll))
            open_position = tr
    return result
