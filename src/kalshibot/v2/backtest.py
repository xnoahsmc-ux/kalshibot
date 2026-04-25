"""Event-driven backtester. Reads SQLite snapshots, replays them in order,
runs the v2 strategy, simulates fills with fees + half-spread + queue
position, and writes a markdown report.

Snapshots are sparse (every 30s in live), so we treat each snapshot as a
discrete decision point — passive limits, escalations and crosses are
modeled with a simplified queue model.
"""
from __future__ import annotations

import math
import sqlite3
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ..genres import classify
from .config import V2Config, load
from .data import Market
from .risk import DayState, Position
from .signals import Signal, combine_signals, edge_cents, microstructure_signal, mispricing_signal
from .strategy import evaluate


@dataclass
class BacktestTrade:
    ts: int
    ticker: str
    series: str
    genre: str
    side: str
    contracts: int
    entry_price_cents: int
    exit_price_cents: int | None = None
    pnl_cents: float = 0.0
    fees_cents: int = 0
    contributing: list[str] = field(default_factory=list)


@dataclass
class BacktestResult:
    trades: list[BacktestTrade]
    bankroll_curve: list[tuple[int, float]]    # (ts, bankroll)
    metrics: dict
    pnl_by_signal: dict
    pnl_by_genre: dict


def _equity_metrics(trades: list[BacktestTrade], starting_bankroll: float) -> dict:
    if not trades:
        return {"total_pnl_usd": 0.0, "trade_count": 0, "hit_rate": 0.0,
                  "sharpe": 0.0, "max_drawdown_pct": 0.0,
                  "avg_edge_cents": 0.0, "win_count": 0, "loss_count": 0}
    pnls = [t.pnl_cents / 100.0 for t in trades]
    cum = 0.0
    cumlist = []
    peak = starting_bankroll
    max_dd = 0.0
    bankroll = starting_bankroll
    for p in pnls:
        cum += p
        bankroll = starting_bankroll + cum
        peak = max(peak, bankroll)
        dd = (peak - bankroll) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
        cumlist.append(cum)
    daily_pnls = pnls   # treat each trade as a 'day' for Sharpe approximation
    sd = statistics.pstdev(daily_pnls) if len(daily_pnls) > 1 else 0.0
    sharpe = (statistics.mean(daily_pnls) / sd * math.sqrt(252)) if sd > 0 else 0.0
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p <= 0)
    return {
        "total_pnl_usd": round(sum(pnls), 2),
        "trade_count": len(trades),
        "hit_rate": round(wins / len(trades), 3) if trades else 0.0,
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "avg_edge_cents": round(sum(t.entry_price_cents for t in trades) / len(trades), 1)
                            if trades else 0.0,
        "win_count": wins, "loss_count": losses,
    }


def _settle_at(t: BacktestTrade, exit_cents: int, fee_cents: int) -> None:
    t.exit_price_cents = exit_cents
    if t.side == "yes":
        gross = t.contracts * (exit_cents - t.entry_price_cents)
    else:
        gross = t.contracts * ((100 - exit_cents) - (100 - t.entry_price_cents))
    t.pnl_cents = gross - fee_cents
    t.fees_cents = fee_cents


def run_backtest(cfg: V2Config, starting_bankroll: float = 1000.0,
                  db_path: str | None = None) -> BacktestResult:
    """Replay snapshots from the SQLite store. Each ticker's terminal price
    is treated as the settlement (good proxy for short-window markets in
    our 1-week test data; longer markets need a real settlement table)."""
    path = db_path or cfg.storage.snapshots_db
    if not Path(path).exists():
        return BacktestResult([], [], {"error": f"no snapshot db at {path}"},
                                {}, {})

    conn = sqlite3.connect(path)
    cur = conn.execute(
        "SELECT ts, ticker, yes_bid, yes_ask, last_price, volume_24h_usd, "
        "open_interest, minutes_to_close, series_ticker FROM snapshots "
        "ORDER BY ts"
    )
    rows = list(cur.fetchall())

    history: dict[str, list[float]] = defaultdict(list)
    open_trades: dict[str, BacktestTrade] = {}
    closed: list[BacktestTrade] = []
    bankroll = starting_bankroll
    bankroll_curve: list[tuple[int, float]] = []
    day_state = DayState()

    fee = cfg.backtest.fee_cents_per_trade
    slippage = cfg.backtest.slippage_cents

    for ts, ticker, yb, ya, last, vol, oi, ttc, series in rows:
        market = Market(ticker=ticker, title=ticker, event_ticker="",
                          series_ticker=series or ticker.split("-", 1)[0],
                          yes_bid=yb, yes_ask=ya, last_price=last,
                          volume_24h_usd=vol, open_interest=oi,
                          minutes_to_close=ttc)
        history[ticker].append(market.mid)
        recent_mids = history[ticker][-10:]
        # Existing trade on this ticker tracks against current mid.
        if ticker in open_trades and ttc <= 1:
            # Settle on terminal price proxy: 100 if mid > 0.5 else 0
            terminal = 100 if market.mid >= 0.5 else 0
            t = open_trades.pop(ticker)
            _settle_at(t, terminal + (slippage if t.side == "yes" else -slippage), fee)
            closed.append(t)
            bankroll += t.pnl_cents / 100.0
            day_state.record(t.pnl_cents / 100.0)
            bankroll_curve.append((ts, bankroll))
            continue
        if ticker in open_trades:
            continue   # already in this market; don't double up

        positions = [Position(ticker=t.ticker, series=t.series, genre=t.genre,
                                contracts=t.contracts, entry_price_cents=t.entry_price_cents,
                                side=t.side.upper())
                       for t in open_trades.values()]
        decision = evaluate(market, cfg, bankroll, positions, day_state, recent_mids)
        if decision.intent is None or day_state.halted:
            continue
        intent = decision.intent
        # Realistic fill: cross spread for the take side, half-spread + slippage
        # for passive (we assume queue position 50%).
        spread = max(1, market.spread_cents)
        if abs(decision.edge_cents) >= cfg.execution.take_if_edge_cents:
            fill_cents = intent.target_price_cents
        else:
            queue_cost = math.ceil(spread * cfg.backtest.queue_position_assumption)
            if intent.side == "yes":
                fill_cents = min(intent.target_price_cents + queue_cost, int(round(market.yes_ask * 100)))
            else:
                fill_cents = min(intent.target_price_cents + queue_cost,
                                  100 - int(round(market.yes_bid * 100)))
        fill_cents += slippage
        t = BacktestTrade(
            ts=ts, ticker=ticker, series=market.series_ticker,
            genre=classify(ticker).genre, side=intent.side,
            contracts=intent.contracts, entry_price_cents=fill_cents,
            contributing=[s.name for s in decision.contributors],
        )
        open_trades[ticker] = t
        bankroll -= t.contracts * fill_cents / 100.0

    # Force-close any still-open trades at last seen mid.
    for ticker, t in open_trades.items():
        last_mid = history[ticker][-1] if history[ticker] else 0.5
        terminal = 100 if last_mid >= 0.5 else 0
        _settle_at(t, terminal, fee)
        closed.append(t)
        bankroll += t.pnl_cents / 100.0

    # Aggregate
    pnl_by_signal: dict[str, float] = defaultdict(float)
    pnl_by_genre: dict[str, float] = defaultdict(float)
    for t in closed:
        pnl_by_genre[t.genre] += t.pnl_cents / 100.0
        for s in t.contributing:
            pnl_by_signal[s] += t.pnl_cents / 100.0
    metrics = _equity_metrics(closed, starting_bankroll)
    conn.close()
    return BacktestResult(trades=closed, bankroll_curve=bankroll_curve,
                            metrics=metrics, pnl_by_signal=dict(pnl_by_signal),
                            pnl_by_genre=dict(pnl_by_genre))


def write_report(result: BacktestResult, path: str = "BACKTEST_REPORT.md") -> Path:
    p = Path(path)
    lines = ["# Kalshibot v2 Backtest Report",
              "", f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}_",
              "", "## Headline metrics", ""]
    for k, v in result.metrics.items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "## P&L by signal", ""]
    for name, pnl in sorted(result.pnl_by_signal.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {name}: ${pnl:+.2f}")
    lines += ["", "## P&L by genre", ""]
    for name, pnl in sorted(result.pnl_by_genre.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {name}: ${pnl:+.2f}")
    lines += ["", f"## Trades ({len(result.trades)})", "",
                "| ts | ticker | side | contracts | entry | exit | pnl |",
                "|---|---|---|---|---|---|---|"]
    for t in result.trades[:200]:
        lines.append(f"| {t.ts} | {t.ticker} | {t.side} | {t.contracts} | "
                       f"{t.entry_price_cents}¢ | "
                       f"{(str(t.exit_price_cents)+'¢') if t.exit_price_cents is not None else '—'} | "
                       f"${t.pnl_cents/100:+.2f} |")
    p.write_text("\n".join(lines))
    return p
