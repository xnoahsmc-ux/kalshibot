"""Universe filtering + SQLite snapshot store.

Pulls open Kalshi markets, filters by spread / volume / TTR / OI, and
persists 30s snapshots to data/markets.db so we can backtest on the same
distribution we live-trade in.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..bot import _extract_price, _extract_volume
from ..kalshi_client import KalshiClient
from .config import V2Config


@dataclass
class Market:
    ticker: str
    title: str
    event_ticker: str
    series_ticker: str
    yes_bid: float       # 0-1
    yes_ask: float
    last_price: float
    volume_24h_usd: float
    open_interest: int
    minutes_to_close: float

    @property
    def spread_cents(self) -> int:
        return int(round((self.yes_ask - self.yes_bid) * 100))

    @property
    def mid(self) -> float:
        return 0.5 * (self.yes_bid + self.yes_ask)


def _minutes_to_close(m: dict) -> float:
    close = m.get("close_time") or m.get("expiration_time")
    if not close:
        return float("inf")
    try:
        import pandas as pd
        ct = pd.to_datetime(close, utc=True)
        now = pd.Timestamp.now(tz="UTC")
        return max(0.0, (ct - now).total_seconds() / 60.0)
    except Exception:
        return float("inf")


def _to_market(m: dict) -> Market | None:
    ticker = m.get("ticker")
    if not ticker:
        return None
    yes_bid = _extract_price(m, "yes_bid", "yes_bid_dollars", 0.5)
    yes_ask = _extract_price(m, "yes_ask", "yes_ask_dollars", min(0.99, yes_bid + 0.02))
    last = _extract_price(m, "last_price", "last_price_dollars", 0.5 * (yes_bid + yes_ask))
    volume = _extract_volume(m)  # already in contracts
    last_dollars = last  # already 0-1
    return Market(
        ticker=ticker,
        title=m.get("title") or m.get("subtitle") or ticker,
        event_ticker=m.get("event_ticker") or "",
        series_ticker=m.get("series_ticker") or
                       (m.get("event_ticker", "").split("-", 1)[0] if m.get("event_ticker") else
                        ticker.split("-", 1)[0]),
        yes_bid=yes_bid, yes_ask=yes_ask, last_price=last,
        volume_24h_usd=volume * last_dollars,
        open_interest=int(m.get("open_interest") or m.get("open_interest_fp") or 0),
        minutes_to_close=_minutes_to_close(m),
    )


def filter_universe(markets: Iterable[Market], cfg: V2Config) -> list[Market]:
    u = cfg.universe
    out: list[Market] = []
    for m in markets:
        if m.spread_cents > u.max_spread_cents:
            continue
        if m.volume_24h_usd < u.min_volume_usd_24h:
            continue
        if not (u.min_time_to_resolution_minutes <= m.minutes_to_close
                 <= u.max_time_to_resolution_minutes):
            continue
        if m.open_interest < u.min_open_interest:
            continue
        out.append(m)
    return out


def fetch_filtered_universe(client: KalshiClient, cfg: V2Config,
                             pool: int = 600) -> list[Market]:
    markets: list[Market] = []
    cursor = None
    seen: set[str] = set()
    while len(markets) < pool:
        page = client.get_markets(limit=200, status="open", cursor=cursor)
        rows = page.get("markets", []) if isinstance(page, dict) else []
        for raw in rows:
            m = _to_market(raw)
            if m is None or m.ticker in seen:
                continue
            seen.add(m.ticker)
            markets.append(m)
        cursor = page.get("cursor") if isinstance(page, dict) else None
        if not cursor or not rows:
            break
    return filter_universe(markets, cfg)


# -------- SQLite snapshots ----------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS snapshots (
  ts INTEGER NOT NULL,
  ticker TEXT NOT NULL,
  yes_bid REAL, yes_ask REAL, last_price REAL,
  volume_24h_usd REAL, open_interest INTEGER,
  minutes_to_close REAL,
  series_ticker TEXT,
  PRIMARY KEY (ts, ticker)
);
CREATE INDEX IF NOT EXISTS idx_ticker_ts ON snapshots (ticker, ts);
"""


class SnapshotStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), isolation_level=None,
                                      check_same_thread=False)
        for stmt in _DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self.conn.execute(stmt)

    def write(self, markets: Iterable[Market], ts: int | None = None) -> int:
        ts = ts or int(time.time())
        rows = [(ts, m.ticker, m.yes_bid, m.yes_ask, m.last_price,
                  m.volume_24h_usd, m.open_interest, m.minutes_to_close,
                  m.series_ticker) for m in markets]
        self.conn.executemany(
            "INSERT OR REPLACE INTO snapshots "
            "(ts, ticker, yes_bid, yes_ask, last_price, volume_24h_usd, "
            "open_interest, minutes_to_close, series_ticker) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        return len(rows)

    def history(self, ticker: str, since_ts: int = 0) -> list[tuple]:
        cur = self.conn.execute(
            "SELECT ts, yes_bid, yes_ask, last_price, volume_24h_usd, "
            "open_interest, minutes_to_close FROM snapshots "
            "WHERE ticker = ? AND ts >= ? ORDER BY ts",
            (ticker, since_ts),
        )
        return list(cur.fetchall())

    def all_tickers(self, since_ts: int = 0) -> list[str]:
        cur = self.conn.execute(
            "SELECT DISTINCT ticker FROM snapshots WHERE ts >= ?",
            (since_ts,),
        )
        return [r[0] for r in cur.fetchall()]

    def close(self) -> None:
        self.conn.close()
