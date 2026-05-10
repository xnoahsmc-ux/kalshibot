"""Pull resolved (settled) Kalshi markets and cache them in SQLite for the
backtester. Separate from v2/data.py SnapshotStore, which is for live
polling — this module is purely historical.

Kalshi exposes settled markets via `/trade-api/v2/markets?status=settled`.
Each market record contains the resolution (yes/no/void), the close
timestamp, and the last-traded price at close. We hydrate per-market
trade history via `/markets/trades?ticker=…` for replay.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from ..kalshi_client import KalshiAPIError, KalshiClient


_DDL = """
CREATE TABLE IF NOT EXISTS settled_markets (
  ticker TEXT PRIMARY KEY,
  title TEXT,
  series_ticker TEXT,
  event_ticker TEXT,
  open_ts INTEGER,
  close_ts INTEGER,
  result TEXT,            -- 'yes' | 'no' | 'void'
  yes_settle_price_cents INTEGER,
  category TEXT,
  raw_json TEXT,
  fetched_at INTEGER
);

CREATE TABLE IF NOT EXISTS settled_trades (
  ticker TEXT NOT NULL,
  ts INTEGER NOT NULL,
  yes_price_cents INTEGER,
  count INTEGER,
  taker_side TEXT,
  PRIMARY KEY (ticker, ts, yes_price_cents, count)
);
CREATE INDEX IF NOT EXISTS idx_trades_ticker_ts ON settled_trades (ticker, ts);
"""


@dataclass
class SettledMarket:
    ticker: str
    title: str
    series_ticker: str
    event_ticker: str
    open_ts: int | None
    close_ts: int | None
    result: str | None
    yes_settle_price_cents: int | None
    category: str
    raw: dict


@dataclass
class TradeTick:
    ts: int
    yes_price_cents: int
    count: int
    taker_side: str


class HistoricalStore:
    def __init__(self, path: str = "data/kalshi_historical.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), isolation_level=None,
                                      check_same_thread=False)
        for stmt in _DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self.conn.execute(stmt)

    # ---- markets ---------------------------------------------------------
    def upsert_markets(self, markets: list[SettledMarket]) -> int:
        rows = [(m.ticker, m.title, m.series_ticker, m.event_ticker,
                  m.open_ts, m.close_ts, m.result,
                  m.yes_settle_price_cents, m.category,
                  json.dumps(m.raw), int(time.time()))
                 for m in markets]
        self.conn.executemany(
            "INSERT OR REPLACE INTO settled_markets "
            "(ticker, title, series_ticker, event_ticker, open_ts, close_ts, "
            "result, yes_settle_price_cents, category, raw_json, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
        return len(rows)

    def list_markets(self, series_prefix: str | None = None,
                      since_ts: int = 0, limit: int = 1000) -> list[SettledMarket]:
        q = "SELECT ticker, title, series_ticker, event_ticker, open_ts, close_ts, "\
            "result, yes_settle_price_cents, category, raw_json " \
            "FROM settled_markets WHERE 1=1"
        params: list = []
        if series_prefix:
            q += " AND series_ticker LIKE ?"; params.append(series_prefix + "%")
        if since_ts:
            q += " AND COALESCE(close_ts, 0) >= ?"; params.append(since_ts)
        q += " ORDER BY close_ts DESC LIMIT ?"; params.append(limit)
        out = []
        for row in self.conn.execute(q, params):
            try:
                raw = json.loads(row[9] or "{}")
            except Exception:
                raw = {}
            out.append(SettledMarket(
                ticker=row[0], title=row[1] or "", series_ticker=row[2] or "",
                event_ticker=row[3] or "", open_ts=row[4], close_ts=row[5],
                result=row[6], yes_settle_price_cents=row[7],
                category=row[8] or "", raw=raw,
            ))
        return out

    # ---- trades ----------------------------------------------------------
    def upsert_trades(self, ticker: str, trades: list[TradeTick]) -> int:
        rows = [(ticker, t.ts, t.yes_price_cents, t.count, t.taker_side)
                 for t in trades]
        self.conn.executemany(
            "INSERT OR REPLACE INTO settled_trades "
            "(ticker, ts, yes_price_cents, count, taker_side) "
            "VALUES (?, ?, ?, ?, ?)", rows)
        return len(rows)

    def trades_for(self, ticker: str, start_ts: int = 0,
                    end_ts: int | None = None) -> list[TradeTick]:
        end_ts = end_ts or int(time.time()) + 86400
        cur = self.conn.execute(
            "SELECT ts, yes_price_cents, count, taker_side "
            "FROM settled_trades WHERE ticker = ? AND ts BETWEEN ? AND ? "
            "ORDER BY ts", (ticker, start_ts, end_ts))
        return [TradeTick(ts=r[0], yes_price_cents=r[1], count=r[2],
                            taker_side=r[3] or "") for r in cur.fetchall()]

    def close(self) -> None:
        self.conn.close()


# --------------------------------------------------------------------------
# Fetchers
# --------------------------------------------------------------------------

def _ts(v) -> int | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v)
    try:
        import datetime as dt
        return int(dt.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
                    .timestamp())
    except Exception:
        return None


def _parse_market(raw: dict) -> SettledMarket | None:
    if not raw.get("ticker"):
        return None
    return SettledMarket(
        ticker=raw["ticker"],
        title=raw.get("title") or raw.get("subtitle") or "",
        series_ticker=raw.get("series_ticker")
            or (raw.get("event_ticker", "") or "").split("-", 1)[0]
            or raw["ticker"].split("-", 1)[0],
        event_ticker=raw.get("event_ticker") or "",
        open_ts=_ts(raw.get("open_time") or raw.get("opened_at")),
        close_ts=_ts(raw.get("close_time") or raw.get("expiration_time")),
        result=(raw.get("result") or "").lower() or None,
        yes_settle_price_cents=(100 if (raw.get("result") or "").lower() == "yes"
                                 else 0 if (raw.get("result") or "").lower() == "no"
                                 else None),
        category=raw.get("category") or "",
        raw=raw,
    )


def fetch_settled(
    client: KalshiClient, store: HistoricalStore, *,
    series_prefix: str | None = None,
    pages: int = 8, page_size: int = 200,
) -> int:
    """Pull `pages × page_size` settled markets via /markets?status=settled.
    Optionally filter by series prefix client-side."""
    total = 0
    cursor = None
    for _ in range(pages):
        page = client.get_markets(limit=page_size, status="settled", cursor=cursor)
        markets = page.get("markets", []) if isinstance(page, dict) else []
        parsed = []
        for raw in markets:
            m = _parse_market(raw)
            if m is None:
                continue
            if series_prefix and not m.series_ticker.startswith(series_prefix):
                continue
            parsed.append(m)
        if parsed:
            total += store.upsert_markets(parsed)
        cursor = page.get("cursor") if isinstance(page, dict) else None
        if not cursor or not markets:
            break
    return total


def fetch_trades_for(client: KalshiClient, store: HistoricalStore,
                      ticker: str, max_pages: int = 6) -> int:
    total = 0
    cursor = None
    for _ in range(max_pages):
        try:
            page = client.get_trades(ticker=ticker, limit=1000, cursor=cursor)
        except KalshiAPIError:
            break
        trades = page.get("trades", []) if isinstance(page, dict) else []
        parsed = []
        for t in trades:
            ts = _ts(t.get("created_time") or t.get("ts"))
            if ts is None:
                continue
            parsed.append(TradeTick(
                ts=ts,
                yes_price_cents=int(t.get("yes_price") or t.get("price") or 0),
                count=int(t.get("count") or 0),
                taker_side=str(t.get("taker_side") or ""),
            ))
        if parsed:
            total += store.upsert_trades(ticker, parsed)
        cursor = page.get("cursor") if isinstance(page, dict) else None
        if not cursor or not trades:
            break
    return total
