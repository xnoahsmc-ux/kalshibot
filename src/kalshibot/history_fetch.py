"""Fetch real historical Kalshi candlestick data and build MarketHistory
objects suitable for the backtester.

Kalshi's candlesticks endpoint returns per-period OHLC plus open-interest and
volume. We normalize 1-99 cent quotes into [0,1] prices and fabricate a
bid/ask by offsetting by a small spread so existing strategies Just Work.
"""
from __future__ import annotations

import datetime as dt
from typing import Iterable

import pandas as pd

from .data import MarketHistory
from .kalshi_client import KalshiClient


def _cents(v, default=None) -> float | None:
    if v is None:
        return default
    try:
        return float(v) / 100.0
    except (TypeError, ValueError):
        return default


def fetch_history(
    client: KalshiClient,
    series_ticker: str,
    ticker: str,
    lookback_hours: int = 72,
    period_minutes: int = 1,
    synthetic_spread: float = 0.01,
) -> MarketHistory:
    """Fetch candles and produce a MarketHistory in our internal format."""
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(hours=lookback_hours)
    payload = client.get_candles(series_ticker=series_ticker, ticker=ticker,
                                 start=start, end=end, period_minutes=period_minutes)
    candles = payload.get("candlesticks", []) if isinstance(payload, dict) else []
    rows = []
    idx = []
    for c in candles:
        # Kalshi returns e.g. { "end_period_ts": ..., "yes_bid": { "close": .. },
        #                       "yes_ask": { "close": .. }, "price": {...},
        #                       "volume": .., "open_interest": .. }
        ts = c.get("end_period_ts") or c.get("end_ts") or c.get("open_ts")
        if ts is None:
            continue
        yb = _cents(((c.get("yes_bid") or {}).get("close")))
        ya = _cents(((c.get("yes_ask") or {}).get("close")))
        last = _cents(((c.get("price") or {}).get("close")))
        if last is None and yb is not None and ya is not None:
            last = 0.5 * (yb + ya)
        if last is None:
            continue
        if yb is None:
            yb = max(0.01, last - synthetic_spread / 2)
        if ya is None:
            ya = min(0.99, last + synthetic_spread / 2)
        rows.append({
            "yes_bid": yb, "yes_ask": ya, "last": last,
            "volume": float(c.get("volume") or 0),
            "open_interest": float(c.get("open_interest") or 0),
            "minutes_to_close": float("inf"),
        })
        idx.append(pd.Timestamp(int(ts), unit="s", tz="UTC").tz_convert(None))
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx))
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return MarketHistory(ticker=ticker, df=df)


def fetch_open_markets(client: KalshiClient, limit: int = 50) -> list[dict]:
    """List current open markets. Each dict includes ticker + series_ticker."""
    out = []
    cursor = None
    while len(out) < limit:
        page = client.get_markets(limit=min(200, limit - len(out)),
                                  status="open", cursor=cursor)
        markets = page.get("markets", []) if isinstance(page, dict) else []
        out.extend(markets)
        cursor = page.get("cursor")
        if not cursor or not markets:
            break
    return out[:limit]


def fetch_universe(
    client: KalshiClient,
    tickers: Iterable[str] | None = None,
    series_map: dict[str, str] | None = None,
    limit: int = 24,
    lookback_hours: int = 72,
) -> list[MarketHistory]:
    """Build a universe of MarketHistory objects from a list of tickers."""
    histories: list[MarketHistory] = []
    if tickers is None:
        markets = fetch_open_markets(client, limit=limit)
        tickers = [m["ticker"] for m in markets]
        series_map = {m["ticker"]: m.get("event_ticker") or m.get("series_ticker")
                      for m in markets}
    for t in tickers:
        series = (series_map or {}).get(t)
        if not series:
            continue
        try:
            h = fetch_history(client, series_ticker=series, ticker=t,
                              lookback_hours=lookback_hours)
            if len(h.df) >= 30:
                histories.append(h)
        except Exception as e:
            print(f"warn: could not fetch {t}: {e}")
    return histories
