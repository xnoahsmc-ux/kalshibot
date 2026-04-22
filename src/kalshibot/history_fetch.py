"""Fetch real historical Kalshi candlestick data and build MarketHistory
objects suitable for the backtester.

Kalshi's candlesticks endpoint is:
  GET /series/{series_ticker}/markets/{ticker}/candlesticks

The series ticker isn't always on the market object. We try several
fallbacks: the explicit `series_ticker` field, the first segment of
`event_ticker` (a Kalshi convention - e.g. KXBTC-25DEC31H-T110K has series
KXBTC), and finally the first segment of the market ticker itself.
"""
from __future__ import annotations

import datetime as dt
from typing import Iterable

import pandas as pd

from .data import MarketHistory
from .kalshi_client import KalshiAPIError, KalshiClient


def _cents(v, default=None) -> float | None:
    if v is None:
        return default
    try:
        return float(v) / 100.0
    except (TypeError, ValueError):
        return default


def extract_series_ticker(m: dict) -> str | None:
    """Best-effort extraction of the series ticker from a market dict."""
    if not m:
        return None
    candidate = m.get("series_ticker")
    if candidate:
        return candidate
    et = m.get("event_ticker") or ""
    if et:
        return et.split("-", 1)[0]
    tk = m.get("ticker") or ""
    if tk:
        return tk.split("-", 1)[0]
    return None


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
    if not rows:
        return MarketHistory(ticker=ticker)
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx))
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return MarketHistory(ticker=ticker, df=df)


def fetch_open_markets(client: KalshiClient, limit: int = 50) -> list[dict]:
    """List current open markets. Each dict includes ticker + event_ticker."""
    out: list[dict] = []
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
    min_bars: int = 30,
    verbose: bool = True,
) -> list[MarketHistory]:
    """Build a universe of MarketHistory objects from live Kalshi markets.

    Verbose logging makes it obvious which markets yielded data and why
    others were skipped.
    """
    if tickers is None:
        markets = fetch_open_markets(client, limit=limit)
        if verbose:
            print(f"[fetch-history] found {len(markets)} open markets")
        if not markets:
            return []
        series_map = {m["ticker"]: extract_series_ticker(m) for m in markets}
        tickers = [m["ticker"] for m in markets]

    histories: list[MarketHistory] = []
    attempts = 0
    for t in tickers:
        attempts += 1
        series = (series_map or {}).get(t)
        if not series:
            if verbose:
                print(f"  [skip] {t}: no series_ticker resolved")
            continue
        try:
            h = fetch_history(client, series_ticker=series, ticker=t,
                              lookback_hours=lookback_hours)
            if len(h.df) >= min_bars:
                histories.append(h)
                if verbose:
                    print(f"  [ok]   {t}: {len(h.df)} bars (series={series})")
            else:
                if verbose:
                    print(f"  [skip] {t}: only {len(h.df)} bars (series={series})")
        except KalshiAPIError as e:
            if verbose:
                print(f"  [skip] {t}: {e.status} {e.body[:120]}")
        except Exception as e:
            if verbose:
                print(f"  [skip] {t}: {type(e).__name__}: {e}")
    if verbose:
        print(f"[fetch-history] produced {len(histories)}/{attempts} usable histories")
    return histories
