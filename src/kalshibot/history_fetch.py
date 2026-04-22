"""Fetch real historical Kalshi candlestick data and build MarketHistory
objects suitable for the backtester.

Kalshi's candlesticks endpoint:
  GET /series/{series_ticker}/markets/{ticker}/candlesticks
  params: start_ts, end_ts (unix seconds), period_interval (1, 60, or 1440 min)

Kalshi limits how much data you can pull per call — small intervals support
a smaller window. We default to 60-minute candles over 7 days which lands
well inside every limit and also means low-activity markets still produce
enough bars to backtest on.

A lot of Kalshi's open markets (all those `KXMV…` multi-variant baskets)
have never traded and therefore have zero candles. We filter by 24h
volume so those markets don't even get fetched.
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


def _int_field(m: dict, *keys: str) -> int:
    for k in keys:
        v = m.get(k)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    return 0


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
    lookback_hours: int = 168,       # 7 days
    period_minutes: int = 60,         # hourly candles: widely supported
    synthetic_spread: float = 0.01,
) -> MarketHistory:
    """Fetch candles and produce a MarketHistory in our internal format."""
    end = dt.datetime.now(dt.timezone.utc)
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


def fetch_open_markets(client: KalshiClient, limit: int = 500) -> list[dict]:
    """List current open markets, paginated as needed. Caller can filter/sort."""
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


def liquidity_score(m: dict) -> int:
    """Catch-all liquidity proxy. Combines every field Kalshi might publish
    that hints the market has been traded on. We check a bunch of variants
    because the exact schema has drifted between API versions.
    """
    score = 0
    score += _int_field(m, "volume_24h", "volume24h", "volume24H", "recent_volume") * 10
    score += _int_field(m, "open_interest", "openInterest")
    score += _int_field(m, "volume") // 10
    score += _int_field(m, "liquidity", "liquidity_cents") // 100
    # Presence of a last price / bid-ask is itself a signal the market trades.
    if m.get("last_price") not in (None, 0):
        score += 5
    if m.get("yes_bid") and m.get("yes_ask"):
        score += 3
    return score


def rank_markets(markets: list[dict]) -> list[dict]:
    """Sort by our composite liquidity score so the most-traded markets are
    tried first."""
    return sorted(markets, key=liquidity_score, reverse=True)


def fetch_universe(
    client: KalshiClient,
    tickers: Iterable[str] | None = None,
    series_map: dict[str, str] | None = None,
    limit: int = 24,
    candidate_pool: int = 400,
    lookback_hours: int = 168,
    period_minutes: int = 60,
    min_volume_24h: int = 1,
    min_bars: int = 20,
    verbose: bool = True,
) -> list[MarketHistory]:
    """Pull a liquidity-sorted universe of markets and return MarketHistory
    objects for those that actually have backtestable candle data.

    * `candidate_pool`: how many open markets to fetch and rank before we
      start requesting candlesticks. 400 is enough to get past all the
      multi-variant basket markets that have zero trades.
    * `min_volume_24h`: skip markets that haven't traded at all in 24h -
      candlesticks for those come back empty.
    * Stops early once we have `limit` usable histories.
    """
    if tickers is None:
        markets = fetch_open_markets(client, limit=candidate_pool)
        if verbose:
            print(f"[fetch-history] fetched {len(markets)} open markets")
        if not markets:
            return []
        if verbose:
            sample_keys = sorted(markets[0].keys()) if markets else []
            print(f"[fetch-history] sample market keys: {sample_keys[:24]}")
        ranked = rank_markets(markets)
        # Any liquidity signal at all qualifies the market as worth a try.
        tradeable = [m for m in ranked if liquidity_score(m) >= min_volume_24h]
        if verbose:
            top_sample = [(m.get('ticker'), liquidity_score(m),
                            _int_field(m, 'volume'),
                            _int_field(m, 'open_interest'),
                            _int_field(m, 'volume_24h'))
                           for m in ranked[:5]]
            print(f"[fetch-history] {len(tradeable)} have liquidity >= {min_volume_24h}")
            print(f"[fetch-history] top5 by liquidity score: {top_sample}")
        # If nothing passes the filter, still try the highest-ranked ones
        # rather than giving up. Empty candle responses are handled downstream.
        tickers_list = [m["ticker"] for m in (tradeable or ranked[:limit * 3])]
        series_map = {m["ticker"]: extract_series_ticker(m) for m in (tradeable or ranked)}
        tickers = tickers_list

    histories: list[MarketHistory] = []
    attempts = 0
    for t in tickers:
        if len(histories) >= limit:
            break
        attempts += 1
        series = (series_map or {}).get(t)
        if not series:
            if verbose:
                print(f"  [skip] {t}: no series_ticker resolved")
            continue
        try:
            h = fetch_history(client, series_ticker=series, ticker=t,
                              lookback_hours=lookback_hours,
                              period_minutes=period_minutes)
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
        print(f"[fetch-history] kept {len(histories)} usable histories "
              f"after {attempts} attempts")
    return histories
