"""Background service that polls Kalshi's REST API, builds rolling
MarketHistory per ticker, and applies the ensemble to produce live signals.

Design:
  * One poller thread per LiveFeed instance. Call `start()` / `stop()`.
  * Thread-safe `snapshot()` returns the latest view for the web layer.
  * Histories are bounded to `max_bars` entries per ticker.
  * No trading is performed here - this is data + signal only. The live
    trade loop in `bot.py` is what actually submits orders.
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .bot import kelly_fraction, snapshot_from_market
from .combiner import Combination
from .config import Config
from .data import MarketHistory
from .genres import classify
from .kalshi_client import KalshiClient
from .strategies.registry import by_name


@dataclass
class LiveSignal:
    ticker: str
    genre: str
    title: str
    yes_bid: float
    yes_ask: float
    mid: float
    last: float
    volume: float
    ensemble_prob: float
    dispersion: float
    edge: float
    kelly: float
    suggested_side: str            # "YES" | "NO" | "flat"
    suggested_notional: float
    minutes_to_close: float
    ts: str


@dataclass
class LiveStatus:
    running: bool = False
    last_poll_at: float | None = None
    last_error: str | None = None
    markets_seen: int = 0
    polls: int = 0
    authed: bool = False


class LiveFeed:
    def __init__(
        self,
        cfg: Config,
        combo: Combination | None = None,
        client: KalshiClient | None = None,
        max_markets: int = 40,
        max_bars: int = 400,
    ) -> None:
        self.cfg = cfg
        self.combo = combo
        self.client = client or KalshiClient(cfg)
        self.max_markets = max_markets
        self.max_bars = max_bars
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._histories: dict[str, MarketHistory] = {}
        self._titles: dict[str, str] = {}
        self._signals: dict[str, LiveSignal] = {}
        self._status = LiveStatus()

    # ------------------------------------------------------------------ API
    def status(self) -> LiveStatus:
        return self._status

    def signals(self) -> list[LiveSignal]:
        with self._lock:
            # Sorted by absolute edge so the most opinionated markets bubble up
            return sorted(self._signals.values(), key=lambda s: -abs(s.edge))

    def history(self, ticker: str) -> MarketHistory | None:
        return self._histories.get(ticker)

    def set_combination(self, combo: Combination) -> None:
        self.combo = combo

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._status.running = True

    def stop(self) -> None:
        self._stop.set()
        self._status.running = False

    # --------------------------------------------------------------- worker
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll_once()
                self._status.polls += 1
                self._status.last_poll_at = time.time()
                self._status.last_error = None
            except Exception as e:
                self._status.last_error = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            # Wait, but wake up promptly on stop
            self._stop.wait(max(5, self.cfg.poll_seconds))

    def _poll_once(self) -> None:
        # Pull a bigger candidate pool so the most-traded markets surface,
        # then keep only the ones with any liquidity signal.
        from .history_fetch import liquidity_score, rank_markets
        page = self.client.get_markets(limit=max(200, self.max_markets * 4),
                                         status="open")
        all_markets = page.get("markets", []) if isinstance(page, dict) else []
        self._status.authed = True
        # Rank by liquidity and keep the top N - otherwise Kalshi's default
        # ordering buries real markets behind hundreds of empty baskets.
        ranked = rank_markets(all_markets)
        tradeable = [m for m in ranked if liquidity_score(m) > 0]
        chosen = (tradeable or ranked)[: self.max_markets]
        self._status.markets_seen = len(chosen)
        now = pd.Timestamp.now()
        for m in chosen:
            ticker = m.get("ticker")
            if not ticker:
                continue
            try:
                self._titles[ticker] = m.get("title") or m.get("subtitle") or ticker
                snap = snapshot_from_market(m, now)
                hist = self._histories.setdefault(ticker, MarketHistory(ticker))
                hist.append(snap)
                if len(hist.df) > self.max_bars:
                    hist.df = hist.df.iloc[-self.max_bars:]
                self._signals[ticker] = self._signal_for(ticker, hist, snap)
            except Exception as e:
                # One flaky ticker should never kill the whole poll
                self._status.last_error = f"{ticker}: {type(e).__name__}: {e}"

    def _signal_for(self, ticker: str, h: MarketHistory, snap) -> LiveSignal:
        title = self._titles.get(ticker, ticker)
        mid = snap.mid
        genre = classify(ticker, title).genre
        prob, disp = self._ensemble_prob(h)
        edge = prob - mid
        f = kelly_fraction(prob, mid)
        disp_scale = max(0.1, 1.0 - 2 * disp)
        f_damped = f * disp_scale
        notional = min(self.cfg.max_position_usd, abs(f_damped) * self.cfg.bankroll_usd)
        if abs(edge) < self.cfg.min_edge or disp > 0.18 or abs(f_damped) < 1e-6:
            side = "flat"
            notional = 0.0
        else:
            side = "YES" if f_damped > 0 else "NO"
        return LiveSignal(
            ticker=ticker,
            genre=genre,
            title=title,
            yes_bid=snap.yes_bid,
            yes_ask=snap.yes_ask,
            mid=mid,
            last=snap.last,
            volume=snap.volume,
            ensemble_prob=float(prob),
            dispersion=float(disp),
            edge=float(edge),
            kelly=float(f_damped),
            suggested_side=side,
            suggested_notional=float(notional),
            minutes_to_close=float(snap.minutes_to_close),
            ts=pd.Timestamp.now(tz="UTC").isoformat(),
        )

    def _ensemble_prob(self, h: MarketHistory) -> tuple[float, float]:
        if self.combo is None or len(h.df) < 5:
            return float(h.mid.iloc[-1]), 0.0
        import numpy as np
        ws = self.combo.weights
        probs: list[float] = []
        weights: list[float] = []
        for name, w in ws.items():
            if w <= 0:
                continue
            try:
                s = by_name(name).predict(h)
            except KeyError:
                continue
            probs.append(s.prob_yes)
            weights.append(float(w))
        if not probs:
            return float(h.mid.iloc[-1]), 0.0
        p = np.asarray(probs)
        ww = np.asarray(weights)
        ww = ww / ww.sum()
        mean = float((p * ww).sum())
        var = float((ww * (p - mean) ** 2).sum())
        return mean, float(var ** 0.5)
