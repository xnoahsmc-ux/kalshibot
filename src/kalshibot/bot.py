"""The live trading loop.

Every poll cycle we fetch open markets, build a `MarketHistory` per ticker by
appending the current snapshot to a rolling cache, ask the ensemble for a
probability, compute edge vs. the market mid, and (if not in dry-run) submit a
limit order sized by Kelly capped at the configured per-market budget.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from .combiner import Combination
from .config import Config
from .data import MarketHistory, MarketSnapshot
from .genres import classify
from .kalshi_client import KalshiClient
from .strategies.registry import all_strategies, by_name


def cents_to_p(c: float) -> float:
    return max(0.01, min(0.99, float(c) / 100.0))


def _extract_price(m: dict, cents_key: str, dollars_key: str,
                    fallback: float = 0.5) -> float:
    """Kalshi's newer responses use e.g. `yes_bid_dollars` (0.01-0.99) while
    older ones use `yes_bid` (1-99 cents). Handle both and clip to [0.01, 0.99]."""
    if dollars_key in m and m[dollars_key] is not None:
        try:
            return max(0.01, min(0.99, float(m[dollars_key])))
        except (TypeError, ValueError):
            pass
    if cents_key in m and m[cents_key] is not None:
        try:
            return cents_to_p(float(m[cents_key]))
        except (TypeError, ValueError):
            pass
    return fallback


def _extract_volume(m: dict) -> float:
    for k in ("volume_fp", "volume24h_fp", "volume_24h_fp",
              "volume", "volume_24h", "volume24h"):
        v = m.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return 0.0


def snapshot_from_market(m: dict, ts: pd.Timestamp) -> MarketSnapshot:
    yes_bid = _extract_price(m, "yes_bid", "yes_bid_dollars", 0.5)
    yes_ask = _extract_price(m, "yes_ask", "yes_ask_dollars",
                              min(0.99, yes_bid + 0.02))
    last = _extract_price(m, "last_price", "last_price_dollars",
                           0.5 * (yes_bid + yes_ask))
    return MarketSnapshot(
        ticker=m["ticker"],
        ts=ts,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        last=last,
        volume=_extract_volume(m),
        open_interest=float(m.get("open_interest") or m.get("open_interest_fp") or 0),
        minutes_to_close=_minutes_to_close(m),
    )


def _minutes_to_close(m: dict) -> float:
    close = m.get("close_time") or m.get("expiration_time")
    if not close:
        return float("inf")
    try:
        ct = pd.to_datetime(close, utc=True)
        now = pd.Timestamp.now(tz="UTC")
        return max(0.0, (ct - now).total_seconds() / 60.0)
    except Exception:
        return float("inf")


def kelly_fraction(p: float, mid: float, max_frac: float = 0.10) -> float:
    """Fractional Kelly for a binary contract priced at `mid` (prob units).

    Edge = p - mid; we long YES if edge > 0, NO otherwise. Variance of a
    one-shot binary trade ~ p(1-p), so we use a damped Kelly cap.
    """
    edge = p - mid
    if abs(edge) < 1e-6:
        return 0.0
    side = 1.0 if edge > 0 else -1.0
    f = abs(edge) / max(0.01, mid * (1 - mid))
    f = min(max_frac, f) * 0.5     # half-Kelly safety
    return side * f


@dataclass
class Trader:
    cfg: Config
    combo: Combination
    client: KalshiClient
    histories: dict[str, MarketHistory] = field(default_factory=dict)
    # Running exposure per genre in USD; keeps one hot genre from eating the book
    exposure_by_genre: dict[str, float] = field(default_factory=dict)
    max_exposure_per_genre: float = 0.35           # fraction of bankroll
    trade_log: list[dict] = field(default_factory=list)

    def ensemble_prob(self, h: MarketHistory) -> tuple[float, float]:
        """Returns (probability, dispersion). Dispersion is the weighted std
        across strategies and acts as a confidence gate.
        """
        ws = self.combo.weights
        probs = []
        weights = []
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
        probs_np = np.asarray(probs)
        w_np = np.asarray(weights)
        w_np = w_np / w_np.sum()
        mean = float((probs_np * w_np).sum())
        var = float((w_np * (probs_np - mean) ** 2).sum())
        return mean, float(np.sqrt(var))

    def consider_market(self, m: dict) -> dict | None:
        snap = snapshot_from_market(m, pd.Timestamp.now())
        h = self.histories.setdefault(snap.ticker, MarketHistory(snap.ticker))
        h.append(snap)
        if len(h.df) < 30:                # need warm-up
            return None
        p, disp = self.ensemble_prob(h)
        mid = snap.mid
        edge = p - mid
        if abs(edge) < self.cfg.min_edge:
            return None
        # Dispersion gate: if strategies disagree strongly, refuse to trade.
        if disp > 0.18:
            return None
        # Penalize size by disagreement - size scales like (1 - 2*disp), floored
        disp_scale = max(0.1, 1.0 - 2 * disp)
        f = kelly_fraction(p, mid) * disp_scale
        if f == 0.0:
            return None
        genre = classify(snap.ticker).genre
        used = self.exposure_by_genre.get(genre, 0.0)
        cap = self.max_exposure_per_genre * self.cfg.bankroll_usd
        room = max(0.0, cap - used)
        if room <= 0:
            return None
        notional = min(self.cfg.max_position_usd, abs(f) * self.cfg.bankroll_usd, room)
        if notional < 1.0:
            return None
        side = "yes" if f > 0 else "no"
        ref_price = snap.yes_ask if side == "yes" else (1.0 - snap.yes_bid)
        contracts = max(1, math.floor(notional / max(0.01, ref_price)))
        price_cents = int(round(ref_price * 100))
        decision = {
            "ticker": snap.ticker,
            "genre": genre,
            "side": side,
            "count": contracts,
            "price_cents": price_cents,
            "ensemble_prob": round(p, 4),
            "dispersion": round(disp, 4),
            "mid": round(mid, 4),
            "edge": round(edge, 4),
            "notional": round(notional, 2),
            "ts": pd.Timestamp.now(tz="UTC").isoformat(),
        }
        self.exposure_by_genre[genre] = used + notional
        self.trade_log.append(decision)
        if len(self.trade_log) > 500:
            self.trade_log = self.trade_log[-500:]
        if not self.cfg.dry_run:
            decision["order_response"] = self.client.place_order(
                ticker=snap.ticker, side=side, action="buy",
                count=contracts, type_="limit", price_cents=price_cents,
            )
        return decision

    def step(self, markets: Iterable[dict]) -> list[dict]:
        out = []
        for m in markets:
            d = self.consider_market(m)
            if d:
                out.append(d)
        return out


def run_forever(cfg: Config, combo: Combination) -> None:
    client = KalshiClient(cfg)
    trader = Trader(cfg=cfg, combo=combo, client=client)
    print(f"Starting kalshibot (dry_run={cfg.dry_run}, env={cfg.env}). "
          f"{int((combo.weights > 0).sum())} active strategies in ensemble.")
    while True:
        try:
            page = client.get_markets(limit=200, status="open")
            decisions = trader.step(page.get("markets", []))
            for d in decisions:
                print(d)
        except Exception as e:
            print(f"poll error: {e}")
        time.sleep(cfg.poll_seconds)
