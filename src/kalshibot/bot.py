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

import pandas as pd

from .combiner import Combination
from .config import Config
from .data import MarketHistory, MarketSnapshot
from .kalshi_client import KalshiClient
from .strategies.registry import all_strategies, by_name


def cents_to_p(c: float) -> float:
    return max(0.01, min(0.99, float(c) / 100.0))


def snapshot_from_market(m: dict, ts: pd.Timestamp) -> MarketSnapshot:
    return MarketSnapshot(
        ticker=m["ticker"],
        ts=ts,
        yes_bid=cents_to_p(m.get("yes_bid", 50)),
        yes_ask=cents_to_p(m.get("yes_ask", 50)),
        last=cents_to_p(m.get("last_price", m.get("yes_bid", 50))),
        volume=float(m.get("volume", 0)),
        open_interest=float(m.get("open_interest", 0)),
        minutes_to_close=_minutes_to_close(m),
    )


def _minutes_to_close(m: dict) -> float:
    close = m.get("close_time") or m.get("expiration_time")
    if not close:
        return float("inf")
    try:
        ct = pd.to_datetime(close, utc=True)
        now = pd.Timestamp.utcnow().tz_localize("UTC")
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

    def ensemble_prob(self, h: MarketHistory) -> float:
        ws = self.combo.weights
        total = 0.0
        used = 0.0
        for name, w in ws.items():
            if w <= 0:
                continue
            try:
                s = by_name(name).predict(h)
            except KeyError:
                continue
            total += w * s.prob_yes
            used += w
        if used == 0:
            return float(h.mid.iloc[-1])
        return total / used

    def consider_market(self, m: dict) -> dict | None:
        snap = snapshot_from_market(m, pd.Timestamp.utcnow().tz_localize(None))
        h = self.histories.setdefault(snap.ticker, MarketHistory(snap.ticker))
        h.append(snap)
        if len(h.df) < 30:                # need warm-up
            return None
        p = self.ensemble_prob(h)
        mid = snap.mid
        edge = p - mid
        if abs(edge) < self.cfg.min_edge:
            return None
        f = kelly_fraction(p, mid)
        if f == 0.0:
            return None
        notional = min(self.cfg.max_position_usd, abs(f) * self.cfg.bankroll_usd)
        side = "yes" if f > 0 else "no"
        ref_price = snap.yes_ask if side == "yes" else (1.0 - snap.yes_bid)
        contracts = max(1, math.floor(notional / max(0.01, ref_price)))
        price_cents = int(round(ref_price * 100))
        decision = {
            "ticker": snap.ticker,
            "side": side,
            "count": contracts,
            "price_cents": price_cents,
            "ensemble_prob": round(p, 4),
            "mid": round(mid, 4),
            "edge": round(edge, 4),
        }
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
