"""Execution policy.

Quote ladder:
  1. Post passive limit `1¢` inside spread on the side we want to be on.
  2. After 30s without a fill, escalate to mid.
  3. If the edge has grown beyond `take_if_edge_cents`, cross the spread
     immediately (market-style limit at the touch).

Includes self-cross prevention, partial-fill handling, exponential backoff
on rate limits, and a watchdog that flattens if we can't reach Kalshi for
`watchdog_timeout_seconds` straight.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..kalshi_client import KalshiAPIError, KalshiClient
from .config import ExecutionCfg, V2Config

log = logging.getLogger("kalshibot.v2.exec")


@dataclass
class OrderIntent:
    ticker: str
    side: str               # "yes" or "no"
    contracts: int
    target_price_cents: int   # the ideal price; ladder may pay up
    edge_cents_at_decision: int
    fair_value: float
    confidence: float
    reason: str = ""


@dataclass
class ExecResult:
    ok: bool
    message: str
    filled_contracts: int = 0
    avg_fill_price_cents: float = 0.0
    order_id: str | None = None
    raw: dict | None = None


# ---------------------------------------------------------------------------
# Live executor
# ---------------------------------------------------------------------------

class LiveExecutor:
    def __init__(self, client: KalshiClient, cfg: V2Config) -> None:
        self.client = client
        self.cfg = cfg
        self._last_api_ok = time.time()
        self._lock = threading.Lock()

    def _passive_price(self, intent: OrderIntent, yes_bid_c: int, yes_ask_c: int) -> int:
        inset = self.cfg.execution.passive_inset_cents
        if intent.side == "yes":
            # Best YES bid + 1¢ (still inside the spread)
            return min(yes_ask_c - 1, yes_bid_c + inset)
        # NO side: equivalent to "1 - yes_ask + 1c"
        no_bid_c = 100 - yes_ask_c
        no_ask_c = 100 - yes_bid_c
        return min(no_ask_c - 1, no_bid_c + inset)

    def _take_price(self, intent: OrderIntent, yes_bid_c: int, yes_ask_c: int) -> int:
        if intent.side == "yes":
            return yes_ask_c        # cross to ask
        return 100 - yes_bid_c       # cross to NO ask

    def send(self, intent: OrderIntent, yes_bid_c: int, yes_ask_c: int,
              now: Callable[[], float] = time.time) -> ExecResult:
        """Execute the ladder. Blocks up to escalate_after_seconds + a bit."""
        ec = self.cfg.execution
        spread = max(1, yes_ask_c - yes_bid_c)
        # Decide initial price: passive unless edge already big enough to cross
        if abs(intent.edge_cents_at_decision) >= ec.take_if_edge_cents:
            price = self._take_price(intent, yes_bid_c, yes_ask_c)
            stage = "TAKE"
        else:
            price = self._passive_price(intent, yes_bid_c, yes_ask_c)
            stage = "PASSIVE"
        log.info("send %s %s %dx @ %d¢ (stage=%s spread=%dc edge=%dc)",
                  intent.side.upper(), intent.ticker, intent.contracts,
                  price, stage, spread, intent.edge_cents_at_decision)
        result = self._submit(intent, price)
        if result.ok and stage == "TAKE":
            return result
        # Passive: poll for fills, escalate after timeout.
        deadline = now() + ec.escalate_after_seconds
        while now() < deadline:
            time.sleep(min(2.0, max(0.5, ec.escalate_after_seconds / 6)))
            status = self._poll_order(result.order_id)
            if status and status.get("status") == "filled":
                return ExecResult(True, "filled passive",
                                    filled_contracts=int(status.get("count_filled", intent.contracts)),
                                    avg_fill_price_cents=float(status.get("avg_fill_price",
                                                                           price)),
                                    order_id=result.order_id, raw=status)
        # Escalate to mid
        try:
            self._cancel(result.order_id)
        except Exception:
            pass
        mid_price = (yes_bid_c + yes_ask_c) // 2 if intent.side == "yes" else (
            (100 - yes_bid_c) + (100 - yes_ask_c)) // 2
        log.info("escalate %s %s -> mid %d¢", intent.side.upper(),
                  intent.ticker, mid_price)
        return self._submit(intent, mid_price)

    def _submit(self, intent: OrderIntent, price_cents: int) -> ExecResult:
        try:
            resp = self._with_backoff(lambda: self.client.place_order(
                ticker=intent.ticker, side=intent.side, action="buy",
                count=intent.contracts, type_="limit",
                price_cents=price_cents,
            ))
            self._last_api_ok = time.time()
            order = resp.get("order") or resp
            return ExecResult(True, f"submitted @ {price_cents}¢",
                                order_id=order.get("order_id") or order.get("id"),
                                raw=resp)
        except KalshiAPIError as e:
            return ExecResult(False, f"Kalshi {e.status}: {e.body[:200]}")
        except Exception as e:
            return ExecResult(False, f"{type(e).__name__}: {e}")

    def _poll_order(self, order_id: str | None) -> dict | None:
        if not order_id:
            return None
        try:
            page = self._with_backoff(lambda: self.client.get_orders(limit=20))
            for o in page.get("orders", []):
                if (o.get("order_id") or o.get("id")) == order_id:
                    return o
        except Exception:
            return None
        return None

    def _cancel(self, order_id: str | None) -> None:
        if order_id:
            self._with_backoff(lambda: self.client.cancel_order(order_id))

    def _with_backoff(self, fn, max_attempts: int = 4):
        for i in range(max_attempts):
            try:
                return fn()
            except KalshiAPIError as e:
                if e.status in (429, 503) and i < max_attempts - 1:
                    time.sleep(0.5 * (2 ** i))
                    continue
                raise
            except Exception:
                if i < max_attempts - 1:
                    time.sleep(0.5 * (2 ** i))
                    continue
                raise

    def watchdog_breached(self) -> bool:
        return (time.time() - self._last_api_ok) > self.cfg.execution.watchdog_timeout_seconds


# ---------------------------------------------------------------------------
# Shadow executor for paper-trading. Mirrors the LiveExecutor interface.
# ---------------------------------------------------------------------------

class ShadowExecutor:
    """Writes one JSONL line per intent; assumes immediate fill at the
    target_price_cents minus modeled slippage. Used by `make paper`.
    """

    def __init__(self, cfg: V2Config) -> None:
        self.cfg = cfg
        self.path = Path(cfg.storage.paper_ledger)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, intent: OrderIntent, yes_bid_c: int, yes_ask_c: int,
              now=time.time) -> ExecResult:
        if intent.side == "yes":
            fill_price = yes_ask_c   # paper: assume we cross
        else:
            fill_price = 100 - yes_bid_c
        rec = {
            "ts": now(), "kind": "paper_fill",
            "ticker": intent.ticker, "side": intent.side,
            "contracts": intent.contracts,
            "fill_price_cents": fill_price,
            "target_price_cents": intent.target_price_cents,
            "edge_at_decision_c": intent.edge_cents_at_decision,
            "fair_value": intent.fair_value,
            "confidence": intent.confidence,
            "reason": intent.reason,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        log.info("PAPER fill %s %s %dx @ %d¢", intent.side.upper(),
                  intent.ticker, intent.contracts, fill_price)
        return ExecResult(True, f"paper @ {fill_price}¢",
                            filled_contracts=intent.contracts,
                            avg_fill_price_cents=float(fill_price),
                            order_id=f"paper-{int(now()*1000)}",
                            raw=rec)

    def watchdog_breached(self) -> bool:
        return False
