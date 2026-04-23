"""Places real Kalshi orders, guarded by RiskLimits and logged to TradeJournal.

Used by:
  - Manual "Place bet" button on the Live page -> `execute_manual`
  - Auto-trade loop -> `execute_signal` (only when auto_trade_enabled)
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .config import Config
from .kalshi_client import KalshiAPIError, KalshiClient
from .risk_limits import RiskLimits, RiskState, check_order
from .trade_journal import JournalEntry, TradeJournal

if TYPE_CHECKING:
    from .live import LiveSignal


@dataclass
class ExecutionResult:
    ok: bool
    message: str
    entry: JournalEntry | None = None
    kalshi_response: dict | None = None


class TradeExecutor:
    def __init__(self, cfg: Config, client: KalshiClient | None = None,
                  limits: RiskLimits | None = None,
                  state: RiskState | None = None,
                  journal: TradeJournal | None = None) -> None:
        self.cfg = cfg
        self.client = client or KalshiClient(cfg)
        self.limits = limits or RiskLimits()
        self.risk_state = state or RiskState()
        self.journal = journal or TradeJournal()
        self._seen_signal_ids: set[str] = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ utils
    def _current_positions(self) -> list[dict]:
        """Fetch current open positions from Kalshi for the risk check."""
        try:
            page = self.client.get_positions(limit=200)
            raw = page.get("market_positions", [])
        except Exception:
            raw = []
        out = []
        from .genres import classify
        for p in raw:
            if (p.get("position") or 0) == 0:
                continue
            ticker = p.get("ticker", "")
            notional = abs((p.get("position") or 0) * (p.get("market_exposure") or 0) / 100.0)
            out.append({
                "ticker": ticker,
                "genre": classify(ticker).genre,
                "notional": notional,
            })
        return out

    def _current_cash(self) -> float:
        try:
            bal = self.client.get_balance()
            return float(bal.get("balance", 0)) / 100.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------ manual path
    def execute_manual(
        self,
        ticker: str,
        side: str,                    # "YES" | "NO"
        stake_usd: float,
        *,
        genre: str = "other",
        title: str = "",
        limit_price_cents: int | None = None,
        confidence: str = "manual",
        ensemble_prob: float = 0.5,
        reason: str = "manual override",
        strategy_probs: dict[str, float] | None = None,
    ) -> ExecutionResult:
        side_upper = side.upper()
        if side_upper not in ("YES", "NO"):
            return ExecutionResult(False, f"Invalid side: {side}")

        # Pull fresh orderbook to know the current best ask / bid if no price given.
        orderbook = None
        if limit_price_cents is None:
            try:
                orderbook = self.client.get_orderbook(ticker, depth=5)
            except Exception:
                orderbook = None
        if limit_price_cents is None and orderbook:
            book = orderbook.get("orderbook") or orderbook.get("orderbook_fp") or {}
            if side_upper == "YES":
                # best ask is the YES price we'd pay to buy
                levels = book.get("yes_ask") or book.get("yes_dollars") or []
                if levels:
                    top = levels[0]
                    limit_price_cents = int(round((top[0] if isinstance(top[0], int)
                                                    else top[0] * 100)))
            else:
                levels = book.get("no_ask") or book.get("no_dollars") or []
                if levels:
                    top = levels[0]
                    limit_price_cents = int(round((top[0] if isinstance(top[0], int)
                                                    else top[0] * 100)))
        if limit_price_cents is None:
            limit_price_cents = 55  # safe default if API gave nothing

        limit_price_cents = max(1, min(99, int(limit_price_cents)))
        price_dollars = limit_price_cents / 100.0
        contracts = max(1, math.floor(stake_usd / max(0.01, price_dollars)))
        notional = contracts * price_dollars

        cash = self._current_cash()
        positions = self._current_positions()
        ok, why = check_order(self.limits, self.risk_state,
                               ticker=ticker, genre=genre, notional_usd=notional,
                               cash_available=max(cash, 0.01),
                               open_positions=positions)
        if not ok:
            return ExecutionResult(False, why)

        if self.cfg.dry_run:
            entry = self.journal.add(
                ticker=ticker, genre=genre, title=title,
                side=side_upper, contracts=contracts,
                entry_price_cents=limit_price_cents,
                notional_usd=notional, source="manual",
                confidence=confidence, ensemble_prob=ensemble_prob,
                fair_value_prob=None, reason=reason + " [DRY_RUN]",
                kalshi_status="dry_run",
                contributing_strategies=list((strategy_probs or {}).keys()),
            )
            self.risk_state.today_orders_sent += 1
            return ExecutionResult(True, f"[DRY RUN] Would buy {contracts} {side_upper} "
                                           f"@ {limit_price_cents}¢ for ${notional:.2f}", entry)

        # Live send.
        try:
            resp = self.client.place_order(
                ticker=ticker, side=side_upper.lower(), action="buy",
                count=contracts, type_="limit",
                price_cents=limit_price_cents,
            )
        except KalshiAPIError as e:
            return ExecutionResult(False, f"Kalshi rejected order: {e.status} {e.body[:160]}")
        except Exception as e:
            return ExecutionResult(False, f"Order failed: {type(e).__name__}: {e}")

        order = resp.get("order") or resp
        entry = self.journal.add(
            ticker=ticker, genre=genre, title=title,
            side=side_upper, contracts=contracts,
            entry_price_cents=limit_price_cents,
            notional_usd=notional, source="manual",
            confidence=confidence, ensemble_prob=ensemble_prob,
            fair_value_prob=None, reason=reason,
            kalshi_order_id=order.get("order_id") or order.get("id"),
            kalshi_status=order.get("status") or "submitted",
            contributing_strategies=list((strategy_probs or {}).keys()),
        )
        self.risk_state.today_orders_sent += 1
        return ExecutionResult(True,
            f"Order submitted: {contracts} {side_upper} @ {limit_price_cents}¢",
            entry, resp)

    # ------------------------------------------------------------ auto path
    def execute_signal(self, sig: "LiveSignal") -> ExecutionResult | None:
        """Used by the auto-trade loop. Returns None if the signal is
        filtered out before it even reaches the risk checks."""
        if not self.limits.auto_trade_enabled:
            return None
        if sig.suggested_side == "flat":
            return None
        conf_rank = {"low": 0, "med": 1, "high": 2}
        if conf_rank.get(sig.confidence, 0) < conf_rank.get(self.limits.auto_trade_min_conf, 2):
            return None
        if abs(sig.edge) < self.limits.auto_trade_min_edge:
            return None
        # Debounce: don't re-submit on the same signal repeatedly.
        sig_id = f"{sig.ticker}:{round(sig.ensemble_prob, 2)}:{sig.suggested_side}"
        with self._lock:
            if sig_id in self._seen_signal_ids:
                return None
            self._seen_signal_ids.add(sig_id)
        entry_price = sig.yes_ask if sig.suggested_side == "YES" else (1 - sig.yes_bid)
        price_cents = int(round(entry_price * 100))
        stake = min(sig.suggested_notional, self.limits.max_notional_per_trade_usd)
        if stake < 1.0:
            return None
        return self.execute_manual(
            ticker=sig.ticker, side=sig.suggested_side,
            stake_usd=stake, genre=sig.genre, title=sig.title,
            limit_price_cents=price_cents, confidence=sig.confidence,
            ensemble_prob=sig.ensemble_prob, reason="auto: " + sig.rationale,
        )
