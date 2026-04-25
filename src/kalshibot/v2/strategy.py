"""Strategy combiner: turns a Market + signals into either an OrderIntent
or a 'pass'. Applies risk gates and Kelly sizing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..genres import classify
from .config import V2Config
from .data import Market
from .execution import OrderIntent
from .risk import DayState, GateResult, Position, gate, kelly_size
from .signals import (
    Signal,
    combine_signals,
    edge_cents,
    external_anchor_signal,
    microstructure_signal,
    mispricing_signal,
)


@dataclass
class Decision:
    intent: OrderIntent | None
    gate: GateResult
    fair_value: float
    confidence: float
    edge_cents: int
    contributors: list[Signal]
    market: Market


def evaluate(
    market: Market,
    cfg: V2Config,
    bankroll_usd: float,
    open_positions: list[Position],
    day_state: DayState,
    recent_mids: list[float],
) -> Decision:
    genre = classify(market.ticker, market.title).genre
    sigs = [
        mispricing_signal(market, genre),
        microstructure_signal(market, recent_mids),
        external_anchor_signal(market),
    ]
    fv, confidence, contribs = combine_signals(sigs, cfg.signals, market.mid)
    e = edge_cents(fv, market.mid)

    # Sub-threshold ⇒ no trade.
    if abs(e) < cfg.signals.min_edge_cents or confidence < cfg.signals.min_confidence:
        return Decision(None,
                          GateResult(False,
                                       f"sub-threshold edge={e}c conf={confidence:.2f}"),
                          fv, confidence, e, contribs, market)

    side = "yes" if e > 0 else "no"
    if side == "yes":
        entry_price = market.yes_ask
    else:
        entry_price = 1 - market.yes_bid
    if entry_price <= 0.01 or entry_price >= 0.99:
        return Decision(None, GateResult(False, "entry price at extreme"),
                          fv, confidence, e, contribs, market)

    p_win = fv if side == "yes" else (1 - fv)
    notional = kelly_size(p_win, entry_price, bankroll_usd, cfg.risk)
    if notional < 1.0:
        return Decision(None, GateResult(False, "Kelly size < $1"),
                          fv, confidence, e, contribs, market)

    g = gate(cfg.risk, day_state,
              bankroll_usd=bankroll_usd,
              target_notional_usd=notional,
              ticker=market.ticker,
              series=market.series_ticker,
              genre=genre,
              open_positions=open_positions)
    if not g.ok:
        return Decision(None, g, fv, confidence, e, contribs, market)

    contracts = max(1, math.floor(notional / max(0.01, entry_price)))
    target_cents = int(round(entry_price * 100))
    intent = OrderIntent(
        ticker=market.ticker, side=side, contracts=contracts,
        target_price_cents=target_cents,
        edge_cents_at_decision=e, fair_value=fv, confidence=confidence,
        reason=" | ".join(f"{s.name}:{s.fair_value:.2f}({s.confidence:.2f})"
                            for s in contribs),
    )
    return Decision(intent, g, fv, confidence, e, contribs, market)
