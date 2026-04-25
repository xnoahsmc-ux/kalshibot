"""Three signal families. Each returns a fair-value probability for YES and a
confidence in [0,1]. Combined via a logistic stacker.

1. mispricing  - vs base rate (favorite-longshot bias) for recurring event types
2. microstructure - order-book imbalance + 5-min momentum (low-liq fade,
                    high-liq follow)
3. external_anchor - NWS / ESPN / FRED -> fair value vs mid

Combine: `combine_signals(market, signals)` returns (fair_value, confidence,
edge_cents, contributors). We trade only when |edge_cents| >= cfg.signals.min_edge_cents
AND confidence >= cfg.signals.min_confidence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..external_data import ESPNClient, NWSClient
from ..fair_value import sports_fair_value, weather_fair_value
from .config import SignalsCfg

if TYPE_CHECKING:
    from .data import Market


@dataclass
class Signal:
    name: str
    fair_value: float       # P(YES)
    confidence: float       # 0..1
    detail: str = ""


# --------------------------------------------------------------------------
# 1. Mispricing vs base rate (favorite-longshot bias) and per-genre prior
# --------------------------------------------------------------------------

# Empirical base rates from prediction-market literature: long-shots <15%
# resolve YES less often than priced; favorites >85% resolve YES more often.
BASE_RATE_PRIOR_BY_GENRE: dict[str, float] = {
    "weather":  0.50,
    "crypto":   0.50,
    "economy":  0.45,
    "sports":   0.50,
    "politics": 0.50,
    "other":    0.50,
}


def mispricing_signal(market: "Market", genre: str) -> Signal:
    p = market.mid
    base = BASE_RATE_PRIOR_BY_GENRE.get(genre, 0.5)
    # FLB shrinkage: extreme prices drift toward 0.5; near-mid prices toward
    # the per-genre base rate.
    extreme = (2 * (p - 0.5)) ** 2  # 0 at 0.5, 1 at 0/1
    fv = p - 0.06 * (p - 0.5) * extreme + 0.02 * (base - p) * (1 - extreme)
    fv = max(0.02, min(0.98, fv))
    conf = max(0.0, min(1.0, 0.4 + 0.6 * extreme))
    detail = f"FLB shrinkage from {p*100:.0f}% (extremeness={extreme:.2f}, base={base:.2f})"
    return Signal("mispricing", fv, conf, detail)


# --------------------------------------------------------------------------
# 2. Microstructure: order-book imbalance + 5-min momentum
# --------------------------------------------------------------------------

def microstructure_signal(market: "Market", recent_mids: list[float]) -> Signal:
    """recent_mids should be the last ~5 minutes of mid prices (one entry per
    poll). High-liquidity markets follow momentum; low-liquidity fade extremes.
    """
    if len(recent_mids) < 2:
        return Signal("microstructure", market.mid, 0.0, "insufficient history")
    momentum = recent_mids[-1] - recent_mids[0]
    spread = max(0.005, market.yes_ask - market.yes_bid)
    # Imbalance proxy: where last sits within the spread (closer to ask = YES pressure)
    pos = (market.last_price - market.yes_bid) / spread
    pos = max(0.0, min(1.0, pos))
    high_liq = market.volume_24h_usd > 5000 and market.open_interest > 500
    if high_liq:
        # Follow: YES pressure + positive momentum → up the fair value
        fv = market.mid + 0.5 * momentum + (pos - 0.5) * spread
    else:
        # Fade: low-liq extremes mean-revert
        fv = market.mid - 0.4 * momentum - (pos - 0.5) * spread * 0.5
    fv = max(0.02, min(0.98, fv))
    conf = min(1.0, abs(fv - market.mid) / max(0.01, spread))
    detail = (f"mom={momentum*100:+.1f}c imb={pos:.2f} "
                f"{'follow' if high_liq else 'fade'}")
    return Signal("microstructure", fv, conf * 0.8, detail)


# --------------------------------------------------------------------------
# 3. External anchors: NWS / ESPN / FRED
# --------------------------------------------------------------------------

# Module-level singletons so we share the cache.
_NWS = NWSClient()
_ESPN = ESPNClient()


def external_anchor_signal(market: "Market") -> Signal:
    fv = weather_fair_value(market.ticker, market.title, nws=_NWS)
    if fv is not None:
        return Signal("external_anchor", fv.prob_yes, fv.confidence, fv.detail)
    fv = sports_fair_value(market.ticker, market.title, espn=_ESPN)
    if fv is not None:
        return Signal("external_anchor", fv.prob_yes, fv.confidence, fv.detail)
    # FRED hook reserved for future econ markets.
    return Signal("external_anchor", market.mid, 0.0, "no anchor available")


# --------------------------------------------------------------------------
# Logistic stacker
# --------------------------------------------------------------------------

def _logit(p: float) -> float:
    p = max(1e-4, min(1 - 1e-4, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def combine_signals(
    signals: list[Signal],
    cfg: SignalsCfg,
    market_mid: float,
) -> tuple[float, float, list[Signal]]:
    """Return (fair_value, confidence_0_1, contributing_signals_with_nonzero_weight).

    Combine in logit-space, weighted by `weight × confidence` from each signal.
    The market mid is included as a regularizer with weight 1.0 so a single
    weak signal can't override the prevailing market without conviction.
    """
    contributors: list[Signal] = []
    weighted_logit = _logit(market_mid) * 1.0
    total_weight = 1.0
    for s in signals:
        w = cfg.weights.get(s.name, 0.0) * max(0.0, min(1.0, s.confidence))
        if w <= 0:
            continue
        contributors.append(s)
        weighted_logit += w * _logit(s.fair_value)
        total_weight += w
    fv = _sigmoid(weighted_logit / total_weight)
    # Confidence: how strongly the contributors agreed in direction
    if not contributors:
        return market_mid, 0.0, []
    same_side = sum(1 for s in contributors
                     if (s.fair_value - market_mid) * (fv - market_mid) >= 0)
    direction_agreement = same_side / len(contributors)
    avg_signal_conf = sum(s.confidence for s in contributors) / len(contributors)
    confidence = 0.6 * direction_agreement + 0.4 * avg_signal_conf
    return fv, confidence, contributors


def edge_cents(fair_value: float, market_mid: float) -> int:
    return int(round((fair_value - market_mid) * 100))
