"""The three named strategies from the v2 backtest spec, in one module.

Each takes a `MarketTick` (mid + bid + ask + minutes_to_close + optional
nws_forecast_p) and returns a `StrategyDecision` (side, edge, confidence)
OR None if the strategy doesn't fire.

  1. weather_signal       — NWS-derived fair value vs market mid
  2. complementary_arb    — exploit YES+NO ≠ $1.00 mispricings
  3. overreaction_fade    — fade sharp moves with no fresh news
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class MarketTick:
    """Minimal snapshot consumed by every strategy."""
    ticker: str
    title: str
    ts: int
    yes_bid_cents: int
    yes_ask_cents: int
    last_price_cents: int
    minutes_to_close: float
    nws_forecast_p: float | None = None   # 0..1 fair value if known, else None
    recent_prices: tuple[int, ...] = ()    # last-N yes_price_cents (oldest→newest)
    no_bid_cents: int | None = None        # if exchange returns it directly
    no_ask_cents: int | None = None
    seconds_since_news: int | None = None


@dataclass
class StrategyDecision:
    name: str
    side: str               # "yes" | "no"
    edge_cents: int         # signed; positive = bullish on chosen side
    confidence: float       # 0..1
    fair_value_cents: int
    rationale: str


def _no_side_prices(t: MarketTick) -> tuple[int, int]:
    """Return (no_bid_cents, no_ask_cents). Falls back to 100-yes_ask /
    100-yes_bid when the exchange doesn't return NO prices separately."""
    nb = t.no_bid_cents if t.no_bid_cents is not None else (100 - t.yes_ask_cents)
    na = t.no_ask_cents if t.no_ask_cents is not None else (100 - t.yes_bid_cents)
    return nb, na


# --------------------------------------------------------------------------
# 1. Weather signal
# --------------------------------------------------------------------------

def weather_signal(tick: MarketTick, *, min_edge_cents: int = 3) -> StrategyDecision | None:
    """Fire when the NWS-derived fair value disagrees with mid by >= min_edge.
    Caller is responsible for filling tick.nws_forecast_p (use kalshibot's
    fair_value.weather_fair_value for live; the backtester pulls historical
    forecasts the same way).
    """
    if tick.nws_forecast_p is None:
        return None
    fair_cents = int(round(tick.nws_forecast_p * 100))
    mid = (tick.yes_bid_cents + tick.yes_ask_cents) / 2
    edge = fair_cents - mid
    if abs(edge) < min_edge_cents:
        return None
    side = "yes" if edge > 0 else "no"
    return StrategyDecision(
        name="weather_signal", side=side,
        edge_cents=int(round(edge)), confidence=min(1.0, abs(edge) / 15),
        fair_value_cents=fair_cents,
        rationale=f"NWS fair value {fair_cents}¢ vs mid {mid:.0f}¢ "
                    f"({'YES' if side == 'yes' else 'NO'} edge {edge:+.0f}¢)",
    )


# --------------------------------------------------------------------------
# 2. Complementary arb
# --------------------------------------------------------------------------

def complementary_arb(tick: MarketTick, *,
                       sum_high_threshold_cents: int = 107,
                       sum_low_threshold_cents: int = 93) -> StrategyDecision | None:
    """If YES_ask + NO_ask > 107¢ -> sell both (we lose ε on the spread but
    get paid > $1 in total). If YES_bid + NO_bid < 93¢ -> buy both (we pay
    ε but one side must resolve $1).

    Returns the leg we should TAKE (the bigger of the two sides) and prices
    the edge in cents. The caller is expected to execute both legs.
    """
    nb, na = _no_side_prices(tick)
    sum_asks = tick.yes_ask_cents + na
    sum_bids = tick.yes_bid_cents + nb
    if sum_asks <= sum_low_threshold_cents:
        # buy both — we pay sum_asks, guaranteed payout 100¢
        edge = 100 - sum_asks
        side = "yes" if tick.yes_ask_cents < na else "no"   # cheaper leg first
        return StrategyDecision(
            name="complementary_arb", side=side,
            edge_cents=edge, confidence=1.0,
            fair_value_cents=100 - sum_asks,
            rationale=f"YES_ask+NO_ask={sum_asks}¢ — buy both, "
                        f"guaranteed +${edge/100:.2f} per pair",
        )
    if sum_bids >= sum_high_threshold_cents:
        # sell both — we collect sum_bids vs guaranteed 100¢ payout owed
        edge = sum_bids - 100
        side = "no" if tick.yes_bid_cents < nb else "yes"
        return StrategyDecision(
            name="complementary_arb", side=side,
            edge_cents=edge, confidence=1.0,
            fair_value_cents=sum_bids - 100,
            rationale=f"YES_bid+NO_bid={sum_bids}¢ — sell both, "
                        f"collect +${edge/100:.2f} per pair",
        )
    return None


# --------------------------------------------------------------------------
# 3. Overreaction fade
# --------------------------------------------------------------------------

def overreaction_fade(tick: MarketTick, *,
                       min_move_cents: int = 5,
                       lookback_ticks: int = 10,
                       max_seconds_since_news: int = 600,
                       min_minutes_to_close: float = 30.0) -> StrategyDecision | None:
    """If price moved >= min_move_cents in the last `lookback_ticks` AND
    no fresh news arrived (seconds_since_news either unknown or > 10 min),
    fade the move.

    Guards: don't fade in the last 30 minutes before close (resolution
    proximity dominates microstructure).
    """
    if tick.minutes_to_close < min_minutes_to_close:
        return None
    if not tick.recent_prices or len(tick.recent_prices) < 3:
        return None
    window = list(tick.recent_prices[-lookback_ticks:])
    if len(window) < 2:
        return None
    move = window[-1] - window[0]
    if abs(move) < min_move_cents:
        return None
    if tick.seconds_since_news is not None and tick.seconds_since_news < max_seconds_since_news:
        return None
    # Fade: position against the direction of the move.
    side = "no" if move > 0 else "yes"
    yes_mid = (tick.yes_bid_cents + tick.yes_ask_cents) / 2
    # Half-reversion is the realistic short-term target (in YES-cents).
    fair_yes = window[-1] - move // 2
    if side == "yes":
        edge = int(round(fair_yes - yes_mid))
    else:
        edge = int(round(yes_mid - fair_yes))
    if edge <= 0:
        return None
    return StrategyDecision(
        name="overreaction_fade", side=side,
        edge_cents=edge,
        confidence=min(1.0, abs(move) / 12),
        fair_value_cents=int(round(fair_yes)),
        rationale=f"Fading {move:+d}¢ move with no fresh news; "
                    f"half-reversion target {fair_yes:.0f}¢, fade {side.upper()}",
    )


# --------------------------------------------------------------------------
# Combine
# --------------------------------------------------------------------------

def combine_all(tick: MarketTick,
                 enabled: set[str] | None = None) -> list[StrategyDecision]:
    """Run every enabled strategy; return all decisions that fired.
    The backtester will pick the highest-confidence non-conflicting one
    (or take the arb as priority since it's risk-free in theory)."""
    enabled = enabled or {"weather_signal", "complementary_arb", "overreaction_fade"}
    out: list[StrategyDecision] = []
    if "weather_signal" in enabled:
        d = weather_signal(tick)
        if d: out.append(d)
    if "complementary_arb" in enabled:
        d = complementary_arb(tick)
        if d: out.append(d)
    if "overreaction_fade" in enabled:
        d = overreaction_fade(tick)
        if d: out.append(d)
    # Prioritize the arb (it's near risk-free), else best edge × confidence.
    out.sort(key=lambda d: (
        d.name != "complementary_arb",
        -(d.edge_cents * d.confidence),
    ))
    return out
