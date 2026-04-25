"""Top picks + combo discovery.

Given a snapshot of live signals, rank them by expected-profit-per-dollar
and find the best 2-3-leg combination of independent markets. Combos
multiply edge the way a parlay does, but only when the markets are
genuinely uncorrelated (different genres / different events).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .live import LiveSignal


@dataclass
class Pick:
    ticker: str
    title: str
    genre: str
    side: str
    entry_price: float          # 0-1 fraction
    win_prob: float
    edge: float
    confidence: str
    rationale: str
    suggested_stake_usd: float
    expected_profit_usd: float
    fair_value_source: str = ""
    fair_value_detail: str = ""

    def ev_per_dollar(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return ((self.win_prob * (1 - self.entry_price)) / self.entry_price
                 - (1 - self.win_prob))


@dataclass
class Combo:
    legs: list[Pick]
    combined_win_prob: float       # P(all legs win) assuming independence
    combined_payout_per_dollar: float
    expected_value_per_dollar: float
    suggested_stake_usd: float
    expected_profit_usd: float
    rationale: str = ""


def signals_to_picks(signals: list["LiveSignal"], bankroll: float) -> list[Pick]:
    picks: list[Pick] = []
    for s in signals:
        if s.suggested_side == "flat":
            continue
        entry = s.yes_ask if s.suggested_side == "YES" else (1 - s.yes_bid)
        wp = s.win_prob if s.win_prob > 0 else (
            s.ensemble_prob if s.suggested_side == "YES" else 1 - s.ensemble_prob
        )
        # EV per $1 staked
        ev = ((wp * (1 - entry)) / max(0.01, entry)) - (1 - wp)
        if ev <= 0:
            continue
        stake = max(1.0, min(s.suggested_notional, bankroll * 0.05))
        picks.append(Pick(
            ticker=s.ticker, title=s.title, genre=s.genre,
            side=s.suggested_side, entry_price=entry,
            win_prob=wp, edge=s.edge, confidence=s.confidence,
            rationale=s.rationale,
            suggested_stake_usd=stake,
            expected_profit_usd=stake * ev,
            fair_value_source=s.fair_value_source,
            fair_value_detail=s.fair_value_detail,
        ))
    picks.sort(key=lambda p: -p.expected_profit_usd)
    return picks


def _series_root(ticker: str) -> str:
    return ticker.split("-", 1)[0] if "-" in ticker else ticker


def find_combos(picks: list[Pick], bankroll: float,
                 max_legs: int = 3, max_combos: int = 5) -> list[Combo]:
    """Greedy independent-market combos.

    We require legs to come from different genres AND different series so
    the assumption of independence is at least defensible.
    """
    out: list[Combo] = []
    pool = [p for p in picks if p.confidence in ("med", "high")]
    if len(pool) < 2:
        return out
    used: set[tuple[str, ...]] = set()
    for n in (3, 2):
        if n > max_legs:
            continue
        # Greedy: take top picks, then walk down the list adding the next
        # pick whose genre + series we haven't used yet.
        i = 0
        while i < len(pool) and len(out) < max_combos:
            seed = pool[i]
            legs: list[Pick] = [seed]
            used_g, used_s = {seed.genre}, {_series_root(seed.ticker)}
            for cand in pool[i + 1:]:
                if len(legs) >= n:
                    break
                if cand.genre in used_g or _series_root(cand.ticker) in used_s:
                    continue
                legs.append(cand)
                used_g.add(cand.genre)
                used_s.add(_series_root(cand.ticker))
            i += 1
            if len(legs) < n:
                continue
            key = tuple(sorted(p.ticker for p in legs))
            if key in used:
                continue
            used.add(key)
            wp = math.prod(p.win_prob for p in legs)
            payout = math.prod(1.0 / max(0.01, p.entry_price) for p in legs)
            ev_per_dollar = wp * (payout - 1) - (1 - wp)
            if ev_per_dollar <= 0:
                continue
            stake = max(1.0, min(bankroll * 0.02, 25.0))
            out.append(Combo(
                legs=legs,
                combined_win_prob=wp,
                combined_payout_per_dollar=payout,
                expected_value_per_dollar=ev_per_dollar,
                suggested_stake_usd=stake,
                expected_profit_usd=stake * ev_per_dollar,
                rationale=(
                    f"All {n} legs win with prob {wp*100:.1f}%, payout "
                    f"{payout:.2f}x. Markets span {len(used_g)} different genres "
                    f"so they should resolve independently."
                ),
            ))
    out.sort(key=lambda c: -c.expected_profit_usd)
    return out[:max_combos]


def build_betting_slip(signals: list["LiveSignal"], bankroll: float = 100.0,
                       max_picks: int = 8, max_combos: int = 3) -> dict:
    picks = signals_to_picks(signals, bankroll)
    top_picks = picks[:max_picks]
    combos = find_combos(picks, bankroll=bankroll, max_combos=max_combos)
    return {
        "picks": [
            {
                "ticker": p.ticker, "title": p.title, "genre": p.genre,
                "side": p.side,
                "entry_price_cents": int(round(p.entry_price * 100)),
                "win_prob": round(p.win_prob, 3),
                "edge": round(p.edge, 3),
                "confidence": p.confidence,
                "rationale": p.rationale,
                "stake_usd": round(p.suggested_stake_usd, 2),
                "ev_per_dollar": round(p.ev_per_dollar(), 3),
                "expected_profit_usd": round(p.expected_profit_usd, 2),
                "fair_value_source": p.fair_value_source,
                "fair_value_detail": p.fair_value_detail,
            } for p in top_picks
        ],
        "combos": [
            {
                "legs": [{
                    "ticker": leg.ticker, "title": leg.title,
                    "side": leg.side, "genre": leg.genre,
                    "entry_price_cents": int(round(leg.entry_price * 100)),
                    "win_prob": round(leg.win_prob, 3),
                } for leg in c.legs],
                "combined_win_prob": round(c.combined_win_prob, 4),
                "payout": round(c.combined_payout_per_dollar, 2),
                "ev_per_dollar": round(c.expected_value_per_dollar, 3),
                "stake_usd": round(c.suggested_stake_usd, 2),
                "expected_profit_usd": round(c.expected_profit_usd, 2),
                "rationale": c.rationale,
            } for c in combos
        ],
    }
