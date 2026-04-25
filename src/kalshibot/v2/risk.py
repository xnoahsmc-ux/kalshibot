"""Strict risk gates. Non-negotiable.

* Quarter-Kelly sizing capped at hard $/contract.
* Daily realized loss > 3% of bankroll → flatten + halt for the day.
* Concurrent exposure cap, per-genre cap, per-series correlation cap.
* HALT file checked on every gate call.
"""
from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import RiskCfg


@dataclass
class Position:
    ticker: str
    series: str
    genre: str
    contracts: int
    entry_price_cents: int
    side: str               # "YES" or "NO"

    def notional_usd(self) -> float:
        return self.contracts * self.entry_price_cents / 100.0


@dataclass
class DayState:
    started_at: float = field(default_factory=time.time)
    realized_pnl_usd: float = 0.0
    halted: bool = False

    def reset_if_new_day(self) -> None:
        now = dt.datetime.utcnow()
        start = dt.datetime.utcfromtimestamp(self.started_at)
        if now.date() != start.date():
            self.started_at = time.time()
            self.realized_pnl_usd = 0.0
            self.halted = False

    def record(self, pnl_usd: float) -> None:
        self.realized_pnl_usd += pnl_usd


@dataclass
class GateResult:
    ok: bool
    reason: str
    contracts_allowed: int = 0
    notional_usd: float = 0.0


def kelly_size(
    fair_value: float,
    entry_price: float,           # 0..1
    bankroll_usd: float,
    cfg: RiskCfg,
) -> float:
    """Return the dollar size to stake. Quarter-Kelly with hard contract cap.

    Edge expressed per dollar staked: (p_win × payout - (1-p_win)) / payout
    where payout = (1 - entry) / entry for a win.
    """
    p = max(0.01, min(0.99, fair_value))
    e = max(0.01, min(0.99, entry_price))
    payout = (1 - e) / e
    full_kelly = (p * payout - (1 - p)) / max(payout, 0.01)
    if full_kelly <= 0:
        return 0.0
    fraction = full_kelly * cfg.kelly_fraction
    notional = bankroll_usd * fraction
    return min(notional, cfg.hard_max_per_contract_usd * 1)


def halt_flag_present(cfg: RiskCfg) -> bool:
    return Path(cfg.halt_flag_path).exists()


def gate(
    cfg: RiskCfg,
    state: DayState,
    *,
    bankroll_usd: float,
    target_notional_usd: float,
    ticker: str,
    series: str,
    genre: str,
    open_positions: list[Position],
) -> GateResult:
    state.reset_if_new_day()

    if halt_flag_present(cfg):
        return GateResult(False, f"HALT file present at {cfg.halt_flag_path}")

    if state.halted:
        return GateResult(False, "Daily halt active (loss limit hit)")

    daily_cap = -abs(cfg.daily_loss_limit_pct) / 100.0 * bankroll_usd
    if state.realized_pnl_usd <= daily_cap:
        state.halted = True
        return GateResult(False, f"Daily loss limit hit: ${state.realized_pnl_usd:+.2f} ≤ ${daily_cap:.2f}")

    if target_notional_usd <= 0:
        return GateResult(False, "Target notional non-positive")

    if target_notional_usd > cfg.hard_max_per_contract_usd * 4:
        return GateResult(False,
            f"Notional ${target_notional_usd:.2f} exceeds 4× per-contract cap")

    if len(open_positions) >= cfg.max_concurrent_positions:
        return GateResult(False,
            f"At max concurrent positions ({cfg.max_concurrent_positions})")

    same_genre = sum(1 for p in open_positions if p.genre == genre)
    if same_genre >= cfg.max_per_genre:
        return GateResult(False, f"At max {cfg.max_per_genre} positions in {genre}")

    same_series = sum(1 for p in open_positions if p.series == series)
    if same_series >= cfg.max_same_series:
        return GateResult(False,
            f"At max {cfg.max_same_series} positions in {series} (correlation cap)")

    return GateResult(True, "ok",
                       contracts_allowed=0,    # caller computes from price
                       notional_usd=target_notional_usd)
