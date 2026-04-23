"""Persistent journal of every trade we submit to Kalshi (manual or auto).

Used for:
  - User-facing trade history
  - Realized PnL tracking
  - Learning: when a trade settles we update a per-strategy hit-rate
    table so the ensemble can adjust its weights over time.

Writes to `data/journal.jsonl` (gitignored) - one JSON record per line.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class JournalEntry:
    id: str
    created_at: float
    ticker: str
    genre: str
    title: str
    side: str                          # "YES" | "NO"
    contracts: int
    entry_price_cents: int             # 1-99
    notional_usd: float
    source: str                        # "manual" | "auto"
    confidence: str
    ensemble_prob: float
    fair_value_prob: float | None
    reason: str                        # free text (recommendation rationale)
    kalshi_order_id: str | None = None
    kalshi_status: str | None = None
    # Post-settlement fields, filled in later:
    settled_at: float | None = None
    settlement_price_cents: int | None = None   # 0 (NO won) or 100 (YES won)
    realized_pnl_usd: float | None = None
    outcome: str | None = None         # "win" | "loss" | None
    contributing_strategies: list[str] = field(default_factory=list)


class TradeJournal:
    def __init__(self, path: str | Path = "data/journal.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._entries: list[JournalEntry] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        self._entries.append(JournalEntry(**data))
                    except Exception:
                        continue
        except Exception:
            pass

    def _append(self, entry: JournalEntry) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def add(self, **kw: Any) -> JournalEntry:
        entry = JournalEntry(
            id=kw.get("id") or f"j-{int(time.time()*1000)}",
            created_at=kw.get("created_at", time.time()),
            **{k: v for k, v in kw.items() if k not in ("id", "created_at")},
        )
        with self._lock:
            self._entries.append(entry)
            self._append(entry)
        return entry

    def update_settlement(self, entry_id: str, *,
                           settlement_price_cents: int,
                           settled_at: float | None = None) -> JournalEntry | None:
        with self._lock:
            for e in self._entries:
                if e.id != entry_id:
                    continue
                e.settlement_price_cents = settlement_price_cents
                e.settled_at = settled_at or time.time()
                value_per_contract = settlement_price_cents / 100.0
                if e.side == "YES":
                    gross = e.contracts * value_per_contract
                else:
                    gross = e.contracts * (1 - value_per_contract)
                paid = e.contracts * (e.entry_price_cents / 100.0)
                e.realized_pnl_usd = gross - paid
                e.outcome = "win" if e.realized_pnl_usd > 0 else "loss"
                # Rewrite file fresh to reflect update.
                self._rewrite()
                return e
        return None

    def _rewrite(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for e in self._entries:
                f.write(json.dumps(asdict(e)) + "\n")
        tmp.replace(self.path)

    # ------------------------------------------------------------------ views

    def entries(self) -> list[JournalEntry]:
        with self._lock:
            return list(self._entries)

    def open_entries(self) -> list[JournalEntry]:
        return [e for e in self.entries() if e.settled_at is None]

    def settled_entries(self) -> list[JournalEntry]:
        return [e for e in self.entries() if e.settled_at is not None]

    def stats(self) -> dict:
        settled = self.settled_entries()
        wins = [e for e in settled if e.outcome == "win"]
        losses = [e for e in settled if e.outcome == "loss"]
        realized = sum(e.realized_pnl_usd or 0 for e in settled)
        total_staked = sum(e.notional_usd for e in settled)
        roi = (realized / total_staked) if total_staked > 0 else 0.0
        return {
            "total_trades": len(self.entries()),
            "open_trades": len(self.open_entries()),
            "settled_trades": len(settled),
            "wins": len(wins),
            "losses": len(losses),
            "hit_rate": (len(wins) / len(settled)) if settled else 0.0,
            "realized_pnl_usd": realized,
            "roi_pct": roi * 100,
        }

    def today_realized_pnl(self) -> float:
        import datetime as dt
        today = dt.datetime.utcnow().date()
        total = 0.0
        for e in self.settled_entries():
            st = dt.datetime.utcfromtimestamp(e.settled_at or 0).date()
            if st == today:
                total += (e.realized_pnl_usd or 0.0)
        return total
