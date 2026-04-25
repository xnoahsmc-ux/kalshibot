"""Background scheduler. Wakes every 2 hours, generates the current
top-picks summary, and dispatches an SMS / email alert.
"""
from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING

from .notifications import send_alert

if TYPE_CHECKING:
    from .web.state import AppState


def _format_picks_message(slip: dict) -> str:
    picks = slip.get("picks", [])[:3]
    combos = slip.get("combos", [])[:1]
    if not picks and not combos:
        return "Kalshi Bot: nothing actionable right now."
    lines = ["Kalshi Bot picks:"]
    for p in picks:
        lines.append(
            f"• {p['side']} {p['ticker']} @{p['entry_price_cents']}¢ "
            f"({p['confidence']}, EV +${p['expected_profit_usd']:.2f})"
        )
    if combos:
        c = combos[0]
        legs = " + ".join(f"{l['side']} {l['ticker'].split('-')[0]}" for l in c["legs"])
        lines.append(
            f"COMBO: {legs} | win {c['combined_win_prob']*100:.1f}% | "
            f"EV +${c['expected_profit_usd']:.2f}"
        )
    return "\n".join(lines)


class AlertScheduler:
    def __init__(self, state: "AppState", interval_seconds: int = 7200) -> None:
        self.state = state
        self.interval = max(60, interval_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_run: float | None = None
        self._last_message: str | None = None

    def status(self) -> dict:
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "interval_seconds": self.interval,
            "last_run": self._last_run,
            "last_message": self._last_message,
            "phone": os.getenv("KALSHIBOT_PHONE", ""),
            "twilio_configured": bool(os.getenv("KALSHIBOT_TWILIO_SID")
                                        and os.getenv("KALSHIBOT_TWILIO_TOKEN")
                                        and os.getenv("KALSHIBOT_TWILIO_FROM")),
            "smtp_configured": bool(os.getenv("KALSHIBOT_SMTP_HOST")
                                     and os.getenv("KALSHIBOT_NOTIFY_EMAIL")),
        }

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def trigger_now(self) -> str:
        return self._tick()

    def _loop(self) -> None:
        # First send happens after a small warmup delay so the live feed has
        # a chance to populate signals.
        first_delay = min(120, self.interval)
        if self._stop.wait(first_delay):
            return
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:
                pass
            self._stop.wait(self.interval)

    def _tick(self) -> str:
        from .combo_engine import build_betting_slip
        feed = self.state.live()
        if feed is None:
            msg = "Kalshi Bot alert: live feed not running. Open the dashboard to start it."
        else:
            slip = build_betting_slip(feed.signals(), bankroll=200.0)
            msg = _format_picks_message(slip)
        send_alert(msg, kind="periodic")
        self._last_run = time.time()
        self._last_message = msg
        return msg
