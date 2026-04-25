"""JSON structured logging + simple Rich CLI dashboard."""
from __future__ import annotations

import json
import logging
import os
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


def setup_json_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    class _JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload: dict[str, Any] = {
                "ts": record.created,
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            if record.exc_info:
                payload["exc"] = self.formatException(record.exc_info)
            return json.dumps(payload)

    handler = RotatingFileHandler(
        Path(log_dir) / "kalshibot.log", maxBytes=2_000_000, backupCount=5,
    )
    handler.setFormatter(_JsonFormatter())
    handler.setLevel(level)
    root = logging.getLogger()
    # Remove any pre-existing JSON handlers we may have added.
    for h in list(root.handlers):
        if isinstance(h, RotatingFileHandler):
            root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level)


def render_dashboard(state: dict) -> None:
    """Print a minimal dashboard. Uses rich if available, plain otherwise."""
    try:
        from rich.console import Console
        from rich.table import Table
        c = Console()
        c.clear()
        c.rule(f"[bold cyan]Kalshibot v2[/bold cyan]   {time.strftime('%H:%M:%S')}")
        kpi = Table.grid(padding=(0, 4))
        kpi.add_row("[bold]Bankroll[/bold]", f"${state.get('bankroll', 0):.2f}",
                     "[bold]Today PnL[/bold]",
                     f"${state.get('today_pnl', 0):+.2f}",
                     "[bold]Open[/bold]",  str(state.get('open_count', 0)))
        c.print(kpi)
        positions = state.get("positions") or []
        t = Table(title="Open positions")
        for col in ("ticker", "side", "qty", "entry", "live"):
            t.add_column(col)
        for p in positions:
            t.add_row(p["ticker"], p["side"], str(p["contracts"]),
                       f"{p['entry']:.2f}", f"{p.get('live', '—')}")
        c.print(t)
        sigs = state.get("recent_signals") or []
        if sigs:
            s = Table(title="Recent signals")
            for col in ("ticker", "side", "edge¢", "fv", "conf"):
                s.add_column(col)
            for r in sigs[:10]:
                s.add_row(r["ticker"], r["side"], f"{r['edge']:+d}",
                           f"{r['fv']:.2f}", f"{r['conf']:.2f}")
            c.print(s)
    except Exception:
        # Plain fallback
        os.system("clear" if os.name != "nt" else "cls")
        print(f"Kalshibot v2  Bankroll ${state.get('bankroll', 0):.2f}  "
                f"Today ${state.get('today_pnl', 0):+.2f}  "
                f"Open {state.get('open_count', 0)}")
