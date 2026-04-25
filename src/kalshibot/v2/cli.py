"""v2 CLI. Three commands: paper, run, backtest. Invoked via Makefile."""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from ..config import load_config as load_legacy_cfg
from ..kalshi_client import KalshiClient
from .backtest import run_backtest, write_report
from .config import V2Config, load
from .data import SnapshotStore, fetch_filtered_universe
from .execution import LiveExecutor, ShadowExecutor
from .monitoring import render_dashboard, setup_json_logging
from .risk import DayState, Position
from .strategy import evaluate

log = logging.getLogger("kalshibot.v2.cli")


def _bootstrap() -> tuple[V2Config, KalshiClient]:
    cfg = load("config.yaml")
    setup_json_logging(cfg.monitoring.log_dir)
    legacy = load_legacy_cfg()
    cfg.bankroll_usd = legacy.bankroll_usd
    if not (legacy.api_key_id and legacy.api_private_key_path):
        print("[v2] credentials missing. Set KALSHI_API_KEY_ID and "
                "KALSHI_API_PRIVATE_KEY_PATH in .env.", file=sys.stderr)
        sys.exit(2)
    return cfg, KalshiClient(legacy)


def _run_loop(cfg: V2Config, client: KalshiClient, paper: bool) -> int:
    """Shared loop body for `paper` and `run`. Returns process exit code."""
    store = SnapshotStore(cfg.storage.snapshots_db)
    executor = ShadowExecutor(cfg) if paper else LiveExecutor(client, cfg)
    day = DayState()
    history: dict[str, list[float]] = {}
    open_positions: list[Position] = []
    recent_signals: list[dict] = []
    bankroll = cfg.bankroll_usd
    mode = "PAPER" if paper else "LIVE"
    print(f"[v2 {mode}] starting. bankroll=${bankroll:.2f} "
            f"min_edge={cfg.signals.min_edge_cents}c "
            f"min_conf={cfg.signals.min_confidence}")
    try:
        while True:
            if executor.watchdog_breached():
                log.error("Watchdog breached - flatten signal would fire here")
            try:
                universe = fetch_filtered_universe(client, cfg, pool=600)
            except Exception as e:
                log.error("universe fetch failed: %s", e)
                time.sleep(15); continue
            store.write(universe)
            for m in universe:
                history.setdefault(m.ticker, []).append(m.mid)
                if len(history[m.ticker]) > 60:
                    history[m.ticker] = history[m.ticker][-60:]
                decision = evaluate(m, cfg, bankroll, open_positions, day,
                                      history[m.ticker][-10:])
                recent_signals.append({
                    "ticker": m.ticker, "side": "—" if decision.intent is None else decision.intent.side,
                    "edge": decision.edge_cents, "fv": decision.fair_value,
                    "conf": decision.confidence,
                })
                if decision.intent is None:
                    continue
                yes_bid_c = int(round(m.yes_bid * 100))
                yes_ask_c = int(round(m.yes_ask * 100))
                result = executor.send(decision.intent, yes_bid_c, yes_ask_c)
                if result.ok and result.filled_contracts:
                    open_positions.append(Position(
                        ticker=m.ticker, series=m.series_ticker,
                        genre=decision.market.title or "other",
                        contracts=result.filled_contracts,
                        entry_price_cents=int(result.avg_fill_price_cents),
                        side=decision.intent.side.upper(),
                    ))
                    bankroll -= result.filled_contracts * (result.avg_fill_price_cents / 100)
            recent_signals = recent_signals[-50:]
            if cfg.monitoring.cli_dashboard:
                render_dashboard({
                    "bankroll": bankroll,
                    "today_pnl": day.realized_pnl_usd,
                    "open_count": len(open_positions),
                    "positions": [{
                        "ticker": p.ticker, "side": p.side,
                        "contracts": p.contracts,
                        "entry": p.entry_price_cents / 100,
                    } for p in open_positions[-12:]],
                    "recent_signals": recent_signals[-10:],
                })
            time.sleep(30)
    except KeyboardInterrupt:
        print(f"\n[v2 {mode}] stopping. final bankroll=${bankroll:.2f}")
        return 0


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print("usage: python -m kalshibot.v2.cli {paper|run|backtest}")
        return 2
    cmd = args[0]
    if cmd == "backtest":
        cfg = load("config.yaml")
        setup_json_logging(cfg.monitoring.log_dir)
        result = run_backtest(cfg, starting_bankroll=cfg.bankroll_usd)
        path = write_report(result)
        print(f"[v2 backtest] wrote {path}")
        for k, v in result.metrics.items():
            print(f"  {k}: {v}")
        return 0
    if cmd == "report":
        from .reports import generate
        date = args[1] if len(args) > 1 else None
        try:
            from ..notifications import send_alert
        except Exception:
            send_alert = None
        text, out = generate(date=date)
        print(f"[v2 report] wrote {out}")
        print(text)
        # Optional broadcast — only if a backend is configured.
        if send_alert is not None and any(os.getenv(k) for k in (
                "KALSHIBOT_DISCORD_WEBHOOK", "KALSHIBOT_SLACK_WEBHOOK",
                "KALSHIBOT_TWILIO_SID", "KALSHIBOT_NOTIFY_EMAIL")):
            send_alert(text[:1800], kind="daily_report")
        return 0
    cfg, client = _bootstrap()
    if cmd == "paper":
        return _run_loop(cfg, client, paper=True)
    if cmd == "run":
        return _run_loop(cfg, client, paper=False)
    print(f"unknown command: {cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
