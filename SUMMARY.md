# Kalshibot — Summary

A disciplined Kalshi trading bot. Two layers:

* **v1 (legacy)** — the web dashboard, paper-bot leaderboard, NWS / ESPN
  integration, journal + learner. Useful for monitoring and education.
* **v2 (production)** — focused 3-signal alpha stack with strict risk
  controls and queue-aware execution. This is what `make paper` and
  `make run` invoke. Live trading uses v2 only.

## Quick start

```
make install          # one-time: deps + editable install
make backtest         # event-driven backtest, writes BACKTEST_REPORT.md
make paper            # live polling + signals + shadow ledger
make run              # SAME pipeline but real Kalshi orders (DRY_RUN=false)
make halt             # touch data/HALT to stop all new orders globally
make resume           # delete data/HALT
make test             # pytest, full suite
```

## v2 architecture (under `src/kalshibot/v2/`)

| Module | Responsibility |
| --- | --- |
| `config.py` | Load `config.yaml` + `.env`. No PyYAML hard dep. |
| `data.py` | Universe filter (spread / volume / TTR / OI), `SnapshotStore` (SQLite at `data/markets.db`). |
| `signals.py` | Three signals: mispricing-vs-base-rate, microstructure (OB imbalance + 5-min momentum, fade low-liq / follow high-liq), external anchors (NWS / ESPN / FRED). Logistic stacker. |
| `strategy.py` | Combine signals → fair-value, confidence, edge. Apply risk gate + Kelly sizing. Emit OrderIntent or pass. |
| `risk.py` | Quarter-Kelly, hard $/contract, daily 3% loss cap with auto-halt, per-genre + per-series correlation caps, `data/HALT` flag file checked every loop. |
| `execution.py` | Passive (1¢ inside spread) → escalate to mid after 30s → cross spread if edge ≥ 5¢. Watchdog flattens on API outage. `LiveExecutor` and `ShadowExecutor` (paper). |
| `backtest.py` | Event-driven replay of SQLite snapshots with fees + half-spread + queue-position fills. Generates `BACKTEST_REPORT.md`. |
| `monitoring.py` | Structured JSON logs in `logs/`, optional Rich CLI dashboard. |
| `cli.py` | `python -m kalshibot.v2.cli {paper|run|backtest}`. |

## What changed from v1

The legacy 151-strategy ensemble produced noise on event markets (technical
indicators have little edge on a binary that resolves in days). v2 ditches
those and trusts only signals backed by published research:

1. **Mispricing** vs base rate (Wolfers-Zitzewitz; FLB)
2. **Microstructure** (Cont, Lehalle)
3. **External anchors** (real-world data → fair value)

See `DECISIONS.md` for the full rationale and the bar to challenge them.

## Risk envelope (default config)

| Knob | Value |
| --- | --- |
| Sizing | Quarter-Kelly |
| Hard cap per contract | $25 |
| Daily realized-loss limit | 3% of bankroll |
| Max concurrent positions | 12 |
| Max per genre | 4 |
| Max per series (correlation) | 3 |
| Min edge to trade | 3¢ |
| Min stacker confidence | 0.55 |
| Universe spread cap | 4¢ |
| Universe vol_24h floor | $500 |
| TTR window | 1h – 14d |

All of these live in `config.yaml`.

## Tests

`make test` → 54/54 passing, including:

* Universe-filter behavior on bad markets
* All three signals (extreme fade, low/high-liq momentum)
* Logistic stacker pulls toward strong signals, not without them
* Quarter-Kelly capped at hard $/contract
* Risk gate blocks on HALT file, daily loss, per-genre cap, per-series cap
* `ShadowExecutor` writes JSONL ledger
* End-to-end **integration backtest** on a seeded SQLite scenario:
  no exceptions, report generated.

## Going live safely

Read `RUNBOOK.md` first. The 2-week ramp is non-negotiable.
