# Runbook — going live safely

This runbook is the difference between a profitable bot and a smoking
crater. Follow it.

## 0. Prerequisites

* Kalshi account funded.
* API key created (Account → API Keys), PEM saved to
  `secrets/kalshi_private_key.pem`.
* `.env` populated:
  ```
  KALSHI_API_KEY_ID=...
  KALSHI_API_PRIVATE_KEY_PATH=secrets/kalshi_private_key.pem
  KALSHIBOT_BANKROLL_USD=<your real bankroll>
  KALSHIBOT_DRY_RUN=true       # we'll flip this only at step 4
  ```

## 1. Verify auth

```
python -m kalshibot.cli auth-check
```

You should see "Public OK" and "Auth OK" with a balance. If not, stop and
fix credentials.

## 2. Backtest on collected data

Run `make paper` for at least 3 trading days so the SQLite snapshot store
fills with real market history. Then:

```
make backtest
```

Open `BACKTEST_REPORT.md`. **Do not proceed to live unless:**

* Total PnL is positive on the period.
* Sharpe ≥ 0.5.
* Max drawdown ≤ 5% of bankroll.
* P&L by genre shows the stack making money in at least 2 niches
  (concentration in a single niche is a red flag).

If any of these fail, leave `make paper` running for another week and
re-evaluate. Tune `config.yaml` only with single-variable changes
(otherwise you're overfitting to recent noise).

## 3. Paper trade for at least 5 sessions

`make paper` runs the full live polling, signal stack, and risk gates;
the only difference from live is that orders are written to
`data/paper_ledger.jsonl` instead of being sent to Kalshi.

Open the web dashboard: `python -m kalshibot.cli web`. Watch the
Journal page. Specifically look for:

* **Hit rate ≥ 55%** on settled paper trades.
* No "BLOCKED" reasons that you don't understand. Tail
  `logs/kalshibot.log` and grep for "BLOCKED".
* Average edge captured (entry vs settlement) is positive.

## 4. Ramp to live at 10% sizing for 2 weeks

Edit `config.yaml`:

```yaml
risk:
  hard_max_per_contract_usd: 2.50    # 10% of the eventual $25 cap
```

Edit `.env`:

```
KALSHIBOT_DRY_RUN=false
```

Then:

```
make run
```

For the first 14 calendar days:

* Check the dashboard / Discord webhook **at least every 4 hours**.
* If realized PnL < -1% of bankroll on any single day, run `make halt`
  and investigate before resuming.
* If the watchdog ever flattens (API outage), do not resume until you
  have confirmed Kalshi is healthy and our process is reachable.

## 5. After 2 weeks

If realized cumulative PnL > 0 and max-drawdown ≤ 5%:

```yaml
risk:
  hard_max_per_contract_usd: 25.0    # full cap
```

If not, go back to step 3 with a single-variable adjustment to
`config.yaml`.

## Emergency procedures

| Situation | Action |
| --- | --- |
| You see something weird and want to stop | `make halt` |
| You want to resume after halt | `make resume` |
| Drawdown looks unrecoverable | `make halt` + flatten manually in Kalshi UI |
| API outage / network broken | bot's watchdog will flatten; verify with `kalshi-cli auth-check` |
| You suspect a strategy is broken | Set its weight to 0 in `config.yaml`, restart |

## Daily ops

* Tail logs: `tail -f logs/kalshibot.log | jq .`
* Check fills: web dashboard → My trades.
* Backup: `data/markets.db`, `data/journal.jsonl`, `data/learner.json`.
* Update fees if Kalshi changes pricing: `config.yaml` →
  `backtest.fee_cents_per_trade`.

## Things that should never happen

* The bot trading after a `data/HALT` file exists.
* The bot trading with `KALSHIBOT_DRY_RUN=false` AND no `KALSHIBOT_PHONE`
  set (you should always have an SMS channel for fail alerts).
* A single trade > 4× the per-contract hard cap.
* Daily realized loss exceeding `daily_loss_limit_pct` without halting.

If any of these happen, the risk module has a bug. File it, halt, and
fix.
