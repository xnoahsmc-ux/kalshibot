# Kalshibot

Multi-strategy ensemble trading bot for [Kalshi](https://kalshi.com) event markets.

It bundles ~110 parameterized strategies across nine families (trend, mean
reversion, momentum, volatility, microstructure, statistical, regression,
Bayesian drift, time-decay anchors), backtests them on every market, and uses
a constrained Sharpe + log-loss optimizer to pick the best weighted
combination. Markets are auto-classified into genres (sports, weather,
politics, economy, crypto, equities, entertainment, science, world, culture)
so you can see which niche the ensemble actually makes money in. Results
render to a self-contained HTML dashboard.

## Quick start

### Windows (Command Prompt / PowerShell)

```cmd
git clone https://github.com/xnoahsmc-ux/kalshibot.git
cd kalshibot
git checkout claude/kalshi-trading-bot-LowZW
run.bat
```

Or step-by-step without the helper:

```cmd
python -m pip install -r requirements.txt
python -m pip install -e .
kalshibot web --port 8080 --auto-backtest
```

### macOS / Linux

```bash
git clone https://github.com/xnoahsmc-ux/kalshibot.git
cd kalshibot
git checkout claude/kalshi-trading-bot-LowZW
./run.sh
```

Or:

```bash
pip install -r requirements.txt
pip install -e .
kalshibot web --port 8080 --auto-backtest
```

Then open http://127.0.0.1:8080.

## Feeding in live Kalshi data

1. In Kalshi (Account → API Keys) create a new key. Kalshi gives you a UUID
   **Key ID** and a **PEM private key** file.
2. Save the PEM somewhere in the project — e.g. `secrets/kalshi_private_key.pem`.
3. Copy `.env.example` to `.env` and fill in:
   ```
   KALSHI_ENV=prod
   KALSHI_API_KEY_ID=<your-uuid>
   KALSHI_API_PRIVATE_KEY_PATH=secrets/kalshi_private_key.pem
   ```
4. Verify credentials with the CLI:
   ```bash
   kalshibot auth-check
   ```
   You should see `Public /markets OK — N markets visible` and
   `Auth /portfolio/balance OK`.
5. Pull historical candles for real Kalshi markets and backtest on them:
   ```bash
   kalshibot fetch-history --markets 24 --lookback-hours 72
   kalshibot backtest-live
   ```
6. Start the web UI and open the **Live** page. Click **Start feed** — the
   server will begin polling Kalshi every `POLL_SECONDS` and render
   live-updating signals with suggested YES/NO/flat calls.
7. Flip `KALSHIBOT_DRY_RUN=false` and run `kalshibot run` to actually submit
   orders. Start with a tiny `KALSHIBOT_MAX_POSITION_USD` and a high
   `KALSHIBOT_MIN_EDGE` while you get comfortable.

## Other commands

```bash
kalshibot backtest --markets 32    # synthetic-data backtest
kalshibot dashboard --serve        # self-contained HTML dashboard
kalshibot auth-check               # verify Kalshi credentials
kalshibot fetch-history            # pull real historical candles
kalshibot backtest-live            # backtest on the fetched real data
kalshibot run                      # live trade loop (dry-run by default)
```

The web UI (`kalshibot web`) serves a Flask dashboard at
`http://127.0.0.1:8080` with pages for Overview, Strategies, Genres, Markets,
Ensemble, and Run-backtest (submits a background job and polls progress).
`kalshibot list-strategies` and `kalshibot list-genres` are handy
introspection commands.

## How the ensemble is picked

1. Every strategy outputs a probability-of-YES series for each market.
2. The backtester turns those into per-bar PnL using `position = sign(p - mid)`
   sized by `|p - mid|`, with a configurable threshold and fee.
3. `combiner.best_combination` filters to the top-K by Sharpe, then blends
   a Sharpe-maximizing solution with a risk-parity (equal risk contribution)
   solution — the hybrid is noticeably less overfit than pure Sharpe maxing.
   If outcomes are available it further blends with a log-loss-optimal
   weight vector, then a correlation-cap step folds near-duplicate
   strategies into their best representative.
4. `walkforward.walk_forward` provides true out-of-sample Sharpe by refitting
   the ensemble on rolling train windows and evaluating on the next block.
5. Live: `bot.Trader` re-uses the saved weights, computes the blended
   probability per snapshot, and sizes orders via half-Kelly scaled by
   ensemble dispersion (strategies disagreeing → smaller bets) and capped by
   both `MAX_POSITION_USD` and a per-genre exposure budget.

## Profit-oriented extras

- **Regime detection** (`regime.py`): rolling autocorrelation + Lo-MacKinlay
  variance ratio label each bar as trending / mixed / mean-reverting.
  Regime-gated strategies (`regime_trend_*`, `regime_mr_*`, `rsi_regime_*`)
  only fire in the appropriate regime and survive walk-forward much better.
- **Probability calibration** (`calibration.py`): isotonic regression maps
  raw ensemble probabilities to empirical frequencies so Kelly sizing acts
  on well-calibrated edges.
- **Dispersion gate**: the live trader refuses to trade when the weighted
  standard deviation of strategy probabilities exceeds a cap.
- **Per-genre exposure cap**: one hot genre can't blow the book.

## Genre / niche analysis

`genres.classify` maps tickers like `KXNFL-…`, `KXWEATHERHIGH-…`,
`KXBTC-…` into nine genres plus an `other` bucket, and the dashboard breaks
PnL, Sharpe, and best-strategy-per-niche out of the backtest.

## Safety

- Defaults to **dry-run**. Orders are only submitted when
  `KALSHIBOT_DRY_RUN=false`.
- Per-market USD cap (`KALSHIBOT_MAX_POSITION_USD`) and minimum edge
  (`KALSHIBOT_MIN_EDGE`) gate every order.
- Trade at your own risk. This is a research tool; markets can resolve
  adversely and you can lose your full stake on every contract.
