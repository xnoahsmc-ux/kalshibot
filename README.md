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

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

```bash
# 1) Backtest 100+ strategies on a synthetic universe and pick the best combo
kalshibot backtest --markets 32

# 2) Render a dashboard with PnL, per-genre, per-strategy, equity curves
kalshibot dashboard --serve     # opens http://localhost:8765/dashboard.html

# 3) Live trade loop (reads .env; defaults to dry-run = no orders sent)
cp .env.example .env             # edit credentials
kalshibot run
```

`kalshibot list-strategies` and `kalshibot list-genres` are handy
introspection commands.

## How the ensemble is picked

1. Every strategy outputs a probability-of-YES series for each market.
2. The backtester turns those into per-bar PnL using `position = sign(p - mid)`
   sized by `|p - mid|`, with a configurable threshold and fee.
3. `combiner.best_combination` filters to the top-K by Sharpe, then runs a
   projected-gradient SLSQP solver to maximize Sharpe on the simplex with an
   L1 penalty for sparsity. If outcomes are available it blends with a
   log-loss-optimal weight vector.
4. Live: `bot.Trader` re-uses the saved weights, computes the blended
   probability per snapshot, and sizes orders via half-Kelly capped by your
   `MAX_POSITION_USD`.

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
