"""Command-line entry point.

  kalshibot backtest                # backtest 100 strategies on synthetic data
  kalshibot backtest --markets 30   # more synthetic markets
  kalshibot dashboard               # backtest then write & open dashboard.html
  kalshibot run                     # live loop (uses .env)
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .analytics import build_report
from .backtester import evaluate_all, stack_pnls, stack_probs
from .bot import run_forever
from .combiner import Combination, best_combination
from .config import load_config
from .dashboard import render, serve as serve_dashboard
from .data import MarketHistory, synthesize_history
from .genres import classify
from .strategies.registry import all_strategies

app = typer.Typer(add_completion=False, help="Multi-strategy Kalshi trading bot")
console = Console()


def _synth_universe(n_markets: int, seed: int = 0) -> list[MarketHistory]:
    """Synthesize a universe of markets that span our genres for backtesting."""
    archetypes = [
        ("KXNFL-DEMO", 0.0, 0.025, 0.55),
        ("KXNBA-DEMO", 0.001, 0.020, 0.48),
        ("KXWEATHERHIGH-DEMO", -0.0005, 0.012, 0.40),
        ("KXWEATHERSNOW-DEMO", 0.0, 0.018, 0.35),
        ("KXPOLPRES-DEMO", 0.0008, 0.030, 0.52),
        ("KXSENATE-DEMO", -0.0003, 0.022, 0.45),
        ("KXCPI-DEMO", 0.0, 0.015, 0.50),
        ("KXFEDRATE-DEMO", 0.0002, 0.018, 0.42),
        ("KXBTC-DEMO", 0.001, 0.035, 0.60),
        ("KXETH-DEMO", -0.001, 0.030, 0.55),
        ("KXSPX-DEMO", 0.0005, 0.014, 0.55),
        ("KXOSCAR-DEMO", 0.0, 0.020, 0.30),
        ("KXAI-DEMO", 0.0008, 0.022, 0.65),
        ("KXSPACE-DEMO", 0.0, 0.025, 0.40),
        ("KXOIL-DEMO", -0.0006, 0.020, 0.50),
        ("KXMUSK-DEMO", 0.0, 0.030, 0.55),
    ]
    out = []
    for i in range(n_markets):
        a = archetypes[i % len(archetypes)]
        ticker, drift, vol, start = a
        h = synthesize_history(
            ticker=f"{ticker}-{i:03d}",
            n=600,
            seed=seed + i,
            drift=drift,
            vol=vol,
            start_p=start,
        )
        out.append(h)
    return out


@app.command()
def backtest(
    markets: int = typer.Option(16, help="Number of synthetic markets to backtest"),
    threshold: float = typer.Option(0.02, help="Minimum edge to take a position"),
    fee_bps: float = typer.Option(5.0, help="Per-turn cost in basis points"),
    top_k: int = typer.Option(30, help="Pre-filter top-k strategies before optimization"),
    save: Path = typer.Option(Path("backtest_report.pkl"), help="Where to pickle the report"),
):
    """Run a full backtest across all strategies and pick the optimal combo."""
    strats = all_strategies()
    console.print(f"[cyan]Loaded {len(strats)} strategies[/cyan]")

    universe = _synth_universe(markets)
    tickers = [h.ticker for h in universe]
    console.print(f"[cyan]Backtesting on {len(universe)} markets[/cyan]")

    evals = evaluate_all(strats, universe, threshold=threshold, fee_bps=fee_bps)
    pnl = stack_pnls(evals)
    probs = stack_probs(evals)

    # Outcomes: terminal mid > 0.5 → 1
    outcomes_per_market = [1.0 if h.mid.iloc[-1] >= 0.5 else 0.0 for h in universe]
    bar_outcomes = []
    for h, y in zip(universe, outcomes_per_market):
        bar_outcomes.extend([y] * len(h.df))
    import pandas as pd
    outcomes = pd.Series(bar_outcomes[:len(probs)])

    combo = best_combination(pnl, probs, outcomes, top_k=top_k)
    report = build_report(evals, tickers, combo)

    # Print top-level tables
    t = Table(title="Top 15 strategies by Sharpe")
    t.add_column("strategy"); t.add_column("sharpe"); t.add_column("total"); t.add_column("hit"); t.add_column("max_dd")
    for name, row in report.per_strategy.head(15).iterrows():
        t.add_row(name, f"{row.sharpe:+.2f}", f"{row.total:+.4f}",
                  f"{row.hit_rate:.2%}", f"{row.max_dd:+.4f}")
    console.print(t)

    g = Table(title="Genre performance (ensemble)")
    g.add_column("genre"); g.add_column("markets"); g.add_column("total"); g.add_column("sharpe")
    g.add_column("max_dd"); g.add_column("best_strategy")
    for name, row in report.per_genre.iterrows():
        g.add_row(name, str(int(row.n_markets)), f"{row.total:+.4f}",
                  f"{row.sharpe:+.2f}", f"{row.max_dd:+.4f}",
                  report.best_strategy_per_genre.get(name, ""))
    console.print(g)

    w = Table(title="Top 10 weights in optimal ensemble")
    w.add_column("strategy"); w.add_column("weight")
    for name, val in combo.top(10).items():
        w.add_row(name, f"{val:.3f}")
    console.print(w)

    console.print(
        f"[green]Ensemble Sharpe: {combo.sharpe:+.2f}  "
        f"Total PnL: {report.equity_curve.iloc[-1]:+.4f}[/green]"
    )

    save.write_bytes(pickle.dumps(report))
    console.print(f"[dim]Report saved to {save}[/dim]")


@app.command()
def dashboard(
    report_path: Path = typer.Option(Path("backtest_report.pkl"),
                                     help="Backtest report from `backtest`"),
    out: Path = typer.Option(Path("dashboard.html")),
    serve: bool = typer.Option(False, "--serve", help="Serve dashboard.html on localhost"),
    port: int = typer.Option(8765),
    markets: int = typer.Option(16, help="If no report exists, run a fresh backtest"),
):
    """Generate (and optionally serve) the HTML dashboard."""
    if not report_path.exists():
        console.print("[yellow]No backtest report found, running one now…[/yellow]")
        backtest(markets=markets, threshold=0.02, fee_bps=5.0, top_k=30, save=report_path)
    report = pickle.loads(report_path.read_bytes())
    path = render(report, out)
    console.print(f"[green]Wrote {path}[/green]")
    if serve:
        serve_dashboard(path, port=port)


@app.command()
def run():
    """Live trade loop (uses optimal ensemble from disk; runs backtest if absent)."""
    cfg = load_config()
    report_path = Path("backtest_report.pkl")
    if not report_path.exists():
        console.print("[yellow]No saved backtest, computing one first…[/yellow]")
        backtest(markets=16, threshold=0.02, fee_bps=5.0, top_k=30, save=report_path)
    report = pickle.loads(report_path.read_bytes())
    run_forever(cfg, report.combo)


@app.command("auth-check")
def auth_check_cmd():
    """Verify your Kalshi credentials work. Prints what succeeded/failed."""
    from .kalshi_client import KalshiClient
    cfg = load_config()
    console.print(f"[cyan]Base URL:[/cyan] {cfg.base_url}")
    console.print(f"[cyan]Key ID:[/cyan]   {cfg.api_key_id or '(unset)'}")
    console.print(f"[cyan]PEM path:[/cyan] {cfg.api_private_key_path or '(unset)'}")
    client = KalshiClient(cfg)
    result = client.auth_check()
    if result.get("public_ok"):
        console.print(f"[green]Public /markets OK[/green] — {result['markets_visible']} market(s) visible")
    else:
        console.print(f"[red]Public /markets failed:[/red] {result.get('public_error')}")
        return
    if result.get("auth_ok"):
        console.print(f"[green]Auth /portfolio/balance OK[/green]")
        console.print(result.get("balance"))
    else:
        console.print(f"[red]Auth failed:[/red] {result.get('auth_error')}")


@app.command("fetch-history")
def fetch_history_cmd(
    markets: int = typer.Option(16, help="How many live markets to fetch history for"),
    lookback_hours: int = typer.Option(72),
    save: Path = typer.Option(Path("real_universe.pkl")),
):
    """Pull historical candlesticks from Kalshi for N open markets and save."""
    import pickle
    from .history_fetch import fetch_universe
    from .kalshi_client import KalshiClient
    cfg = load_config()
    client = KalshiClient(cfg)
    console.print(f"[cyan]Fetching up to {markets} markets' history from {cfg.base_url}[/cyan]")
    universe = fetch_universe(client, limit=markets, lookback_hours=lookback_hours)
    console.print(f"[green]Got {len(universe)} markets with usable history[/green]")
    save.write_bytes(pickle.dumps(universe))
    console.print(f"[dim]Saved to {save}[/dim]")


@app.command("backtest-live")
def backtest_live_cmd(
    universe_path: Path = typer.Option(Path("real_universe.pkl"),
                                        help="From kalshibot fetch-history"),
    top_k: int = typer.Option(30),
):
    """Backtest all strategies on REAL Kalshi history previously fetched."""
    import pickle
    from .analytics import build_report
    from .backtester import evaluate_all, stack_pnls, stack_probs
    from .combiner import best_combination
    if not universe_path.exists():
        console.print(f"[red]{universe_path} not found. Run `kalshibot fetch-history` first.[/red]")
        raise typer.Exit(1)
    universe = pickle.loads(universe_path.read_bytes())
    console.print(f"[cyan]Loaded {len(universe)} real markets[/cyan]")
    if not universe:
        console.print("[red]Universe is empty - nothing to backtest.[/red]")
        raise typer.Exit(1)
    strats = all_strategies()
    evals = evaluate_all(strats, universe)
    pnl = stack_pnls(evals)
    probs = stack_probs(evals)
    if pnl.empty or probs.empty:
        console.print("[red]Strategies produced no evaluations - skipping.[/red]")
        raise typer.Exit(1)
    outcomes_bars = []
    for h in universe:
        if len(h.df) == 0:
            continue
        y = 1.0 if h.mid.iloc[-1] >= 0.5 else 0.0
        outcomes_bars.extend([y] * len(h.df))
    import pandas as pd
    outcomes = pd.Series(outcomes_bars[:len(probs)])
    combo = best_combination(pnl, probs, outcomes, top_k=top_k)
    report = build_report(evals, [h.ticker for h in universe], combo)
    Path("backtest_report.pkl").write_bytes(pickle.dumps(report))
    console.print(f"[green]Ensemble Sharpe: {combo.sharpe:+.2f}  "
                  f"Total PnL: {report.equity_curve.iloc[-1]:+.4f}[/green]")
    console.print("[dim]Saved to backtest_report.pkl. Start the web UI or `kalshibot run`.[/dim]")


@app.command("setup-live")
def setup_live_cmd(
    markets: int = typer.Option(24, help="Markets to fetch history for"),
    lookback_hours: int = typer.Option(168, help="Candle lookback window"),
    period_minutes: int = typer.Option(60, help="Candle interval (1, 60, 1440)"),
    candidate_pool: int = typer.Option(400,
        help="How many open markets to scan before picking ones with real data"),
    min_volume_24h: int = typer.Option(1,
        help="Skip markets without any trades in the last 24h"),
    fallback_synth_markets: int = typer.Option(24,
        help="If real data fails, synthesize this many to still build an ensemble"),
    port: int = typer.Option(8080),
):
    """One command does it all: auth check -> fetch real Kalshi history ->
    backtest -> save ensemble -> launch the web UI.

    Falls back to a synthetic backtest if real data isn't available, so
    you always end up on a working dashboard."""
    import pickle
    from .analytics import build_report
    from .backtester import evaluate_all, stack_pnls, stack_probs
    from .combiner import best_combination
    from .history_fetch import fetch_universe
    from .kalshi_client import KalshiClient
    cfg = load_config()

    # 1. Auth check
    console.print("[bold cyan]1/4  Auth check[/bold cyan]")
    client = KalshiClient(cfg)
    try:
        chk = client.auth_check()
    except Exception as e:
        console.print(f"[red]Auth check raised: {e}[/red]")
        chk = {"public_ok": False}
    if chk.get("public_ok"):
        console.print(f"  [green]Public OK[/green] - {chk.get('markets_visible', '?')} market(s) visible")
    else:
        console.print(f"  [yellow]Public FAIL[/yellow] - {chk.get('public_error', 'unknown')}")
    if chk.get("auth_ok"):
        console.print(f"  [green]Auth  OK[/green] - balance={chk.get('balance')}")
    elif chk.get("auth_error"):
        console.print(f"  [yellow]Auth  FAIL[/yellow] - {chk['auth_error']}")

    # 2. Fetch real history (best-effort)
    console.print("[bold cyan]2/4  Fetch real Kalshi history[/bold cyan]")
    universe = []
    if chk.get("public_ok"):
        try:
            universe = fetch_universe(
                client,
                limit=markets,
                candidate_pool=candidate_pool,
                lookback_hours=lookback_hours,
                period_minutes=period_minutes,
                min_volume_24h=min_volume_24h,
                verbose=True,
            )
        except Exception as e:
            console.print(f"  [yellow]fetch raised: {e}[/yellow]")
    else:
        console.print("  [dim]skipped (public endpoint unreachable)[/dim]")

    # 3. Backtest on real if we got enough, else synthetic
    console.print("[bold cyan]3/4  Backtest ensemble[/bold cyan]")
    strats = all_strategies()
    if len(universe) >= 4:
        console.print(f"  using {len(universe)} real markets")
    else:
        console.print(f"  [yellow]only {len(universe)} real markets usable - "
                      f"falling back to {fallback_synth_markets} synthetic[/yellow]")
        universe = _synth_universe(fallback_synth_markets)
    evals = evaluate_all(strats, universe)
    pnl = stack_pnls(evals)
    probs = stack_probs(evals)
    outcomes_bars = []
    for h in universe:
        if len(h.df) == 0:
            continue
        y = 1.0 if h.mid.iloc[-1] >= 0.5 else 0.0
        outcomes_bars.extend([y] * len(h.df))
    import pandas as pd
    outcomes = pd.Series(outcomes_bars[:len(probs)])
    combo = best_combination(pnl, probs, outcomes, top_k=30)
    report = build_report(evals, [h.ticker for h in universe], combo)
    Path("backtest_report.pkl").write_bytes(pickle.dumps(report))
    console.print(f"  [green]ensemble saved[/green] - Sharpe {combo.sharpe:+.2f}, "
                  f"{int((combo.weights > 0).sum())} active strategies")

    # 4. Launch the web server and auto-start the live feed
    console.print("[bold cyan]4/4  Launching web UI[/bold cyan]")
    from .web import create_app
    from .web.state import AppState
    state = AppState()
    web_app = create_app(state)

    if chk.get("public_ok"):
        try:
            from .live import LiveFeed
            feed = LiveFeed(cfg=cfg, combo=state.report.combo if state.report else None,
                            client=client)
            feed.start()
            state.set_live(feed)
            console.print("[green]  live feed auto-started; "
                          "open Live tab to watch signals[/green]")
        except Exception as e:
            console.print(f"  [yellow]could not auto-start live feed: {e}[/yellow]")

    console.print(f"[green]Open http://127.0.0.1:{port}  (Ctrl+C to stop)[/green]")
    web_app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)


@app.command()
def web(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8080),
    debug: bool = typer.Option(False),
    auto_backtest: bool = typer.Option(False, help="Kick off a backtest on startup if none exists"),
    markets: int = typer.Option(24, help="Markets for the auto-backtest"),
):
    """Launch the kalshibot web dashboard (Flask)."""
    from .web import create_app
    from .web.state import AppState
    from .web import jobs as webjobs
    state = AppState()
    web_app = create_app(state)
    if auto_backtest and state.report is None:
        console.print("[yellow]No report yet — starting a backtest in the background…[/yellow]")
        webjobs.run_backtest(state, markets=markets)
    console.print(f"[green]Serving dashboard at http://{host}:{port}[/green]")
    web_app.run(host=host, port=port, debug=debug, use_reloader=False)


@app.command()
def list_strategies():
    """Print every available strategy."""
    for s in all_strategies():
        console.print(f"  {s.name}")


@app.command()
def list_genres(tickers: list[str] = typer.Argument(None)):
    """Classify a list of tickers into genres."""
    if not tickers:
        tickers = [h.ticker for h in _synth_universe(16)]
    for t in tickers:
        info = classify(t)
        console.print(f"{t:32}  -> {info.genre}  ({info.matched_token})")


if __name__ == "__main__":
    app()
