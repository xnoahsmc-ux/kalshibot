"""Strategy lab. Isolates each signal and measures it on real Kalshi
candle history. Outputs:

  - per-signal Sharpe, hit rate, total PnL, max drawdown
  - recommended config.yaml signal weights (zero out losers)
  - STRATEGY_REPORT.md

The lab pulls real candlestick history for high-liquidity markets via
the existing history_fetch module so it works without waiting for the
snapshot store to fill up.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..config import load_config as load_legacy_cfg
from ..data import MarketHistory
from ..genres import classify
from ..history_fetch import fetch_universe
from ..kalshi_client import KalshiClient
from .config import V2Config, load
from .data import Market
from .risk import DayState, kelly_size
from .signals import (
    Signal,
    edge_cents,
    external_anchor_signal,
    microstructure_signal,
    mispricing_signal,
)


@dataclass
class SignalResult:
    name: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl_usd: float = 0.0
    pnls: list[float] = field(default_factory=list)
    edges_taken: list[int] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl_usd / self.trades if self.trades else 0.0

    @property
    def sharpe(self) -> float:
        if len(self.pnls) < 2:
            return 0.0
        sd = statistics.pstdev(self.pnls)
        if sd <= 0:
            return 0.0
        return (statistics.mean(self.pnls) / sd) * math.sqrt(252)

    @property
    def max_dd_pct(self) -> float:
        if not self.pnls:
            return 0.0
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in self.pnls:
            cum += p
            peak = max(peak, cum)
            dd = peak - cum
            max_dd = max(max_dd, dd)
        return max_dd / max(1.0, abs(peak)) * 100 if peak else 0.0


def _signal_fn(name: str) -> Callable[[Market, list[float], str], Signal]:
    """Return a single-signal scorer with a uniform interface."""
    if name == "mispricing":
        return lambda m, mids, genre: mispricing_signal(m, genre)
    if name == "microstructure":
        return lambda m, mids, genre: microstructure_signal(m, mids)
    if name == "external_anchor":
        return lambda m, mids, genre: external_anchor_signal(m)
    raise ValueError(name)


def _market_from_history_bar(h: MarketHistory, idx: int) -> Market:
    """Build a Market dataclass from a row of the historical candles df."""
    row = h.df.iloc[idx]
    return Market(
        ticker=h.ticker, title=h.ticker,
        event_ticker=h.ticker.split("-", 1)[0],
        series_ticker=h.ticker.split("-", 1)[0],
        yes_bid=float(row["yes_bid"]), yes_ask=float(row["yes_ask"]),
        last_price=float(row["last"]),
        volume_24h_usd=float(row.get("volume", 0)),
        open_interest=int(row.get("open_interest", 0)),
        minutes_to_close=float(row.get("minutes_to_close", 60)),
    )


def evaluate_signal_on_history(
    name: str, histories: list[MarketHistory], cfg: V2Config,
    bankroll: float = 1000.0,
) -> SignalResult:
    """Trade every bar where the isolated signal disagrees with the mid
    by more than min_edge_cents. Close at the next bar's mid (1-step
    look-ahead) so we measure short-horizon predictive power."""
    fn = _signal_fn(name)
    res = SignalResult(name=name)
    for h in histories:
        if len(h.df) < 4:
            continue
        genre = classify(h.ticker).genre
        mids: list[float] = []
        for i in range(len(h.df) - 1):
            mids.append(float(h.df["last"].iloc[i]))
            mids = mids[-10:]
            m = _market_from_history_bar(h, i)
            sig = fn(m, mids, genre)
            if sig.confidence < 0.30:
                continue
            e = edge_cents(sig.fair_value, m.mid)
            if abs(e) < cfg.signals.min_edge_cents:
                continue
            side = "yes" if e > 0 else "no"
            entry = m.yes_ask if side == "yes" else (1 - m.yes_bid)
            if entry <= 0.01 or entry >= 0.99:
                continue
            p_win = sig.fair_value if side == "yes" else (1 - sig.fair_value)
            notional = kelly_size(p_win, entry, bankroll, cfg.risk)
            if notional < 1.0:
                continue
            contracts = max(1, int(notional / max(0.01, entry)))
            # Exit: next bar's mid as the realised "settlement" proxy
            next_mid = float(h.df["last"].iloc[i + 1])
            if side == "yes":
                pnl = contracts * (next_mid - entry)
            else:
                pnl = contracts * ((1 - next_mid) - (1 - m.yes_bid))
            # Fees + half-spread slippage
            fee_per_trade = cfg.backtest.fee_cents_per_trade / 100.0
            pnl -= fee_per_trade
            res.trades += 1
            res.total_pnl_usd += pnl
            res.pnls.append(pnl)
            res.edges_taken.append(e)
            if pnl > 0:
                res.wins += 1
            else:
                res.losses += 1
    return res


def run_lab(client: KalshiClient | None = None,
             markets: int = 30, lookback_hours: int = 24 * 30,
             period_minutes: int = 60) -> dict:
    cfg = load("config.yaml")
    legacy = load_legacy_cfg()
    cfg.bankroll_usd = legacy.bankroll_usd
    if client is None:
        client = KalshiClient(legacy)
    print(f"[lab] pulling {markets} markets × {lookback_hours}h of candles…")
    histories = fetch_universe(
        client, limit=markets, candidate_pool=600,
        lookback_hours=lookback_hours, period_minutes=period_minutes,
        min_volume_24h=0, min_bars=10, verbose=True,
    )
    print(f"[lab] got {len(histories)} usable histories")

    results: dict[str, SignalResult] = {}
    for name in ("mispricing", "microstructure", "external_anchor"):
        print(f"[lab] evaluating signal: {name}")
        results[name] = evaluate_signal_on_history(name, histories, cfg)

    return {
        "results": results,
        "n_markets": len(histories),
        "config_used": cfg,
    }


def write_report(lab_out: dict, path: str = "STRATEGY_REPORT.md") -> Path:
    res: dict[str, SignalResult] = lab_out["results"]
    n_markets = lab_out["n_markets"]
    md = ["# Per-Signal Backtest Report", "",
            f"Universe: {n_markets} markets pulled from live Kalshi candles.", "",
            "## Headline", "",
            "| Signal | Trades | Hit rate | Total PnL | Avg/trade | Sharpe | Max DD% |",
            "|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        md.append(f"| {name} | {r.trades} | {r.hit_rate*100:.1f}% | "
                    f"${r.total_pnl_usd:+.2f} | ${r.avg_pnl:+.3f} | "
                    f"{r.sharpe:+.2f} | {r.max_dd_pct:.1f}% |")
    md += ["", "## Recommendation", ""]
    keep, kill = [], []
    for name, r in res.items():
        # Senior threshold: Sharpe >= 0.5 AND hit rate >= 52% AND total > 0
        if r.sharpe >= 0.5 and r.hit_rate >= 0.52 and r.total_pnl_usd > 0:
            keep.append(name)
        else:
            kill.append(name)
    if keep:
        md.append("### Keep (enable in config.yaml)")
        for n in keep:
            md.append(f"- **{n}** — Sharpe {res[n].sharpe:+.2f}, hit {res[n].hit_rate*100:.0f}%")
    if kill:
        md.append("")
        md.append("### Disable (set weight to 0.0)")
        for n in kill:
            r = res[n]
            md.append(f"- **{n}** — Sharpe {r.sharpe:+.2f}, hit {r.hit_rate*100:.0f}%, "
                        f"total ${r.total_pnl_usd:+.2f}")
    md += ["", "## Suggested config.yaml weights", "",
            "```yaml", "signals:", "  weights:"]
    for name in res:
        w = 1.0 if name in keep else 0.0
        md.append(f"    {name}: {w}")
    md.append("```")
    md += ["", "## Trade-level edge distribution", ""]
    for name, r in res.items():
        if not r.edges_taken:
            continue
        avg_e = statistics.mean(r.edges_taken)
        md.append(f"- {name}: avg taken edge {avg_e:+.1f}¢, "
                    f"min {min(r.edges_taken)}¢, max {max(r.edges_taken)}¢")
    p = Path(path)
    p.write_text("\n".join(md))
    return p


def autotune_config(lab_out: dict, config_path: str = "config.yaml") -> None:
    """Rewrite signals.weights in config.yaml: losers go to 0.0, winners
    stay at their current weight. Conservative — never raises a weight,
    only zeroes underperformers. The user can re-raise them after they
    earn it.
    """
    import re
    res: dict[str, SignalResult] = lab_out["results"]
    keep = {n: (r.sharpe >= 0.5 and r.hit_rate >= 0.52 and r.total_pnl_usd > 0)
              for n, r in res.items()}
    p = Path(config_path)
    text = p.read_text()
    # Replace weights block
    new_lines = []
    in_weights = False
    for line in text.splitlines():
        if line.strip().startswith("weights:"):
            in_weights = True
            new_lines.append(line)
            continue
        if in_weights and line.startswith("    "):
            m = re.match(r"^(\s+)(\w+):\s*([\d.]+)", line)
            if m:
                indent, name, cur = m.group(1), m.group(2), float(m.group(3))
                new = cur if keep.get(name, False) else 0.0
                new_lines.append(f"{indent}{name}: {new}")
                continue
            in_weights = False
        new_lines.append(line)
    p.write_text("\n".join(new_lines) + "\n")
    print(f"[lab] rewrote {p} — kept "
            f"{[n for n,k in keep.items() if k] or '(none)'}, "
            f"zeroed {[n for n,k in keep.items() if not k] or '(none)'}.")
