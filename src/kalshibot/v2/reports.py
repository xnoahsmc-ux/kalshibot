"""Daily P&L markdown report. Reads journal + paper ledger, writes
data/reports/YYYY-MM-DD.md, returns the markdown string for SMS / Discord.
"""
from __future__ import annotations

import datetime as dt
import json
import statistics
from collections import defaultdict
from pathlib import Path

from ..trade_journal import TradeJournal


def _today_iso() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d")


def generate(date: str | None = None,
              journal_path: str = "data/journal.jsonl",
              paper_path: str = "data/paper_ledger.jsonl",
              out_dir: str = "data/reports") -> tuple[str, Path]:
    target = date or _today_iso()
    target_date = dt.datetime.strptime(target, "%Y-%m-%d").date()

    j = TradeJournal(path=journal_path)
    settled = [e for e in j.settled_entries()
                 if e.settled_at and
                    dt.datetime.utcfromtimestamp(e.settled_at).date() == target_date]
    opens = [e for e in j.entries()
              if dt.datetime.utcfromtimestamp(e.created_at).date() == target_date]

    paper_lines = []
    p = Path(paper_path)
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                rec = json.loads(line)
                if dt.datetime.utcfromtimestamp(rec["ts"]).date() == target_date:
                    paper_lines.append(rec)
            except Exception:
                pass

    realized = sum(e.realized_pnl_usd or 0 for e in settled)
    wins = sum(1 for e in settled if e.outcome == "win")
    losses = sum(1 for e in settled if e.outcome == "loss")
    avg_pnl = realized / len(settled) if settled else 0.0
    notional = sum(e.notional_usd for e in opens)
    by_genre: dict[str, float] = defaultdict(float)
    for e in settled:
        by_genre[e.genre] += e.realized_pnl_usd or 0.0
    by_source: dict[str, int] = defaultdict(int)
    for e in opens:
        by_source[e.source] += 1
    pnls = [e.realized_pnl_usd or 0 for e in settled]
    sd = statistics.pstdev(pnls) if len(pnls) > 1 else 0.0

    md = []
    md.append(f"# Daily P&L — {target}")
    md.append("")
    md.append(f"- **Realized PnL:** ${realized:+.2f}")
    md.append(f"- **Trades opened today:** {len(opens)} (notional ${notional:.2f})")
    md.append(f"- **Trades settled today:** {len(settled)} — {wins}W / {losses}L")
    md.append(f"- **Average realized:** ${avg_pnl:+.2f}")
    md.append(f"- **Stdev of realized:** ${sd:.2f}")
    md.append(f"- **Hit rate:** {(wins/max(1,len(settled)))*100:.1f}%")
    md.append(f"- **Paper-ledger ticks today:** {len(paper_lines)}")
    md.append("")
    md.append("## P&L by genre (settled)")
    md.append("")
    if by_genre:
        for g, v in sorted(by_genre.items(), key=lambda kv: -kv[1]):
            md.append(f"- {g}: ${v:+.2f}")
    else:
        md.append("- (no settled trades)")
    md.append("")
    md.append("## Order sources")
    md.append("")
    if by_source:
        for s, n in by_source.items():
            md.append(f"- {s}: {n}")
    else:
        md.append("- (no orders today)")
    md.append("")
    md.append("## Settled trades (this UTC day)")
    md.append("")
    md.append("| Ticker | Side | Contracts | Entry | Settle | PnL |")
    md.append("|---|---|---|---|---|---|")
    for e in settled[:50]:
        md.append(f"| {e.ticker} | {e.side} | {e.contracts} | "
                    f"{e.entry_price_cents}¢ | "
                    f"{(str(e.settlement_price_cents)+'¢') if e.settlement_price_cents is not None else '—'} | "
                    f"${(e.realized_pnl_usd or 0):+.2f} |")

    text = "\n".join(md)
    out = Path(out_dir) / f"{target}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return text, out
