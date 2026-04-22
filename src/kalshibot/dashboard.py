"""Render a self-contained HTML dashboard from a backtest report.

No external server required - the file embeds Chart.js from a CDN and all
data lives inline as JSON. Open `dashboard.html` in any browser.

Optional: `serve(path, port)` starts a tiny stdlib HTTP server.
"""
from __future__ import annotations

import http.server
import json
import socketserver
from html import escape
from pathlib import Path

import pandas as pd

from .analytics import BacktestReport


def _df_table(df: pd.DataFrame, title: str, max_rows: int = 50) -> str:
    df = df.head(max_rows).copy()
    for col in df.select_dtypes(include="float").columns:
        df[col] = df[col].map(lambda v: f"{v:,.4f}")
    headers = "".join(f"<th>{escape(str(c))}</th>" for c in [df.index.name or ""] + list(df.columns))
    rows = []
    for idx, r in df.iterrows():
        cells = "".join(f"<td>{escape(str(v))}</td>" for v in [idx] + list(r.values))
        rows.append(f"<tr>{cells}</tr>")
    return f"""
<section>
  <h2>{escape(title)}</h2>
  <div class="tablewrap"><table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
</section>
"""


def _series_to_xy(s: pd.Series) -> dict:
    s = s.reset_index(drop=True)
    return {"x": list(range(len(s))), "y": [float(v) for v in s.values]}


def render(report: BacktestReport, out_path: str | Path = "dashboard.html",
           title: str = "Kalshibot Dashboard") -> Path:
    out_path = Path(out_path)

    equity = _series_to_xy(report.equity_curve)
    # Top-N strategy equity curves (by terminal PnL)
    if not report.strategy_equity.empty:
        terminals = report.strategy_equity.iloc[-1].sort_values(ascending=False)
        top = list(terminals.head(10).index)
        bottom = list(terminals.tail(5).index)
        strat_curves = {n: _series_to_xy(report.strategy_equity[n]) for n in top + bottom}
    else:
        strat_curves = {}

    genre_curves = {n: _series_to_xy(report.genre_equity[n]) for n in report.genre_equity.columns}

    weights_top = report.combo.weights.sort_values(ascending=False).head(20)
    weights_data = {
        "labels": [str(i) for i in weights_top.index],
        "values": [float(v) for v in weights_top.values],
    }

    per_genre_bars = {
        "labels": list(report.per_genre.index),
        "totals": [float(v) for v in report.per_genre["total"].values],
        "sharpes": [float(v) for v in report.per_genre["sharpe"].values],
    }

    payload = {
        "equity": equity,
        "strategy_curves": strat_curves,
        "genre_curves": genre_curves,
        "weights": weights_data,
        "per_genre": per_genre_bars,
        "summary": {
            "total_pnl": float(report.equity_curve.iloc[-1]) if len(report.equity_curve) else 0.0,
            "sharpe": float(report.combo.sharpe),
            "n_strategies": int((report.combo.weights > 0).sum()),
            "best_genre": str(report.per_genre.index[0]) if len(report.per_genre) else "n/a",
        },
        "best_strategy_per_genre": report.best_strategy_per_genre,
    }

    tables = (
        _df_table(report.per_strategy.head(25), "Top strategies (by Sharpe)") +
        _df_table(report.per_genre, "Per-genre performance (ensemble)") +
        _df_table(report.per_market.sort_values("total", ascending=False).head(25),
                  "Top markets by ensemble PnL")
    )

    html = _TEMPLATE.format(
        title=escape(title),
        data_json=json.dumps(payload),
        tables=tables,
    )
    out_path.write_text(html, encoding="utf-8")
    return out_path


def serve(path: str | Path = "dashboard.html", port: int = 8765) -> None:
    path = Path(path).resolve()
    directory = str(path.parent)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=directory, **kw)

    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Serving {path.name} at http://localhost:{port}/{path.name}")
        httpd.serve_forever()


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {{ color-scheme: dark; }}
  body {{ font-family: -apple-system, system-ui, Segoe UI, Roboto, sans-serif;
         background: #0b0d10; color: #e6e6e6; margin: 0; padding: 24px; }}
  h1 {{ margin: 0 0 4px; font-size: 28px; }}
  h2 {{ margin: 32px 0 12px; font-size: 18px; color: #9adcff; }}
  .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 16px 0; }}
  .kpi {{ background: #11161c; border: 1px solid #1f2933; padding: 16px;
          border-radius: 10px; }}
  .kpi .label {{ font-size: 12px; color: #9aa5b1; text-transform: uppercase; }}
  .kpi .value {{ font-size: 22px; font-weight: 600; margin-top: 6px; }}
  .charts {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }}
  .card {{ background: #11161c; border: 1px solid #1f2933; padding: 16px;
           border-radius: 10px; height: 360px; }}
  .card.full {{ grid-column: 1 / -1; height: 420px; }}
  .tablewrap {{ overflow-x: auto; background: #11161c; border: 1px solid #1f2933;
                border-radius: 10px; padding: 8px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #1f2933; }}
  th {{ color: #9aa5b1; font-weight: 600; }}
  tr:hover td {{ background: #161c24; }}
  footer {{ margin-top: 32px; color: #6b7280; font-size: 12px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="grid" id="kpis"></div>

<div class="charts">
  <div class="card full"><canvas id="equity"></canvas></div>
  <div class="card"><canvas id="genres"></canvas></div>
  <div class="card"><canvas id="weights"></canvas></div>
  <div class="card full"><canvas id="strategies"></canvas></div>
  <div class="card full"><canvas id="genre_curves"></canvas></div>
</div>

{tables}

<footer>Generated by kalshibot. All figures from backtest simulation.</footer>

<script>
const D = {data_json};

function kpi(label, value) {{
  return `<div class="kpi"><div class="label">${{label}}</div><div class="value">${{value}}</div></div>`;
}}
document.getElementById('kpis').innerHTML = [
  kpi('Total Ensemble PnL', D.summary.total_pnl.toFixed(4)),
  kpi('Sharpe (annualized)', D.summary.sharpe.toFixed(2)),
  kpi('Active strategies', D.summary.n_strategies),
  kpi('Best genre', D.summary.best_genre),
].join('');

const palette = ['#7dd3fc','#fca5a5','#86efac','#fcd34d','#c4b5fd','#fda4af',
                 '#a5f3fc','#fdba74','#bef264','#f0abfc','#67e8f9','#fde68a',
                 '#d9f99d','#f9a8d4','#a7f3d0','#bae6fd'];

new Chart(document.getElementById('equity'), {{
  type: 'line',
  data: {{ labels: D.equity.x, datasets: [{{
    label: 'Ensemble equity', data: D.equity.y,
    borderColor: '#7dd3fc', backgroundColor: 'rgba(125,211,252,0.15)',
    fill: true, pointRadius: 0, tension: 0.15
  }}] }},
  options: {{ responsive: true, maintainAspectRatio: false,
              plugins: {{ title: {{ display: true, text: 'Backtest equity curve' }} }} }}
}});

new Chart(document.getElementById('genres'), {{
  type: 'bar',
  data: {{ labels: D.per_genre.labels, datasets: [
    {{ label: 'Total PnL', data: D.per_genre.totals, backgroundColor: '#7dd3fc' }},
    {{ label: 'Sharpe', data: D.per_genre.sharpes, backgroundColor: '#fca5a5' }},
  ] }},
  options: {{ responsive: true, maintainAspectRatio: false,
              plugins: {{ title: {{ display: true, text: 'PnL & Sharpe by genre' }} }} }}
}});

new Chart(document.getElementById('weights'), {{
  type: 'bar',
  data: {{ labels: D.weights.labels, datasets: [{{
    label: 'Ensemble weight', data: D.weights.values, backgroundColor: '#86efac'
  }}] }},
  options: {{ indexAxis: 'y', responsive: true, maintainAspectRatio: false,
              plugins: {{ title: {{ display: true, text: 'Top strategy weights' }} }} }}
}});

function multiLine(canvasId, curves, title) {{
  const names = Object.keys(curves);
  const datasets = names.map((n, i) => ({{
    label: n, data: curves[n].y, borderColor: palette[i % palette.length],
    backgroundColor: palette[i % palette.length], pointRadius: 0, tension: 0.1, borderWidth: 1.5
  }}));
  const x = names.length ? curves[names[0]].x : [];
  new Chart(document.getElementById(canvasId), {{
    type: 'line',
    data: {{ labels: x, datasets }},
    options: {{ responsive: true, maintainAspectRatio: false,
                plugins: {{ title: {{ display: true, text: title }},
                           legend: {{ display: names.length <= 12 }} }} }}
  }});
}}

multiLine('strategies', D.strategy_curves, 'Top & bottom strategy equity (cumulative PnL)');
multiLine('genre_curves', D.genre_curves, 'Per-genre ensemble equity');
</script>
</body>
</html>
"""
