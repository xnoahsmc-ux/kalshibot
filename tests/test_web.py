from pathlib import Path

import pickle

from kalshibot.analytics import build_report
from kalshibot.backtester import evaluate_all, stack_pnls
from kalshibot.combiner import best_combination
from kalshibot.data import synthesize_history
from kalshibot.strategies.registry import all_strategies
from kalshibot.web import create_app
from kalshibot.web.state import AppState


def _seed_state(tmp_path: Path) -> AppState:
    strats = all_strategies()[:25]
    hs = [synthesize_history(ticker=f"KXNFL-{i}", n=200, seed=i) for i in range(4)]
    evals = evaluate_all(strats, hs)
    pnl = stack_pnls(evals)
    combo = best_combination(pnl, top_k=10)
    report = build_report(evals, [h.ticker for h in hs], combo)
    state = AppState(report_path=tmp_path / "r.pkl")
    state.set_report(report)
    return state


def test_pages_render(tmp_path):
    state = _seed_state(tmp_path)
    app = create_app(state)
    client = app.test_client()
    for path in ["/", "/strategies", "/genres", "/markets", "/ensemble",
                 "/reliability", "/live", "/trades", "/popular",
                 "/paperbots", "/weather", "/settings", "/backtest"]:
        r = client.get(path)
        assert r.status_code == 200, (path, r.status_code)
        assert b"Kalshi Bot" in r.data


def test_apis_return_json(tmp_path):
    state = _seed_state(tmp_path)
    app = create_app(state)
    client = app.test_client()
    r = client.get("/api/summary")
    assert r.status_code == 200
    j = r.get_json()
    assert j["has_report"] is True
    assert j["n_markets"] == 4
    for p in ["/api/equity", "/api/weights", "/api/genres", "/api/genre_equity",
              "/api/reliability", "/api/walkforward", "/api/sparks",
              "/api/suggestions"]:
        rr = client.get(p)
        assert rr.status_code == 200
        assert rr.get_json() is not None


def test_csv_exports(tmp_path):
    state = _seed_state(tmp_path)
    app = create_app(state)
    client = app.test_client()
    for kind in ["strategies", "genres", "markets", "weights"]:
        r = client.get(f"/export/{kind}.csv")
        assert r.status_code == 200
        assert r.mimetype == "text/csv"
        assert b"," in r.data


def test_empty_state_renders(tmp_path):
    state = AppState(report_path=tmp_path / "nope.pkl")
    app = create_app(state)
    client = app.test_client()
    r = client.get("/")
    assert r.status_code == 200
    assert b"No backtest yet" in r.data
