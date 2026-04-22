from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, redirect, render_template, request, url_for

from . import jobs
from .state import AppState


def create_app(state: AppState | None = None) -> Flask:
    here = Path(__file__).parent
    app = Flask(__name__,
                template_folder=str(here / "templates"),
                static_folder=str(here / "static"))
    state = state or AppState()
    app.config["STATE"] = state

    def _series_xy(s: pd.Series) -> dict:
        s = s.reset_index(drop=True)
        return {"x": list(range(len(s))), "y": [float(v) for v in s.values]}

    def _df_records(df: pd.DataFrame) -> list[dict]:
        d = df.reset_index().to_dict(orient="records")
        # Replace NaN/inf so the JSON is valid
        for row in d:
            for k, v in list(row.items()):
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    row[k] = None
        return d

    # -------------------- HTML pages --------------------

    @app.route("/")
    def index() -> str:
        r = state.report
        return render_template("index.html", page="overview", report=r,
                               has_report=r is not None)

    @app.route("/strategies")
    def strategies_page() -> str:
        r = state.report
        rows = _df_records(r.per_strategy) if r is not None else []
        return render_template("strategies.html", page="strategies", rows=rows,
                               has_report=r is not None)

    @app.route("/genres")
    def genres_page() -> str:
        r = state.report
        rows = _df_records(r.per_genre) if r is not None else []
        best_per_genre = r.best_strategy_per_genre if r is not None else {}
        return render_template("genres.html", page="genres", rows=rows,
                               best_per_genre=best_per_genre,
                               has_report=r is not None)

    @app.route("/markets")
    def markets_page() -> str:
        r = state.report
        rows = _df_records(r.per_market) if r is not None else []
        genres = sorted({row["genre"] for row in rows}) if rows else []
        return render_template("markets.html", page="markets", rows=rows,
                               genres=genres, has_report=r is not None)

    @app.route("/backtest", methods=["GET", "POST"])
    def backtest_page():
        if request.method == "POST":
            markets = int(request.form.get("markets", 24))
            top_k = int(request.form.get("top_k", 30))
            method = request.form.get("method", "hybrid")
            jid = jobs.run_backtest(state, markets=markets, top_k=top_k, method=method)
            return redirect(url_for("backtest_page", jid=jid))
        jid = request.args.get("jid")
        job = state.job(jid) if jid else None
        return render_template("backtest.html", page="backtest", job=job,
                               recent=state.jobs()[:10])

    @app.route("/ensemble")
    def ensemble_page() -> str:
        r = state.report
        return render_template("ensemble.html", page="ensemble", report=r,
                               has_report=r is not None)

    # -------------------- JSON API --------------------

    @app.route("/api/summary")
    def api_summary():
        r = state.report
        if r is None:
            return jsonify({"has_report": False})
        per_genre = r.per_genre
        return jsonify({
            "has_report": True,
            "total_pnl": float(r.equity_curve.iloc[-1]) if len(r.equity_curve) else 0.0,
            "sharpe": float(r.combo.sharpe),
            "active_strategies": int((r.combo.weights > 0).sum()),
            "best_genre": str(per_genre.index[0]) if len(per_genre) else "n/a",
            "oos_sharpe": float(per_genre.attrs.get("oos_sharpe")) if per_genre.attrs.get("oos_sharpe") is not None else None,
            "n_markets": int(len(r.per_market)),
        })

    @app.route("/api/equity")
    def api_equity():
        r = state.report
        if r is None:
            return jsonify({})
        # Top-5 strategies by terminal PnL, plus ensemble
        data = {"ensemble": _series_xy(r.equity_curve)}
        if not r.strategy_equity.empty:
            terms = r.strategy_equity.iloc[-1].sort_values(ascending=False)
            for n in list(terms.head(5).index):
                data[n] = _series_xy(r.strategy_equity[n])
        return jsonify(data)

    @app.route("/api/genre_equity")
    def api_genre_equity():
        r = state.report
        if r is None or r.genre_equity.empty:
            return jsonify({})
        return jsonify({c: _series_xy(r.genre_equity[c]) for c in r.genre_equity.columns})

    @app.route("/api/weights")
    def api_weights():
        r = state.report
        if r is None:
            return jsonify({"labels": [], "values": []})
        top = r.combo.weights.sort_values(ascending=False).head(20)
        return jsonify({
            "labels": [str(i) for i in top.index],
            "values": [float(v) for v in top.values],
        })

    @app.route("/api/genres")
    def api_genres():
        r = state.report
        if r is None:
            return jsonify({"labels": [], "totals": [], "sharpes": []})
        return jsonify({
            "labels": list(r.per_genre.index),
            "totals": [float(v) for v in r.per_genre["total"].values],
            "sharpes": [float(v) for v in r.per_genre["sharpe"].values],
        })

    @app.route("/api/job/<jid>")
    def api_job(jid: str):
        j = state.job(jid)
        if j is None:
            return jsonify({"unknown": True}), 404
        return jsonify({
            "id": j.id, "kind": j.kind, "progress": j.progress,
            "message": j.message, "error": j.error,
            "finished": j.finished is not None,
        })

    return app
