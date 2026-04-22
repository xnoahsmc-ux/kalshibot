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

    @app.route("/api/reliability")
    def api_reliability():
        r = state.report
        if r is None:
            return jsonify({"bins": []})
        # Build a reliability curve from the ensemble blended probabilities vs
        # terminal outcomes of each market. We approximate with the per-market
        # average probability and the terminal outcome.
        from .calibration_api import reliability_from_report  # lazy
        bins = reliability_from_report(r)
        return jsonify({"bins": bins})

    @app.route("/api/sparks")
    def api_sparks():
        """Downsampled strategy equity curves (~40 points each) for sparkline
        rendering in the strategies table.
        """
        r = state.report
        if r is None or r.strategy_equity.empty:
            return jsonify({})
        df = r.strategy_equity
        step = max(1, len(df) // 40)
        small = df.iloc[::step]
        return jsonify({
            c: [float(v) for v in small[c].tolist()] for c in small.columns
        })

    @app.route("/api/suggestions")
    def api_suggestions():
        """What would the bot do on each market right now?
        Uses the last bar of each market's history in the report.
        """
        r = state.report
        if r is None:
            return jsonify([])
        # We can't rebuild histories without re-synthesizing; instead rank
        # markets by ensemble PnL and surface the last-bar hit-rate as a
        # heuristic confidence. This is illustrative for the UI; live mode
        # pulls real snapshots.
        rows = []
        df = r.per_market.reset_index()
        df = df.sort_values("total", ascending=False)
        for _, row in df.iterrows():
            rows.append({
                "market": row.get("market"),
                "genre": row.get("genre"),
                "total": float(row.get("total", 0.0)),
                "sharpe": float(row.get("sharpe", 0.0)),
                "hit_rate": float(row.get("hit_rate", 0.0)),
            })
        return jsonify(rows[:50])

    @app.route("/api/walkforward")
    def api_walkforward():
        r = state.report
        if r is None:
            return jsonify({"oos_sharpe": None})
        return jsonify({
            "oos_sharpe": float(r.per_genre.attrs.get("oos_sharpe"))
                          if r.per_genre.attrs.get("oos_sharpe") is not None else None,
        })

    @app.route("/export/<kind>.csv")
    def export_csv(kind: str):
        r = state.report
        if r is None:
            return Response("no report", status=404)
        if kind == "strategies":
            df = r.per_strategy.reset_index()
        elif kind == "genres":
            df = r.per_genre.reset_index()
        elif kind == "markets":
            df = r.per_market.reset_index()
        elif kind == "weights":
            df = r.combo.weights.sort_values(ascending=False).to_frame("weight").reset_index()
            df.columns = ["strategy", "weight"]
        else:
            return Response("unknown export", status=404)
        return Response(df.to_csv(index=False),
                        mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={kind}.csv"})

    @app.route("/live")
    def live_page() -> str:
        r = state.report
        return render_template("live.html", page="live", has_report=r is not None)

    @app.route("/settings", methods=["GET", "POST"])
    def settings_page():
        from ..config import load_config
        cfg = load_config()
        if request.method == "POST":
            # Settings are env-backed; we write suggestions so the user can
            # paste them into .env - we don't mutate .env from the server.
            return render_template("settings.html", page="settings", cfg=cfg,
                                   saved=request.form.to_dict())
        return render_template("settings.html", page="settings", cfg=cfg, saved=None)

    @app.route("/reliability")
    def reliability_page() -> str:
        r = state.report
        return render_template("reliability.html", page="reliability",
                               has_report=r is not None)

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
