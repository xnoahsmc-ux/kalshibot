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

    @app.route("/trades")
    def trades_page() -> str:
        return render_template("trades.html", page="trades")

    @app.route("/popular")
    def popular_page() -> str:
        return render_template("popular.html", page="popular")

    @app.route("/paperbots")
    def paperbots_page() -> str:
        from ..paper import PRESETS
        return render_template("paperbots.html", page="paperbots", presets=PRESETS)

    @app.route("/journal")
    def journal_page() -> str:
        return render_template("journal.html", page="journal")

    @app.route("/picks")
    def picks_page() -> str:
        return render_template("picks.html", page="picks")

    @app.route("/brain")
    def brain_page() -> str:
        return render_template("brain.html", page="brain")

    @app.route("/alerts")
    def alerts_page() -> str:
        return render_template("alerts.html", page="alerts")

    @app.route("/api/streaks")
    def api_streaks():
        """Win/loss streak + today/week realized PnL for the streak tracker."""
        import datetime as dt
        settled = sorted(state.journal.settled_entries(),
                          key=lambda e: e.settled_at or 0)
        if not settled:
            return jsonify({"current_streak": 0, "streak_kind": "none",
                              "best_win_streak": 0, "today_pnl": 0.0,
                              "week_pnl": 0.0, "lifetime_pnl": 0.0,
                              "wins": 0, "losses": 0})
        cur = 0
        kind = "none"
        for e in reversed(settled):
            if e.outcome == "win":
                if kind in ("none", "win"): kind = "win"; cur += 1
                else: break
            elif e.outcome == "loss":
                if kind in ("none", "loss"): kind = "loss"; cur += 1
                else: break
            else: break
        # Best historical win streak
        best, run = 0, 0
        for e in settled:
            if e.outcome == "win": run += 1; best = max(best, run)
            else: run = 0
        today = dt.datetime.utcnow().date()
        week_start = today - dt.timedelta(days=today.weekday())
        today_pnl = sum(e.realized_pnl_usd or 0 for e in settled
                          if dt.datetime.utcfromtimestamp(e.settled_at or 0).date() == today)
        week_pnl  = sum(e.realized_pnl_usd or 0 for e in settled
                          if dt.datetime.utcfromtimestamp(e.settled_at or 0).date() >= week_start)
        lifetime  = sum(e.realized_pnl_usd or 0 for e in settled)
        return jsonify({
            "current_streak": cur, "streak_kind": kind,
            "best_win_streak": best,
            "today_pnl": round(today_pnl, 2),
            "week_pnl": round(week_pnl, 2),
            "lifetime_pnl": round(lifetime, 2),
            "wins": sum(1 for e in settled if e.outcome == "win"),
            "losses": sum(1 for e in settled if e.outcome == "loss"),
        })

    @app.route("/api/goal", methods=["GET", "POST"])
    def api_goal():
        from pathlib import Path
        import json as _json
        path = Path("data/goal.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        if request.method == "POST":
            payload = request.get_json(silent=True) or {}
            data = {
                "daily_goal_usd": float(payload.get("daily_goal_usd", 50)),
                "weekly_goal_usd": float(payload.get("weekly_goal_usd", 250)),
            }
            path.write_text(_json.dumps(data))
            return jsonify({"ok": True, **data})
        if path.exists():
            try:
                return jsonify(_json.loads(path.read_text()))
            except Exception:
                pass
        return jsonify({"daily_goal_usd": 50.0, "weekly_goal_usd": 250.0})

    @app.route("/api/picks")
    def api_picks():
        """Top picks, preferring the v2 stack when its universe + signals
        are healthy. Falls back to the legacy combo engine.
        """
        from ..combo_engine import build_betting_slip
        from ..config import load_config
        feed = state.live()
        signals = feed.signals() if feed else []
        cfg = load_config()
        slip = build_betting_slip(signals, bankroll=cfg.bankroll_usd)

        # v2 augmentation: when we have a live feed AND credentials,
        # also produce a v2 evaluation for the same markets and surface
        # whichever picks v2 endorses with positive edge + min confidence.
        v2_picks: list[dict] = []
        try:
            if feed and cfg.api_key_id and cfg.api_private_key_path:
                from ..v2 import config as v2cfg, strategy as v2strategy
                from ..v2.data import Market
                from ..v2.risk import DayState, Position
                vc = v2cfg.load("config.yaml")
                vc.bankroll_usd = cfg.bankroll_usd
                day = DayState()
                # We rebuild Market objects from live signals so v2 can score
                # them without making a separate Kalshi roundtrip.
                for s in signals[:30]:
                    m = Market(
                        ticker=s.ticker, title=s.title,
                        event_ticker=s.ticker.split("-", 1)[0],
                        series_ticker=s.ticker.split("-", 1)[0],
                        yes_bid=s.yes_bid, yes_ask=s.yes_ask,
                        last_price=s.last,
                        volume_24h_usd=s.volume * s.last,
                        open_interest=200,
                        minutes_to_close=s.minutes_to_close,
                    )
                    hist = feed.history(s.ticker)
                    mids = [float(v) for v in hist.mid.tolist()[-10:]] if hist is not None else [m.mid]
                    d = v2strategy.evaluate(m, vc, vc.bankroll_usd, [], day, mids)
                    if d.intent is not None:
                        v2_picks.append({
                            "ticker": d.intent.ticker,
                            "side": d.intent.side.upper(),
                            "fair_value": round(d.fair_value, 3),
                            "edge_cents": d.edge_cents,
                            "confidence": round(d.confidence, 2),
                            "stake_usd": round(d.intent.contracts * d.intent.target_price_cents / 100.0, 2),
                            "contracts": d.intent.contracts,
                            "target_price_cents": d.intent.target_price_cents,
                            "contributors": [s.name for s in d.contributors],
                            "reason": d.intent.reason,
                        })
        except Exception:
            v2_picks = []

        return jsonify({
            "running": bool(feed and feed.status().running),
            "v2_picks": v2_picks,
            **slip,
        })

    @app.route("/api/brain")
    def api_brain():
        # Snapshot of what the bot has learned and how it's connecting markets
        feed = state.live()
        signals = feed.signals() if feed else []
        # Genre breakdown
        genre_counts: dict[str, int] = {}
        ev_by_genre: dict[str, float] = {}
        for s in signals:
            genre_counts[s.genre] = genre_counts.get(s.genre, 0) + 1
            if s.suggested_side != "flat":
                ev_by_genre[s.genre] = ev_by_genre.get(s.genre, 0.0) + s.expected_profit
        # Champion ensemble snapshot
        champ = state.champion
        # Learner summary
        learner = state.learner.summary()
        # Journal stats
        jstats = state.journal.stats()
        # Recent fair-value attributions (which signals used NWS / ESPN)
        attributions: list[dict] = []
        for s in signals[:20]:
            if s.fair_value_source:
                attributions.append({
                    "ticker": s.ticker, "title": s.title,
                    "source": s.fair_value_source,
                    "detail": s.fair_value_detail,
                    "fair_prob": s.fair_value_prob,
                    "ensemble_prob": s.ensemble_prob,
                    "edge": s.edge,
                })
        # Strategy correlation cluster: rough — group by name prefix
        from collections import defaultdict
        clusters: dict[str, list[str]] = defaultdict(list)
        if champ:
            for name in champ.weights:
                prefix = name.split("_")[0]
                clusters[prefix].append(name)
        return jsonify({
            "champion": champ.to_json() if champ else None,
            "learner": learner,
            "journal_stats": jstats,
            "genre_counts": genre_counts,
            "ev_by_genre": ev_by_genre,
            "fair_value_attributions": attributions,
            "strategy_clusters": {k: v for k, v in clusters.items() if len(v) > 0},
            "live_signals": len(signals),
            "actionable_signals": len([s for s in signals
                                         if s.suggested_side != "flat"]),
        })

    @app.route("/api/champion/refit", methods=["POST"])
    def api_champion_refit():
        """Refit the champion ensemble from live Kalshi history. Runs in
        the background so the UI doesn't block."""
        from ..alpha import fit_from_kalshi_history
        from ..config import load_config
        from ..kalshi_client import KalshiClient
        cfg = load_config()
        if not (cfg.api_key_id and cfg.api_private_key_path):
            return jsonify({"ok": False, "error": "Credentials not configured"}), 400
        import threading
        def _job():
            try:
                champ = fit_from_kalshi_history(KalshiClient(cfg),
                    markets=40, lookback_hours=24 * 100)
                if champ:
                    state.champion = champ
            except Exception:
                pass
        threading.Thread(target=_job, daemon=True).start()
        return jsonify({"ok": True})

    @app.route("/api/alerts")
    def api_alerts():
        from ..notifications import recent_alerts
        return jsonify({
            "scheduler": state.scheduler.status(),
            "alerts": recent_alerts(50),
        })

    @app.route("/api/alerts/test", methods=["POST"])
    def api_alerts_test():
        from ..notifications import send_alert
        msg = (request.get_json(silent=True) or {}).get(
            "message", "Test alert from Kalshi Bot")
        result = send_alert(msg, kind="test")
        return jsonify({"ok": result.ok, "backend": result.backend,
                        "message": result.message})

    @app.route("/api/alerts/trigger", methods=["POST"])
    def api_alerts_trigger():
        msg = state.scheduler.trigger_now()
        return jsonify({"ok": True, "message": msg})

    @app.route("/api/scheduler", methods=["POST"])
    def api_scheduler_set():
        payload = request.get_json(silent=True) or {}
        if "interval_seconds" in payload:
            try:
                state.scheduler.interval = max(60, int(payload["interval_seconds"]))
            except Exception:
                pass
        action = payload.get("action")
        if action == "start":
            state.scheduler.start()
        elif action == "stop":
            state.scheduler.stop()
        return jsonify(state.scheduler.status())

    @app.route("/api/execute/diagnose")
    def api_execute_diagnose():
        ex = state.executor()
        if ex is None:
            from ..config import load_config
            cfg = load_config()
            return jsonify({
                "ok": False,
                "error": "Credentials not configured",
                "config": {
                    "api_key_id_set": bool(cfg.api_key_id),
                    "private_key_path_set": bool(cfg.api_private_key_path),
                    "dry_run": cfg.dry_run,
                },
            })
        return jsonify({"ok": True, **ex.diagnose()})

    @app.route("/api/execute", methods=["POST"])
    def api_execute():
        payload = request.get_json(silent=True) or {}
        ticker = payload.get("ticker")
        side   = payload.get("side")
        stake  = float(payload.get("stake", 0) or 0)
        price_cents = payload.get("price_cents")
        if not ticker or not side or stake < 1.0:
            return jsonify({"ok": False, "error": "Need ticker, side, stake≥$1"}), 400
        ex = state.executor()
        if ex is None:
            return jsonify({"ok": False, "error": "Credentials not configured"}), 400
        # Get fresh metadata from live feed if available
        feed = state.live()
        sig = None
        if feed is not None:
            for s in feed.signals():
                if s.ticker == ticker:
                    sig = s; break
        result = ex.execute_manual(
            ticker=ticker, side=side, stake_usd=stake,
            limit_price_cents=int(price_cents) if price_cents else None,
            genre=sig.genre if sig else "other",
            title=sig.title if sig else "",
            confidence=(sig.confidence if sig else "manual"),
            ensemble_prob=(sig.ensemble_prob if sig else 0.5),
            reason=(sig.rationale if sig else "manual"),
        )
        return jsonify({
            "ok": result.ok, "message": result.message,
            "entry": None if result.entry is None else {
                "id": result.entry.id, "ticker": result.entry.ticker,
                "side": result.entry.side, "contracts": result.entry.contracts,
                "entry_price_cents": result.entry.entry_price_cents,
                "notional_usd": result.entry.notional_usd,
                "kalshi_status": result.entry.kalshi_status,
            },
        })

    @app.route("/api/risk", methods=["GET", "POST"])
    def api_risk():
        if request.method == "POST":
            payload = request.get_json(silent=True) or {}
            for key in ("kill_switch", "auto_trade_enabled"):
                if key in payload:
                    setattr(state.limits, key, bool(payload[key]))
            for key in ("daily_loss_limit_usd", "max_notional_per_trade_usd",
                         "auto_trade_min_edge"):
                if key in payload:
                    setattr(state.limits, key, float(payload[key]))
            for key in ("max_open_positions", "max_open_per_genre",
                         "max_same_series"):
                if key in payload:
                    setattr(state.limits, key, int(payload[key]))
            if "auto_trade_min_conf" in payload:
                conf = str(payload["auto_trade_min_conf"]).lower()
                if conf in ("low", "med", "high"):
                    state.limits.auto_trade_min_conf = conf
        L = state.limits
        state.risk_state.rolling_reset_if_needed()
        return jsonify({
            "kill_switch": L.kill_switch,
            "auto_trade_enabled": L.auto_trade_enabled,
            "auto_trade_min_conf": L.auto_trade_min_conf,
            "auto_trade_min_edge": L.auto_trade_min_edge,
            "daily_loss_limit_usd": L.daily_loss_limit_usd,
            "max_open_positions": L.max_open_positions,
            "max_open_per_genre": L.max_open_per_genre,
            "max_same_series": L.max_same_series,
            "max_notional_per_trade_usd": L.max_notional_per_trade_usd,
            "today_realized_pnl_usd":
                state.journal.today_realized_pnl(),
            "today_orders_sent": state.risk_state.today_orders_sent,
        })

    @app.route("/api/journal")
    def api_journal():
        entries = state.journal.entries()
        entries.sort(key=lambda e: -e.created_at)
        return jsonify({
            "stats": state.journal.stats(),
            "entries": [{
                "id": e.id, "created_at": e.created_at,
                "ticker": e.ticker, "genre": e.genre, "title": e.title,
                "side": e.side, "contracts": e.contracts,
                "entry_price_cents": e.entry_price_cents,
                "notional_usd": e.notional_usd,
                "source": e.source, "confidence": e.confidence,
                "ensemble_prob": e.ensemble_prob,
                "kalshi_status": e.kalshi_status,
                "settled_at": e.settled_at,
                "settlement_price_cents": e.settlement_price_cents,
                "realized_pnl_usd": e.realized_pnl_usd,
                "outcome": e.outcome,
                "reason": e.reason,
            } for e in entries[:200]],
            "learner": state.learner.summary(),
        })

    @app.route("/weather")
    def weather_page() -> str:
        return render_template("weather.html", page="weather")

    @app.route("/api/paperbots")
    def api_paperbots_list():
        pm = state.paper()
        return jsonify({"bots": [
            {
                "id": b.id, "name": b.name, "style": b.style,
                "starting_cash": b.starting_cash,
                "cash": round(b.cash, 2),
                "open_value": round(b.open_position_value(), 2),
                "equity": round(b.equity, 2),
                "pnl": round(b.total_pnl, 2),
                "pnl_pct": round(b.pnl_pct * 100, 2),
                "wins": b.wins, "losses": b.losses,
                "hit_rate": round(b.hit_rate * 100, 1),
                "open_count": len(b.open_trades),
                "closed_count": len(b.closed_trades),
                "equity_curve": [[round(t, 0), round(e, 2)] for t, e in b.equity_history[-200:]],
                "open_trades": [{
                    "ticker": t.ticker, "side": t.side, "contracts": t.contracts,
                    "entry_price": round(t.entry_price, 3),
                    "current_price": round(t.current_price, 3),
                    "unrealized_pnl": round(t.unrealized_pnl, 2),
                } for t in list(b.open_trades.values())[-20:]],
            } for b in pm.bots()
        ]})

    @app.route("/api/paperbots/create", methods=["POST"])
    def api_paperbots_create():
        from ..paper import PRESETS
        pm = state.paper()
        payload = request.get_json(silent=True) or {}
        style = payload.get("style", "balanced")
        if style not in PRESETS:
            return jsonify({"ok": False, "error": f"Unknown style {style}"}), 400
        try:
            bankroll = float(payload.get("bankroll", 100))
        except (TypeError, ValueError):
            bankroll = 100.0
        name = payload.get("name") or PRESETS[style]["label"]
        bot = pm.add(name=name, style=style, bankroll=bankroll)
        return jsonify({"ok": True, "id": bot.id})

    @app.route("/api/paperbots/<bot_id>", methods=["DELETE"])
    def api_paperbots_delete(bot_id: str):
        state.paper().remove(bot_id)
        return jsonify({"ok": True})

    @app.route("/api/paperbots/reset", methods=["POST"])
    def api_paperbots_reset():
        state.paper().reset_all()
        state.paper().seed_defaults()
        return jsonify({"ok": True})

    @app.route("/api/weather")
    def api_weather():
        from ..external_data import NWSClient
        nws = NWSClient()
        out = []
        for cf in nws.forecast_all():
            out.append({
                "code": cf.code, "name": cf.name,
                "lat": cf.lat, "lon": cf.lon,
                "today_high": cf.today_high, "today_low": cf.today_low,
                "today_summary": cf.today_summary,
                "tomorrow_high": cf.tomorrow_high,
                "tomorrow_low": cf.tomorrow_low,
                "updated_at": cf.updated_at,
            })
        return jsonify({"cities": out})

    @app.route("/live")
    def live_page() -> str:
        r = state.report
        feed = state.live()
        live_status = feed.status() if feed else None
        return render_template("live.html", page="live", has_report=r is not None,
                               live_status=live_status)

    @app.route("/api/live/status")
    def api_live_status():
        feed = state.live()
        if feed is None:
            from ..config import load_config
            cfg = load_config()
            return jsonify({
                "configured": bool(cfg.api_key_id and cfg.api_private_key_path),
                "running": False,
                "env": cfg.env,
                "base_url": cfg.base_url,
                "has_report": state.report is not None,
            })
        st = feed.status()
        from ..config import load_config
        cfg = load_config()
        return jsonify({
            "configured": bool(cfg.api_key_id and cfg.api_private_key_path),
            "running": st.running,
            "env": cfg.env,
            "base_url": cfg.base_url,
            "last_poll_at": st.last_poll_at,
            "last_error": st.last_error,
            "markets_seen": st.markets_seen,
            "polls": st.polls,
            "has_report": state.report is not None,
        })

    @app.route("/api/live/start", methods=["POST"])
    def api_live_start():
        from ..config import load_config
        cfg = load_config()
        if not (cfg.api_key_id and cfg.api_private_key_path):
            return jsonify({"ok": False,
                            "error": "Missing KALSHI_API_KEY_ID or KALSHI_API_PRIVATE_KEY_PATH. "
                                     "Fill them into your .env then restart the server."}), 400
        from ..kalshi_client import KalshiClient
        from ..live import LiveFeed
        client = KalshiClient(cfg)
        combo = state.report.combo if state.report is not None else None
        feed = state.live() or LiveFeed(cfg=cfg, combo=combo, client=client)
        if combo is not None:
            feed.set_combination(combo)
        # Do a synchronous sanity check so we return a useful error if auth fails
        try:
            check = client.auth_check()
        except Exception as e:
            return jsonify({"ok": False, "error": f"Auth check raised: {e}"}), 500
        if not check.get("public_ok"):
            return jsonify({"ok": False, "error": check.get("public_error", "network error"),
                            "check": check}), 502
        feed.start()
        state.set_live(feed)
        return jsonify({"ok": True, "check": check})

    @app.route("/api/live/stop", methods=["POST"])
    def api_live_stop():
        feed = state.live()
        if feed:
            feed.stop()
        return jsonify({"ok": True})

    @app.route("/api/live/signals")
    def api_live_signals():
        feed = state.live()
        if feed is None:
            return jsonify({"signals": [], "running": False})
        sigs = feed.signals()
        return jsonify({
            "running": feed.status().running,
            "signals": [s.__dict__ for s in sigs],
        })

    @app.route("/api/balance")
    def api_balance():
        from ..config import load_config
        from ..kalshi_client import KalshiClient
        cfg = load_config()
        if not (cfg.api_key_id and cfg.api_private_key_path):
            return jsonify({"balance_text": "—", "portfolio_text": "—",
                            "configured": False})
        client = KalshiClient(cfg)
        try:
            bal = client.get_balance()
            b = bal.get("balance", 0)
            pv = bal.get("portfolio_value", 0)
            # Kalshi returns balance in cents (pennies-of-dollar).
            return jsonify({
                "balance_cents": b, "portfolio_cents": pv,
                "balance_text": f"${b/100:,.2f}",
                "portfolio_text": f"${pv/100:,.2f}",
                "configured": True,
            })
        except Exception as e:
            return jsonify({"balance_text": "err", "portfolio_text": "—",
                            "error": str(e), "configured": True})

    @app.route("/api/trades")
    def api_trades():
        from ..config import load_config
        from ..kalshi_client import KalshiClient
        cfg = load_config()
        if not (cfg.api_key_id and cfg.api_private_key_path):
            return jsonify({"positions": [], "orders": [], "configured": False})
        client = KalshiClient(cfg)
        try:
            positions = client.get_positions(limit=200).get("market_positions", [])
        except Exception as e:
            positions = []
            pos_err = str(e)
        else:
            pos_err = None
        try:
            orders = client.get_orders(limit=100).get("orders", [])
        except Exception as e:
            orders = []
            ord_err = str(e)
        else:
            ord_err = None
        return jsonify({
            "positions": positions, "orders": orders,
            "configured": True,
            "positions_error": pos_err, "orders_error": ord_err,
        })

    @app.route("/api/popular")
    def api_popular():
        """Snapshot of the most-traded markets from the live feed, one per
        popular series, with their recent mid price series."""
        feed = state.live()
        if feed is None or not feed.signals():
            return jsonify({"markets": []})
        signals = feed.signals()[:12]
        out = []
        for s in signals:
            hist = feed.history(s.ticker)
            mids = []
            if hist is not None and len(hist.df) > 0:
                mids = [float(v) for v in hist.mid.tolist()[-60:]]
            out.append({
                "ticker": s.ticker, "title": s.title, "genre": s.genre,
                "bid": s.yes_bid, "ask": s.yes_ask, "last": s.last,
                "volume": s.volume,
                "prob": s.ensemble_prob, "edge": s.edge,
                "suggested_side": s.suggested_side,
                "confidence": s.confidence,
                "mids": mids,
            })
        return jsonify({"markets": out})

    @app.route("/api/live/auth_check")
    def api_live_auth_check():
        from ..config import load_config
        from ..kalshi_client import KalshiClient
        cfg = load_config()
        client = KalshiClient(cfg)
        try:
            return jsonify(client.auth_check())
        except Exception as e:
            return jsonify({"error": str(e)}), 500

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
