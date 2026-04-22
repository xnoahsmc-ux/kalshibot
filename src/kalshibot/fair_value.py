"""Convert real external data into P(YES) for Kalshi markets.

Parses a market ticker + title, pulls the relevant forecast/odds, and
produces a fair value probability the ensemble can blend with.

Weather example:
  KXHIGHLAX-26APR22-T64   "Will the high temp in LA be <64° on Apr 22?"
  Interpreted as: target=64, direction="<", city=LAX, date=Apr 22.
  NWS forecast high for LAX on that day is, say, 72°F with sigma=3°F.
  P(YES) = P(actual < 64) = norm_cdf(64, 72, 3) = ~0.4%.

Sports example:
  KXNFLGAME-25-SFO-DAL    "Will SF beat DAL?"
  ESPN implied probabilities for SFO vs DAL derived from spread / moneyline.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .external_data import CITIES, CityForecast, ESPNClient, NWSClient

# Typical 1-day forecast error for NWS temperature predictions.
# Real values are ~2-4°F at 24h horizon, widening for further out.
WEATHER_SIGMA_F = 3.0


def _norm_cdf(x: float, mu: float, sigma: float) -> float:
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2))))


@dataclass
class FairValue:
    prob_yes: float             # model's estimate
    confidence: float = 0.6     # 0-1 how much to trust this
    source: str = ""
    detail: str = ""


# ----- Weather ---------------------------------------------------------------

_CITY_BY_SERIES: dict[str, dict] = {c["kalshi"]: c for c in CITIES}


def parse_weather_ticker(ticker: str, title: str = "") -> dict | None:
    """Extract (series, city, direction, target_temp) from a weather ticker.

    Handles shapes we've observed on Kalshi:
      KXHIGHLAX-26APR22-T64     -> target=64, direction=">" (title says "above")
      KXHIGHLAX-26APR22-B64.5   -> target=64.5, direction="<" (title says "below")
      KXHIGHLAX-26APR22-B65-66  -> range 65-66
      KXHIGHLAX-26APR22-T72     -> target=72, direction=">"
    Falls back to parsing the title for direction if ticker is ambiguous.
    """
    t = ticker.upper()
    for series in _CITY_BY_SERIES:
        if t.startswith(series + "-"):
            m = re.search(r"[-_](T|B)([\d.]+)(?:-([\d.]+))?", t)
            if not m:
                return None
            letter = m.group(1)
            lo = float(m.group(2))
            hi = float(m.group(3)) if m.group(3) else None
            title_up = (title or "").upper()
            direction = ">"
            if letter == "B" or "BELOW" in title_up or "<" in title:
                direction = "<"
            elif "BETWEEN" in title_up or hi is not None:
                direction = "between"
            return {
                "series": series,
                "city": _CITY_BY_SERIES[series],
                "direction": direction,
                "lo": lo,
                "hi": hi,
            }
    return None


def weather_fair_value(ticker: str, title: str,
                        nws: NWSClient | None = None) -> FairValue | None:
    parsed = parse_weather_ticker(ticker, title)
    if parsed is None:
        return None
    nws = nws or NWSClient()
    try:
        forecast = nws.city_forecast(parsed["city"])
    except Exception:
        return None
    forecast_high = forecast.today_high or forecast.tomorrow_high
    if forecast_high is None:
        return None
    mu, sigma = float(forecast_high), WEATHER_SIGMA_F
    if parsed["direction"] == ">":
        p = 1.0 - _norm_cdf(parsed["lo"], mu, sigma)
    elif parsed["direction"] == "<":
        p = _norm_cdf(parsed["lo"], mu, sigma)
    else:
        hi = parsed["hi"] or (parsed["lo"] + 1.0)
        p = _norm_cdf(hi, mu, sigma) - _norm_cdf(parsed["lo"], mu, sigma)
    p = max(0.02, min(0.98, p))
    detail = (f"NWS forecast high {mu:.0f}°F · "
               f"{parsed['direction']} {parsed['lo']}"
               f"{('-'+str(parsed['hi'])) if parsed['hi'] else ''}°F")
    return FairValue(prob_yes=p, confidence=0.85, source="NWS", detail=detail)


# ----- Sports ----------------------------------------------------------------

# Kalshi NFL tickers we've seen use variants like:
#   KXNFLGAME-25-KAN  (Kansas City to win today)
#   KXNFLWIN-25SEP-SFO-DAL  (SF vs DAL)
# We just look for any two three-letter team codes in the ticker.

_TEAM_CODES = {
    # NFL teams
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB",  "HOU", "IND", "JAX", "KAN", "LAC", "LAR", "LV",  "MIA",
    "MIN", "NE",  "NO",  "NYG", "NYJ", "PHI", "PIT", "SEA", "SF",  "TB",
    "TEN", "WAS", "SFO", "NOR", "KC",
    # NBA (common abbrevs; not exhaustive)
    "LAL", "BOS", "MIL", "MEM", "PHX", "GSW", "BKN", "NYK", "DAL",
}


def extract_team_codes(ticker: str) -> list[str]:
    codes = re.findall(r"[A-Z]{2,4}", ticker.upper())
    return [c for c in codes if c in _TEAM_CODES]


def sports_fair_value(ticker: str, title: str,
                       espn: ESPNClient | None = None) -> FairValue | None:
    teams = extract_team_codes(ticker)
    if not teams:
        return None
    espn = espn or ESPNClient()
    try:
        games = espn.all_games()
    except Exception:
        return None
    for g in games:
        comps = (g.get("competitions") or [{}])[0]
        competitors = comps.get("competitors") or []
        if len(competitors) < 2:
            continue
        abbreviations = [c.get("team", {}).get("abbreviation", "").upper()
                          for c in competitors]
        # Match any team in ticker against this game.
        match = [t for t in teams if t in abbreviations]
        if not match:
            continue
        # Pull implied probability from ESPN odds if present.
        odds = comps.get("odds")
        home_fav = None
        if odds and isinstance(odds, list):
            o = odds[0]
            details = o.get("details")
            spread = o.get("spread")
            if details and spread:
                home_idx = 0 if competitors[0].get("homeAway") == "home" else 1
                home_team = abbreviations[home_idx]
                # Convert spread to implied P(home win) via a light heuristic:
                # each 3 points of spread is ~12% probability.
                try:
                    s = float(spread)
                    p_home = max(0.05, min(0.95, 0.5 - s * 0.04))
                except (TypeError, ValueError):
                    p_home = 0.5
                home_fav = (home_team, p_home)
        # If the ticker references a specific team, align direction.
        target = match[0]
        if home_fav is not None:
            home_team, p_home = home_fav
            p = p_home if target == home_team else (1 - p_home)
        else:
            # Fall back to live win-probability if present, else 50/50.
            wp = comps.get("situation", {}).get("lastPlay", {}) \
                  .get("probability", {}).get("homeWinPercentage")
            if wp is not None:
                home_idx = 0 if competitors[0].get("homeAway") == "home" else 1
                home_team = abbreviations[home_idx]
                p = (float(wp) / 100.0) if target == home_team else \
                    (1.0 - float(wp) / 100.0)
            else:
                p = 0.5
        p = max(0.03, min(0.97, p))
        return FairValue(prob_yes=p, confidence=0.7, source="ESPN",
                         detail=f"ESPN odds {' vs '.join(abbreviations)}")
    return None


def fair_value(ticker: str, title: str,
                nws: NWSClient | None = None,
                espn: ESPNClient | None = None) -> FairValue | None:
    """Try each data source in turn; return the first that produces a value."""
    fv = weather_fair_value(ticker, title, nws=nws)
    if fv is not None:
        return fv
    fv = sports_fair_value(ticker, title, espn=espn)
    if fv is not None:
        return fv
    return None


def blend(ensemble_prob: float, fair: FairValue | None,
           base_weight: float = 0.35) -> float:
    """Blend the technical ensemble probability with an external fair value.
    Weight on the fair value scales with its confidence."""
    if fair is None:
        return ensemble_prob
    w = base_weight * max(0.0, min(1.0, fair.confidence))
    return max(0.01, min(0.99, (1 - w) * ensemble_prob + w * fair.prob_yes))
