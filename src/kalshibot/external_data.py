"""External data sources for computing fair-value probabilities.

- NWS (api.weather.gov): US government weather forecasts. No API key required.
  Used for KXHIGHNY / KXHIGHLAX / KXRAINNY style markets.
- ESPN (site.api.espn.com): free scoreboard + odds for major sports.
  Used for KXNFLGAME / KXNBAGAME style markets.

All calls are best-effort with short timeouts so a slow external API never
blocks the live feed loop.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import requests


# -- US cities with associated Kalshi weather series --------------------------
CITIES: list[dict[str, Any]] = [
    {"code": "NYC", "name": "New York",    "lat": 40.7794, "lon": -73.9691,
     "kalshi": "KXHIGHNY"},
    {"code": "LAX", "name": "Los Angeles", "lat": 33.9381, "lon": -118.3889,
     "kalshi": "KXHIGHLAX"},
    {"code": "CHI", "name": "Chicago",     "lat": 41.9742, "lon": -87.9073,
     "kalshi": "KXHIGHCHI"},
    {"code": "MIA", "name": "Miami",       "lat": 25.7932, "lon": -80.2906,
     "kalshi": "KXHIGHMIA"},
    {"code": "AUS", "name": "Austin",      "lat": 30.1975, "lon": -97.6664,
     "kalshi": "KXHIGHAUS"},
    {"code": "DEN", "name": "Denver",      "lat": 39.8561, "lon": -104.6737,
     "kalshi": "KXHIGHDEN"},
    {"code": "DCA", "name": "Washington",  "lat": 38.8512, "lon": -77.0402,
     "kalshi": "KXHIGHDC"},
    {"code": "PHL", "name": "Philadelphia","lat": 39.8719, "lon": -75.2411,
     "kalshi": "KXHIGHPHL"},
    {"code": "SEA", "name": "Seattle",     "lat": 47.4502, "lon": -122.3088,
     "kalshi": "KXHIGHSEA"},
    {"code": "BOS", "name": "Boston",      "lat": 42.3643, "lon": -71.0052,
     "kalshi": "KXHIGHBOS"},
    {"code": "SFO", "name": "San Francisco","lat": 37.6213, "lon": -122.3790,
     "kalshi": "KXHIGHSF"},
    {"code": "ATL", "name": "Atlanta",     "lat": 33.6407, "lon": -84.4277,
     "kalshi": "KXHIGHATL"},
]


@dataclass
class CityForecast:
    code: str
    name: str
    lat: float
    lon: float
    today_high: float | None = None
    today_low: float | None = None
    today_summary: str = ""
    tomorrow_high: float | None = None
    tomorrow_low: float | None = None
    updated_at: float = 0.0


# Simple in-memory cache so we don't hammer NWS every live tick.
_NWS_CACHE: dict[str, tuple[float, dict]] = {}
_NWS_LOCK = threading.Lock()
_NWS_TTL = 900  # 15 minutes


class NWSClient:
    BASE = "https://api.weather.gov"
    UA = "kalshibot/0.1 (https://github.com/xnoahsmc-ux/kalshibot)"

    def __init__(self, session: requests.Session | None = None) -> None:
        self.s = session or requests.Session()

    def _get(self, url: str) -> dict:
        # NWS strongly recommends a descriptive User-Agent.
        r = self.s.get(url, headers={"User-Agent": self.UA, "Accept": "application/geo+json"},
                        timeout=8)
        r.raise_for_status()
        return r.json()

    def points(self, lat: float, lon: float) -> dict:
        key = f"points/{lat:.3f},{lon:.3f}"
        with _NWS_LOCK:
            hit = _NWS_CACHE.get(key)
            if hit and time.time() - hit[0] < 3600 * 24:
                return hit[1]
        data = self._get(f"{self.BASE}/points/{lat:.4f},{lon:.4f}")
        with _NWS_LOCK:
            _NWS_CACHE[key] = (time.time(), data)
        return data

    def forecast(self, lat: float, lon: float) -> dict:
        key = f"forecast/{lat:.3f},{lon:.3f}"
        with _NWS_LOCK:
            hit = _NWS_CACHE.get(key)
            if hit and time.time() - hit[0] < _NWS_TTL:
                return hit[1]
        pts = self.points(lat, lon)
        url = pts.get("properties", {}).get("forecast")
        if not url:
            return {}
        data = self._get(url)
        with _NWS_LOCK:
            _NWS_CACHE[key] = (time.time(), data)
        return data

    def city_forecast(self, city: dict) -> CityForecast:
        cf = CityForecast(code=city["code"], name=city["name"],
                           lat=city["lat"], lon=city["lon"])
        try:
            payload = self.forecast(city["lat"], city["lon"])
            periods = payload.get("properties", {}).get("periods", [])
            # NWS alternates day/night periods; isDaytime tells us which.
            days = [p for p in periods if p.get("isDaytime")]
            nights = [p for p in periods if not p.get("isDaytime")]
            if days:
                cf.today_high = float(days[0].get("temperature") or 0)
                cf.today_summary = days[0].get("shortForecast", "")
            if nights:
                cf.today_low = float(nights[0].get("temperature") or 0)
            if len(days) > 1:
                cf.tomorrow_high = float(days[1].get("temperature") or 0)
            if len(nights) > 1:
                cf.tomorrow_low = float(nights[1].get("temperature") or 0)
            cf.updated_at = time.time()
        except Exception:
            pass
        return cf

    def forecast_all(self) -> list[CityForecast]:
        return [self.city_forecast(c) for c in CITIES]


# -- ESPN (no auth) -----------------------------------------------------------

_ESPN_CACHE: dict[str, tuple[float, dict]] = {}
_ESPN_TTL = 120  # 2 minutes


class ESPNClient:
    BASE = "https://site.api.espn.com/apis/site/v2/sports"

    def __init__(self, session: requests.Session | None = None) -> None:
        self.s = session or requests.Session()

    def _get(self, path: str) -> dict:
        with _NWS_LOCK:
            hit = _ESPN_CACHE.get(path)
            if hit and time.time() - hit[0] < _ESPN_TTL:
                return hit[1]
        try:
            r = self.s.get(f"{self.BASE}/{path}", timeout=8)
            r.raise_for_status()
            data = r.json()
        except Exception:
            data = {}
        with _NWS_LOCK:
            _ESPN_CACHE[path] = (time.time(), data)
        return data

    def nfl_scoreboard(self) -> list[dict]:
        return self._get("football/nfl/scoreboard").get("events", [])

    def nba_scoreboard(self) -> list[dict]:
        return self._get("basketball/nba/scoreboard").get("events", [])

    def mlb_scoreboard(self) -> list[dict]:
        return self._get("baseball/mlb/scoreboard").get("events", [])

    def all_games(self) -> list[dict]:
        out = []
        for game in self.nfl_scoreboard():
            game["league"] = "NFL"; out.append(game)
        for game in self.nba_scoreboard():
            game["league"] = "NBA"; out.append(game)
        for game in self.mlb_scoreboard():
            game["league"] = "MLB"; out.append(game)
        return out
