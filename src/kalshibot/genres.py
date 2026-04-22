"""Classify Kalshi markets into genres (niches).

Kalshi tickers and series names embed strong category hints (e.g. NFL*, KXWEATHER*,
KXPOL*, KXBTC*). We map by token rules with a fallback to "other".
"""
from __future__ import annotations

import re
from dataclasses import dataclass


GENRE_RULES: list[tuple[str, list[str]]] = [
    ("sports",        ["NFL", "NBA", "MLB", "NHL", "UFC", "PGA", "MLS", "NCAA",
                       "SOCCER", "TENNIS", "BOX", "GOLF", "FIGHT", "GAME", "MATCH"]),
    ("weather",       ["WEATHER", "TEMP", "HIGH", "LOW", "RAIN", "SNOW", "HURR",
                       "STORM", "WIND", "CLIMATE", "HOT", "COLD"]),
    ("politics",      ["POL", "ELEC", "PRES", "SENATE", "HOUSE", "GOV", "TRUMP",
                       "BIDEN", "VOTE", "BALLOT", "CONGRESS", "PRIMARY"]),
    ("economy",       ["CPI", "GDP", "FED", "RATE", "JOBS", "JOBLESS", "INFL",
                       "UNRATE", "PPI", "PAYROLL", "ECON", "RECESS"]),
    ("crypto",        ["BTC", "ETH", "CRYPTO", "DOGE", "SOL", "BITCOIN", "ETHER"]),
    ("equities",      ["SPX", "NDX", "DJI", "STOCK", "MARKET", "RUSSELL", "VIX"]),
    ("entertainment", ["OSCAR", "EMMY", "GRAMMY", "BOXOFF", "MOVIE", "MUSIC",
                       "TV", "STREAM", "BILLBOARD", "ALBUM"]),
    ("science",       ["SPACE", "NASA", "LAUNCH", "ROCKET", "AI", "CHATGPT",
                       "OPENAI", "GPT", "TECH", "QUANTUM"]),
    ("world",         ["WORLD", "WAR", "UKR", "ISR", "GAZA", "CHINA", "RUSSIA",
                       "IRAN", "OPEC", "OIL", "GAS"]),
    ("culture",       ["CELEB", "TWITTER", "X_", "MUSK", "TAYLOR", "KANYE"]),
]


@dataclass(frozen=True)
class GenreInfo:
    genre: str
    matched_token: str | None


def classify(ticker: str, title: str | None = None) -> GenreInfo:
    """Classify a market into one of the known genres.

    `ticker` is the Kalshi market or series ticker; `title` (optional) lets us
    catch human-readable hints when the ticker is opaque.
    """
    needle = (ticker or "").upper()
    if title:
        needle += " " + title.upper()
    needle = re.sub(r"[^A-Z0-9 _]", " ", needle)
    # Tickers like "KXNFL-..." or "KX_NFL" should match "NFL", so we look for
    # each token as a substring rather than requiring a strict word boundary.
    for genre, tokens in GENRE_RULES:
        for tok in tokens:
            if tok in needle:
                return GenreInfo(genre, tok)
    return GenreInfo("other", None)


def genres_in(tickers: list[str]) -> dict[str, str]:
    return {t: classify(t).genre for t in tickers}
