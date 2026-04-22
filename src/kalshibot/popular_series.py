"""Known-popular Kalshi series to seed the live feed with.

Kalshi's default `/markets?status=open` pagination buries real markets under
hundreds of auto-generated multi-variant baskets. Targeting specific popular
series with `?series_ticker=X` guarantees we get markets with actual trading
activity.

Series here were chosen because they:
  * trade every day (weather, crypto) or every week (economy, sports)
  * have tight bid/ask spreads and reliable last_price
  * cover every genre in our classifier

Feel free to add more — the fetch is tolerant of unknown tickers.
"""
from __future__ import annotations

POPULAR_SERIES: list[str] = [
    # Weather (daily resolution, always tradeable)
    "KXHIGHNY",        # NYC high temp
    "KXHIGHCHI",       # Chicago high temp
    "KXHIGHLAX",       # LAX high temp
    "KXHIGHMIA",       # Miami high temp
    "KXHIGHAUS",       # Austin high temp
    "KXHIGHDEN",       # Denver high temp
    "KXHIGHDC",        # DC high temp
    # Crypto (24/7, deep liquidity)
    "KXBTCD",          # BTC end-of-day price
    "KXBTC",
    "KXETHD",
    "KXETH",
    "KXBTCMAX",
    # Economy (Fed, CPI, jobs)
    "KXFEDDECISION",
    "KXCPIYOY",
    "KXJOBS",
    "KXGDPNOW",
    # Equities / markets
    "KXSPDAY",
    "KXNDQDAY",
    # Sports
    "KXNFLGAME",
    "KXNBAGAME",
    "KXMLBGAME",
    "KXNHLGAME",
    # Entertainment
    "KXMOVIEBOX",
    "KXOSCARS",
    # Science / tech
    "KXAICHAT",
    "KXSPACEX",
    # World / oil
    "KXOPEC",
    "KXOILWTI",
]
