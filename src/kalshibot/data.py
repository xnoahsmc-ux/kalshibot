from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class MarketSnapshot:
    """Point-in-time view of a Kalshi binary market.

    Prices live on [0,1] - we convert from Kalshi's 1-99 cent quotes upstream.
    """
    ticker: str
    ts: pd.Timestamp
    yes_bid: float
    yes_ask: float
    last: float
    volume: float = 0.0
    open_interest: float = 0.0
    minutes_to_close: float = float("inf")

    @property
    def mid(self) -> float:
        return 0.5 * (self.yes_bid + self.yes_ask)

    @property
    def spread(self) -> float:
        return max(0.0, self.yes_ask - self.yes_bid)


@dataclass
class MarketHistory:
    """Time-indexed history for a single binary market.

    `df` columns: yes_bid, yes_ask, last, volume, open_interest, minutes_to_close.
    All prices are normalized to [0,1].
    """
    ticker: str
    df: pd.DataFrame = field(default_factory=pd.DataFrame)

    def append(self, snap: MarketSnapshot) -> None:
        row = {
            "yes_bid": snap.yes_bid,
            "yes_ask": snap.yes_ask,
            "last": snap.last,
            "volume": snap.volume,
            "open_interest": snap.open_interest,
            "minutes_to_close": snap.minutes_to_close,
        }
        if self.df.empty:
            # `.loc[idx] = dict` fails on a frame with no columns.
            self.df = pd.DataFrame([row], index=pd.DatetimeIndex([snap.ts]))
        else:
            self.df.loc[snap.ts] = row

    @property
    def mid(self) -> pd.Series:
        return 0.5 * (self.df["yes_bid"] + self.df["yes_ask"])

    @property
    def spread(self) -> pd.Series:
        return (self.df["yes_ask"] - self.df["yes_bid"]).clip(lower=0)


def synthesize_history(
    ticker: str = "DEMO",
    n: int = 600,
    seed: int = 0,
    drift: float = 0.0,
    vol: float = 0.02,
    spread: float = 0.01,
    start_p: float = 0.5,
) -> MarketHistory:
    """Generate a plausible synthetic market for tests/backtests."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(drift, vol, size=n)
    # Random-walk in logit space so prices stay in (0,1)
    logit = np.log(start_p / (1 - start_p)) + np.cumsum(shocks)
    p = 1.0 / (1.0 + np.exp(-logit))
    ts = pd.date_range("2025-01-01", periods=n, freq="1min")
    df = pd.DataFrame(
        {
            "yes_bid": np.clip(p - spread / 2, 0.01, 0.99),
            "yes_ask": np.clip(p + spread / 2, 0.01, 0.99),
            "last": p,
            "volume": rng.integers(0, 200, size=n).astype(float),
            "open_interest": rng.integers(100, 5000, size=n).astype(float),
            "minutes_to_close": np.linspace(n, 1, n),
        },
        index=ts,
    )
    return MarketHistory(ticker=ticker, df=df)


def histories_from_snapshots(snaps: Iterable[MarketSnapshot]) -> dict[str, MarketHistory]:
    out: dict[str, MarketHistory] = {}
    for s in snaps:
        h = out.setdefault(s.ticker, MarketHistory(ticker=s.ticker))
        h.append(s)
    return out
