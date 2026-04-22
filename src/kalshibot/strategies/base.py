from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from ..data import MarketHistory


@dataclass(frozen=True)
class Signal:
    """A strategy's view of a market at a moment in time.

    `prob_yes` is the strategy's estimate that YES resolves true (in [0,1]).
    `confidence` weights the call (0=no opinion, 1=max conviction).
    """
    prob_yes: float
    confidence: float = 1.0

    @property
    def edge_vs(self) -> float:
        return self.prob_yes


class Strategy(Protocol):
    name: str

    def predict(self, h: MarketHistory) -> Signal: ...

    def predict_series(self, h: MarketHistory) -> pd.Series: ...
