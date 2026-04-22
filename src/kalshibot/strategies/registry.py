"""Library of strategies. Each emits a per-bar probability series for YES.

Strategies are intentionally small and parameterized so that hundreds of
distinct instances arise from a handful of families. The combiner blends them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .. import features as F
from ..data import MarketHistory
from .base import Signal


# A single concrete strategy is just a function (history) -> probability series,
# wrapped with a name. Wrapping it lets us add metadata later without a refactor.
PredictFn = Callable[[MarketHistory], pd.Series]


@dataclass
class Strat:
    name: str
    fn: PredictFn

    def predict_series(self, h: MarketHistory) -> pd.Series:
        s = self.fn(h).astype(float)
        return s.clip(0.01, 0.99).fillna(0.5)

    def predict(self, h: MarketHistory) -> Signal:
        s = self.predict_series(h)
        if len(s) == 0:
            return Signal(0.5, 0.0)
        p = float(s.iloc[-1])
        # Confidence: how far the call is from 0.5 and how stable it has been.
        recent = s.iloc[-min(20, len(s)):]
        stability = 1.0 - float(recent.std())
        conf = max(0.0, min(1.0, abs(p - 0.5) * 2 * max(0.1, stability)))
        return Signal(p, conf)


# ---------------------------------------------------------------------------
# Strategy families. Each function returns a probability-of-YES series.
# ---------------------------------------------------------------------------

def _mid(h: MarketHistory) -> pd.Series:
    return h.mid


def sma_crossover(fast: int, slow: int) -> PredictFn:
    def fn(h: MarketHistory) -> pd.Series:
        m = _mid(h)
        f = F.sma(m, fast)
        s = F.sma(m, slow)
        # Trend-following: lean toward whichever side the fast MA has crossed
        diff = (f - s) / s.replace(0, np.nan)
        return F.sigmoid(diff * 8)
    return fn


def ema_crossover(fast: int, slow: int) -> PredictFn:
    def fn(h: MarketHistory) -> pd.Series:
        m = _mid(h)
        diff = F.ema(m, fast) - F.ema(m, slow)
        return F.sigmoid(diff * 12)
    return fn


def rsi_strategy(n: int, lo: float, hi: float, contrarian: bool = True) -> PredictFn:
    def fn(h: MarketHistory) -> pd.Series:
        r = F.rsi(_mid(h), n)
        # Contrarian: fade extremes. Otherwise: ride momentum.
        if contrarian:
            score = (50 - r) / 50.0  # +1 at oversold -> bullish for YES
        else:
            score = (r - 50) / 50.0
        # Damp inside [lo,hi] dead-zone
        mask = (r > lo) & (r < hi)
        score = score.where(~mask, 0.0)
        return F.sigmoid(score * 2.0)
    return fn


def bollinger_strategy(n: int, k: float, contrarian: bool = True) -> PredictFn:
    def fn(h: MarketHistory) -> pd.Series:
        m = _mid(h)
        mid, hi, lo = F.bollinger(m, n, k)
        width = (hi - lo).replace(0, np.nan)
        z = (m - mid) / width
        return F.sigmoid((-z if contrarian else z) * 4)
    return fn


def donchian_breakout(n: int) -> PredictFn:
    def fn(h: MarketHistory) -> pd.Series:
        m = _mid(h)
        hi, lo = F.donchian(m, n)
        score = ((m - lo) / (hi - lo).replace(0, np.nan) - 0.5) * 2
        return F.sigmoid(score.fillna(0) * 3)
    return fn


def macd_strategy(fast: int, slow: int, sig: int) -> PredictFn:
    def fn(h: MarketHistory) -> pd.Series:
        line, signal = F.macd(_mid(h), fast, slow, sig)
        return F.sigmoid((line - signal) * 20)
    return fn


def zscore_meanrev(n: int, scale: float = 1.0) -> PredictFn:
    def fn(h: MarketHistory) -> pd.Series:
        z = F.zscore(_mid(h), n)
        return F.sigmoid(-z * scale)
    return fn


def momentum_strategy(n: int, scale: float = 8.0) -> PredictFn:
    def fn(h: MarketHistory) -> pd.Series:
        return F.sigmoid(F.momentum(_mid(h), n) * scale)
    return fn


def vol_target(n: int, target: float) -> PredictFn:
    def fn(h: MarketHistory) -> pd.Series:
        v = F.realized_vol(_mid(h), n)
        # In high vol, lean toward 0.5 (no opinion); in low vol, ride trend
        trend = F.ema(_mid(h), n) - F.ema(_mid(h), n * 3)
        damp = (target / (v + 1e-4)).clip(0, 4)
        return F.sigmoid(trend * damp * 6)
    return fn


def orderbook_imbalance() -> PredictFn:
    def fn(h: MarketHistory) -> pd.Series:
        # Approximate via the bid/ask position vs. last traded price.
        last = h.df["last"]
        mid = h.mid
        return F.sigmoid((last - mid) * 8)
    return fn


def spread_aware_fade(threshold: float) -> PredictFn:
    def fn(h: MarketHistory) -> pd.Series:
        sp = h.spread
        m = h.mid
        # When spread widens past threshold, mean-revert to 0.5 (uncertainty);
        # otherwise lean toward last trade.
        score = (h.df["last"] - m) * (1 - (sp / threshold).clip(0, 1))
        return F.sigmoid(score * 10)
    return fn


def time_decay_anchor(half_life_min: float) -> PredictFn:
    def fn(h: MarketHistory) -> pd.Series:
        m = h.mid
        ttc = h.df["minutes_to_close"].clip(lower=0)
        # As resolution approaches, anchor to current price (less noise reverts);
        # far from resolution, regress toward 0.5.
        weight = 0.5 ** (ttc / max(1.0, half_life_min))
        anchored = m * weight + 0.5 * (1 - weight)
        return anchored
    return fn


def volume_weighted_drift(n: int) -> PredictFn:
    def fn(h: MarketHistory) -> pd.Series:
        m = h.mid
        v = h.df["volume"].replace(0, 1.0)
        vwap = (m * v).rolling(n, min_periods=2).sum() / v.rolling(n, min_periods=2).sum()
        return F.sigmoid((m - vwap) * 6)
    return fn


def overround_correction(scale: float = 6.0) -> PredictFn:
    def fn(h: MarketHistory) -> pd.Series:
        # Kalshi YES+NO mid usually sums close to 1; deviation hints at edge.
        bid = h.df["yes_bid"]
        ask = h.df["yes_ask"]
        # NO-side implied = 1 - yes mid. Edge = (true mid is closer to last trade).
        edge = h.df["last"] - 0.5 * (bid + ask)
        return F.sigmoid(edge * scale)
    return fn


def bayesian_trade_drift(n: int, prior_strength: float = 5.0) -> PredictFn:
    def fn(h: MarketHistory) -> pd.Series:
        # Treat each minute's last price as a noisy observation; do a rolling
        # beta-binomial style update around the mid.
        m = h.mid
        last = h.df["last"]
        evidence = (last - m).rolling(n, min_periods=2).mean().fillna(0)
        return F.sigmoid(evidence * prior_strength)
    return fn


def kelly_contrarian(n: int, k: float = 1.5) -> PredictFn:
    def fn(h: MarketHistory) -> pd.Series:
        m = h.mid
        z = F.zscore(m, n)
        # Bigger fade when z is extreme.
        return F.sigmoid(-z * k)
    return fn


def trend_with_vol_filter(fast: int, slow: int, vol_n: int) -> PredictFn:
    def fn(h: MarketHistory) -> pd.Series:
        m = h.mid
        diff = F.ema(m, fast) - F.ema(m, slow)
        v = F.realized_vol(m, vol_n)
        gate = (1 - (v * 50).clip(0, 1))
        return F.sigmoid(diff * 14 * gate)
    return fn


def regression_to_long_term(n: int) -> PredictFn:
    def fn(h: MarketHistory) -> pd.Series:
        m = h.mid
        long = F.sma(m, n)
        return F.sigmoid((long - m) * 6)
    return fn


def keltner_channel(n: int, k: float) -> PredictFn:
    def fn(h: MarketHistory) -> pd.Series:
        m = h.mid
        atr = F.realized_vol(m, n).rolling(n, min_periods=2).mean()
        center = F.ema(m, n)
        upper = center + k * atr
        lower = center - k * atr
        score = (m - center) / (upper - lower).replace(0, np.nan)
        return F.sigmoid(score.fillna(0) * 6)
    return fn


# ---------------------------------------------------------------------------
# Build the catalog. Aim for ~100 distinct strategies.
# ---------------------------------------------------------------------------

def _build() -> list[Strat]:
    out: list[Strat] = []

    # SMA crossovers (12)
    for fast, slow in [(3, 10), (5, 20), (5, 50), (8, 21), (10, 30), (10, 50),
                       (12, 26), (15, 60), (20, 50), (20, 100), (30, 90), (50, 200)]:
        out.append(Strat(f"sma_x_{fast}_{slow}", sma_crossover(fast, slow)))

    # EMA crossovers (12)
    for fast, slow in [(3, 10), (5, 20), (8, 21), (9, 26), (10, 30), (12, 26),
                       (12, 48), (15, 60), (20, 50), (20, 80), (30, 90), (50, 200)]:
        out.append(Strat(f"ema_x_{fast}_{slow}", ema_crossover(fast, slow)))

    # RSI variants - both contrarian and momentum (10)
    for n in [7, 14, 21, 28]:
        out.append(Strat(f"rsi_fade_{n}", rsi_strategy(n, 40, 60, contrarian=True)))
        out.append(Strat(f"rsi_ride_{n}", rsi_strategy(n, 40, 60, contrarian=False)))
    out.append(Strat("rsi_fade_aggressive_14", rsi_strategy(14, 30, 70, True)))
    out.append(Strat("rsi_ride_aggressive_14", rsi_strategy(14, 30, 70, False)))

    # Bollinger (8)
    for n in [10, 20, 30, 50]:
        out.append(Strat(f"bb_fade_{n}_2", bollinger_strategy(n, 2.0, True)))
        out.append(Strat(f"bb_ride_{n}_2", bollinger_strategy(n, 2.0, False)))

    # Donchian breakouts (6)
    for n in [10, 20, 30, 50, 100, 200]:
        out.append(Strat(f"donchian_{n}", donchian_breakout(n)))

    # MACD (6)
    for fast, slow, sig in [(8, 21, 5), (12, 26, 9), (5, 35, 5),
                            (10, 40, 9), (20, 60, 9), (3, 10, 16)]:
        out.append(Strat(f"macd_{fast}_{slow}_{sig}", macd_strategy(fast, slow, sig)))

    # Z-score mean reversion (8)
    for n in [10, 20, 30, 50]:
        out.append(Strat(f"z_mr_{n}_1", zscore_meanrev(n, 1.0)))
        out.append(Strat(f"z_mr_{n}_2", zscore_meanrev(n, 2.0)))

    # Momentum (8)
    for n in [3, 5, 10, 20, 30, 60, 90, 120]:
        out.append(Strat(f"mom_{n}", momentum_strategy(n)))

    # Volatility-targeted trend (6)
    for n, t in [(20, 0.01), (20, 0.02), (50, 0.01), (50, 0.02), (100, 0.01), (100, 0.02)]:
        out.append(Strat(f"voltgt_{n}_{t}", vol_target(n, t)))

    # Microstructure (6)
    out.append(Strat("ob_imbalance", orderbook_imbalance()))
    for thr in [0.005, 0.01, 0.02, 0.04]:
        out.append(Strat(f"spread_fade_{int(thr*1000)}", spread_aware_fade(thr)))
    out.append(Strat("overround_corr", overround_correction()))

    # Time-decay anchors (4)
    for hl in [10, 30, 120, 720]:
        out.append(Strat(f"time_anchor_{hl}", time_decay_anchor(hl)))

    # Volume drift (4)
    for n in [10, 30, 60, 120]:
        out.append(Strat(f"vwap_drift_{n}", volume_weighted_drift(n)))

    # Bayesian drift (4)
    for n in [10, 30, 60, 120]:
        out.append(Strat(f"bayes_drift_{n}", bayesian_trade_drift(n)))

    # Kelly contrarian (3)
    for n in [20, 50, 100]:
        out.append(Strat(f"kelly_fade_{n}", kelly_contrarian(n)))

    # Trend with vol filter (6)
    for fast, slow, vn in [(5, 20, 30), (8, 21, 30), (10, 30, 60),
                           (12, 26, 60), (20, 50, 100), (50, 200, 100)]:
        out.append(Strat(f"trend_volf_{fast}_{slow}_{vn}", trend_with_vol_filter(fast, slow, vn)))

    # Regression to long mean (4)
    for n in [60, 120, 240, 480]:
        out.append(Strat(f"longmean_{n}", regression_to_long_term(n)))

    # Keltner (4)
    for n, k in [(20, 1.5), (20, 2.0), (50, 1.5), (50, 2.5)]:
        out.append(Strat(f"keltner_{n}_{k}", keltner_channel(n, k)))

    return out


_CATALOG: list[Strat] = _build()


def all_strategies() -> list[Strat]:
    return list(_CATALOG)


def by_name(name: str) -> Strat:
    for s in _CATALOG:
        if s.name == name:
            return s
    raise KeyError(name)
