"""Probability calibration.

Raw strategy probabilities tend to be overconfident near the tails. We fit
isotonic regression on out-of-sample (p, y) pairs so the calibrated
probability equals the empirical frequency of YES at that signal level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


class Calibrator:
    def __init__(self) -> None:
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
        self.fitted = False

    def fit(self, probs: np.ndarray | pd.Series, outcomes: np.ndarray | pd.Series) -> "Calibrator":
        p = np.asarray(probs, dtype=float)
        y = np.asarray(outcomes, dtype=float)
        mask = np.isfinite(p) & np.isfinite(y)
        if mask.sum() < 10:
            return self
        self.iso.fit(p[mask], y[mask])
        self.fitted = True
        return self

    def transform(self, probs: np.ndarray | pd.Series) -> np.ndarray:
        p = np.asarray(probs, dtype=float)
        if not self.fitted:
            return p
        return self.iso.predict(np.clip(p, 0.001, 0.999))

    def reliability(self, probs: np.ndarray, outcomes: np.ndarray,
                    bins: int = 10) -> pd.DataFrame:
        p = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
        y = np.asarray(outcomes, dtype=float)
        edges = np.linspace(0, 1, bins + 1)
        idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
        rows = []
        for b in range(bins):
            sel = idx == b
            if sel.sum() == 0:
                continue
            rows.append({
                "bin": b,
                "p_mean": float(p[sel].mean()),
                "y_mean": float(y[sel].mean()),
                "n": int(sel.sum()),
            })
        return pd.DataFrame(rows)
