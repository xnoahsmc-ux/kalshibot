import numpy as np
import pandas as pd

from kalshibot.calibration import Calibrator
from kalshibot.data import synthesize_history
from kalshibot.regime import regime_label, regime_score


def test_regime_label_range():
    h = synthesize_history(n=400, seed=7)
    labels = regime_label(h.mid, n=30)
    assert set(labels.unique()).issubset({-1, 0, 1})
    scores = regime_score(h.mid, n=30)
    assert scores.between(-1, 1).all()


def test_calibrator_monotone():
    # Perfectly calibrated toy example: p = y / n for bins.
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=500)
    y = (rng.uniform(0, 1, size=500) < p).astype(float)
    c = Calibrator().fit(p, y)
    out = c.transform(np.linspace(0, 1, 50))
    # Isotonic must be non-decreasing
    assert np.all(np.diff(out) >= -1e-9)


def test_calibrator_no_op_when_unfit():
    c = Calibrator()
    inp = np.array([0.1, 0.5, 0.9])
    assert np.allclose(c.transform(inp), inp)
