"""Calibration metrics: Brier/log-loss extremes and ECE on a known calibration."""
import math

import numpy as np

from hermes.research.eval.calibration import brier, log_loss, reliability


def test_brier_perfect_is_zero():
    p = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([0, 1])
    assert brier(p, y) == 0.0


def test_brier_worst_is_two():
    p = np.array([[0.0, 1.0], [1.0, 0.0]])
    y = np.array([0, 1])
    assert math.isclose(brier(p, y), 2.0)          # each row: (0-1)^2 + (1-0)^2 = 2


def test_log_loss_confident_correct_is_small():
    p = np.array([[0.99, 0.01], [0.01, 0.99]])
    y = np.array([0, 1])
    assert log_loss(p, y) < 0.02


def test_reliability_ece_matches_overconfidence():
    # always predict class 0 at confidence 0.9 and always be right -> ECE = |1.0 - 0.9|
    p = np.array([[0.9, 0.1]] * 100)
    y = np.zeros(100, dtype=int)
    ece, rows = reliability(p, y, n_bins=5)
    assert math.isclose(ece, 0.1, abs_tol=1e-9)
    assert len(rows) == 1                           # all samples fall in one confidence bin
