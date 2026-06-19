"""Probability-calibration metrics — ported/generalized from odds-pipeline.

Use these to check whether a signal's predicted probabilities are *trustworthy*
before sizing any position on them. A model that says "70% up" should be right
~70% of the time; if it is not, the edge is illusory. Most quant beginners skip
this step — don't.

  - log_loss / brier : proper scoring rules vs the realized class
  - reliability      : confidence-binned predicted-vs-empirical table + ECE

`p` is an (n, k) array of class probabilities (rows sum to 1); `y` is an (n,)
array of integer class indices. For binary up/down use k=2.
"""
from __future__ import annotations

import numpy as np


def log_loss(p, y) -> float:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y)
    idx = np.arange(len(y))
    return float(-np.mean(np.log(np.clip(p[idx, y], 1e-15, 1.0))))


def brier(p, y) -> float:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y)
    oh = np.zeros_like(p)
    oh[np.arange(len(y)), y] = 1.0
    return float(np.mean(((p - oh) ** 2).sum(axis=1)))


def reliability(p, y, n_bins: int = 10):
    """Bin by the predicted (argmax) confidence; compare predicted vs empirical.

    Returns (ece, rows) where each row is (lo, hi, n, predicted, empirical).
    A well-calibrated model has predicted ≈ empirical in every bin (ECE → 0).
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y)
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    edges = np.linspace(conf.min(), 1.0, n_bins + 1)
    ece, rows = 0.0, []
    for i in range(n_bins):
        last = i == n_bins - 1
        m = (conf >= edges[i]) & (conf <= edges[i + 1] if last else conf < edges[i + 1])
        if m.sum() == 0:
            continue
        pred, actual, w = conf[m].mean(), correct[m].mean(), m.mean()
        ece += w * abs(actual - pred)
        rows.append((float(edges[i]), float(edges[i + 1]), int(m.sum()), float(pred), float(actual)))
    return float(ece), rows
