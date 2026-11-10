"""Vectorised helpers shared by the backtests."""

from __future__ import annotations

import numpy as np


def devig_power(odds: np.ndarray) -> np.ndarray:
    """Row-wise no-vig probabilities (power method) for an (n, k) array of decimal odds."""
    q = 1.0 / np.asarray(odds, float)
    lo, hi = np.full(len(q), 0.5), np.full(len(q), 3.0)
    for _ in range(50):
        mid = (lo + hi) / 2
        s = (q ** mid[:, None]).sum(axis=1)
        hi = np.where(s < 1, mid, hi)
        lo = np.where(s >= 1, mid, lo)
    p = q ** ((lo + hi) / 2)[:, None]
    return p / p.sum(axis=1, keepdims=True)


def logloss_multi(p: np.ndarray, y: np.ndarray) -> float:
    """Mean log loss; p is (n, k), y the index of the outcome."""
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).mean())
