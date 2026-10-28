"""Odds conversion, vig removal, expected value and Kelly staking."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def implied(odds: float) -> float:
    return 1.0 / odds


def fair_odds(p: float) -> float:
    return float("inf") if p <= 0 else 1.0 / p


def devig(odds: list[float], method: str = "power") -> list[float]:
    """No-vig probabilities from a full set of decimal odds for one market."""
    q = np.array([1.0 / o for o in odds])
    if method == "multiplicative" or len(q) < 2:
        return list(q / q.sum())
    if method == "power":
        # Find k such that sum(q_i ** k) == 1; shades favourite/longshot bias better.
        if abs(q.sum() - 1.0) < 1e-9:
            return list(q)
        f = lambda k: (q ** k).sum() - 1.0  # noqa: E731
        if f(0.2) * f(20.0) > 0:  # degenerate prices: fall back to normalising
            return list(q / q.sum())
        k = brentq(f, 0.2, 20.0)
        return list(q ** k)
    raise ValueError(f"unknown devig method {method}")


def expected_value(p_win: float, odds: float, p_push: float = 0.0) -> float:
    """Expected profit per unit staked."""
    p_lose = 1.0 - p_win - p_push
    return p_win * (odds - 1.0) - p_lose


def kelly(p_win: float, odds: float, p_push: float = 0.0) -> float:
    """Full-Kelly fraction of bankroll, accounting for a possible push (stake returned)."""
    b = odds - 1.0
    p_lose = 1.0 - p_win - p_push
    if b <= 0 or p_win * b <= p_lose:
        return 0.0
    return (p_win * b - p_lose) / (b * (p_win + p_lose))


def stake(p_win: float, odds: float, bankroll: float, fraction: float, cap: float,
          p_push: float = 0.0) -> float:
    f = min(kelly(p_win, odds, p_push) * fraction, cap)
    return round(bankroll * f, 2)


def american_to_decimal(odds: float) -> float:
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / -odds)
