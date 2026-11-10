"""Walk-forward evaluation of the football model against the market."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from ..vec import devig_power, logloss_multi
from .model import MAX_GOALS, GoalModel, RatingConfig, run_ratings

TUNE = range(2015, 2020)
HOLDOUT = range(2020, 2027)


def score_grid(lh: np.ndarray, la: np.ndarray, rho: float) -> np.ndarray:
    """(n, G, G) score probabilities with the Dixon-Coles correction."""
    g = np.arange(MAX_GOALS + 1)
    ph = poisson.pmf(g[None, :], lh[:, None])
    pa = poisson.pmf(g[None, :], la[:, None])
    M = ph[:, :, None] * pa[:, None, :]
    M[:, 0, 0] *= 1 - lh * la * rho
    M[:, 0, 1] *= 1 + lh * rho
    M[:, 1, 0] *= 1 + la * rho
    M[:, 1, 1] *= 1 - rho
    return M / M.sum(axis=(1, 2), keepdims=True)


def add_probs(df: pd.DataFrame, lh: np.ndarray, la: np.ndarray, rho: float) -> pd.DataFrame:
    M = score_grid(lh, la, rho)
    g = np.arange(MAX_GOALS + 1)
    diff, tot = g[:, None] - g[None, :], g[:, None] + g[None, :]
    out = df.copy()
    out["lh"], out["la"] = lh, la
    out["p_h"] = M[:, diff > 0].sum(axis=1)
    out["p_d"] = M[:, diff == 0].sum(axis=1)
    out["p_a"] = M[:, diff < 0].sum(axis=1)
    out["p_over"] = M[:, tot > 2.5].sum(axis=1)
    out["p_btts"] = M[:, 1:, 1:].sum(axis=(1, 2))
    return out


def walk_forward(matches: pd.DataFrame, cfg: RatingConfig | None = None,
                 first: int = 2015) -> pd.DataFrame:
    """Out-of-sample predictions: ratings are pre-match by construction; the stage-2
    calibration for season s is fitted on seasons before s."""
    rated, _ = run_ratings(matches, cfg)
    preds = []
    for s in sorted(x for x in rated["season"].unique() if x >= first):
        train = rated[rated["season"] < s]
        test = rated[(rated["season"] == s) & rated["played"]]
        if test.empty:
            continue
        gm = GoalModel().fit(train)
        lh, la = gm.rates(test)
        preds.append(add_probs(test, lh, la, gm.rho))
    return pd.concat(preds)


def metrics(p: pd.DataFrame, seasons=None, min_games: int = 6) -> dict:
    d = p[(p["n_h"] >= min_games) & (p["n_a"] >= min_games)]
    if seasons is not None:
        d = d[d["season"].isin(seasons)]
    d = d.dropna(subset=["c_h", "c_d", "c_a", "c_over", "c_under"])
    d = d[(d[["c_h", "c_d", "c_a", "c_over", "c_under"]] > 1.0).all(axis=1)]
    y = np.where(d["hg"] > d["ag"], 0, np.where(d["hg"] == d["ag"], 1, 2))
    pm = d[["p_h", "p_d", "p_a"]].to_numpy()
    mk = devig_power(d[["c_h", "c_d", "c_a"]].to_numpy())
    over = (d["hg"] + d["ag"] > 2.5).to_numpy()
    po = d["p_over"].to_numpy()
    mo = devig_power(d[["c_over", "c_under"]].to_numpy())[:, 0]
    ll2 = lambda q: float(-np.mean(np.where(over, np.log(q), np.log(1 - q))))  # noqa: E731
    return {"n": len(d), "ll_1x2": logloss_multi(pm, y), "ll_1x2_mkt": logloss_multi(mk, y),
            "ll_ou": ll2(po), "ll_ou_mkt": ll2(mo),
            "goals_bias": float((d["hg"] + d["ag"] - d["lh"] - d["la"]).mean())}
