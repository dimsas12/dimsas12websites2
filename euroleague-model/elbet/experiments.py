"""Walk-forward experiments for model changes.

Every variant is scored the same way as `backtest.walk_forward`: each season is
predicted by a model trained only on earlier seasons, and every feature uses only
games before tip-off. Variants are *selected* on the tune seasons and *reported*
on the holdout seasons, which the selection never sees.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd

from .config import Settings
from .features import MARGIN_FEATURES, TOTAL_FEATURES
from .models import GameModel
from .ratings import efficiency_ratings

TUNE = range(2015, 2023)
HOLDOUT = range(2023, 2027)


def _logloss(p, y) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def evaluate(feats: pd.DataFrame, margin_features=None, total_features=None) -> dict:
    """Walk-forward metrics on the tune and holdout seasons."""
    mf = list(margin_features or MARGIN_FEATURES)
    tf = list(total_features or TOTAL_FEATURES)
    f = feats[feats["has_ratings"] & feats["played"]]
    preds = []
    for s in sorted(set(TUNE) | set(HOLDOUT)):
        train, test = f[f["season"] < s], f[f["season"] == s]
        if len(train) < 200 or test.empty:
            continue
        m = GameModel(margin_features=mf, total_features=tf).fit(train)
        preds.append(test.join(m.predict(test)))
    p = pd.concat(preds)
    out = {}
    for name, seasons in (("tune", TUNE), ("holdout", HOLDOUT)):
        d = p[p["season"].isin(seasons)]
        y = d["home_win"].to_numpy()
        out[f"{name}_logloss"] = _logloss(d["p_home"].to_numpy(), y)
        out[f"{name}_acc"] = float(((d["p_home"] > 0.5) == (y == 1)).mean())
        out[f"{name}_mae_margin"] = float((d["margin"] - d["pred_margin"]).abs().mean())
        out[f"{name}_mae_total"] = float((d["total"] - d["pred_total"]).abs().mean())
        early = d[np.minimum(d["gp_home"], d["gp_away"]) < 8]
        out[f"{name}_early_logloss"] = _logloss(early["p_home"].to_numpy(),
                                                early["home_win"].to_numpy())
    return out


# ---------------------------------------------------------------- research items

def add_form_features(feats: pd.DataFrame, games: pd.DataFrame, settings: Settings,
                      half_life_days: float) -> pd.DataFrame:
    """Short-window ('last ~5 games') ratings next to the season-long ones:
    form_margin / form_total = short-window projection minus long-window projection."""
    cfg = replace(settings.ratings, half_life_days=half_life_days, window_days=120)
    short = efficiency_ratings(games.sort_values(["date", "game_code"]).reset_index(drop=True),
                               cfg).reset_index(drop=True)
    sm = (short["pts_home_exp"] - short["pts_away_exp"]).to_numpy()
    st = (short["pts_home_exp"] + short["pts_away_exp"]).to_numpy()
    out = feats.copy()
    out["form_margin"] = np.nan_to_num(sm - out["rating_margin"].to_numpy())
    out["form_total"] = np.nan_to_num(st - out["rating_total"].to_numpy())
    return out


def elo_538(games: pd.DataFrame, k: float = 20.0, hfa: float = 100.0, carry: float = 0.75,
            mean: float = 1505.0) -> pd.Series:
    """FiveThirtyEight NBA Elo: MOV mult = (MOV + 3)^0.8 / (7.5 + 0.006 * EloDiff)."""
    ratings: dict[str, float] = {}
    season_seen = None
    diff = np.empty(len(games))
    for i, g in enumerate(games.itertuples(index=False)):
        if season_seen is not None and g.season != season_seen:
            ratings = {t: mean + carry * (r - mean) for t, r in ratings.items()}
        season_seen = g.season
        rh, ra = ratings.setdefault(g.home, mean), ratings.setdefault(g.away, mean)
        h = 0.0 if g.neutral else hfa
        diff[i] = rh + h - ra
        if not g.played:
            continue
        margin = g.home_pts - g.away_pts
        exp_h = 1.0 / (1.0 + 10 ** (-(rh + h - ra) / 400.0))
        win_diff = (rh + h - ra) * (1 if margin > 0 else -1)
        mult = (abs(margin) + 3) ** 0.8 / (7.5 + 0.006 * win_diff)
        delta = k * mult * ((1.0 if margin > 0 else 0.0) - exp_h)
        ratings[g.home], ratings[g.away] = rh + delta, ra - delta
    return pd.Series(diff, index=games.index)


def logistic_vs_normal(feats: pd.DataFrame) -> dict:
    """Compare Phi(margin / sd) with the logistic 1 / (1 + e^(-margin / s)), s fitted."""
    f = feats[feats["has_ratings"] & feats["played"]]
    out = {}
    preds = []
    for s in HOLDOUT:
        train, test = f[f["season"] < s], f[f["season"] == s]
        m = GameModel().fit(train)
        tr = train.join(m.predict(train))
        # fit logistic scale on training predictions (1-D grid search on log loss)
        grid = np.linspace(4, 10, 61)
        y = tr["home_win"].to_numpy()
        best = min(grid, key=lambda sc: _logloss(1 / (1 + np.exp(-tr["pred_margin"] / sc)), y))
        te = test.join(m.predict(test))
        te["p_logit"] = 1 / (1 + np.exp(-te["pred_margin"] / best))
        te["scale"] = best
        preds.append(te)
    p = pd.concat(preds)
    y = p["home_win"].to_numpy()
    out["normal"] = _logloss(p["p_home"].to_numpy(), y)
    out["logistic"] = _logloss(p["p_logit"].to_numpy(), y)
    out["logistic_scale"] = float(p["scale"].mean())
    return out


def nan_safe(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x
