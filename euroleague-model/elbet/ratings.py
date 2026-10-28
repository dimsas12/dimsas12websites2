"""Pre-game team strength ratings.

Every rating attached to a game uses only games played strictly before it, so the
same code produces leakage-free features for backtesting and ratings for upcoming
games.

* Elo with a margin-of-victory multiplier.
* Opponent-adjusted offensive/defensive efficiency (points per 100 possessions)
  and pace, fitted by recency-weighted ridge regression on recent games.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import EloConfig, RatingConfig


def _elo_delta(rh: float, ra: float, margin: float, neutral: bool, cfg: EloConfig) -> float:
    hfa = 0.0 if neutral else cfg.hfa
    exp_h = 1.0 / (1.0 + 10 ** (-(rh + hfa - ra) / 400.0))
    winner_diff = (rh + hfa - ra) * (1 if margin > 0 else -1)
    mov = math.log(abs(margin) + 1.0) * 2.2 / (winner_diff * 0.001 + 2.2)
    return cfg.k * mov * ((1.0 if margin > 0 else 0.0) - exp_h)


def run_elo(games: pd.DataFrame, cfg: EloConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    """Pre-game Elo for each game (rows in date order) and the ratings after the last game."""
    ratings: dict[str, float] = {}
    season_seen = None
    out_h = np.empty(len(games))
    out_a = np.empty(len(games))
    for i, g in enumerate(games.itertuples(index=False)):
        if season_seen is not None and g.season != season_seen:
            ratings = {t: cfg.init + cfg.carryover * (r - cfg.init) for t, r in ratings.items()}
        season_seen = g.season
        rh = ratings.setdefault(g.home, cfg.new_team)
        ra = ratings.setdefault(g.away, cfg.new_team)
        out_h[i], out_a[i] = rh, ra
        if g.played:
            delta = _elo_delta(rh, ra, g.home_pts - g.away_pts, g.neutral, cfg)
            ratings[g.home] = rh + delta
            ratings[g.away] = ra - delta
    return pd.DataFrame({"elo_home": out_h, "elo_away": out_a}, index=games.index), ratings


def _solve_efficiency(hist: pd.DataFrame, weights: np.ndarray, cfg: RatingConfig):
    """Weighted ridge fit of points/100 and pace on a window of past games.

    Points/100 model, two rows per game:
        home_eff = mu + hca/2 + off[home] - def[away]
        away_eff = mu - hca/2 + off[away] - def[home]
    Pace model, one row per game:
        pace = mu_p + p[home] + p[away]
    """
    teams = pd.unique(np.concatenate([hist["home"].values, hist["away"].values]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    h = hist["home"].map(idx).to_numpy()
    a = hist["away"].map(idx).to_numpy()
    side = np.where(hist["neutral"].to_numpy(), 0.0, 0.5)
    poss = hist["poss"].to_numpy(dtype=float)
    poss = np.where(np.isfinite(poss) & (poss > 40), poss, cfg.default_pace)
    eff_h = 100.0 * hist["home_pts"].to_numpy(dtype=float) / poss
    eff_a = 100.0 * hist["away_pts"].to_numpy(dtype=float) / poss
    has_pace = np.isfinite(hist["poss"].to_numpy(dtype=float))

    # Parameters: [mu, hca, off_0..off_n-1, def_0..def_n-1]
    m = 2 + 2 * n
    rows = len(hist)
    X = np.zeros((2 * rows, m))
    r = np.arange(rows)
    X[r, 0] = 1.0
    X[r, 1] = side
    X[r, 2 + h] = 1.0
    X[r, 2 + n + a] = -1.0
    X[rows + r, 0] = 1.0
    X[rows + r, 1] = -side
    X[rows + r, 2 + a] = 1.0
    X[rows + r, 2 + n + h] = -1.0
    y = np.concatenate([eff_h, eff_a])
    w = np.concatenate([weights, weights])
    pen = np.full(m, cfg.ridge_team)
    pen[0], pen[1] = 1e-6, 0.5
    A = X.T @ (X * w[:, None]) + np.diag(pen)
    b = X.T @ (w * y)
    beta = np.linalg.solve(A, b)

    pw = weights * has_pace
    Xp = np.zeros((rows, 1 + n))
    Xp[r, 0] = 1.0
    np.add.at(Xp, (r, 1 + h), 1.0)
    np.add.at(Xp, (r, 1 + a), 1.0)
    yp = np.where(has_pace, poss, 0.0)
    penp = np.full(1 + n, cfg.ridge_pace)
    penp[0] = 1e-6
    if pw.sum() > 0:
        bp = np.linalg.solve(Xp.T @ (Xp * pw[:, None]) + np.diag(penp), Xp.T @ (pw * yp))
    else:
        bp = np.zeros(1 + n)
        bp[0] = cfg.default_pace

    return {
        "mu": beta[0], "hca": beta[1], "mu_pace": bp[0],
        "off": dict(zip(teams, beta[2:2 + n])),
        "def": dict(zip(teams, beta[2 + n:])),
        "pace": dict(zip(teams, bp[1:])),
    }


def efficiency_ratings(games: pd.DataFrame, cfg: RatingConfig) -> pd.DataFrame:
    """Pre-game efficiency ratings for every game in `games` (played or not)."""
    games = games.sort_values("date")
    played = games[games["played"]].copy()
    played["day"] = played["date"].dt.normalize()
    out = {}
    days = games["date"].dt.normalize()
    snapshots: dict = {}
    for day in pd.unique(days):
        lo = day - pd.Timedelta(days=cfg.window_days)
        hist = played[(played["day"] < day) & (played["day"] >= lo)]
        if len(hist) < 20:
            snapshots[day] = None
            continue
        age = (day - hist["day"]).dt.days.to_numpy(dtype=float)
        weights = 0.5 ** (age / cfg.half_life_days)
        cur_season = games.loc[days == day, "season"].max()
        weights = weights * np.where(hist["season"].to_numpy() < cur_season,
                                     cfg.prev_season_weight, 1.0)
        snapshots[day] = _solve_efficiency(hist, weights, cfg)
    for gid, day, home, away, neutral in zip(games["game_id"], days, games["home"],
                                             games["away"], games["neutral"]):
        snap = snapshots.get(day)
        if snap is None:
            out[gid] = {}
            continue
        out[gid] = rating_row(snap, home, away, bool(neutral))
    return pd.DataFrame.from_dict(out, orient="index").reindex(games["game_id"])


def rating_row(snap: dict, home: str, away: str, neutral: bool) -> dict:
    hca = 0.0 if neutral else snap["hca"]
    off_h, def_h = snap["off"].get(home, 0.0), snap["def"].get(home, 0.0)
    off_a, def_a = snap["off"].get(away, 0.0), snap["def"].get(away, 0.0)
    pace = snap["mu_pace"] + snap["pace"].get(home, 0.0) + snap["pace"].get(away, 0.0)
    eff_h = snap["mu"] + hca / 2 + off_h - def_a
    eff_a = snap["mu"] - hca / 2 + off_a - def_h
    return {
        "mu_eff": snap["mu"], "mu_pace": snap["mu_pace"], "hca_eff": hca,
        "off_home": off_h, "def_home": def_h, "off_away": off_a, "def_away": def_a,
        "pace_exp": pace, "eff_home_exp": eff_h, "eff_away_exp": eff_a,
        "pts_home_exp": eff_h * pace / 100.0, "pts_away_exp": eff_a * pace / 100.0,
    }


def latest_snapshot(games: pd.DataFrame, cfg: RatingConfig, as_of: pd.Timestamp | None = None):
    """Efficiency fit using every played game before `as_of` (default: now)."""
    as_of = (as_of or pd.Timestamp.now()).normalize()
    played = games[games["played"]].copy()
    played["day"] = played["date"].dt.normalize()
    hist = played[(played["day"] < as_of)
                  & (played["day"] >= as_of - pd.Timedelta(days=cfg.window_days))]
    age = (as_of - hist["day"]).dt.days.to_numpy(dtype=float)
    weights = 0.5 ** (age / cfg.half_life_days)
    weights = weights * np.where(hist["season"].to_numpy() < hist["season"].max(),
                                 cfg.prev_season_weight, 1.0)
    return _solve_efficiency(hist, weights, cfg)
