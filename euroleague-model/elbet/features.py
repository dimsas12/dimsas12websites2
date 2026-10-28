"""Assemble the pre-game feature table for every game."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .availability import AvailabilityModel
from .config import Settings
from .ratings import efficiency_ratings, run_elo

MARGIN_FEATURES = ["rating_margin", "elo_diff", "neutral", "short_rest_diff", "playoff",
                   "avail_diff", "bu_margin", "bu_out_diff"]
TOTAL_FEATURES = ["rating_total", "pace_exp", "playoff", "early_season", "scoring_trend",
                  "missing_pts_total", "bu_total_dev", "bu_out_total"]


def _rest_days(games: pd.DataFrame) -> pd.DataFrame:
    """Days since each team's previous EuroLeague game (domestic games are not visible)."""
    long = pd.concat([
        games[["game_id", "date", "home"]].rename(columns={"home": "team"}).assign(side="home"),
        games[["game_id", "date", "away"]].rename(columns={"away": "team"}).assign(side="away"),
    ]).sort_values("date")
    long["prev"] = long.groupby("team")["date"].shift()
    long["rest"] = ((long["date"] - long["prev"]).dt.total_seconds() / 86400).clip(upper=10)
    long["rest"] = long["rest"].fillna(10)
    wide = long.pivot(index="game_id", columns="side", values="rest")
    return wide.rename(columns={"home": "rest_home", "away": "rest_away"})


def _recent_league_total(games: pd.DataFrame, half_life_days: float = 30.0) -> pd.Series:
    """Recency-weighted league average total from games before each game day.

    Team ratings use a long memory; league-wide scoring drifts faster (it has
    risen ~25 points since 2012), so the totals model gets this short-memory view.
    """
    played = games[games["played"]]
    pdays = played["date"].dt.normalize().to_numpy()
    totals = (played["home_pts"] + played["away_pts"]).to_numpy(float)
    days = games["date"].dt.normalize()
    cache = {}
    for day in pd.unique(days):
        mask = pdays < day
        if mask.sum() < 20:
            cache[day] = np.nan
            continue
        age = (day - pdays[mask]) / np.timedelta64(1, "D")
        w = 0.5 ** (age / half_life_days)
        cache[day] = float((w * totals[mask]).sum() / w.sum())
    return days.map(cache)


def _games_played(games: pd.DataFrame) -> pd.DataFrame:
    long = pd.concat([
        games[["game_id", "date", "season", "home"]].rename(columns={"home": "team"})
        .assign(side="home"),
        games[["game_id", "date", "season", "away"]].rename(columns={"away": "team"})
        .assign(side="away"),
    ]).sort_values("date")
    long["gp"] = long.groupby(["season", "team"]).cumcount()
    wide = long.pivot(index="game_id", columns="side", values="gp")
    return wide.rename(columns={"home": "gp_home", "away": "gp_away"})


def build_features(games: pd.DataFrame, settings: Settings, box: pd.DataFrame | None = None,
                   availability: AvailabilityModel | None = None) -> pd.DataFrame:
    """Pre-game features. With player box scores, also player-availability features
    (`availability` is filled in place so its end state can be reused for live games)."""
    games = games.sort_values(["date", "game_code"]).reset_index(drop=True)
    elo, _ = run_elo(games, settings.elo)
    eff = efficiency_ratings(games, settings.ratings).reset_index(drop=True)
    feats = pd.concat([games, elo.reset_index(drop=True), eff], axis=1)
    feats = feats.merge(_rest_days(games), left_on="game_id", right_index=True, how="left")
    feats = feats.merge(_games_played(games), left_on="game_id", right_index=True, how="left")

    feats["elo_diff"] = feats["elo_home"] - feats["elo_away"]
    feats["rating_margin"] = feats["pts_home_exp"] - feats["pts_away_exp"]
    feats["rating_total"] = feats["pts_home_exp"] + feats["pts_away_exp"]
    feats["neutral"] = feats["neutral"].astype(float)
    feats["short_rest_diff"] = ((feats["rest_home"] <= 2).astype(float)
                                - (feats["rest_away"] <= 2).astype(float))
    feats["playoff"] = feats["phase"].isin(["PO", "FF", "PI"]).astype(float)
    feats["early_season"] = (np.minimum(feats["gp_home"], feats["gp_away"]) < 5).astype(float)
    feats["league_total_recent"] = _recent_league_total(feats).to_numpy()
    # Recent league scoring vs. the league level implied by the (slow) ratings.
    feats["scoring_trend"] = (feats["league_total_recent"]
                              - feats["mu_eff"] * feats["mu_pace"] / 50).fillna(0.0)
    if box is not None and not box.empty:
        avail = (availability or AvailabilityModel()).run(games, box)
        feats = feats.merge(avail, on="game_id", how="left")
    for col in ("missing_home", "missing_away", "missing_pts_home", "missing_pts_away",
                "n_missing_home", "n_missing_away"):
        feats[col] = feats[col].fillna(0.0) if col in feats else 0.0
    set_availability(feats)
    set_bottom_up(feats)
    feats["margin"] = feats["home_pts"] - feats["away_pts"]
    feats["total"] = feats["home_pts"] + feats["away_pts"]
    feats["home_win"] = (feats["margin"] > 0).astype(float).where(feats["played"])
    feats["has_ratings"] = feats["rating_margin"].notna()
    return feats


def set_availability(feats: pd.DataFrame) -> None:
    """Derive model inputs from missing_home / missing_away (positive avail_diff favours home)."""
    feats["avail_diff"] = feats["missing_away"] - feats["missing_home"]
    feats["missing_total"] = feats["missing_home"] + feats["missing_away"]
    # Scoring lost to absences (points per game above replacement) drives the total.
    feats["missing_pts_total"] = feats["missing_pts_home"] + feats["missing_pts_away"]


def add_bottom_up(feats: pd.DataFrame, team_rows: pd.DataFrame) -> pd.DataFrame:
    """Bottom-up features from the players expected to play (see props_backtest.replay):
    expected player points (chance of playing x projection) per team, and the points of
    players ruled out. Walk-forward tests: holdout log loss 0.6112 -> 0.6109, margin MAE
    9.018 -> 8.999, total MAE 13.279 -> 13.257, and better on 2015-22 too."""
    feats = feats.drop(columns=[c for c in ("bu_home", "bu_away", "bu_out_home", "bu_out_away")
                                if c in feats], errors="ignore")
    t = team_rows[["game_id", "team", "bottom_pts", "bottom_out_pts"]]
    feats = feats.merge(t.rename(columns={"team": "home", "bottom_pts": "bu_home",
                                          "bottom_out_pts": "bu_out_home"}),
                        on=["game_id", "home"], how="left")
    feats = feats.merge(t.rename(columns={"team": "away", "bottom_pts": "bu_away",
                                          "bottom_out_pts": "bu_out_away"}),
                        on=["game_id", "away"], how="left")
    set_bottom_up(feats)
    return feats


def set_bottom_up(feats: pd.DataFrame) -> None:
    for c in ("bu_home", "bu_away", "bu_out_home", "bu_out_away"):
        if c not in feats:
            feats[c] = np.nan
    feats["bu_margin"] = (feats["bu_home"] - feats["bu_away"]).fillna(0.0)
    feats["bu_total_dev"] = (feats["bu_home"] + feats["bu_away"]
                             - feats["rating_total"]).fillna(0.0)
    feats["bu_out_diff"] = (feats["bu_out_away"] - feats["bu_out_home"]).fillna(0.0)
    feats["bu_out_total"] = (feats["bu_out_away"] + feats["bu_out_home"]).fillna(0.0)
