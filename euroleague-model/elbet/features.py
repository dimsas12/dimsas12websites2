"""Assemble the pre-game feature table for every game."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings
from .ratings import efficiency_ratings, run_elo

MARGIN_FEATURES = ["rating_margin", "elo_diff", "neutral", "short_rest_diff", "playoff"]
TOTAL_FEATURES = ["rating_total", "pace_exp", "playoff", "early_season"]


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


def build_features(games: pd.DataFrame, settings: Settings) -> pd.DataFrame:
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
    feats["margin"] = feats["home_pts"] - feats["away_pts"]
    feats["total"] = feats["home_pts"] + feats["away_pts"]
    feats["home_win"] = (feats["margin"] > 0).astype(float).where(feats["played"])
    feats["has_ratings"] = feats["rating_margin"].notna()
    return feats
