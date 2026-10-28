"""Turn raw API JSON into tidy game, team-box and player-box tables."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from . import config
from .fetch import boxscore_path, schedule_path

log = logging.getLogger(__name__)

BOX_FIELDS = {
    "FieldGoalsMade2": "fgm2", "FieldGoalsAttempted2": "fga2",
    "FieldGoalsMade3": "fgm3", "FieldGoalsAttempted3": "fga3",
    "FreeThrowsMade": "ftm", "FreeThrowsAttempted": "fta",
    "OffensiveRebounds": "oreb", "DefensiveRebounds": "dreb", "TotalRebounds": "reb",
    "Assistances": "ast", "Steals": "stl", "Turnovers": "tov",
    "BlocksFavour": "blk", "FoulsCommited": "pf", "Valuation": "pir", "Points": "pts",
}
PLAYER_STATS = ["pts", "reb", "ast", "fgm3", "stl", "blk", "tov", "pir"]


def parse_minutes(value) -> float:
    if not value or not isinstance(value, str) or ":" not in value:
        return 0.0
    m, s = value.split(":")[:2]
    try:
        return int(m) + int(s) / 60.0
    except ValueError:
        return 0.0


def possessions(fga2, fga3, fta, oreb, tov) -> float:
    return fga2 + fga3 + 0.44 * fta - oreb + tov


def _club_names(club: dict) -> str:
    names = {club.get("name", ""), club.get("abbreviatedName", ""), club.get("editorialName", "")}
    return "|".join(sorted(n for n in names if n))


def parse_schedule(season: int, games: list[dict]) -> pd.DataFrame:
    rows = []
    for g in games:
        loc, road = g["local"], g["road"]
        rows.append({
            "game_id": f"{config.season_code(season)}_{g['gameCode']}",
            "season": season,
            "game_code": g["gameCode"],
            "date": pd.Timestamp(g["utcDate"]).tz_convert(None) if g.get("utcDate")
            else pd.Timestamp(g["date"]),
            "phase": g["phaseType"]["code"],
            "round": g["round"],
            "neutral": bool(g.get("isNeutralVenue")),
            "home": loc["club"]["code"],
            "away": road["club"]["code"],
            "home_name": loc["club"]["name"],
            "away_name": road["club"]["name"],
            "home_names": _club_names(loc["club"]),
            "away_names": _club_names(road["club"]),
            "played": bool(g["played"]),
            "home_pts": loc.get("score") if g["played"] else np.nan,
            "away_pts": road.get("score") if g["played"] else np.nan,
        })
    return pd.DataFrame(rows)


def parse_boxscore(game: pd.Series, box: dict) -> tuple[dict, list[dict]] | None:
    """Return (team-level box stats, player rows) or None if the box is unusable."""
    stats = box.get("Stats") or []
    if len(stats) != 2:
        return None
    sides = {}
    players = []
    for idx, side in enumerate(("home", "away")):
        tot = stats[idx].get("totr") or {}
        team_stats = {v: float(tot.get(k) or 0) for k, v in BOX_FIELDS.items()}
        sides[side] = team_stats
        team = game[side]
        opp = game["away" if side == "home" else "home"]
        for p in stats[idx].get("PlayersStats") or []:
            pid = (p.get("Player_ID") or "").strip()
            if not pid:
                continue
            row = {
                "game_id": game["game_id"], "season": game["season"], "date": game["date"],
                "player_id": pid, "player": p.get("Player", ""), "team": team, "opp": opp,
                "is_home": side == "home", "starter": bool(p.get("IsStarter")),
                "min": parse_minutes(p.get("Minutes")),
            }
            for k, v in BOX_FIELDS.items():
                if v in PLAYER_STATS:
                    row[v] = float(p.get(k) or 0)
            row["pm"] = float(p.get("Plusminus") or 0)  # on-court point differential
            players.append(row)
    # Sanity check: box score order must match home/away points.
    if not (np.isclose(sides["home"]["pts"], game["home_pts"])
            and np.isclose(sides["away"]["pts"], game["away_pts"])):
        if (np.isclose(sides["home"]["pts"], game["away_pts"])
                and np.isclose(sides["away"]["pts"], game["home_pts"])):
            log.warning("box score sides swapped for %s; skipping", game["game_id"])
        return None
    poss = 0.5 * sum(
        possessions(s["fga2"], s["fga3"], s["fta"], s["oreb"], s["tov"]) for s in sides.values()
    )
    team = {"game_id": game["game_id"], "poss": poss}
    for side, s in sides.items():
        for k in ("fga2", "fgm2", "fga3", "fgm3", "fta", "ftm", "oreb", "dreb", "ast", "tov"):
            team[f"{side}_{k}"] = s[k]
    return team, players


def build_tables(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build games (with possessions) and player box tables from the raw cache."""
    games, players = [], []
    for season in seasons:
        path = schedule_path(season)
        if not path.exists():
            log.warning("no schedule cached for %s; run `elbet fetch`", season)
            continue
        sched = parse_schedule(season, json.loads(path.read_text()))
        team_rows = []
        for _, g in sched[sched["played"]].iterrows():
            bpath = boxscore_path(season, g["game_code"])
            if not bpath.exists():
                continue
            parsed = parse_boxscore(g, json.loads(bpath.read_text()))
            if parsed is None:
                continue
            team_rows.append(parsed[0])
            players.extend(parsed[1])
        if team_rows:
            sched = sched.merge(pd.DataFrame(team_rows), on="game_id", how="left")
        games.append(sched)
    games_df = pd.concat(games, ignore_index=True).sort_values(["date", "game_code"])
    if "poss" not in games_df:
        games_df["poss"] = np.nan
    players_df = pd.DataFrame(players)
    if not players_df.empty:
        players_df = players_df.sort_values(["date", "game_id"]).reset_index(drop=True)
    return games_df.reset_index(drop=True), players_df


def save_tables(games: pd.DataFrame, players: pd.DataFrame) -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    games.to_csv(config.PROCESSED_DIR / "games.csv", index=False)
    players.to_csv(config.PROCESSED_DIR / "players.csv", index=False)


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    gpath = config.PROCESSED_DIR / "games.csv"
    if not gpath.exists():
        raise FileNotFoundError("processed data missing; run `elbet build` first")
    games = pd.read_csv(gpath, parse_dates=["date"])
    players = pd.read_csv(config.PROCESSED_DIR / "players.csv", parse_dates=["date"])
    return games, players
