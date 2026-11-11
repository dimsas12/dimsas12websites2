"""Games, lines and player box scores for NBA / NCAA basketball / NCAA football (ESPN)
and the NFL (nflverse)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .. import espn

ROOT = espn.ROOT
DIR = ROOT / "data" / "us"

LEAGUES = {
    "nba": {"sport": "basketball", "league": "nba", "months": (10, 6), "first": 2019,
            "extra": None},
    "ncaab": {"sport": "basketball", "league": "mens-college-basketball", "months": (11, 4),
              "first": 2022, "extra": {"groups": 50}},
    "ncaaf": {"sport": "football", "league": "college-football", "months": (8, 1),
              "first": 2021, "extra": {"groups": 80}},
}


def _season_dates(code: str, first: int | None = None) -> list[pd.Timestamp]:
    cfg = LEAGUES[code]
    start_m, end_m = cfg["months"]
    today = pd.Timestamp.now(tz="UTC").tz_convert(None).normalize()
    out = []
    for y in range(first or cfg["first"], today.year + 1):
        a = pd.Timestamp(year=y, month=start_m, day=1)
        b = pd.Timestamp(year=y + 1, month=end_m, day=28)
        out += [d for d in pd.date_range(a, min(b, today + pd.Timedelta(days=8)))]
    return out


def parse_scoreboard(data: dict, code: str) -> list[dict]:
    rows = []
    for ev in (data or {}).get("events", []):
        c = ev["competitions"][0]
        comp = {t["homeAway"]: t for t in c["competitors"]}
        if "home" not in comp or "away" not in comp:
            continue
        st = ev["status"]["type"]
        row = {"event_id": ev["id"], "code": code, "date": ev["date"],
               "season": ev.get("season", {}).get("year"),
               "season_type": ev.get("season", {}).get("type"),
               "state": st.get("state"), "completed": bool(st.get("completed")),
               "neutral": bool(c.get("neutralSite", False)),
               "periods": ev["status"].get("period")}
        for side, t in comp.items():
            p = side[0]
            row[f"{p}_id"] = t["team"]["id"]
            row[f"{p}_abbr"] = t["team"].get("abbreviation")
            row[f"{p}_name"] = t["team"].get("displayName")
            row[f"{p}_pts"] = pd.to_numeric(t.get("score"), errors="coerce") if st.get(
                "completed") else np.nan
        rows.append(row)
    return rows


def download_schedule(code: str, refresh_h: float = 3.0, workers: int = 8) -> pd.DataFrame:
    cfg = LEAGUES[code]
    today = pd.Timestamp.now(tz="UTC").tz_convert(None).normalize()
    dates = _season_dates(code)

    def one(d):
        age = refresh_h if d >= today - pd.Timedelta(days=3) else None
        return parse_scoreboard(espn.scoreboard(cfg["sport"], cfg["league"], f"{d:%Y%m%d}", age,
                                                cfg["extra"]), code)
    with ThreadPoolExecutor(workers) as ex:
        rows = [r for part in ex.map(one, dates) for r in part]
    df = pd.DataFrame(rows).drop_duplicates("event_id", keep="last")
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    df = df.sort_values(["date", "event_id"]).reset_index(drop=True)
    DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DIR / f"{code}_games.csv", index=False)
    return df


def load_games(code: str) -> pd.DataFrame:
    return pd.read_csv(DIR / f"{code}_games.csv", parse_dates=["date"],
                       dtype={"event_id": str, "h_id": str, "a_id": str})


def _american(x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return np.nan
    if x == 0 or x != x:
        return np.nan
    return 1 + x / 100 if x > 0 else 1 + 100 / -x


def parse_odds(d: dict | None) -> dict:
    """First non-live provider: home spread, total and decimal prices."""
    for x in (d or {}).get("items", []):
        if x.get("spread") is None and x.get("overUnder") is None:
            continue
        h, a = x.get("homeTeamOdds") or {}, x.get("awayTeamOdds") or {}
        spread = x.get("spread")
        # ESPN's spread is from the favourite's side: negative = home favoured only if
        # the home team is the favourite.
        if spread is not None and h.get("favorite") is False and spread < 0:
            spread = -spread
        elif spread is not None and h.get("favorite") is True and spread > 0:
            spread = -spread
        return {"spread": spread, "total": x.get("overUnder"),
                "ml_h": _american(h.get("moneyLine")), "ml_a": _american(a.get("moneyLine")),
                "sp_h": _american(h.get("spreadOdds")), "sp_a": _american(a.get("spreadOdds")),
                "o_over": _american(x.get("overOdds")), "o_under": _american(x.get("underOdds")),
                "book": (x.get("provider") or {}).get("name")}
    return {}


def download_lines(code: str, first_season: int | None = None, workers: int = 8
                   ) -> pd.DataFrame:
    cfg = LEAGUES[code]
    g = load_games(code)
    g = g[g["completed"]]
    if first_season:
        g = g[g["season"] >= first_season]

    def one(eid):
        return {"event_id": eid, **parse_odds(espn.event_odds(cfg["sport"], cfg["league"], eid))}
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, g["event_id"]))
    df = pd.DataFrame(rows)
    df.to_csv(DIR / f"{code}_lines.csv", index=False)
    return df


BOX_COLS = {"MIN": "min", "PTS": "pts", "REB": "reb", "AST": "ast", "TO": "tov", "STL": "stl",
            "BLK": "blk", "3PT": "fg3", "FG": "fg", "FT": "ft", "OREB": "oreb", "PF": "pf"}


def parse_box(summary: dict | None, event_id: str) -> list[dict]:
    rows = []
    for team in ((summary or {}).get("boxscore") or {}).get("players", []):
        tid = team.get("team", {}).get("id")
        for block in team.get("statistics", []):
            names = block.get("names") or block.get("labels") or []
            for a in block.get("athletes", []):
                stats = a.get("stats") or []
                if not a.get("athlete", {}).get("id"):
                    continue
                row = {"event_id": event_id, "team_id": tid,
                       "player_id": a["athlete"]["id"],
                       "player": a["athlete"].get("displayName"),
                       "pos": (a["athlete"].get("position") or {}).get("abbreviation"),
                       "starter": bool(a.get("starter")), "dnp": bool(a.get("didNotPlay"))}
                for nm, v in zip(names, stats):
                    col = BOX_COLS.get(nm)
                    if not col:
                        continue
                    if col in ("fg3", "fg", "ft"):
                        made, _, att = str(v).partition("-")
                        row[col + "m"] = pd.to_numeric(made, errors="coerce")
                        row[col + "a"] = pd.to_numeric(att, errors="coerce")
                    else:
                        row[col] = pd.to_numeric(v, errors="coerce")
                rows.append(row)
    return rows


def download_boxes(code: str, first_season: int | None = None, workers: int = 8
                   ) -> pd.DataFrame:
    cfg = LEAGUES[code]
    g = load_games(code)
    g = g[g["completed"]]
    if first_season:
        g = g[g["season"] >= first_season]

    def one(eid):
        return parse_box(espn.summary(cfg["sport"], cfg["league"], eid), eid)
    with ThreadPoolExecutor(workers) as ex:
        rows = [r for part in ex.map(one, g["event_id"]) for r in part]
    df = pd.DataFrame(rows)
    df = df.merge(g[["event_id", "date", "season"]], on="event_id")
    df.to_csv(DIR / f"{code}_players.csv", index=False)
    return df


def load_players(code: str) -> pd.DataFrame:
    return pd.read_csv(DIR / f"{code}_players.csv", parse_dates=["date"],
                       dtype={"event_id": str, "team_id": str, "player_id": str})


NFL_GAMES = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"


def download_nfl() -> pd.DataFrame:
    """NFL schedule, scores and closing lines since 1999 (nflverse)."""
    import io

    import requests

    r = requests.get(NFL_GAMES, timeout=60)
    r.raise_for_status()
    g = pd.read_csv(io.BytesIO(r.content))
    out = pd.DataFrame({
        "event_id": g["game_id"], "code": "nfl",
        "date": pd.to_datetime(g["gameday"]) + pd.to_timedelta(
            pd.to_numeric(g["gametime"].fillna("13:00").str[:2], errors="coerce").fillna(13),
            unit="h"),
        "season": g["season"], "season_type": g["game_type"], "week": g["week"],
        "completed": g["home_score"].notna(), "neutral": g["location"].eq("Neutral"),
        "h_id": g["home_team"], "a_id": g["away_team"], "h_abbr": g["home_team"],
        "a_abbr": g["away_team"], "h_name": g["home_team"], "a_name": g["away_team"],
        "h_pts": g["home_score"], "a_pts": g["away_score"],
        "spread": -g["spread_line"],  # nflverse: positive = home favoured
        "total": g["total_line"],
        "ml_h": g["home_moneyline"].map(_american), "ml_a": g["away_moneyline"].map(_american),
        "sp_h": g["home_spread_odds"].map(_american), "sp_a": g["away_spread_odds"].map(_american),
        "o_over": g["over_odds"].map(_american), "o_under": g["under_odds"].map(_american),
        "roof": g["roof"], "temp": g["temp"], "wind": g["wind"],
        "home_qb": g["home_qb_name"], "away_qb": g["away_qb_name"],
    })
    # kick-off times are US Eastern: store UTC like every other league
    out["date"] = out["date"].dt.tz_localize("America/New_York", ambiguous="NaT",
                                             nonexistent="NaT").dt.tz_convert(None)
    out = out.dropna(subset=["date"])
    out["state"] = np.where(out["completed"], "post", "pre")
    out = out.sort_values(["date", "event_id"]).reset_index(drop=True)
    DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(DIR / "nfl_games.csv", index=False)
    return out


def games_with_lines(code: str) -> pd.DataFrame:
    """Standard game table with the closing line columns for any league."""
    g = load_games(code)
    if code == "nfl":
        return g
    lines = DIR / f"{code}_lines.csv"
    if lines.exists():
        g = g.merge(pd.read_csv(lines, dtype={"event_id": str}), on="event_id", how="left")
    else:
        for c in ("spread", "total"):
            g[c] = np.nan
    return g
