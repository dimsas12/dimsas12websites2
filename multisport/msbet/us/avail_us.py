"""Availability for NBA and NFL: history table, injury reports, and the miss values of
upcoming games (see msbet/avail.py)."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests

from .. import avail, espn
from . import data, props

PARAMS = {"nba": dict(exp_col="min", window=10, recent=3, min_appear=0.5),
          "nfl": dict(exp_col="expo", window=6, recent=2, min_appear=0.5),
          "ncaab": dict(exp_col="min", window=8, recent=2, min_appear=0.5)}
OUT_P = {"out": 1.0, "doubtful": 0.75, "questionable": 0.5, "day-to-day": 0.4, "probable": 0.1,
         "suspension": 1.0}


def box(code: str) -> pd.DataFrame:
    if code == "ncaab":
        b = data.load_players("ncaab")
        b = b[~b["dnp"] & (b["min"].fillna(0) > 0)].copy()
        b["prod"] = b["pts"] + b["reb"] + b["ast"] + b["stl"].fillna(0) + b["blk"].fillna(0) \
            - b["tov"].fillna(0)
        b["red"] = 0
        return b[["event_id", "date", "team_id", "player_id", "player", "min", "prod", "pts",
                  "red"]]
    if code == "nba":
        b = props.nba_table()
        b["prod"] = b["pts"] + b["reb"] + b["ast"] + b["stl"] + b["blk"] - b["tov"]
        return b[["event_id", "date", "team_id", "player_id", "player", "min", "prod", "pts"]]
    p = pd.read_csv(data.DIR / "nfl_players.csv", parse_dates=["date"])
    p["prod"] = (0.04 * p["passing_yards"] + 4 * p["passing_tds"] - 2 * p["passing_interceptions"]
                 + 0.1 * p["rushing_yards"] + 6 * p["rushing_tds"] + 0.5 * p["receptions"]
                 + 0.1 * p["receiving_yards"] + 6 * p["receiving_tds"])
    p["expo"] = p["attempts"] + p["carries"] + p["targets"]
    p["pts"] = p["prod"]
    return p[["event_id", "date", "team_id", "player_id", "player", "pos", "expo", "prod", "pts"]]


def missing_history(code: str, rebuild: bool = False) -> pd.DataFrame:
    """miss_val / miss_pts per (event, team); only the last 45 days are recomputed."""
    path = data.DIR / f"{code}_missing.pkl"
    b = box(code)
    kw = dict(PARAMS[code])
    old = pd.read_pickle(path) if path.exists() and not rebuild else None
    since = None
    if old is not None:
        since = b["date"].max() - pd.Timedelta(days=45)
    new = avail.build_missing(b, prod_col="prod", pts_col="pts", since=since, **kw)
    if old is not None:
        old = old[~old["event_id"].isin(set(new["event_id"]))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_pickle(path)
    return new


def available(code: str) -> bool:
    """Is there player data to build the availability features for this league?"""
    # NCAA basketball was tested too and did not improve the model (margin error 9.641 ->
    # 9.654, 2025-26), so it is left out; PARAMS keeps its settings for reference.
    f = {"nba": "nba_players.csv", "nfl": "nfl_players.csv"}
    return code in f and (data.DIR / f[code]).exists()


def with_missing(g: pd.DataFrame, code: str) -> pd.DataFrame:
    if not available(code):
        return g
    m = missing_history(code)
    h = m.rename(columns={"team_id": "h_id", "miss_val": "miss_h", "miss_pts": "mp_h"})
    a = m.rename(columns={"team_id": "a_id", "miss_val": "miss_a", "miss_pts": "mp_a"})
    g = g.merge(h[["event_id", "h_id", "miss_h"]], on=["event_id", "h_id"], how="left")
    return g.merge(a[["event_id", "a_id", "miss_a"]], on=["event_id", "a_id"], how="left")


# ---------------------------------------------------------------- injury reports

def nba_report(event_id: str) -> dict[str, float]:
    """{player_id: P(out)} from ESPN's report for an upcoming game."""
    s = espn.summary("basketball", "nba", event_id, max_age_h=1.0) or {}
    out = {}
    for team in s.get("injuries", []):
        for inj in team.get("injuries", []):
            pid = str((inj.get("athlete") or {}).get("id", ""))
            st = str(inj.get("status", "")).lower()
            if pid:
                out[pid] = next((v for k, v in OUT_P.items() if k in st), 0.0)
    return out


NFL_PRACTICE = {"did not participate": 0.5, "limited": 0.2, "full": 0.0}


def nfl_report(season: int, week: int) -> dict[tuple, float]:
    """{(team, gsis_id): P(out)} for a week: the game-status report when published
    (Out / Doubtful / Questionable), else the practice report, else last week's Outs."""
    url = ("https://github.com/nflverse/nflverse-data/releases/download/injuries/"
           f"injuries_{season}.csv")
    r = requests.get(url, timeout=60)
    if not r.ok:
        return {}
    d = pd.read_csv(io.BytesIO(r.content))
    out: dict[tuple, float] = {}
    prev = d[d["week"] == week - 1]
    for x in prev.itertuples(index=False):
        if str(x.report_status).lower() == "out":
            out[(x.team, x.gsis_id)] = 0.7  # still out unless he is on this week's list
    for x in d[d["week"] == week].itertuples(index=False):
        st, pr = str(x.report_status).lower(), str(x.practice_status).lower()
        if st in OUT_P:
            p = {"out": 1.0, "doubtful": 0.8, "questionable": 0.35}.get(st, 0.0)
        else:
            p = next((v for k, v in NFL_PRACTICE.items() if k in pr), 0.0)
        out[(x.team, x.gsis_id)] = p
    return out


def miss_for_team(b: pd.DataFrame, code: str, team_id: str, as_of: pd.Timestamp,
                  p_out: dict[str, float]) -> float:
    """Sum of regulars' value weighted by the chance each is out."""
    if not p_out:
        return 0.0
    kw = PARAMS[code]
    pv = avail.current_values(b, team_id, as_of, exp_col=kw["exp_col"], prod_col="prod",
                              pts_col="pts", window=kw["window"], min_appear=0.3)
    return float(sum(pv.at[p, "value"] * w for p, w in p_out.items() if p in pv.index))


def upcoming_missing(code: str, up: pd.DataFrame) -> pd.DataFrame:
    """Adds miss_h / miss_a to upcoming games from the official injury reports."""
    up = up.copy()
    b = box(code)
    mh, ma = [], []
    nfl_cache: dict = {}
    if code == "ncaab":  # no injury report exists: infer from who missed the last games
        from ..soccer import avail_soccer as A
        probs = A.learn_streak_probs(b[b["date"] >= b["date"].max() - pd.Timedelta(days=500)],
                                     regular=5, span=8)
        for g in up.itertuples(index=False):
            vals = []
            for tid in (g.h_id, g.a_id):
                pv = A.regulars(b, tid, g.date, window=8)
                p = A.inferred_p_out(pv, probs) if len(pv) else {}
                vals.append(sum(pv.at[k, "value"] * w for k, w in p.items() if k in pv.index))
            mh.append(vals[0]); ma.append(vals[1])
        up["miss_h"], up["miss_a"] = mh, ma
        return up
    for g in up.itertuples(index=False):
        if code == "nba":
            rep = nba_report(g.event_id)
            ph = {p: w for p, w in rep.items()}
            pa = ph
        else:
            key = (int(g.season), int(g.week))
            if key not in nfl_cache:
                nfl_cache[key] = nfl_report(*key)
            rep = nfl_cache[key]
            ph = {p: w for (t, p), w in rep.items() if t == g.h_abbr}
            pa = {p: w for (t, p), w in rep.items() if t == g.a_abbr}
        mh.append(miss_for_team(b, code, g.h_id, g.date, ph))
        ma.append(miss_for_team(b, code, g.a_id, g.date, pa))
    up["miss_h"], up["miss_a"] = mh, ma
    return up
