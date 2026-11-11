"""Build, predict and price props for NBA / NFL / NCAA."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from .. import espn
from ..window import now as _now
from . import data, props
from .model import CONFIGS, GameModel, run

NAMES = {"nba": "NBA", "nfl": "NFL", "ncaab": "NCAA basketball", "ncaaf": "NCAA football"}


def _state(code: str):
    return data.DIR / f"{code}_state.pkl"


def build(code: str) -> dict:
    g = data.games_with_lines(code)
    from .avail_us import available, with_missing
    if available(code):
        g = with_missing(g, code)
    rated, ratings = run(g, CONFIGS[code])
    gm = GameModel(CONFIGS[code]).fit(rated)
    _state(code).write_bytes(pickle.dumps({"rated": rated, "model": gm, "ratings": ratings}))
    return {"league": code, "games": int(rated["completed"].sum()), "margin_sd": gm.sd_m,
            "total_sd": gm.sd_t, "home_edge": ratings.hfa}


def load(code: str) -> dict:
    return pickle.loads(_state(code).read_bytes())


def upcoming(code: str, days: float = 2.0) -> pd.DataFrame:
    st = load(code)
    r = st["rated"]
    now = _now()
    up = r[~r["completed"] & (r["date"] > now) & (r["state"].fillna("pre") == "pre")
           & (r["date"] <= now + pd.Timedelta(days=days))]
    if up.empty:
        return up
    from .avail_us import available, upcoming_missing
    if available(code) and "miss_h" in st["rated"]:
        up = upcoming_missing(code, up)
    out = st["model"].predict(up)
    out["home_pts"] = (out["pred_total"] + out["pred_margin"]) / 2
    out["away_pts"] = (out["pred_total"] - out["pred_margin"]) / 2
    out["sd_m"], out["sd_t"] = st["model"].sd_m, st["model"].sd_t
    return out


def power(code: str, top: int = 40) -> pd.DataFrame:
    r = load(code)["ratings"]
    df = pd.DataFrame({"team": list(r.off), "off": [r.off[t] for t in r.off],
                       "def": [r.dfn[t] for t in r.off], "games": [r.n[t] for t in r.off]})
    df["net"] = df["off"] + df["def"]
    return df.sort_values("net", ascending=False).head(top).reset_index(drop=True)


# ---------------------------------------------------------------- NBA props

def nba_injuries(event_id: str) -> dict[str, str]:
    """{player_id: status} from ESPN's injury report for an upcoming game."""
    s = espn.summary("basketball", "nba", event_id, max_age_h=1.0) or {}
    out = {}
    for team in s.get("injuries", []):
        for inj in team.get("injuries", []):
            pid = str((inj.get("athlete") or {}).get("id", ""))
            if pid:
                out[pid] = inj.get("status", "")
    return out


OUT_P = {"out": 1.0, "doubtful": 0.75, "questionable": 0.5, "day-to-day": 0.4, "probable": 0.1}


def nba_game_props(game, table: pd.DataFrame, injuries: dict | None = None) -> pd.DataFrame:
    """Projections for both teams of one upcoming NBA game."""
    stats = props.NBA_STATS
    of = props.opponent_factors(table, stats, game.date)
    league_pts = table[table["date"] >= game.date - pd.Timedelta(days=60)].groupby(
        ["event_id", "team_id"])["pts"].sum().mean()
    out = []
    for tid, opp, team_pts, name in ((game.h_id, game.a_id, game.home_pts, game.h_name),
                                     (game.a_id, game.h_id, game.away_pts, game.a_name)):
        t = table[(table["team_id"] == tid) & (table["date"] < game.date)]
        recent = t.drop_duplicates("event_id").tail(4)["event_id"]
        roster = t[t["event_id"].isin(recent)][["player_id", "team_id", "pos"]].drop_duplicates(
            "player_id")
        pr = props.project(table[table["team_id"] == tid], stats, roster, game.date,
                           exposure="min")
        if pr.empty:
            continue
        pr["status"] = pr["player_id"].map(injuries or {}).fillna("")
        pr["p_out"] = pr["status"].str.lower().map(
            lambda s: next((v for k, v in OUT_P.items() if k in s), 0.0))
        # team minutes: scale the expected minutes of available players to 240
        base = pr["min"].to_numpy()
        avail = 1 - pr["p_out"].to_numpy()
        scale = 240.0 / max(float((base * avail).sum()), 1.0)
        new_min = np.minimum(base * np.clip(scale, 0.85, 1.35), 40.0)
        ratio = np.divide(new_min, base, out=np.ones_like(base), where=base > 0)
        pace = team_pts / league_pts if league_pts else 1.0
        for s in stats:
            f = float(of[s].get(opp, 1.0)) if s in of else 1.0
            adj = pace if s in ("pts", "fg3m", "ast") else 1.0
            pr[s] = pr[s] * ratio * (1 + 0.5 * (f - 1)) * (1 + 0.5 * (adj - 1))
        pr["min"] = new_min
        pr["pra"] = pr["pts"] + pr["reb"] + pr["ast"]
        pr["team"] = name
        out.append(pr[pr["p_out"] < 1].sort_values("min", ascending=False))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ---------------------------------------------------------------- NFL props

def nfl_game_props(game, table: pd.DataFrame, report: dict | None = None) -> pd.DataFrame:
    """Projections for both teams. report = {(team, player_id): P(out)} from the injury
    report: ruled-out players get no projection, their workload moves to teammates,
    and only one quarterback (the last game's leader in attempts) is projected."""
    stats = props.NFL_STATS
    of = props.opponent_factors(table, stats, game.date, games=8)
    out = []
    for tid, opp, team_pts, abbr in ((game.h_id, game.a_id, game.home_pts, game.h_abbr),
                                     (game.a_id, game.h_id, game.away_pts, game.a_abbr)):
        t = table[(table["team_id"] == tid) & (table["date"] < game.date)]
        if t.empty:
            continue
        last = t.drop_duplicates("event_id").tail(2)["event_id"]
        roster = t[t["event_id"].isin(last)][["player_id", "team_id", "pos"]].drop_duplicates(
            "player_id")
        pr = props.project(table, stats, roster, game.date, prior_games=2.0, window=10)
        if pr.empty:
            continue
        pr["p_out"] = [(report or {}).get((abbr, p), 0.0) for p in pr["player_id"]]
        full = pr.copy()
        pr = pr[pr["p_out"] < 0.8].copy()
        # one starting quarterback
        lastg = t[t["event_id"] == t.sort_values("date")["event_id"].iloc[-1]]
        qbs = lastg[lastg["pos"] == "QB"].sort_values("attempts", ascending=False)
        pr["starting_qb"] = ""
        if len(qbs):
            starter = qbs["player_id"].iloc[0]
            starter_out = (report or {}).get((abbr, starter), 0.0) >= 0.8
            alive = [q for q in qbs["player_id"] if (report or {}).get((abbr, q), 0.0) < 0.8]
            lead = alive[0] if alive else None
            is_qb = pr["pos"] == "QB"
            keep = pr["player_id"] == lead
            cols = [c for c in stats if c in pr.columns]
            pr.loc[is_qb & ~keep, cols] = pr.loc[is_qb & ~keep, cols] * 0.05
            pr.loc[is_qb & keep, "starting_qb"] = "starter" if not starter_out else "replacement"
        # workload of players who are out goes to teammates in proportion to their own
        for pos, cols in (("RB", ["carries", "receptions", "targets"]),
                          ("WR", ["targets", "receptions", "receiving_yards"]),
                          ("TE", ["targets", "receptions", "receiving_yards"])):
            grp = full[full["pos"] == pos]
            for c in cols:
                gone = float((grp[c] * grp["p_out"].where(grp["p_out"] >= 0.8, 0.0)).sum())
                here = pr["pos"] == pos
                have = float(pr.loc[here, c].sum())
                if gone > 0 and have > 0:
                    pr.loc[here, c] = pr.loc[here, c] * (1 + 0.6 * gone / have)
        for s in stats:
            f = float(of[s].get(opp, 1.0)) if s in of else 1.0
            pr[s] = pr[s] * f
        # scoring environment: touchdowns scale with the team's expected points
        k = float(np.clip(team_pts / 22.5, 0.7, 1.4))
        for s in ("passing_tds", "rushing_tds", "receiving_tds"):
            pr[s] *= k
        pr["anytime_td"] = 1 - np.exp(-(pr["rushing_tds"] + pr["receiving_tds"]))
        pr["team"] = abbr
        out.append(pr)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
