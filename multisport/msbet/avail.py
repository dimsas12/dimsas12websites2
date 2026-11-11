"""Player availability as a team-level feature, learned from box scores.

For every team-game we estimate how much value the team is missing: regulars of
the last games (appeared in at least `min_appear` of the last `window` games and in
one of the last `recent`) who did not play. A player's value is his usual
exposure x (his production per unit of exposure - a replacement-level rate).
For upcoming games the same sum is taken over the players the injury report lists,
weighted by the chance each one is out. The model then learns how many points of
margin / total one unit of missing value costs (walk-forward, earlier seasons only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def player_values(window: pd.DataFrame, exp_col: str, prod_col: str, pts_col: str,
                  r0: float) -> pd.DataFrame:
    """Typical exposure and production per game of each player over `window`
    (the team's last games), as value above replacement."""
    played = window[window[exp_col] > 0]
    pos = played.groupby("player_id")["pos"].last() if "pos" in played else None
    g = played.groupby("player_id").agg(
        appear=("event_id", "nunique"), exp=(exp_col, "mean"), prod=(prod_col, "sum"),
        pts=(pts_col, "sum"), expsum=(exp_col, "sum"), last=("date", "max"))
    g["rate"] = g["prod"] / g["expsum"].clip(lower=1e-9)
    g["pts_rate"] = g["pts"] / g["expsum"].clip(lower=1e-9)
    base = g.index.map(lambda p: r0.get(pos[p], 0.0)) if isinstance(r0, dict) else r0
    g["value"] = (g["exp"] * (g["rate"] - base)).clip(lower=0)
    g["pts_value"] = g["exp"] * g["pts_rate"]
    return g


def build_missing(box: pd.DataFrame, exp_col: str = "min", prod_col: str = "prod",
                  pts_col: str = "pts", window: int = 10, min_appear: float = 0.5,
                  recent: int = 3, r0_quantile: float = 0.4,
                  since: pd.Timestamp | None = None) -> pd.DataFrame:
    """Rows (event_id, team_id) -> miss_val, miss_pts, n_missing, plus a per-player
    table of who was missing (for audits)."""
    box = box.sort_values(["date", "event_id"])
    pl = box[box[exp_col] > 0]
    if "pos" in box:  # replacement level by position (a QB is not comparable to a WR)
        r0_pos = (pl.groupby("pos")[prod_col].sum() / pl.groupby("pos")[exp_col].sum()
                  * r0_quantile).to_dict()
        r0 = r0_pos
    else:
        r0 = float(pl[prod_col].sum() / pl[exp_col].sum()) * r0_quantile
    rows = []
    for team, tb in box.groupby("team_id"):
        games = tb.drop_duplicates("event_id").sort_values("date")["event_id"].tolist()
        pos = {e: i for i, e in enumerate(games)}
        tb = tb.assign(gi=tb["event_id"].map(pos))
        by_game = {i: d for i, d in tb.groupby("gi")}
        gdate = tb.drop_duplicates("event_id").sort_values("date")["date"].tolist()
        for i in range(1, len(games)):
            if since is not None and gdate[i] < since:
                continue
            lo = max(0, i - window)
            frames = [by_game[j] for j in range(lo, i) if j in by_game]
            if not frames:
                continue
            w = pd.concat(frames)
            pv = player_values(w, exp_col, prod_col, pts_col, r0)
            n_games = i - lo
            last_recent = set(pd.concat([by_game[j] for j in range(max(0, i - recent), i)
                                         if j in by_game]).query(f"{exp_col} > 0")["player_id"])
            reg = pv[(pv["appear"] >= min_appear * n_games) & pv.index.isin(last_recent)]
            cur = by_game.get(i)
            here = set(cur.loc[cur[exp_col] > 0, "player_id"]) if cur is not None else set()
            miss = reg[~reg.index.isin(here)]
            rows.append((games[i], team, float(miss["value"].sum()),
                         float(miss["pts_value"].sum()), len(miss)))
    return pd.DataFrame(rows, columns=["event_id", "team_id", "miss_val", "miss_pts",
                                       "n_missing"])


def current_values(box: pd.DataFrame, team_id: str, as_of: pd.Timestamp, exp_col: str = "min",
                   prod_col: str = "prod", pts_col: str = "pts", window: int = 10,
                   min_appear: float = 0.3) -> pd.DataFrame:
    """Value table of the team's regulars before `as_of` (for injury-report weighting)."""
    tb = box[(box["team_id"] == team_id) & (box["date"] < as_of)]
    games = tb.drop_duplicates("event_id").sort_values("date").tail(window)["event_id"]
    w = tb[tb["event_id"].isin(set(games))]
    pl = box[box[exp_col] > 0]
    if "pos" in box:
        r0 = (pl.groupby("pos")[prod_col].sum() / pl.groupby("pos")[exp_col].sum() * 0.4).to_dict()
    else:
        r0 = float(pl[prod_col].sum() / pl[exp_col].sum()) * 0.4
    pv = player_values(w, exp_col, prod_col, pts_col, r0)
    return pv[pv["appear"] >= min_appear * len(games)]
