"""Football player props from ESPN match rosters.

Per-90 rates (goals, shots, shots on target, assists, fouls, cards, saves) from each
player's recent matches, shrunk toward the average of his position, then scaled by
the team's expected goals in this match. Minutes are estimated from the line-up
data (ESPN gives starter / subbed in / subbed out, not minutes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from .. import espn

STATS = {"totalGoals": "goals", "totalShots": "shots", "shotsOnTarget": "sot",
         "goalAssists": "assists", "foulsCommitted": "fouls", "yellowCards": "yellow",
         "redCards": "red", "saves": "saves"}
MIN_FULL, MIN_SUB_OFF, MIN_SUB_ON = 93.0, 68.0, 24.0  # average minutes by line-up role
PRIOR_MIN = {"goals": 900, "assists": 900, "shots": 360, "sot": 540, "fouls": 360,
             "yellow": 900, "red": 3000, "saves": 360}  # minutes of position average mixed in
ATTACK = ("goals", "shots", "sot", "assists")  # scale with the team's expected goals
# Typical per-90 rates by position in top European leagues (used as the prior).
POS_PER90 = {
    "F": {"goals": 0.40, "shots": 2.6, "sot": 1.05, "assists": 0.15, "fouls": 1.1,
          "yellow": 0.12, "red": 0.004, "saves": 0.0},
    "M": {"goals": 0.12, "shots": 1.3, "sot": 0.45, "assists": 0.12, "fouls": 1.2,
          "yellow": 0.17, "red": 0.006, "saves": 0.0},
    "D": {"goals": 0.04, "shots": 0.6, "sot": 0.18, "assists": 0.06, "fouls": 1.1,
          "yellow": 0.18, "red": 0.008, "saves": 0.0},
    "G": {"goals": 0.0, "shots": 0.0, "sot": 0.0, "assists": 0.005, "fouls": 0.05,
          "yellow": 0.04, "red": 0.003, "saves": 3.0},
}
DISPERSION = {"shots": 0.25, "sot": 0.20, "fouls": 0.20, "saves": 0.10}  # NB alpha


def match_players(league: str, event_id: str) -> pd.DataFrame:
    s = espn.summary("soccer", league, event_id)
    rows = []
    for team in (s or {}).get("rosters", []):
        tid = team.get("team", {}).get("id")
        for p in team.get("roster", []):
            st = {x["name"]: x.get("value", 0.0) for x in p.get("stats", [])}
            starter, on, off = bool(p.get("starter")), bool(p.get("subbedIn")), bool(
                p.get("subbedOut"))
            if not starter and not on:
                mins = 0.0
            elif starter:
                mins = MIN_SUB_OFF if off else MIN_FULL
            else:
                mins = MIN_SUB_ON
            row = {"event_id": event_id, "team_id": tid, "player_id": p["athlete"]["id"],
                   "player": p["athlete"].get("displayName"),
                   "pos": (p.get("position") or {}).get("abbreviation", "?"),
                   "starter": starter, "min": mins}
            for k, v in STATS.items():
                row[v] = float(st.get(k, 0.0) or 0.0)
            rows.append(row)
    return pd.DataFrame(rows)


def pos_group(pos: str) -> str:
    p = (pos or "?").upper()
    if p.startswith("G"):
        return "G"
    if p.startswith("F") or p in ("ST", "CF", "LW", "RW", "S"):
        return "F"
    if "M" in p or p in ("AM", "DM"):
        return "M"
    if "D" in p or p in ("CB", "LB", "RB"):
        return "D"
    return "M"


def team_history(matches: pd.DataFrame, team_id: str, before: pd.Timestamp, n: int = 10
                 ) -> pd.DataFrame:
    """Player rows of a team's last `n` played matches before a date (any competition)."""
    m = matches[matches["played"] & (matches["date"] < before)
                & ((matches["home"] == team_id) | (matches["away"] == team_id))].tail(n)
    frames = []
    for r in m.itertuples(index=False):
        d = match_players(r.div, r.event_id)
        if d.empty:
            continue
        d = d[d["team_id"] == team_id].assign(
            date=r.date, team_goals=r.hg if r.home == team_id else r.ag)
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def project(hist: pd.DataFrame, team_lambda: float, pos_avg: dict | None = None,
            decay: float = 0.9) -> pd.DataFrame:
    """Per-player expected stats if he starts (about 80 minutes)."""
    if hist.empty:
        return hist
    games = hist.drop_duplicates("event_id").sort_values("date")
    age = {e: i for i, e in enumerate(games["event_id"].iloc[::-1])}
    h = hist.assign(w=hist["event_id"].map(lambda e: decay ** age[e]),
                    grp=hist["pos"].map(pos_group))
    team_avg_goals = float(np.average(games["team_goals"],
                                      weights=[decay ** age[e] for e in games["event_id"]]))
    scale = team_lambda / max(team_avg_goals, 0.4)
    scale = float(np.clip(scale, 0.6, 1.6))
    stats = list(STATS.values())
    # position averages per minute within this team sample (or supplied league averages)
    pavg = pos_avg or {g: {k: v / 90.0 for k, v in d.items()} for g, d in POS_PER90.items()}
    # Goals are noisy: learn from half goals, half shot-based xG.
    h = h.assign(goals=0.5 * h["goals"] + 0.5 * (0.30 * h["sot"]
                                                  + 0.04 * (h["shots"] - h["sot"]).clip(lower=0)))
    last3 = set(games["event_id"].tail(3))
    rows = []
    for pid, d in h.groupby("player_id"):
        pl = d[d["min"] > 0]
        if pl.empty or not (set(d["event_id"]) & last3):
            continue
        wmin = float((pl["w"] * pl["min"]).sum())
        grp = pl["grp"].iloc[-1]
        row = {"player_id": pid, "player": pl["player"].iloc[-1], "pos": grp,
               "games": len(pl), "starts": int(pl["starter"].sum()),
               "p_start": float((d["w"] * d["starter"]).sum() / d["w"].sum()),
               "avg_min": float(pl["min"].mean())}
        exp_min = 80.0
        for s in stats:
            prior = PRIOR_MIN[s]
            rate = ((pl["w"] * pl[s]).sum() + prior * pavg.get(grp, {}).get(s, 0.0)) / (
                wmin + prior)
            row[s] = rate * exp_min * (scale if s in ATTACK else 1.0)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("p_start", ascending=False).reset_index(drop=True)
    # Team consistency: the likely starting XI scores about 85% of the team's goals
    # (substitutes ~12%, own goals ~3%) and provides assists for about 68% of them.
    xi = out.head(11)
    for stat, share in (("goals", 0.85), ("assists", 0.68)):
        tot = float(xi[stat].sum())
        if tot > 0:
            out[stat] *= float(np.clip(share * team_lambda / tot, 0.5, 2.5))
    return out


def p_at_least(mean: float, k: int, stat: str) -> float:
    """P(stat >= k) for a player with expected value `mean`."""
    if mean <= 0:
        return 0.0
    a = DISPERSION.get(stat)
    if a:
        n = 1.0 / a
        return float(nbinom.sf(k - 1, n, n / (n + mean)))
    return float(poisson.sf(k - 1, mean))


def prop_table(proj: pd.DataFrame) -> pd.DataFrame:
    """Probabilities and fair odds for the usual player markets (if he starts)."""
    rows = []
    for r in proj.itertuples(index=False):
        base = {"player": r.player, "pos": r.pos, "p_start": round(r.p_start, 2),
                "starts": f"{r.starts}/{r.games}"}
        base["xG"] = round(r.goals, 2)
        base["scorer"] = p_at_least(r.goals, 1, "goals")
        base["2+ goals"] = p_at_least(r.goals, 2, "goals")
        base["assist"] = p_at_least(r.assists, 1, "assists")
        base["shots"] = round(r.shots, 2)
        for k in (1, 2, 3):
            base[f"shots {k}+"] = p_at_least(r.shots, k, "shots")
        base["sot"] = round(r.sot, 2)
        for k in (1, 2):
            base[f"sot {k}+"] = p_at_least(r.sot, k, "sot")
        base["fouls"] = round(r.fouls, 2)
        for k in (1, 2):
            base[f"fouls {k}+"] = p_at_least(r.fouls, k, "fouls")
        base["card"] = p_at_least(r.yellow + r.red, 1, "yellow")
        if r.pos == "G":
            base["saves"] = round(r.saves, 2)
        rows.append(base)
    return pd.DataFrame(rows)
