"""Learned minutes model.

The key signal a fixed rule misses: how a team actually split its minutes the
last time it was missing the same players. For every team-game we build, for
each player on the recent roster, features known before tip-off:

* base_min: decayed average minutes when playing (decay 0.88 per game);
* last1 / avg3: minutes in the team's last game / last 3 games (0 if DNP);
* ctx_min: the player's minutes in the team's recent games weighted by how
  similar each game's absences were to tonight's (weighted overlap of the absent
  players' usual minutes); falls back to base_min when nothing is similar;
* gap: usual minutes of tonight's absent rotation players, gap_new: the part of
  it from players who played the team's last game (not yet reflected in anyone's
  recent minutes), and the player's share of the available players' minutes;
* appear: share of recent team games the player appeared in.

A gradient-boosted regressor predicts minutes given that the player plays;
a team step then rescales the available players toward the minutes the team
must fill (201.3 incl. overtime, minus what the model expects outsiders to play).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

ROTATION_MIN = 15.0
DECAY = 0.88
FEATURES = ["base_min", "last1", "avg3", "ctx_min", "ctx_weight", "gap", "gap_new",
            "share", "appear", "n_out", "team_base_sum", "is_home",
            "pos_gap", "pos_gap_new", "pos_share", "pos_avail", "height", "position",
            "exp_margin", "abs_exp_margin"]


class _Hist:
    """Per team-season list of games: {pid: minutes} for players who played."""

    def __init__(self):
        self.games: list[dict[str, float]] = []


def _similarity(absent_now: dict[str, float], played_g: dict[str, float],
                roster_g: set[str]) -> float:
    """Weighted overlap of absences: 1 when the same players (by usual minutes) were
    missing in game g as tonight, lower as the absent sets differ."""
    keys = set(absent_now) | {p for p in roster_g if p not in played_g}
    if not keys:
        return 1.0
    num = den = 0.0
    for p in keys:
        w = absent_now.get(p, 0.0) or 10.0
        a_now = p in absent_now
        a_then = p in roster_g and p not in played_g
        den += w
        num += w * (a_now == a_then)
    return num / den if den else 1.0


def build_dataset(players: pd.DataFrame, absences: dict | None = None,
                  upcoming: list[tuple] | None = None,
                  people: pd.DataFrame | None = None,
                  margins: dict | None = None) -> pd.DataFrame:
    """margins: {(game_id, team): pre-game expected margin for that team}.

    Feature rows for every roster player in every played team-game (with the
    actual minutes as target). With `upcoming` = [(game_id, team, season, is_home)]
    and `absences` = {(game_id, team): {player_id: p_out}}, also rows for those games
    (target NaN) using all history."""
    players = players.sort_values(["date", "game_id"])
    hist: dict[tuple, _Hist] = defaultdict(_Hist)
    rosters: dict[tuple, list[set]] = defaultdict(list)  # roster before each game
    state: dict[str, list[float]] = {}  # pid -> [decayed minutes, decayed games, season]
    rows = []
    pos, hgt = {}, {}
    if people is not None and not people.empty:
        pp = people.drop_duplicates(["season", "player_id"], keep="last")
        pos = {(r.season, r.player_id): r.position for r in pp.itertuples(index=False)}
        hgt = {(r.season, r.player_id): r.height for r in pp.itertuples(index=False)}

    def p_of(season, pid):
        v = pos.get((season, pid))
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = next((pos[(s2, pid)] for s2 in range(season - 1, season - 4, -1)
                      if (s2, pid) in pos), np.nan)
        return v

    def features(team, season, is_home, absent_ids: dict[str, float], played_now, gid,
                 known: bool = False):
        h = hist[(team, season)].games
        rs = rosters[(team, season)]
        roster = set().union(*[set(g) for g in h[-3:]]) if h else set()
        base = {}
        for pid in roster:
            st = state.get(pid)
            if st and st[1] > 0:
                base[pid] = st[0] / st[1]
        rot = {p for p, b in base.items() if b >= ROTATION_MIN and any(p in g for g in h[-4:])}
        # absent_ids: {pid: probability out}; usual minutes are weighted by it.
        # In history an absence is inferred from a rotation player not playing (bench
        # DNPs are coach's decisions nobody knows in advance). For upcoming games the
        # injury report is known, so any reported player's minutes count as freed.
        absent = {p: base[p] * absent_ids[p] for p in absent_ids
                  if p in base and (known or p in rot)}
        gap = sum(absent.values())
        gap_new = sum(b for p, b in absent.items() if h and p in h[-1])
        # Only ruled-out *rotation* players leave the list; a bench player who then
        # did not play stays in it (that is what the play-probability model learns).
        avail = [p for p in base if not (p in absent and absent_ids[p] >= 1.0)]
        team_base = sum(base[p] for p in avail)
        out = []
        ppos = {p: p_of(season, p) for p in base}
        for pid in avail:
            my_pos = ppos.get(pid)
            same = [p for p in absent if ppos.get(p) == my_pos]
            pos_gap = sum(absent[p] for p in same)
            pos_gap_new = sum(absent[p] for p in same if h and p in h[-1])
            pos_avail = sum(base[p] for p in avail if ppos.get(p) == my_pos)
            pos_share = base[pid] / pos_avail if pos_avail else 0.0
            mins = [g.get(pid, 0.0) for g in h]
            last1 = mins[-1] if mins else 0.0
            avg3 = float(np.mean(mins[-3:])) if mins else 0.0
            # context-weighted minutes over the last 6 team games
            num = den = 0.0
            for k, g in enumerate(h[-6:]):
                idx = len(h) - min(6, len(h)) + k
                r_g = rs[idx] if idx < len(rs) else set()
                sim = _similarity(absent, g, r_g)
                w = sim ** 3 * DECAY ** (min(6, len(h)) - 1 - k)
                num += w * g.get(pid, 0.0)
                den += w
            ctx = num / den if den > 0 else base[pid]
            window = h[-6:]
            appear = sum(pid in g for g in window) / len(window) if window else 0.5
            out.append({
                "game_id": gid, "team": team, "season": season, "player_id": pid,
                "base_min": base[pid], "last1": last1, "avg3": avg3, "ctx_min": ctx,
                "ctx_weight": den, "gap": gap, "gap_new": gap_new,
                "share": base[pid] / team_base if team_base else 0.0, "appear": appear,
                "n_out": len(absent), "team_base_sum": team_base, "is_home": float(is_home),
                "pos_gap": pos_gap, "pos_gap_new": pos_gap_new, "pos_share": pos_share,
                "pos_avail": pos_avail, "height": hgt.get((season, pid), np.nan),
                "position": my_pos,
                # Pre-game expected margin from this team's side: blowouts (either way)
                # send starters to the bench early.
                "exp_margin": (margins or {}).get((gid, team), np.nan),
                "abs_exp_margin": abs((margins or {}).get((gid, team), np.nan)),
                "played": pid in played_now if played_now is not None else np.nan,
                "min": played_now.get(pid, np.nan) if played_now is not None else np.nan,
            })
        return out

    for gid, box in players.groupby("game_id", sort=False):
        season = int(box["season"].iloc[0])
        for team, tb in box.groupby("team"):
            played_now = dict(zip(tb.loc[tb["min"] > 0, "player_id"], tb.loc[tb["min"] > 0, "min"]))
            h = hist[(team, season)].games
            roster = set().union(*[set(g) for g in h[-3:]]) if h else set()
            # absences known pre-game: rotation players who then did not play
            absent_ids = {p: 1.0 for p in roster if p not in played_now}
            rows += features(team, season, bool(tb["is_home"].iloc[0]), absent_ids,
                             played_now, gid)
        for team, tb in box.groupby("team"):
            rosters[(team, season)].append(
                set().union(*[set(g) for g in hist[(team, season)].games[-3:]])
                if hist[(team, season)].games else set())
            hist[(team, season)].games.append(
                dict(zip(tb.loc[tb["min"] > 0, "player_id"], tb.loc[tb["min"] > 0, "min"])))
        for r in box.itertuples(index=False):
            st = state.setdefault(r.player_id, [0.0, 0.0, r.season])
            if st[2] != r.season:
                st[0] *= 0.4
                st[1] *= 0.4
                st[2] = r.season
            if r.min > 0:
                st[0] = st[0] * DECAY + r.min
                st[1] = st[1] * DECAY + 1
    for gid, team, season, is_home in (upcoming or []):
        absent = (absences or {}).get((gid, team), {})
        absent_ids = {p: w for p, w in absent.items() if w > 0}
        rows += features(team, season, is_home, absent_ids, None, gid, known=True)
    return pd.DataFrame(rows)


OUTSIDER_LABEL = "Players outside the recent rotation (call-ups, returning players)"


def train(ds: pd.DataFrame):
    """Fit the minutes model on every played roster row from 2013 on."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    d = ds[(ds["season"] >= 2013) & (ds["played"] == True)]  # noqa: E712
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                      min_samples_leaf=80, random_state=0)
    return m.fit(d[FEATURES], d["min"])


PLAY_FEATURES = FEATURES + ["played_last"]


def train_play(ds: pd.DataFrame):
    """Probability that a recent-roster player who is not ruled out actually plays
    (held-out log loss 0.174 vs 0.294 for the plain appearance rate)."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    d = ds[(ds["season"] >= 2013) & ds["played"].notna()].copy()
    d["played_last"] = (d["last1"] > 0).astype(float)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=100, random_state=0)
    return clf.fit(d[PLAY_FEATURES], (d["played"] == True).astype(int))  # noqa: E712
