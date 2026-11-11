"""Football availability: who is missing from the regular XI.

History: from ESPN line-ups, the value of regulars who did not play (attacking
production, minutes). Upcoming: ESPN has no injury feed, so two sources are combined:
 * the Premier League's official availability list (Fantasy Premier League API: injured,
   suspended, doubtful with the club's chance of playing), and
 * inference for every club: a regular who missed the previous match(es) is out with a
   probability learned from history by streak length, and a red card means a ban.
"""

from __future__ import annotations

import difflib

import numpy as np
import pandas as pd
import requests

from .. import avail
from . import props
from .espn_data import DOMESTIC

PARAMS = dict(exp_col="min", window=8, recent=2, min_appear=0.5)
OFFICIAL: set = set()  # (team_id, player_id) taken from the official FPL list in the last call
FPL = "https://fantasy.premierleague.com/api/bootstrap-static/"


def box() -> pd.DataFrame:
    b = props.load_lineups()
    b = b.rename(columns={"team_id": "team_id"})
    b["prod"] = b["min"]
    b["pts"] = (0.5 * b["goals"] + 0.3 * b["sot"] + 0.04 * (b["shots"] - b["sot"]).clip(lower=0)
                + 0.3 * b["assists"])
    return b[["event_id", "date", "team_id", "player_id", "player", "pos", "min", "prod", "pts",
              "red", "yellow"]].drop(columns=["pos"])


def missing_history(rebuild: bool = False) -> pd.DataFrame:
    path = props.espn.ROOT / "data" / "soccer" / "missing.pkl"
    b = box()
    old = pd.read_pickle(path) if path.exists() and not rebuild else None
    since = b["date"].max() - pd.Timedelta(days=30) if old is not None else None
    new = avail.build_missing(b, prod_col="prod", pts_col="pts", since=since, **PARAMS)
    if old is not None:
        old = old[~old["event_id"].isin(set(new["event_id"]))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_pickle(path)
    return new


def learn_streak_probs(b: pd.DataFrame, regular: int = 5, span: int = 8) -> dict[int, float]:
    """P(regular misses the next match | he missed the last k), k = 0..4."""
    hits = {k: [0, 0] for k in range(5)}
    for _, tb in b.groupby("team_id"):
        games = tb.drop_duplicates("event_id").sort_values("date")["event_id"].tolist()
        pos = {e: i for i, e in enumerate(games)}
        played = tb[tb["min"] > 0].assign(gi=lambda d: d["event_id"].map(pos))
        mat = {pid: set(d["gi"]) for pid, d in played.groupby("player_id")}
        for i in range(span + 5, len(games)):
            for pid, g in mat.items():
                if sum(1 for j in range(i - 4 - span, i - 4) if j in g) < regular:
                    continue
                k = 0
                while k < 4 and (i - 1 - k) not in g:
                    k += 1
                if k and (i - 1) in g:
                    continue
                hits[k][0] += (i not in g)
                hits[k][1] += 1
    return {k: (v[0] + 1) / (v[1] + 2) for k, v in hits.items()}


def regulars(b: pd.DataFrame, team_id: str, as_of: pd.Timestamp, window: int = 8) -> pd.DataFrame:
    """Value table of players with enough recent minutes, plus streak and last-match cards."""
    tb = b[(b["team_id"] == team_id) & (b["date"] < as_of)]
    games = tb.drop_duplicates("event_id").sort_values("date").tail(window + 4)["event_id"].tolist()
    if len(games) < 3:
        return pd.DataFrame()
    pv = avail.current_values(b, team_id, as_of, exp_col="min", prod_col="prod", pts_col="pts",
                              window=window, min_appear=0.4)
    played = tb[tb["min"] > 0]
    streak, red = {}, {}
    for pid in pv.index:
        g = set(played.loc[played["player_id"] == pid, "event_id"])
        k = 0
        for e in reversed(games):
            if e in g:
                break
            k += 1
        streak[pid] = k
    last = tb[tb["event_id"] == games[-1]]
    for r in last.itertuples(index=False):
        if r.red and r.red > 0:
            red[r.player_id] = 1
    pv["streak"] = pv.index.map(streak)
    pv["red"] = pv.index.map(lambda p: red.get(p, 0))
    return pv


def inferred_p_out(pv: pd.DataFrame, probs: dict[int, float]) -> dict[str, float]:
    out = {}
    for pid, r in pv.iterrows():
        p = probs.get(min(int(r["streak"]), 4), 0.1) if r["streak"] else 0.0
        if r["red"]:
            p = max(p, 0.95)
        if p > 0.05:
            out[pid] = float(p)
    return out


# ---------------------------------------------------------------- Premier League list

def fpl_status() -> pd.DataFrame:
    """Players not fully available per the official FPL feed: team, name, P(out)."""
    r = requests.get(FPL, timeout=40)
    r.raise_for_status()
    j = r.json()
    teams = {t["id"]: t["name"] for t in j["teams"]}
    rows = []
    for e in j["elements"]:
        if e["status"] == "a":
            continue
        if e["status"] == "u":  # left the club
            continue
        chance = e.get("chance_of_playing_next_round")
        p_out = 1.0 if e["status"] in ("i", "s", "n") else (
            1.0 - chance / 100.0 if chance is not None else 0.5)
        rows.append({"team": teams[e["team"]], "first": e["first_name"], "second": e["second_name"],
                     "web": e["web_name"], "p_out": p_out, "news": e["news"]})
    return pd.DataFrame(rows)


def _toks(name: str) -> set[str]:
    import unicodedata
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return {t for t in s.replace("-", " ").replace(".", " ").replace("'", " ").split() if len(t) > 1}


def match_team(fpl_team: str, names: dict[str, str]) -> str | None:
    """Our team id for an FPL team name (best fuzzy match on ESPN's display names)."""
    best, score = None, 0.0
    f = fpl_team.lower().replace("man ", "manchester ").replace("spurs", "tottenham")
    f = f.replace("nott'm", "nottingham").replace("wolves", "wolverhampton")
    for tid, nm in names.items():
        sc = difflib.SequenceMatcher(None, f, nm.lower()).ratio()
        if f in nm.lower() or nm.lower() in f:
            sc = max(sc, 0.9)
        if sc > score:
            best, score = tid, sc
    return best if score >= 0.6 else None


def fpl_p_out(b: pd.DataFrame, team_names: dict[str, str]) -> dict[str, dict[str, float]]:
    """{team_id: {player_id: P(out)}} for Premier League clubs from the FPL list."""
    try:
        f = fpl_status()
    except Exception:  # noqa: BLE001 - feed unavailable: inference only
        return {}
    out: dict[str, dict[str, float]] = {}
    players = b.drop_duplicates(["team_id", "player_id"], keep="last")
    for team, grp in f.groupby("team"):
        tid = match_team(team, team_names)
        if tid is None:
            continue
        roster = players[players["team_id"] == tid]
        for r in grp.itertuples(index=False):
            q = _toks(r.second) | _toks(r.web)
            for p in roster.itertuples(index=False):
                pt = _toks(p.player)
                if q and q <= pt | _toks(r.first) and (q & pt):
                    out.setdefault(tid, {})[p.player_id] = r.p_out
                    break
    return out


def upcoming_missing(up: pd.DataFrame, team_names: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    """Adds mp_h/mp_a/mm_h/mm_a to club matches; returns (frame, {team_id: {pid: p_out}})."""
    up = up.copy()
    b = box()
    probs = learn_streak_probs(b[b["date"] >= b["date"].max() - pd.Timedelta(days=900)])
    fpl = fpl_p_out(b, team_names)
    OFFICIAL.clear()
    OFFICIAL.update((t, p) for t, d in fpl.items() for p in d)
    outs: dict[str, dict[str, float]] = {}
    cols = {c: [] for c in ("mp_h", "mp_a", "mm_h", "mm_a")}
    for g in up.itertuples(index=False):
        for side, tid in (("h", g.home), ("a", g.away)):
            val = {"mp": 0.0, "mm": 0.0}
            if g.div in DOMESTIC or g.div.startswith("uefa.") and "qual" not in g.div:
                pv = regulars(b, tid, g.date)
                if len(pv):
                    p = inferred_p_out(pv, probs)
                    p.update({k: v for k, v in fpl.get(tid, {}).items()})  # official list wins
                    outs[tid] = p
                    for pid, w in p.items():
                        if pid in pv.index:
                            val["mp"] += w * pv.at[pid, "pts_value"]
                            val["mm"] += w * pv.at[pid, "value"]
            cols[f"mp_{side}"].append(val["mp"])
            cols[f"mm_{side}"].append(val["mm"])
    for c, v in cols.items():
        up[c] = v
    return up, outs


def with_missing(rated: pd.DataFrame) -> pd.DataFrame:
    """Merge the history table into the rated match table (rows keyed by home/away team id)."""
    m = missing_history()
    h = m.rename(columns={"team_id": "home", "miss_val": "mm_h", "miss_pts": "mp_h"})
    a = m.rename(columns={"team_id": "away", "miss_val": "mm_a", "miss_pts": "mp_a"})
    r = rated.merge(h[["event_id", "home", "mm_h", "mp_h"]], on=["event_id", "home"], how="left")
    return r.merge(a[["event_id", "away", "mm_a", "mp_a"]], on=["event_id", "away"], how="left")


def absence_table(outs: dict[str, dict[str, float]], team_names: dict[str, str]) -> pd.DataFrame:
    """Readable list of the players assumed absent, with the source of the information."""
    b = box().drop_duplicates(["team_id", "player_id"], keep="last").set_index(
        ["team_id", "player_id"])
    rows = []
    for tid, d in outs.items():
        for pid, w in d.items():
            nm = b.at[(tid, pid), "player"] if (tid, pid) in b.index else pid
            rows.append({"team": team_names.get(tid, tid), "player": nm, "P(out)": w,
                         "source": "official (Premier League list)" if (tid, pid) in OFFICIAL
                         else "inferred: regular who missed recent matches / red card"})
    return pd.DataFrame(rows).sort_values(["team", "P(out)"], ascending=[True, False]) \
        if rows else pd.DataFrame(columns=["team", "player", "P(out)", "source"])
