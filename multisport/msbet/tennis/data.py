"""Tennis data: ESPN results for ATP and WTA (2020+), TML serve statistics for ATP.

ESPN gives every tour-level match with set scores, round and status, plus today's
schedule. The TML database (github.com/Tennismylife/TML-Database) adds serve
numbers for ATP matches (aces, service points won, break points).
"""

from __future__ import annotations

import io
import re
import unicodedata

import numpy as np
import pandas as pd
import requests

from .. import espn

ROOT = espn.ROOT
DIR = ROOT / "data" / "tennis"
TML = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master"

GRASS = ("wimbledon", "halle", "queen", "eastbourne", "hertogenbosch", "libema", "stuttgart|atp",
         "mallorca", "newport", "nottingham", "birmingham", "berlin", "bad homburg", "boss open",
         "terra wortmann", "cinch", "rothesay", "ilkley", "surbiton", "lexus")
CLAY = ("roland", "french open", "monte", "madrid", "rome", "italia", "barcelona", "hamburg",
        "kitzb", "gstaad", "umag", "bastad", "båstad", "estoril", "munich", "bmw open", "geneva",
        "lyon", "buenos aires", "rio", "santiago", "cordoba", "córdoba", "houston", "marrakech",
        "bucharest", "charleston", "bogota", "bogotá", "stuttgart|wta", "porsche", "strasbourg",
        "rabat", "palermo", "prague", "budapest", "iasi", "lausanne", "parma", "clarins", "lalla",
        "argentina open", "chile open", "tiriac", "swiss open", "nordea", "croatia open",
        "generali open", "paris|wta125", "saint-malo", "bari", "makarska", "valencia", "contrexeville")


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", " ", s.replace("-", " ")).split() and " ".join(
        re.sub(r"[^a-z ]", " ", s.replace("-", " ")).split()) or ""


def surface(tournament: str, date: pd.Timestamp, tour: str) -> str:
    """Surface from the tournament name, else from the calendar."""
    t = str(tournament).lower()
    for k in GRASS:
        name, _, only = k.partition("|")
        if name in t and (not only or only == tour):
            return "Grass"
    for k in CLAY:
        name, _, only = k.partition("|")
        if name in t and (not only or only.startswith(tour)):
            return "Clay"
    doy = date.dayofyear
    if 160 <= doy <= 195:
        return "Grass"
    if 95 <= doy <= 160:
        return "Clay"
    return "Hard"


def _weeks(first: str) -> list[str]:
    end = pd.Timestamp.now(tz="UTC").tz_convert(None).normalize() + pd.Timedelta(days=7)
    out, d = [], pd.Timestamp(first)
    # A single-date query returns every tournament active that day with all of its
    # matches (date ranges only return tournaments that *start* in the range and miss
    # the Grand Slams), so one Thursday per week covers the calendar.
    d += pd.Timedelta(days=(3 - d.dayofweek) % 7)
    while d <= end:
        out.append(f"{d:%Y%m%d}")
        d += pd.Timedelta(days=7)
    today = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d")
    return [x for x in out if x < today] + [today]


def parse(data: dict, tour: str) -> list[dict]:
    rows = []
    for ev in (data or {}).get("events", []):
        for g in ev.get("groupings", []):
            slug = g["grouping"]["slug"]
            if "singles" not in slug:
                continue
            tour = "wta" if "women" in slug else "atp"
            for c in g.get("competitions", []):
                comp = c.get("competitors", [])
                if len(comp) != 2 or not all("athlete" in x for x in comp):
                    continue
                if any(str(x["athlete"].get("displayName", "")).strip().lower() in (
                        "tbd", "bye", "") for x in comp):
                    continue
                st = c["status"]["type"]
                a, b = comp
                sets_a = [x.get("value") for x in a.get("linescores", [])]
                sets_b = [x.get("value") for x in b.get("linescores", [])]
                rows.append({
                    "match_id": c["id"], "tour": tour, "tournament": ev["name"],
                    "major": bool(ev.get("major")), "date": c["date"],
                    "round": (c.get("round") or {}).get("displayName"),
                    "status": st.get("name"), "state": st.get("state"),
                    "best_of": (c.get("format") or {}).get("regulation", {}).get("periods"),
                    "p1_id": a["athlete"].get("guid") or a.get("id"),
                    "p1": a["athlete"]["displayName"],
                    "p2_id": b["athlete"].get("guid") or b.get("id"),
                    "p2": b["athlete"]["displayName"],
                    "winner": 1 if a.get("winner") else 2 if b.get("winner") else 0,
                    "sets1": "-".join(str(int(x)) for x in sets_a if x is not None),
                    "sets2": "-".join(str(int(x)) for x in sets_b if x is not None)})
    return rows


def download(first: str = "2020-01-06", refresh_h: float = 3.0) -> pd.DataFrame:
    weeks = _weeks(first)
    rows = []
    for tour in ("atp", "wta"):
        for i, w in enumerate(weeks):
            age = refresh_h if i >= len(weeks) - 2 else None
            rows += parse(espn.scoreboard("tennis", tour, w, age), tour)
    df = pd.DataFrame(rows).drop_duplicates("match_id", keep="last")
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    df["surface"] = [surface(t, d, tr) for t, d, tr in zip(df["tournament"], df["date"],
                                                           df["tour"])]
    df["qualifying"] = df["round"].fillna("").str.contains("Qualif")
    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)
    DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DIR / "matches.csv", index=False)
    return df


def load() -> pd.DataFrame:
    return pd.read_csv(DIR / "matches.csv", parse_dates=["date"], dtype={"match_id": str})


def download_tml(first: int = 2015) -> pd.DataFrame:
    """ATP tour-level matches with serve statistics."""
    frames = []
    year = pd.Timestamp.now().year
    for y in list(range(first, year + 1)) + ["ongoing_tourneys"]:
        f = DIR / "tml" / f"{y}.csv"
        if not f.exists() or y in (year, "ongoing_tourneys"):
            r = requests.get(f"{TML}/{y}.csv", timeout=60)
            if r.ok and len(r.content) > 1000:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(r.content)
        if f.exists():
            frames.append(pd.read_csv(io.BytesIO(f.read_bytes()), low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates(["tourney_id", "match_num"], keep="last")
    df.to_csv(DIR / "tml_matches.csv", index=False)
    return df


def load_tml() -> pd.DataFrame:
    return pd.read_csv(DIR / "tml_matches.csv", parse_dates=["date"], low_memory=False)


ROUND_ORDER = {"Q1": 0, "Q2": 1, "Q3": 2, "RR": 3, "R128": 3, "R64": 4, "R32": 5, "R16": 6,
               "QF": 7, "SF": 8, "BR": 9, "F": 10}


def combined(espn_first: str = "2020-01-01") -> pd.DataFrame:
    """One table for rating runs: ATP from TML before `espn_first` (longer history),
    then ESPN for both tours. Players are keyed by normalised name."""
    e = load()
    e = e.assign(p1_id=e["p1"].map(norm), p2_id=e["p2"].map(norm))
    e["best_of"] = np.where((e["tour"] == "atp") & e["major"] & ~e["qualifying"], 5, 3)
    cols = ["match_id", "tour", "tournament", "date", "round", "status", "state", "best_of",
            "p1_id", "p1", "p2_id", "p2", "winner", "surface", "qualifying", "sets1", "sets2"]
    frames = [e[cols]]
    tml_path = DIR / "tml_matches.csv"
    if tml_path.exists():
        t = load_tml()
        t = t[t["date"] < pd.Timestamp(espn_first) - pd.Timedelta(days=7)]
        t = t[~t["score"].fillna("").str.contains("W/O")]
        t = t.assign(ord=t["round"].map(ROUND_ORDER).fillna(5)).sort_values(
            ["date", "tourney_id", "ord"])
        frames.insert(0, pd.DataFrame({
            "match_id": "tml" + t["tourney_id"].astype(str) + "_" + t["match_num"].astype(str),
            "tour": "atp", "tournament": t["tourney_name"],
            "date": t["date"] + pd.to_timedelta(t["ord"], unit="D"),
            "round": t["round"], "status": "STATUS_FINAL", "state": "post",
            "best_of": t["best_of"].fillna(3).astype(int),
            "p1_id": t["winner_name"].map(norm), "p1": t["winner_name"],
            "p2_id": t["loser_name"].map(norm), "p2": t["loser_name"], "winner": 1,
            "surface": t["surface"].fillna("Hard"), "qualifying": False,
            "sets1": "", "sets2": ""}))
    return pd.concat(frames, ignore_index=True).sort_values(["date", "match_id"]).reset_index(
        drop=True)
