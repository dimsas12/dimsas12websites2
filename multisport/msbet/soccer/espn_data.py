"""Football matches from ESPN: domestic leagues, European cups, national teams.

Same team ids across competitions, so one rating run can link leagues through
the cups. Each match has goals, shots, shots on target, corners and fouls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import espn

ROOT = espn.ROOT
MATCHES = ROOT / "data" / "soccer" / "espn_matches.csv"

DOMESTIC = {
    "eng.1": "Premier League", "esp.1": "La Liga", "ger.1": "Bundesliga",
    "gre.1": "Super League Greece", "ita.1": "Serie A", "fra.1": "Ligue 1",
    "ned.1": "Eredivisie", "por.1": "Primeira Liga", "bel.1": "Belgian Pro League",
    "tur.1": "Super Lig", "sco.1": "Scottish Premiership", "aut.1": "Austrian Bundesliga",
    "sui.1": "Swiss Super League", "den.1": "Danish Superliga", "nor.1": "Eliteserien",
    "swe.1": "Allsvenskan", "cze.1": "Czech League", "cro.1": "Croatian League",
    "ukr.1": "Ukrainian Premier League", "srb.1": "Serbian SuperLiga", "pol.1": "Ekstraklasa",
    "cyp.1": "Cypriot First Division", "rou.1": "Romanian Liga I", "isr.1": "Israeli Premier",
    "eng.2": "Championship",
}
CUPS = {"uefa.champions": "Champions League", "uefa.europa": "Europa League",
        "uefa.europa.conf": "Conference League",
        "uefa.champions_qual": "Champions League qualifying",
        "uefa.europa_qual": "Europa League qualifying"}
NATIONAL = {
    "fifa.world": "World Cup", "uefa.euro": "European Championship",
    "uefa.nations": "UEFA Nations League", "fifa.worldq.uefa": "World Cup qualifying UEFA",
    "uefa.euroq": "Euro qualifying", "conmebol.america": "Copa America",
    "fifa.worldq.conmebol": "World Cup qualifying CONMEBOL", "caf.nations": "Africa Cup",
    "fifa.worldq.caf": "World Cup qualifying CAF", "fifa.worldq.afc": "World Cup qualifying AFC",
    "fifa.worldq.concacaf": "World Cup qualifying CONCACAF", "concacaf.gold": "Gold Cup",
    "concacaf.nations.league": "CONCACAF Nations League", "afc.asian.cup": "Asian Cup",
    "fifa.friendly": "International friendly",
}
ALL = {**DOMESTIC, **CUPS, **NATIONAL}
STAT = {"totalShots": "s", "shotsOnTarget": "st", "wonCorners": "c", "foulsCommitted": "f"}


def kind(league: str) -> str:
    return "domestic" if league in DOMESTIC else "cup" if league in CUPS else "national"


def parse_events(data: dict, league: str) -> list[dict]:
    rows = []
    for ev in (data or {}).get("events", []):
        c = ev["competitions"][0]
        st = ev["status"]["type"]
        comp = {t["homeAway"]: t for t in c["competitors"]}
        if "home" not in comp or "away" not in comp:
            continue
        row = {"event_id": ev["id"], "league": league, "date": ev["date"],
               "season": ev.get("season", {}).get("year"),
               "state": st.get("state"), "status": st.get("name"),
               "neutral": bool(c.get("neutralSite", False))}
        for side, t in comp.items():
            p = side[0]
            row[f"{p}_id"] = t["team"]["id"]
            row[f"{p}_name"] = t["team"].get("displayName") or t["team"].get("name")
            row[f"{p}g"] = float(t["score"]) if st.get("completed") and t.get("score") not in (
                None, "") else np.nan
            stats = {s["name"]: s.get("displayValue") for s in t.get("statistics", [])}
            for k, v in STAT.items():
                row[f"{p}{v}"] = pd.to_numeric(stats.get(k), errors="coerce")
        rows.append(row)
    return rows


def download(leagues=None, first_year: int = 2017, refresh_h: float = 6.0) -> pd.DataFrame:
    """Fetch every league-year (past years cached forever) and write the match table."""
    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for lg in leagues or ALL:
        for y in range(first_year, now.year + 2):
            age = refresh_h if y >= now.year else None
            rows += parse_events(espn.scoreboard("soccer", lg, str(y), age), lg)
    df = pd.DataFrame(rows).drop_duplicates("event_id")
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    df["played"] = df["hg"].notna() & df["ag"].notna()
    df = df.sort_values(["date", "event_id"]).reset_index(drop=True)
    MATCHES.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MATCHES, index=False)
    return df


def load() -> pd.DataFrame:
    df = pd.read_csv(MATCHES, parse_dates=["date"], dtype={"event_id": str, "h_id": str,
                                                           "a_id": str})
    return df
