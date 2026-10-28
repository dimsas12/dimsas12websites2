"""Injury feed for upcoming games.

Source: the EuroLeague injury table on basketballsphere.com (robots.txt allows
crawling; it is rebuilt from club reports and dated). Each status becomes a
probability of missing the game. A manual file, data/injuries/manual.csv
(columns: player, team, status), overrides or adds entries: use it for news
the feed hasn't caught, or set status "available" to cancel a feed entry.
"""

from __future__ import annotations

import html
import logging
import re

import pandas as pd
import requests

from . import config

log = logging.getLogger(__name__)

FEED_URL = "https://basketballsphere.com/en/injuries/euroleague/"
INJURY_DIR = config.DATA_DIR / "injuries"
CACHE_PATH = INJURY_DIR / "latest.csv"
MANUAL_PATH = INJURY_DIR / "manual.csv"

# Probability that a player with this status misses the game.
STATUS_WEIGHT = {
    "out": 1.0, "suspended": 1.0, "doubtful": 0.75, "questionable": 0.5,
    "game time decision": 0.5, "gtd": 0.5, "day to day": 0.5, "probable": 0.15,
    "cleared": 0.0, "available": 0.0, "returning": 0.0,
}


def status_weight(status: str) -> float:
    s = (status or "").strip().lower()
    if s in STATUS_WEIGHT:
        return STATUS_WEIGHT[s]
    for key, w in STATUS_WEIGHT.items():
        if key in s:
            return w
    return 0.5  # unknown status: treat as a coin flip


def _cell(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _rows(table_html: str) -> list[list[str]]:
    return [[_cell(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
            for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.S)]


def parse_feed(page: str) -> pd.DataFrame:
    """Parse the club-by-club tables (complete list): Player | Status | Injury | Return.
    Falls back to the 'current injuries' summary table (newest entries only)."""
    rows = []
    # Split at each club block; clubs without injuries have no table at all.
    starts = [m.start() for m in re.finditer(r'<div class="bs-team-block bs-inj-club"', page)]
    for a, b in zip(starts, starts[1:] + [len(page)]):
        block = page[a:b]
        title = re.search(r'<h3 class="bs-team-title">(.*?)</h3>', block, re.S)
        tbl = re.search(r'<table class="bs-inj">(.*?)</table>', block, re.S)
        if not title or not tbl:
            continue
        team = _cell(title.group(1))
        for c in _rows(tbl.group(1)):
            if len(c) >= 2:
                rows.append({"player": c[0], "team": team, "status": c[1],
                             "injury": c[2] if len(c) > 2 else "",
                             "back": c[3] if len(c) > 3 else ""})
    if not rows:
        m = re.search(r'<table class="bs-inj">(.*?)</table>', page, re.S)
        if not m:
            raise ValueError("injury table not found (page layout changed?)")
        for c in _rows(m.group(1)):
            if len(c) >= 3:
                rows.append({"player": c[0], "team": c[1], "status": c[2],
                             "injury": c[3] if len(c) > 3 else "",
                             "back": c[4] if len(c) > 4 else ""})
    upd = re.search(r"updated (\d{1,2} \w+ \d{4})", page)
    df = pd.DataFrame(rows, columns=["player", "team", "status", "injury", "back"])
    df = df.replace("\u2013", "").drop_duplicates("player")
    df["source"] = "basketballsphere"
    df["updated"] = upd.group(1) if upd else ""
    return df


def fetch_feed() -> pd.DataFrame:
    """Download the feed (falls back to the last cached copy if the site is unreachable)."""
    try:
        resp = requests.get(FEED_URL, timeout=30,
                            headers={"User-Agent": "Mozilla/5.0 (elbet research model)"})
        resp.raise_for_status()
        df = parse_feed(resp.text)
        INJURY_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE_PATH, index=False)
        return df
    except (requests.RequestException, ValueError) as exc:
        if CACHE_PATH.exists():
            log.warning("injury feed unavailable (%s); using cached copy", exc)
            return pd.read_csv(CACHE_PATH).fillna("")
        raise


def load_injuries(fetch: bool = True) -> pd.DataFrame:
    """Feed + manual overrides, with a `weight` column (probability of missing)."""
    frames = []
    try:
        frames.append(fetch_feed() if fetch else pd.read_csv(CACHE_PATH).fillna(""))
    except Exception as exc:  # the model still runs without injury data
        print(f"warning: no injury feed ({exc})")
    if MANUAL_PATH.exists():
        man = pd.read_csv(MANUAL_PATH).fillna("")
        man["source"] = "manual"
        frames.append(man)
    if not frames:
        return pd.DataFrame(columns=["player", "team", "status", "weight", "source"])
    df = pd.concat(frames, ignore_index=True)
    df["key"] = df["player"].str.lower().str.strip()
    df = df.drop_duplicates("key", keep="last").drop(columns="key")  # manual wins
    df["weight"] = df["status"].map(status_weight)
    return df.reset_index(drop=True)


def absences_for_team(injuries: pd.DataFrame, team_names: str) -> dict[str, float]:
    """{player name: probability of missing} for one club ('|'-joined official names).
    Entries without a team are kept; the roster match decides where they belong."""
    from .odds import name_similarity

    out = {}
    for r in injuries.itertuples(index=False):
        if r.weight <= 0:
            continue
        if not r.team or name_similarity(str(r.team), team_names) >= 0.6:
            out[r.player] = float(r.weight)
    return out
