"""ESPN public API client with a disk cache (scoreboards, event summaries).

One backbone for fixtures, results, team and player box scores across football,
basketball, American football and tennis. No key needed.
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "espn"
BASE = "https://site.api.espn.com/apis/site/v2/sports"
_session = requests.Session()
_session.headers["User-Agent"] = "Mozilla/5.0 (msbet)"


def _get(url: str, params: dict | None = None, retries: int = 3) -> dict | None:
    for i in range(retries):
        try:
            r = _session.get(url, params=params, timeout=40)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (400, 404):
                return None
        except (requests.RequestException, ValueError):
            pass
        time.sleep(1.5 * (i + 1))
    return None


def _cached(path: Path, fetch, max_age_h: float | None) -> dict | None:
    if path.exists() and (max_age_h is None
                          or time.time() - path.stat().st_mtime < max_age_h * 3600):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    data = fetch()
    if data is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(data, f)
    return data


def scoreboard(sport: str, league: str, dates: str, max_age_h: float | None = None,
               extra: dict | None = None) -> dict | None:
    """dates: 'YYYY' (whole year, football), 'YYYYMMDD' or 'YYYYMMDD-YYYYMMDD'."""
    params = {"dates": dates, "limit": 1000, **(extra or {})}
    tag = dates + ("_" + "_".join(f"{k}{v}" for k, v in (extra or {}).items()) if extra else "")
    path = CACHE / sport / league / f"sb_{tag}.json.gz"
    return _cached(path, lambda: _get(f"{BASE}/{sport}/{league}/scoreboard", params), max_age_h)


def summary(sport: str, league: str, event_id: str, max_age_h: float | None = None
            ) -> dict | None:
    path = CACHE / sport / league / "summary" / f"{event_id}.json.gz"

    def fetch():
        d = _get(f"{BASE}/{sport}/{league}/summary", {"event": event_id})
        if d is None:
            return None
        keep = ("boxscore", "rosters", "header", "gameInfo", "odds", "pickcenter", "injuries")
        return {k: d[k] for k in keep if k in d}
    return _cached(path, fetch, max_age_h)


CORE = "https://sports.core.api.espn.com/v2/sports"


def event_odds(sport: str, league: str, event_id: str, max_age_h: float | None = None
               ) -> dict | None:
    """Bookmaker lines ESPN stored for an event (spread, total, moneylines, open/close)."""
    path = CACHE / sport / league / "odds" / f"{event_id}.json.gz"

    def fetch():
        d = _get(f"{CORE}/{sport}/leagues/{league}/events/{event_id}/competitions/"
                 f"{event_id}/odds")
        if d is None:
            return {"items": []}
        items = []
        for x in d.get("items", []):
            if "live" in str(x.get("provider", {}).get("name", "")).lower():
                continue
            items.append({k: x.get(k) for k in ("provider", "spread", "overUnder", "overOdds",
                                                "underOdds", "homeTeamOdds", "awayTeamOdds")})
        return {"items": items}
    return _cached(path, fetch, max_age_h)
