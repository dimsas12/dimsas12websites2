"""Download schedules and box scores from the public EuroLeague APIs, with a local cache."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config

log = logging.getLogger(__name__)

SCHEDULE_URL = "https://api-live.euroleague.net/v2/competitions/{comp}/seasons/{season}/games"
BOXSCORE_URL = "https://live.euroleague.net/api/Boxscore?gamecode={game}&seasoncode={season}"


def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504))
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=16))
    s.headers["User-Agent"] = "elbet/0.1 (research model)"
    return s


def schedule_path(season: int) -> Path:
    return config.RAW_DIR / "schedule" / f"{config.season_code(season)}.json"


def boxscore_path(season: int, game_code: int) -> Path:
    return config.RAW_DIR / "box" / config.season_code(season) / f"{game_code}.json"


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def fetch_schedule(season: int, session: requests.Session, refresh: bool = False) -> list[dict]:
    path = schedule_path(season)
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    url = SCHEDULE_URL.format(comp=config.COMPETITION, season=config.season_code(season))
    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    games = resp.json()["data"]
    _write_json(path, games)
    return games


def fetch_boxscore(season: int, game_code: int, session: requests.Session) -> dict | None:
    path = boxscore_path(season, game_code)
    if path.exists():
        return json.loads(path.read_text())
    url = BOXSCORE_URL.format(game=game_code, season=config.season_code(season))
    resp = session.get(url, timeout=60)
    if resp.status_code != 200 or not resp.text.strip():
        log.warning("no boxscore for %s game %s (HTTP %s)", season, game_code, resp.status_code)
        return None
    box = resp.json()
    if box.get("Live"):  # game still in progress; don't cache
        return None
    _write_json(path, box)
    return box


def fetch_seasons(seasons: list[int], workers: int = 6, refresh_current: bool = True) -> None:
    """Fetch schedules for every season and box scores for every played game."""
    session = _session()
    current = config.current_season()
    for season in seasons:
        refresh = refresh_current and season >= current
        if not refresh and not schedule_path(season).exists():
            refresh = True
        games = fetch_schedule(season, session, refresh=refresh)
        # A cached past season may be stale if it was fetched while still in progress.
        if not refresh and any(not g["played"] for g in games) and season < current:
            games = fetch_schedule(season, session, refresh=True)
        todo = [
            g["gameCode"] for g in games
            if g["played"] and not boxscore_path(season, g["gameCode"]).exists()
        ]
        played = sum(g["played"] for g in games)
        print(f"{config.season_code(season)}: {len(games)} games, {played} played, "
              f"{len(todo)} box scores to download")
        if not todo:
            continue
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(fetch_boxscore, season, code, session) for code in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    fut.result()
                except Exception as exc:  # keep going; missing boxes are tolerated downstream
                    log.warning("boxscore failed: %s", exc)
                if i % 100 == 0:
                    print(f"  {i}/{len(todo)} ({time.time() - t0:.0f}s)")
