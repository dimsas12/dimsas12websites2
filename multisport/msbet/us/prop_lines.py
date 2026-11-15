"""Historical NBA player-prop lines (ESPN BET / DraftKings via ESPN's odds API).

ESPN keeps, for completed games, each book's player props with the closing line and
the two prices (over/under are not labelled, but the main line is the one whose two
prices are closest). Used to test the props model against real lines.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .. import espn
from . import data

TYPES = {"Total Points": "pts", "Total Rebounds": "reb", "Total Assists": "ast",
         "Total 3-Point Field Goals": "fg3m", "Total Points, Rebounds, and Assists": "pra"}


def _props(event_id: str, live: bool = False) -> dict:
    sub = "props_live" if live else "props"
    path = espn.CACHE / "basketball" / "nba" / sub / f"{event_id}.json.gz"

    def fetch():
        base = (f"{espn.CORE}/basketball/leagues/nba/events/{event_id}/competitions/"
                f"{event_id}/odds")
        d = espn._get(base)
        for x in (d or {}).get("items", []):
            pid = str((x.get("provider") or {}).get("id", ""))
            if "propBets" in x and "live" not in str(x.get("provider", {}).get("name")).lower():
                pb = espn._get(f"{base}/{pid}/propBets", {"limit": 2000})
                items = []
                for it in (pb or {}).get("items", []):
                    t = (it.get("type") or {}).get("name")
                    if t not in TYPES or "athlete" not in it:
                        continue
                    items.append({"athlete": it["athlete"]["$ref"], "type": t,
                                  "line": (it.get("odds") or {}).get("total", {}).get("value"),
                                  "open": (it.get("odds") or {}).get("total", {}).get("open"),
                                  "price": (it.get("odds") or {}).get("decimal", {}).get("value"),
                                  "book": x["provider"].get("name")})
                return {"items": items}
        return {"items": []}
    return espn._cached(path, fetch, 1.0 if live else None)


def parse(event_id: str, d: dict) -> list[dict]:
    rows = []
    df = pd.DataFrame(d.get("items", []))
    if df.empty:
        return rows
    df["player_id"] = df["athlete"].map(lambda s: re.search(r"athletes/(\d+)", s).group(1))
    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    for (pid, t), g in df.dropna(subset=["line", "price"]).groupby(["player_id", "type"]):
        best = None
        for ln, h in g.groupby("line"):
            if len(h) < 2:
                continue
            p = sorted(h["price"].tolist())[:2]
            gap = abs(1 / p[0] - 1 / p[1])
            if best is None or gap < best[0]:
                best = (gap, ln, p, h["open"].iloc[0], h["book"].iloc[0])
        if best and best[0] < 0.15:
            rows.append({"event_id": event_id, "player_id": pid, "stat": TYPES[t],
                         "line": best[1], "open_line": pd.to_numeric(best[3], errors="coerce"),
                         "price_lo": best[2][0], "price_hi": best[2][1], "book": best[4]})
    return rows


def download(first: str = "2024-10-20", workers: int = 16) -> pd.DataFrame:
    g = data.load_games("nba")
    g = g[g["completed"] & (g["date"] >= first)]
    with ThreadPoolExecutor(workers) as ex:
        parts = list(ex.map(lambda e: parse(e, _props(e)), g["event_id"]))
    df = pd.DataFrame([r for p in parts for r in p])
    df.to_csv(data.DIR / "nba_prop_lines.csv", index=False)
    return df


def load() -> pd.DataFrame:
    return pd.read_csv(data.DIR / "nba_prop_lines.csv", dtype={"event_id": str, "player_id": str})


def current(event_id: str) -> pd.DataFrame:
    """Today's DraftKings main lines (via ESPN) for an upcoming game: player_id, stat, line."""
    return pd.DataFrame(parse(event_id, _props(event_id, live=True)))
