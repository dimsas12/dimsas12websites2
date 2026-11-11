"""The Odds API client and the sharp-price value logic shared by every sport.

Value rule (the one that is profitable in the football backtest): the fair
probability comes from the sharp books (Pinnacle and the exchanges) with the margin
removed; a bet is flagged when the best price at a book you can use beats that
fair price by a threshold. Needs ODDS_API_KEY in the environment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .betting import devig

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "history" / "odds"
API = "https://api.the-odds-api.com/v4"
SHARP = ["pinnacle", "betfair_ex_eu", "betfair_ex_uk", "matchbook", "smarkets"]
COMMISSION = {"betfair_ex_eu": 0.05, "betfair_ex_uk": 0.05, "matchbook": 0.02, "smarkets": 0.02}
EXCLUDED = {"onexbet", "gtbets", "betonlineag", "mybookieag", "everygame", "bovada", "lowvig",
            "betus", "marathonbet"}

SPORT_KEYS = {
    "eng.1": "soccer_epl", "esp.1": "soccer_spain_la_liga", "ger.1": "soccer_germany_bundesliga",
    "gre.1": "soccer_greece_super_league", "ita.1": "soccer_italy_serie_a",
    "fra.1": "soccer_france_ligue_one", "uefa.champions": "soccer_uefa_champs_league",
    "uefa.europa": "soccer_uefa_europa_league",
    "uefa.europa.conf": "soccer_uefa_europa_conference_league",
    "uefa.nations": "soccer_uefa_nations_league", "fifa.world": "soccer_fifa_world_cup",
    "uefa.euro": "soccer_uefa_european_championship",
    "nba": "basketball_nba", "ncaab": "basketball_ncaab", "nfl": "americanfootball_nfl",
    "ncaaf": "americanfootball_ncaaf",
}


def api_key() -> str:
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        raise RuntimeError("set ODDS_API_KEY in the environment (never commit it)")
    return key


def fetch(sport_key: str, markets: str = "h2h,totals,spreads", regions: str = "eu",
          event_id: str | None = None) -> tuple[list, dict]:
    """Odds for a sport (or one event, for props). Each call costs
    (#markets x #regions) credits; the remaining quota is returned."""
    path = f"/sports/{sport_key}/odds" if event_id is None \
        else f"/sports/{sport_key}/events/{event_id}/odds"
    r = requests.get(API + path, params={"apiKey": api_key(), "regions": regions,
                                         "markets": markets, "oddsFormat": "decimal"},
                     timeout=40)
    r.raise_for_status()
    data = r.json()
    events = data if isinstance(data, list) else [data]
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S")
    (ARCHIVE / f"{sport_key}_{stamp}.json").write_text(json.dumps(events))
    return events, {"remaining": r.headers.get("x-requests-remaining"),
                    "used": r.headers.get("x-requests-used")}


def flatten(events: list) -> pd.DataFrame:
    rows = []
    for e in events:
        for b in e.get("bookmakers", []):
            for m in b.get("markets", []):
                for o in m.get("outcomes", []):
                    price = float(o["price"])
                    c = COMMISSION.get(b["key"], 0.0)
                    rows.append({"event_id": e["id"], "commence": e["commence_time"],
                                 "home_team": e.get("home_team"), "away_team": e.get("away_team"),
                                 "book": b["key"], "market": m["key"], "name": o["name"],
                                 "description": o.get("description"), "point": o.get("point"),
                                 "price": price, "net": round(1 + (price - 1) * (1 - c), 4)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["commence"] = pd.to_datetime(df["commence"], utc=True).dt.tz_convert(None)
    return df


def value_table(rows: pd.DataFrame, allowed: set[str] | None = None,
                min_books: int = 3) -> pd.DataFrame:
    """One row per (event, market, line, outcome): fair probability from the sharp
    books (else the consensus of all books), best net price at a usable book, edge.

    rows: output of `flatten` (one or many events). Outcomes are grouped by
    (event, market, point, description) so each line / player is priced separately."""
    if rows.empty:
        return rows
    out = []
    r = rows.copy()
    for (eid, market, desc), g in r.groupby(["event_id", "market", r["description"].fillna("")],
                                             dropna=False):
        if market.startswith("spreads") or market.startswith("alternate_spreads"):
            # a spread market is the pair (team A at +x, team B at -x)
            groups = _spread_groups(g)
        else:
            groups = [d for _, d in g.groupby(g["point"].fillna(-999))]
        for d in groups:
            names = sorted(d["name"].unique())
            if len(names) < 2:
                continue
            full = d.groupby("book").filter(lambda x: x["name"].nunique() == len(names))
            if full["book"].nunique() < 1:
                continue
            probs = {}
            for book, b in full.groupby("book"):
                o = b.drop_duplicates("name").set_index("name")["price"].reindex(names)
                over = (1 / o).sum()
                if 0.98 <= over <= 1.20:
                    probs[book] = np.array(devig(list(o)))
            if not probs:
                continue
            sharp = [b for b in SHARP if b in probs]
            if sharp:
                fair, source = np.mean([probs[b] for b in sharp], axis=0), "+".join(sharp)
            elif len(probs) >= min_books:
                fair, source = np.median(list(probs.values()), axis=0), f"consensus({len(probs)})"
                fair = fair / fair.sum()
            else:
                continue
            usable = d[~d["book"].isin(EXCLUDED)]
            if allowed:
                usable = usable[usable["book"].isin(allowed)]
            for i, nm in enumerate(names):
                u = usable[usable["name"] == nm]
                if u.empty:
                    continue
                best = u.loc[u["net"].idxmax()]
                out.append({"event_id": eid, "commence": best["commence"],
                            "home_team": best["home_team"], "away_team": best["away_team"],
                            "market": market, "description": desc or None, "selection": nm,
                            "point": best["point"], "fair_p": float(fair[i]),
                            "fair_odds": float(1 / fair[i]), "best_odds": float(best["net"]),
                            "book": best["book"], "edge": float(fair[i] * best["net"] - 1),
                            "fair_source": source, "n_books": len(probs)})
    return pd.DataFrame(out)


def _spread_groups(g: pd.DataFrame) -> list[pd.DataFrame]:
    """Split spread quotes into consistent two-sided lines (home at h, away at -h)."""
    home = g["home_team"].iloc[0]
    g = g.assign(home_line=np.where(g["name"] == home, g["point"], -g["point"]))
    return [d for _, d in g.groupby("home_line")]


def picks(table: pd.DataFrame, min_edge: float = 0.02, max_odds: float = 8.0,
          sharp_only: bool = True) -> pd.DataFrame:
    """Value bets: best price beats the fair price by `min_edge` (backtest threshold 2%)."""
    if table.empty:
        return table
    t = table[(table["edge"] >= min_edge) & (table["best_odds"] <= max_odds)
              & (table["edge"] <= 0.25)]  # >25% over a sharp price is a stale / wrong quote
    if sharp_only:
        t = t[~t["fair_source"].str.startswith("consensus")]
    return t.sort_values("edge", ascending=False).reset_index(drop=True)
