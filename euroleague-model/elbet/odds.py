"""Bookmaker odds from The Odds API (https://the-odds-api.com), matched to EuroLeague games.

Set ODDS_API_KEY in the environment. Live odds cost 1 credit per region per
market; historical snapshots cost 10x that and need a paid plan.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from difflib import SequenceMatcher

import pandas as pd
import requests

from . import config

BASE = "https://api.the-odds-api.com/v4"
SPORT = "basketball_euroleague"
GAME_MARKETS = "h2h,spreads,totals"
PROP_MARKETS = "player_points,player_rebounds,player_assists,player_threes," \
               "player_points_rebounds_assists"
PROP_STAT = {"player_points": "pts", "player_rebounds": "reb", "player_assists": "ast",
             "player_threes": "fgm3", "player_points_rebounds_assists": "pra"}

_STOP = {"bc", "kk", "bk", "fc", "sk", "cb", "basket", "basketball", "club", "the",
         "de", "s.a.d", "sad"}


# Bookmaker names that share no words with the official club name.
_ALIASES = {
    "red star": "crvena zvezda", "milano": "milan", "olimpia": "armani",
    "olympiakos": "olympiacos", "zvezda belgrade": "zvezda", "efes": "anadolu efes",
    "fener": "fenerbahce",
}


class OddsApiError(RuntimeError):
    pass


def api_key() -> str:
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        raise OddsApiError("set ODDS_API_KEY (free key at https://the-odds-api.com)")
    return key


def _get(path: str, **params) -> tuple[object, dict]:
    params["apiKey"] = api_key()
    resp = requests.get(f"{BASE}{path}", params=params, timeout=30)
    if resp.status_code != 200:
        raise OddsApiError(f"Odds API {resp.status_code}: {resp.text[:300]}")
    quota = {k: resp.headers.get(k) for k in ("x-requests-remaining", "x-requests-used")}
    return resp.json(), quota


def fetch_odds(regions: str = "eu", markets: str = GAME_MARKETS,
               bookmakers: str | None = None) -> tuple[list[dict], dict]:
    params = {"regions": regions, "markets": markets, "oddsFormat": "decimal"}
    if bookmakers:
        params = {"bookmakers": bookmakers, "markets": markets, "oddsFormat": "decimal"}
    events, quota = _get(f"/sports/{SPORT}/odds", **params)
    _archive("live", events)
    return events, quota


def fetch_event_props(event_id: str, regions: str = "eu,us") -> tuple[dict, dict]:
    return _get(f"/sports/{SPORT}/events/{event_id}/odds", regions=regions,
                markets=PROP_MARKETS, oddsFormat="decimal")


def fetch_historical(date_iso: str, regions: str = "eu",
                     markets: str = GAME_MARKETS) -> tuple[list[dict], dict]:
    data, quota = _get(f"/historical/sports/{SPORT}/odds", regions=regions, markets=markets,
                       oddsFormat="decimal", date=date_iso)
    events = data.get("data", []) if isinstance(data, dict) else data
    _archive(f"hist_{date_iso.replace(':', '')}", events)
    return events, quota


def _archive(tag: str, events) -> None:
    config.ODDS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S")
    name = f"{tag}_{stamp}.json" if tag == "live" else f"{tag}.json"
    (config.ODDS_DIR / name).write_text(json.dumps(events))


# ---------------------------------------------------------------- parsing & matching

def flatten(events: list[dict]) -> pd.DataFrame:
    """One row per bookmaker x market x outcome."""
    rows = []
    for ev in events:
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                for oc in mk.get("outcomes", []):
                    rows.append({
                        "event_id": ev["id"], "commence": pd.Timestamp(ev["commence_time"]),
                        "home_team": ev["home_team"], "away_team": ev["away_team"],
                        "book": bk["key"], "market": mk["key"], "name": oc["name"],
                        "description": oc.get("description"), "price": float(oc["price"]),
                        "point": oc.get("point"),
                    })
    return pd.DataFrame(rows)


def _norm(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    toks = [t for t in re.split(r"[^a-z0-9]+", name) if t and t not in _STOP]
    out = " ".join(toks)
    for alias, canon in _ALIASES.items():
        out = re.sub(rf"\b{alias}\b", canon, out)
    return out


def name_similarity(a: str, b_names: str) -> float:
    """Best similarity between an Odds API team name and any of a club's '|'-joined names."""
    na = _norm(a)
    best = 0.0
    for b in b_names.split("|"):
        nb = _norm(b)
        if not nb:
            continue
        ta, tb = set(na.split()), set(nb.split())
        overlap = len(ta & tb) / max(1, min(len(ta), len(tb)))
        best = max(best, SequenceMatcher(None, na, nb).ratio(), overlap)
    return best


def match_events(events: pd.DataFrame, games: pd.DataFrame, max_hours: float = 30) -> dict:
    """Map Odds API event_id -> (game_id, swapped) using kickoff time and team names."""
    out = {}
    evs = events.drop_duplicates("event_id")
    for ev in evs.itertuples(index=False):
        ct = ev.commence.tz_convert(None) if ev.commence.tzinfo else ev.commence
        near = games[(games["date"] - ct).abs() <= pd.Timedelta(hours=max_hours)]
        best, best_score = None, 0.0
        for g in near.itertuples(index=False):
            straight = name_similarity(ev.home_team, g.home_names) + \
                name_similarity(ev.away_team, g.away_names)
            swapped = name_similarity(ev.home_team, g.away_names) + \
                name_similarity(ev.away_team, g.home_names)
            score, sw = (straight, False) if straight >= swapped else (swapped, True)
            if score > best_score:
                best, best_score = (g.game_id, sw), score
        if best and best_score >= 1.2:
            out[ev.event_id] = best
    return out


def market_lines(rows: pd.DataFrame) -> dict:
    """Summarise one event's odds into consensus line + best price per outcome.

    Returns {"h2h": {"home": (price, book), "away": ..., "fair_home": p},
             "spreads": {"line": home_line, "home": ..., "away": ..., "fair_home": p},
             "totals": {"line": L, "over": ..., "under": ..., "fair_over": p}}
    """
    from .betting import devig

    ev = rows.iloc[0]
    out = {}
    side = lambda name: "home" if name == ev.home_team else "away"  # noqa: E731

    h2h = rows[rows["market"] == "h2h"]
    if not h2h.empty:
        h2h = h2h.assign(side=h2h["name"].map(side))
        out["h2h"] = _best_and_fair(h2h, "side", ("home", "away"), devig)

    sp = rows[rows["market"] == "spreads"].copy()
    if not sp.empty:
        sp["side"] = sp["name"].map(side)
        sp["home_line"] = sp.apply(lambda r: r["point"] if r["side"] == "home" else -r["point"],
                                   axis=1)
        line = sp["home_line"].mode().iloc[0]
        at = sp[sp["home_line"] == line]
        out["spreads"] = {"line": float(line), **_best_and_fair(at, "side", ("home", "away"),
                                                                  devig)}

    tot = rows[rows["market"] == "totals"].copy()
    if not tot.empty:
        tot["side"] = tot["name"].str.lower()
        line = tot["point"].mode().iloc[0]
        at = tot[tot["point"] == line]
        out["totals"] = {"line": float(line), **_best_and_fair(at, "side", ("over", "under"),
                                                                 devig)}
    return out


def _best_and_fair(df: pd.DataFrame, col: str, sides: tuple[str, str], devig) -> dict:
    res = {}
    for s in sides:
        d = df[df[col] == s]
        if d.empty:
            return {}
        i = d["price"].idxmax()
        res[s] = (float(d.at[i, "price"]), d.at[i, "book"])
    # Consensus no-vig probability: devig each book, then average.
    fair = []
    for _, b in df.groupby("book"):
        prices = [b[b[col] == s]["price"].max() for s in sides]
        if all(pd.notna(prices)):
            fair.append(devig(prices)[0])
    res[f"fair_{sides[0]}"] = float(sum(fair) / len(fair)) if fair else None
    res["n_books"] = len(fair)
    return res
