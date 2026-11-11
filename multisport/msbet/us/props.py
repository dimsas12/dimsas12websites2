"""Player props for the NBA and the NFL.

One engine: each player's recent games are averaged with recency weights and
shrunk toward the typical player of his role; the projection is then scaled for
the opponent and for how many points the team is expected to score. Count stats
use a negative binomial, yardage a normal whose spread grows with the mean; the
spread parameters are fitted on walk-forward residuals.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests
from scipy.stats import gamma, nbinom, norm

from . import data

NFL_STATS = ["attempts", "completions", "passing_yards", "passing_tds", "passing_interceptions",
             "carries", "rushing_yards", "rushing_tds", "receptions", "targets",
             "receiving_yards", "receiving_tds"]
NFL_YARDS = {"passing_yards", "rushing_yards", "receiving_yards"}
NBA_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov"]
NFL_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player"


# ---------------------------------------------------------------- data

def download_nfl_players(seasons=None) -> pd.DataFrame:
    year = pd.Timestamp.now().year
    seasons = seasons or [year - 2, year - 1, year]
    frames = []
    for s in seasons:
        f = data.DIR / "nfl_players" / f"{s}.csv"
        if not f.exists() or s >= year:
            r = requests.get(f"{NFL_URL}/stats_player_week_{s}.csv", timeout=120)
            if r.ok and len(r.content) > 10000:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(r.content)
        if f.exists():
            frames.append(pd.read_csv(io.BytesIO(f.read_bytes()), low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["position"].isin(["QB", "RB", "WR", "TE", "FB"])]
    keep = ["player_id", "player_display_name", "position", "season", "week", "season_type",
            "game_id", "team", "opponent_team"] + NFL_STATS
    df = df[keep].rename(columns={"player_display_name": "player", "game_id": "event_id",
                                  "team": "team_id", "opponent_team": "opp_id",
                                  "position": "pos"})
    df[NFL_STATS] = df[NFL_STATS].fillna(0.0)
    g = data.load_games("nfl")[["event_id", "date"]]
    df = df.merge(g, on="event_id", how="left").dropna(subset=["date"])
    df = df[(df["attempts"] + df["carries"] + df["targets"]) > 0]
    df.to_csv(data.DIR / "nfl_players.csv", index=False)
    return df


def nba_table() -> pd.DataFrame:
    """NBA player games with the opponent attached; DNPs dropped."""
    p = data.load_players("nba")
    p = p[~p["dnp"] & (p["min"].fillna(0) > 0)].copy()
    g = data.load_games("nba")[["event_id", "h_id", "a_id"]]
    p = p.merge(g, on="event_id")
    p["opp_id"] = np.where(p["team_id"] == p["h_id"], p["a_id"], p["h_id"])
    return p.sort_values(["date", "event_id"]).reset_index(drop=True)


# ---------------------------------------------------------------- engine

def project(hist: pd.DataFrame, stats: list[str], roster: pd.DataFrame, as_of: pd.Timestamp,
            decay: float = 0.9, prior_games: float = 3.0, window: int = 12,
            exposure: str | None = None, max_age_days: int = 400) -> pd.DataFrame:
    """Projections for the players in `roster` (player_id, team_id, pos).

    hist: one row per player-game (date, player_id, pos, stats, optional exposure
    column such as minutes). With `exposure`, stats are projected as
    rate-per-exposure x projected exposure (both recency-weighted)."""
    h = hist[(hist["date"] < as_of) & (hist["date"] >= as_of - pd.Timedelta(days=max_age_days))]
    cols = stats + ([exposure] if exposure else [])
    pos_avg = h.groupby("pos")[cols].mean()
    rows = []
    by_player = {pid: d for pid, d in h.groupby("player_id")}
    for r in roster.itertuples(index=False):
        d = by_player.get(r.player_id)
        if d is None or d.empty:
            continue
        d = d.tail(window)
        w = decay ** np.arange(len(d) - 1, -1, -1)
        pos = r.pos if r.pos in pos_avg.index else d["pos"].iloc[-1]
        row = {"player_id": r.player_id, "player": d["player"].iloc[-1], "team_id": r.team_id,
               "pos": pos, "games": len(d), "last_game": d["date"].iloc[-1]}
        if exposure:
            ex = d[exposure].to_numpy(float)
            # minutes: recent games dominate (role changes), light shrink to own average
            w_fast = 0.75 ** np.arange(len(d) - 1, -1, -1)
            row[exposure] = float(0.7 * np.average(ex, weights=w_fast) + 0.3 * np.average(
                ex, weights=w))
            prior_ex = prior_games * float(pos_avg.at[pos, exposure]) if pos in pos_avg.index \
                else 0.0
        for s in stats:
            y = d[s].to_numpy(float)
            prior = float(pos_avg.at[pos, s]) if pos in pos_avg.index else float(np.mean(y))
            if exposure:
                rate = ((w * y).sum() + prior_games * prior) / ((w * ex).sum() + prior_ex)
                row[s] = rate * row[exposure]
            else:
                row[s] = ((w * y).sum() + prior_games * prior) / (w.sum() + prior_games)
        rows.append(row)
    return pd.DataFrame(rows)


def opponent_factors(hist: pd.DataFrame, stats: list[str], as_of: pd.Timestamp,
                     games: int = 12, damp: float = 0.5) -> pd.DataFrame:
    """How much each defence allows relative to the league (1.0 = average), damped."""
    h = hist[hist["date"] < as_of]
    tot = h.groupby(["opp_id", "event_id", "date"], as_index=False)[stats].sum()
    tot = tot.sort_values("date").groupby("opp_id").tail(games)
    allowed = tot.groupby("opp_id")[stats].mean()
    league = allowed.mean()
    return 1 + damp * (allowed / league - 1)


# ---------------------------------------------------------------- distributions

NBA_ALPHA = {"pts": 0.13, "reb": 0.13, "ast": 0.20, "fg3m": 0.25, "stl": 0.15, "blk": 0.30,
             "tov": 0.15, "pra": 0.09, "pr": 0.10, "pa": 0.10, "ra": 0.11}
NFL_ALPHA = {"attempts": 0.02, "completions": 0.03, "passing_tds": 0.10,
             "passing_interceptions": 0.10, "carries": 0.10, "receptions": 0.15,
             "targets": 0.12, "rushing_tds": 0.3, "receiving_tds": 0.3}
# Yardage is right-skewed (median below the mean): gamma with sd = a * sqrt(mean),
# a fitted on 2025-26 walk-forward residuals.
NFL_SD = {"passing_yards": 5.8, "rushing_yards": 4.8, "receiving_yards": 4.9}


def p_over(mean: float, line: float, stat: str, sport: str) -> float:
    """P(stat > line)."""
    if mean <= 0:
        return 0.0
    if sport == "nfl" and stat in NFL_SD:
        a = NFL_SD[stat]
        if stat == "passing_yards":  # close to symmetric
            return float(norm.sf((line - mean) / (a * mean ** 0.5)))
        return float(gamma.sf(max(line, 0.0), mean / a ** 2, scale=a ** 2))
    alpha = (NBA_ALPHA if sport == "nba" else NFL_ALPHA).get(stat, 0.15)
    n = 1.0 / alpha
    return float(nbinom.sf(int(np.floor(line)), n, n / (n + mean)))


def fair_line(mean: float, stat: str, sport: str) -> float:
    """Half-point line closest to a 50/50 bet."""
    cands = np.arange(0.5, max(2.5, mean * 2.5), 1.0 if mean < 60 else 5.0)
    if len(cands) == 0:
        return 0.5
    ps = np.array([p_over(mean, c, stat, sport) for c in cands])
    return float(cands[np.argmin(np.abs(ps - 0.5))])
