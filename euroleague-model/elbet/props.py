"""Player prop projections (points, rebounds, assists, threes, PRA).

projection = expected minutes x per-minute rate x pace factor x opponent factor

* Minutes and per-minute rates are exponentially decayed averages of the
  player's previous games (only games the player actually played, so projections are
  conditional on playing, which is how books settle props).
* Rates are shrunk toward the league per-minute rate for players with little history.
* Outcomes are negative binomial with var = mean + alpha * mean^2, alpha fitted
  per stat on historical projection errors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import nbinom

from .config import PropsConfig

PROP_STATS = ["pts", "reb", "ast", "fgm3", "pra"]
OPP_ADJUSTED = {"pts", "pra"}


@dataclass
class PlayerState:
    player: str
    team: str
    season: int
    s_min: float = 0.0
    s_n: float = 0.0
    s_stat: dict = field(default_factory=lambda: {s: 0.0 for s in PROP_STATS})
    games: int = 0
    last_date: pd.Timestamp | None = None

    def decay(self, factor: float) -> None:
        self.s_min *= factor
        self.s_n *= factor
        for s in self.s_stat:
            self.s_stat[s] *= factor

    def exp_minutes(self) -> float:
        return self.s_min / self.s_n if self.s_n > 0 else float("nan")

    def rate(self, stat: str, league_rate: float, prior_min: float) -> float:
        return (self.s_stat[stat] + prior_min * league_rate) / (self.s_min + prior_min)


def _context(feats: pd.DataFrame) -> dict:
    """Per game, per side: pace factor and opponent defence factor."""
    ctx = {}
    for r in feats.itertuples(index=False):
        if not getattr(r, "has_ratings", False) or not np.isfinite(r.pace_exp):
            ctx[r.game_id] = {True: (1.0, 1.0), False: (1.0, 1.0)}
            continue
        pace_f = r.pace_exp / r.mu_pace
        ctx[r.game_id] = {
            True: (pace_f, (r.mu_eff - r.def_away) / r.mu_eff),
            False: (pace_f, (r.mu_eff - r.def_home) / r.mu_eff),
        }
    return ctx


def _league_rates(players: pd.DataFrame) -> dict[int, dict[str, float]]:
    """Per-minute league rate for each season, computed from earlier seasons only."""
    p = players[players["min"] > 0]
    by = p.groupby("season")[["min"] + PROP_STATS].sum().sort_index()
    cum = by.cumsum().shift().bfill()
    return {int(s): {st: row[st] / row["min"] for st in PROP_STATS} for s, row in cum.iterrows()}


@dataclass
class PropsModel:
    cfg: PropsConfig = field(default_factory=PropsConfig)
    alpha: dict = field(default_factory=dict)
    league: dict = field(default_factory=dict)
    states: dict = field(default_factory=dict)

    def run(self, players: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
        """Walk through history producing a pre-game projection for every played appearance."""
        cfg = self.cfg
        players = players.copy()
        players["pra"] = players["pts"] + players["reb"] + players["ast"]
        players = players.sort_values(["date", "game_id"])
        self.league = _league_rates(players)
        ctx = _context(feats)
        self.states = {}
        rows = []
        for r in players.itertuples(index=False):
            st = self.states.get(r.player_id)
            if st is None:
                st = self.states[r.player_id] = PlayerState(r.player, r.team, r.season)
            if r.season != st.season:
                st.decay(cfg.season_decay)
                st.season = r.season
            st.team, st.player = r.team, r.player
            if r.min <= 0:
                continue
            league = self.league.get(int(r.season)) or next(iter(self.league.values()))
            pace_f, opp_f = ctx.get(r.game_id, {}).get(bool(r.is_home), (1.0, 1.0))
            if st.s_n > 0:
                exp_min = st.exp_minutes()
                row = {"game_id": r.game_id, "season": r.season, "player_id": r.player_id,
                       "player": r.player, "team": r.team, "min": r.min, "exp_min": exp_min,
                       "prior_games": st.games}
                for s in PROP_STATS:
                    f = pace_f * (opp_f if s in OPP_ADJUSTED else 1.0)
                    row[f"proj_{s}"] = exp_min * st.rate(s, league[s], cfg.prior_minutes[s]) * f
                    row[s] = getattr(r, s)
                    row[f"naive_{s}"] = st.s_stat[s] / st.s_n
                rows.append(row)
            st.decay(cfg.decay)
            st.s_min += r.min
            st.s_n += 1
            for s in PROP_STATS:
                st.s_stat[s] += getattr(r, s)
            st.games += 1
            st.last_date = r.date
        return pd.DataFrame(rows)

    def fit_dispersion(self, history: pd.DataFrame) -> None:
        h = history[history["prior_games"] >= 3]
        for s in PROP_STATS:
            m = h[f"proj_{s}"].to_numpy()
            y = h[s].to_numpy()
            ok = m > 0.3
            a = ((y[ok] - m[ok]) ** 2 - m[ok]).sum() / (m[ok] ** 2).sum()
            self.alpha[s] = float(max(a, 0.01))

    def project(self, team: str, opp_def: float, mu_eff: float, pace_f: float,
                season: int, roster: list[str]) -> pd.DataFrame:
        league = self.league.get(season) or self.league[max(self.league)]
        opp_f = (mu_eff - opp_def) / mu_eff
        rows = []
        for pid in roster:
            st = self.states.get(pid)
            if st is None or st.s_n <= 0:
                continue
            decay = self.cfg.season_decay if st.season != season else 1.0
            exp_min = st.exp_minutes()
            row = {"player_id": pid, "player": st.player, "team": team, "exp_min": exp_min}
            for s in PROP_STATS:
                f = pace_f * (opp_f if s in OPP_ADJUSTED else 1.0)
                stat_sum = st.s_stat[s] * decay
                prior = self.cfg.prior_minutes[s]
                rate = (stat_sum + prior * league[s]) / (st.s_min * decay + prior)
                row[s] = exp_min * rate * f
            rows.append(row)
        return pd.DataFrame(rows).sort_values("exp_min", ascending=False)


def nb_probs(mean: float, alpha: float, line: float) -> tuple[float, float, float]:
    """(P over, P push, P under) for a count with NB(mean, alpha)."""
    n = 1.0 / alpha
    p = n / (n + mean)
    if line == math.floor(line):
        k = int(line)
        p_under = float(nbinom.cdf(k - 1, n, p))
        p_push = float(nbinom.pmf(k, n, p))
    else:
        k = math.floor(line)
        p_under = float(nbinom.cdf(k, n, p))
        p_push = 0.0
    return 1.0 - p_under - p_push, p_push, p_under


def nb_median(mean: float, alpha: float) -> float:
    n = 1.0 / alpha
    return float(nbinom.median(n, n / (n + mean)))


def recent_roster(players: pd.DataFrame, team: str, season: int, n_games: int) -> list[str]:
    p = players[(players["team"] == team) & (players["season"] == season) & (players["min"] > 0)]
    if p.empty:
        return []
    recent = p["game_id"].drop_duplicates().iloc[-n_games:]
    return p[p["game_id"].isin(recent)]["player_id"].drop_duplicates().tolist()
