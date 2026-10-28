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
    dist: object = None  # PropDistribution, fitted by `elbet build`

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


def appearance_rates(players: pd.DataFrame, team: str, season: int,
                     window: int = 6) -> dict[str, float]:
    """Share of the team's last `window` games each player appeared in, counting only
    games since their first appearance for the team (so new signings aren't penalised)."""
    t = players[(players["team"] == team) & (players["season"] == season)]
    games = t["game_id"].drop_duplicates().tolist()[-window:]
    if not games:
        return {}
    idx = {g: i for i, g in enumerate(games)}
    played = t[t["min"] > 0]
    out = {}
    first_ever = played.groupby("player_id")["game_id"].first()
    all_games = t["game_id"].drop_duplicates().tolist()
    pos = {g: i for i, g in enumerate(all_games)}
    start = len(all_games) - len(games)
    for pid, d in played[played["game_id"].isin(idx)].groupby("player_id"):
        first = max(pos[first_ever[pid]], start)
        eligible = len(all_games) - first
        out[pid] = min(1.0, d["game_id"].nunique() / max(1, eligible))
    return out


def recent_roster(players: pd.DataFrame, team: str, season: int, n_games: int) -> list[str]:
    p = players[(players["team"] == team) & (players["season"] == season) & (players["min"] > 0)]
    if p.empty:
        return []
    recent = p["game_id"].drop_duplicates().iloc[-n_games:]
    return p[p["game_id"].isin(recent)]["player_id"].drop_duplicates().tolist()


# ---------------------------------------------------------------- team allocation

TEAM_MINUTES = 200.0
MAX_MINUTES = 40.0


def allocate_minutes(base_min: np.ndarray, p_play: np.ndarray, total: float = TEAM_MINUTES,
                     cap: float = MAX_MINUTES) -> np.ndarray:
    """Minutes each player gets *if they play*, so the team's expected minutes
    sum(p_play * minutes) equal `total`: everyone is scaled by one factor (capped
    at `cap` minutes), which hands absent players' minutes to the rest in
    proportion to their usual role."""
    base_min = np.asarray(base_min, float)
    p_play = np.asarray(p_play, float)
    if base_min.sum() <= 0:
        return base_min

    def expected(s: float) -> float:
        return float((p_play * np.minimum(cap, base_min * s)).sum())

    lo, hi = 0.2, 6.0
    if expected(hi) < total:  # roster too thin to fill 200 even at the cap
        return np.minimum(cap, base_min * hi)
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if expected(mid) < total else (lo, mid)
    return np.minimum(cap, base_min * hi)


def nb_sd(mean: float, alpha: float) -> float:
    return float(np.sqrt(mean + alpha * mean ** 2))


def fair_line(mean: float, alpha: float) -> tuple[float, float]:
    """Half-point line whose over probability is closest to 50%, and that probability."""
    base = math.floor(mean)
    cands = [L + 0.5 for L in range(max(0, base - 6), base + 7)]
    probs = [nb_probs(mean, alpha, L)[0] for L in cands]
    i = int(np.argmin([abs(p - 0.5) for p in probs]))
    return cands[i], probs[i]


def ladder_lines(mean: float, stat: str) -> list[float]:
    """Half-point lines around the projection, for the over/under ladder."""
    width = 6 if stat in ("pts", "pra") else 3
    base = math.floor(mean)
    return [L + 0.5 for L in range(max(0, base - width), base + width + 1)]


RATE_Z = np.array([-1.2816, -0.5244, 0.0, 0.5244, 1.2816])  # 5 normal quantiles


def mixture_probs(mean: float, alpha: float, line: float, ratios,
                  rate_sd: float = 0.0) -> tuple[float, float, float]:
    """(P over, P push, P under) for a stat whose minutes and per-minute rate are both
    uncertain: NB(mean x r x k, alpha) averaged over minutes ratios r (actual /
    projected minutes, from history) and rate multipliers k (lognormal, sd rate_sd).
    `alpha` is the stat's dispersion *given* minutes."""
    r = np.asarray(ratios, float)
    k = np.exp(rate_sd * RATE_Z - rate_sd ** 2 / 2) if rate_sd > 0 else np.ones(1)
    mu = np.maximum(mean * r[:, None] * k[None, :], 1e-3).ravel()
    n = 1.0 / alpha
    p = n / (n + mu)
    if line == math.floor(line):
        under = nbinom.cdf(int(line) - 1, n, p).mean()
        push = nbinom.pmf(int(line), n, p).mean()
    else:
        under = nbinom.cdf(math.floor(line), n, p).mean()
        push = 0.0
    return float(1 - under - push), float(push), float(under)


@dataclass
class PropDistribution:
    """Calibrated outcome model for props, fitted on historical replays
    (`elbet build`). Falls back to the plain NB if not fitted."""
    mix: dict = field(default_factory=dict)  # minutes bucket -> ratio quantiles
    alpha_cond: dict = field(default_factory=dict)
    rate_sd: float = 0.0
    alpha_plain: dict = field(default_factory=dict)
    cal: dict = field(default_factory=dict)  # stat -> (a, b): logit recalibration of P(over)

    def probs(self, mean: float, stat: str, minutes: float, line: float):
        if not self.mix:
            return nb_probs(mean, self.alpha_plain[stat], line)
        from .props_backtest import minutes_bucket

        ratios = self.mix.get(minutes_bucket(minutes)) or [1.0]
        po, pp, pu = mixture_probs(mean, self.alpha_cond[stat], line, ratios, self.rate_sd)
        ab = (getattr(self, "cal", None) or {}).get(stat)
        if ab and po > 0 and pu > 0:
            # P(over | no push) through the fitted logit curve, push probability unchanged
            q = po / (po + pu)
            z = math.log(q / (1 - q))
            q = 1 / (1 + math.exp(-(ab[0] + ab[1] * z)))
            po, pu = q * (1 - pp), (1 - q) * (1 - pp)
        return po, pp, pu

    def sd(self, mean: float, stat: str, minutes: float) -> float:
        if not self.mix:
            return nb_sd(mean, self.alpha_plain[stat])
        from .props_backtest import minutes_bucket

        r = np.asarray(self.mix.get(minutes_bucket(minutes)) or [1.0])
        k2 = np.exp(self.rate_sd ** 2)  # E[k^2] for the mean-one lognormal
        mu = mean * r
        a = self.alpha_cond[stat]
        second = (mu + a * mu ** 2 * k2 + mu ** 2 * k2).mean()  # E[X^2]
        return float(np.sqrt(max(second - mean ** 2, 0.0)))

    def fair_line(self, mean: float, stat: str, minutes: float) -> tuple[float, float]:
        base = math.floor(mean)
        cands = [L + 0.5 for L in range(max(0, base - 6), base + 7)]
        probs = [self.probs(mean, stat, minutes, L)[0] for L in cands]
        i = int(np.argmin([abs(p - 0.5) for p in probs]))
        return cands[i], probs[i]
