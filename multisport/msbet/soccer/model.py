"""Team ratings and score distribution for football.

Stage 1 - online attack/defence ratings (one pass, strictly pre-match):
    lambda_home = exp(mu_div + hfa_div + att[home] - dfn[away])
    lambda_away = exp(mu_div           + att[away] - dfn[home])
  After each match the ratings move toward the observed performance, a blend of
  goals and shot-based expected goals (shots on target are a steadier signal of
  strength than goals). Ratings regress toward the league mean between seasons;
  promoted / new teams start below average.
Stage 2 - calibration: a Poisson regression maps the stage-1 rates (plus rest days)
  to goals, fitted only on earlier seasons; Dixon-Coles rho fixes the low scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10
SECOND_TIER = {"E1": "E0", "eng.2": "eng.1"}  # lower division -> its top division


@dataclass
class RatingConfig:
    k: float = 0.03           # learning rate of attack/defence (tuned on 2015-19)
    goal_weight: float = 0.35  # weight of actual goals vs shot-based xG in performance
    sot_value: float = 0.30   # goals per shot on target (fitted on 2012-18)
    off_value: float = 0.0    # goals per shot off target (regression: ~0)
    season_regress: float = 0.80  # share of a rating kept across the summer
    new_team: float = -0.18   # starting attack/defence of promoted / unseen teams
    tier_gap: float = 0.33    # attack/defence shift when moving between E1 and E0
    k_league: float = 0.004   # learning rate of league goal level and home edge
    k_strength: float = 0.012  # learning rate of league strength (from cross-league games)
    mu0: float = 1.30         # starting goals per team per match
    hfa0: float = 0.25        # starting home edge (log scale)


@dataclass
class Ratings:
    cfg: RatingConfig = field(default_factory=RatingConfig)
    att: dict = field(default_factory=dict)
    dfn: dict = field(default_factory=dict)
    mu: dict = field(default_factory=dict)
    hfa: dict = field(default_factory=dict)
    last_div: dict = field(default_factory=dict)
    last_season: dict = field(default_factory=dict)
    last_date: dict = field(default_factory=dict)
    n: dict = field(default_factory=dict)
    dom: dict = field(default_factory=dict)       # team -> its domestic league / pool
    strength: dict = field(default_factory=dict)  # league -> strength (cross-league games)

    def _enter(self, team: str, div: str, season: int, domestic: bool = True) -> None:
        c = self.cfg
        if team not in self.att:
            self.att[team] = self.dfn[team] = c.new_team
            self.n[team] = 0
            self.dom[team] = div if domestic else "minor"
            if not domestic:  # cup-only club from an unlisted league
                self.last_div[team], self.last_season[team] = div, season
            self.strength.setdefault(self.dom[team], 0.0)
        if not domestic:
            if self.last_season[team] != season and self.dom[team] == "minor":
                self.att[team] *= c.season_regress
                self.dfn[team] *= c.season_regress
                self.last_season[team] = season
            return
        if self.n[team] == 0 and team not in self.last_div:
            pass
        else:
            if self.last_season[team] != season:
                self.att[team] *= c.season_regress
                self.dfn[team] *= c.season_regress
            prev = self.last_div[team]
            if prev != div:
                if SECOND_TIER.get(prev) == div:      # promoted
                    self.att[team] -= c.tier_gap
                    self.dfn[team] -= c.tier_gap
                elif SECOND_TIER.get(div) == prev:    # relegated
                    self.att[team] += c.tier_gap
                    self.dfn[team] += c.tier_gap
        self.last_div[team], self.last_season[team] = div, season
        self.dom[team] = div
        self.strength.setdefault(div, 0.0)

    def rates(self, home: str, away: str, div: str, season: int, neutral: bool = False,
              domestic: bool = True) -> tuple[float, float]:
        self._enter(home, div, season, domestic)
        self._enter(away, div, season, domestic)
        mu = self.mu.setdefault(div, np.log(self.cfg.mu0))
        hfa = 0.0 if neutral else self.hfa.setdefault(div, self.cfg.hfa0)
        gap = self.strength[self.dom[home]] - self.strength[self.dom[away]]
        lh = np.exp(mu + hfa + self.att[home] - self.dfn[away] + gap)
        la = np.exp(mu + self.att[away] - self.dfn[home] - gap)
        return float(lh), float(la)

    def update(self, home, away, div, lh, la, perf_h, perf_a, date, neutral=False) -> None:
        c = self.cfg
        eh, ea = perf_h - lh, perf_a - la
        dh, da = self.dom[home], self.dom[away]
        if dh != da:
            self.strength[dh] += c.k_strength * (eh - ea)
            self.strength[da] -= c.k_strength * (eh - ea)
        self.att[home] += c.k * eh
        self.dfn[away] -= c.k * eh
        self.att[away] += c.k * ea
        self.dfn[home] -= c.k * ea
        self.mu[div] += c.k_league * (eh + ea) / 2
        if not neutral:
            self.hfa[div] += c.k_league * (eh - ea) / 2
        for t in (home, away):
            self.n[t] += 1
            self.last_date[t] = date


def performance(m, cfg: RatingConfig) -> tuple[float, float]:
    """Blend of goals and expected goals (real xG if present, else from shots)."""
    def xg(goals, real, shots, sot):
        if real == real:
            return real
        if sot == sot and shots == shots:
            return cfg.sot_value * sot + cfg.off_value * max(shots - sot, 0.0)
        return goals
    xh, xa = xg(m.hg, m.hxg, m.hs, m.hst), xg(m.ag, m.axg, m.as_, m.ast)
    w = cfg.goal_weight
    return w * m.hg + (1 - w) * xh, w * m.ag + (1 - w) * xa


def run_ratings(matches: pd.DataFrame, cfg: RatingConfig | None = None, perf=None
                ) -> tuple[pd.DataFrame, Ratings]:
    """Pre-match stage-1 rates for every row (date order); unplayed rows get rates too.

    Optional columns: `neutral` (no home edge) and `domestic` (False for cup /
    international games, which do not change a club's home league). `perf(m, cfg)`
    returns the (home, away) performance to learn from; default goals + shot xG."""
    r = Ratings(cfg or RatingConfig())
    perf = perf or performance
    has_n, has_d = "neutral" in matches, "domestic" in matches
    lh_, la_, rh_, ra_, nh_, na_ = [], [], [], [], [], []
    for m in matches.itertuples(index=False):
        dh, da = r.last_date.get(m.home), r.last_date.get(m.away)
        neutral = bool(m.neutral) if has_n else False
        lh, la = r.rates(m.home, m.away, m.div, m.season, neutral,
                         bool(m.domestic) if has_d else True)
        lh_.append(lh); la_.append(la)
        rh_.append(min((m.date - dh).days, 14) if dh is not None else 7)
        ra_.append(min((m.date - da).days, 14) if da is not None else 7)
        nh_.append(r.n[m.home]); na_.append(r.n[m.away])
        if m.played:
            ph, pa = perf(m, r.cfg)
            if ph == ph and pa == pa:
                r.update(m.home, m.away, m.div, lh, la, ph, pa, m.date, neutral)
    out = matches.copy()
    out["lh0"], out["la0"], out["rest_h"], out["rest_a"] = lh_, la_, rh_, ra_
    out["n_h"], out["n_a"] = nh_, na_
    return out, r


# ---------------------------------------------------------------- stage 2

AVAIL_COLS = ("mp_h", "mp_a", "mm_h", "mm_a")  # missing attacking value / minutes, home & away


def _design(df: pd.DataFrame, avail: bool = False) -> np.ndarray:
    lh, la = np.log(df["lh0"].to_numpy()), np.log(df["la0"].to_numpy())
    short_h = (df["rest_h"].to_numpy() <= 3).astype(float)
    short_a = (df["rest_a"].to_numpy() <= 3).astype(float)
    one = np.ones(len(df))
    cols = [one, lh, la, short_h, short_a]
    if avail:  # value of regulars missing, scaled to about one per missing key player
        v = {c: (df[c].fillna(0.0).to_numpy(float) if c in df else np.zeros(len(df)))
             for c in AVAIL_COLS}
        cols += [v["mp_h"] / 3.0, v["mp_a"] / 3.0, v["mm_h"] / 60.0, v["mm_a"] / 60.0]
    return np.column_stack(cols)


def _swap(X: np.ndarray) -> np.ndarray:
    """The same design seen from the away side (home/away columns exchanged)."""
    idx = [0, 2, 1, 4, 3] + ([6, 5, 8, 7] if X.shape[1] > 5 else [])
    return X[:, idx]


def _fit_poisson(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    b = np.zeros(X.shape[1])
    for _ in range(50):  # Newton iterations
        mu = np.exp(X @ b)
        g = X.T @ (y - mu) - l2 * b * (np.arange(len(b)) > 0)
        H = X.T @ (X * mu[:, None]) + l2 * np.diag((np.arange(len(b)) > 0).astype(float))
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def score_matrix(lh: float, la: float, rho: float = 0.0, n: int = MAX_GOALS) -> np.ndarray:
    """P(home goals = i, away goals = j) with the Dixon-Coles low-score correction."""
    i = np.arange(n + 1)
    M = np.outer(poisson.pmf(i, lh), poisson.pmf(i, la))
    if rho:
        M[0, 0] *= 1 - lh * la * rho
        M[0, 1] *= 1 + lh * rho
        M[1, 0] *= 1 + la * rho
        M[1, 1] *= 1 - rho
    return M / M.sum()


@dataclass
class GoalModel:
    bh: np.ndarray | None = None
    ba: np.ndarray | None = None
    rho: float = -0.05
    avail: bool = False

    def fit(self, df: pd.DataFrame, use_avail: bool = True) -> "GoalModel":
        df = df[df["played"] & (df["n_h"] >= 6) & (df["n_a"] >= 6)]
        self.avail = bool(use_avail and "mp_h" in df and df["mp_h"].notna().sum() > 3000)
        if self.avail:  # only matches with line-up data carry the availability features
            df = df[df["mp_h"].notna() & df["mp_a"].notna()]
        X = _design(df, self.avail)
        self.bh = _fit_poisson(X, df["hg"].to_numpy(float))
        self.ba = _fit_poisson(_swap(X), df["ag"].to_numpy(float))
        lh, la = self.rates(df)
        hg, ag = df["hg"].to_numpy(int), df["ag"].to_numpy(int)

        def nll(rho):
            t = np.ones(len(df))
            t = np.where((hg == 0) & (ag == 0), 1 - lh * la * rho, t)
            t = np.where((hg == 0) & (ag == 1), 1 + lh * rho, t)
            t = np.where((hg == 1) & (ag == 0), 1 + la * rho, t)
            t = np.where((hg == 1) & (ag == 1), 1 - rho, t)
            return -np.log(np.clip(t, 1e-9, None)).sum()
        self.rho = float(minimize(lambda r: nll(r[0]), [-0.05], bounds=[(-0.2, 0.1)]).x[0])
        return self

    def rates(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = _design(df, self.avail)
        return np.exp(X @ self.bh), np.exp(_swap(X) @ self.ba)


def market_probs(lh: float, la: float, rho: float = 0.0, ah_line: float | None = None,
                 ou_line: float = 2.5) -> dict:
    """Probabilities for the main markets from the two goal rates."""
    M = score_matrix(lh, la, rho)
    i = np.arange(M.shape[0])
    diff = i[:, None] - i[None, :]
    tot = i[:, None] + i[None, :]
    out = {"p_h": float(M[diff > 0].sum()), "p_d": float(M[diff == 0].sum()),
           "p_a": float(M[diff < 0].sum()),
           "p_over": float(M[tot > ou_line].sum()), "p_under": float(M[tot < ou_line].sum()),
           "p_btts": float(M[1:, 1:].sum())}
    if ah_line is not None and ah_line == ah_line:
        out["ah_h"], out["ah_a"] = ah_value(M, ah_line)
    return out


def ah_value(M: np.ndarray, line: float) -> tuple[tuple, tuple]:
    """Asian handicap on the home team at `line` (e.g. -0.75): for each side
    (P win, P half win, P push, P half loss, P loss) so EV can be computed."""
    i = np.arange(M.shape[0])
    diff = (i[:, None] - i[None, :]).astype(float)

    def side(d, ln):
        parts = [ln] if (ln * 2) == int(ln * 2) else [ln - 0.25, ln + 0.25]
        res = np.zeros(5)
        for p in parts:
            adj = d + p
            w, pu, lo = M[adj > 0].sum(), M[adj == 0].sum(), M[adj < 0].sum()
            if len(parts) == 1:
                res += np.array([w, 0, pu, 0, lo])
            else:  # half stake on each quarter line
                res += 0.5 * np.array([w, 0, pu, 0, lo])
        if len(parts) == 2:
            # outcomes where exactly one half wins/pushes -> half win / half loss
            a1, a2 = d + parts[0], d + parts[1]
            win = M[(a1 > 0) & (a2 > 0)].sum()
            hw = M[((a1 == 0) & (a2 > 0)) | ((a1 > 0) & (a2 == 0))].sum()
            hl = M[((a1 == 0) & (a2 < 0)) | ((a1 < 0) & (a2 == 0))].sum()
            lose = M[(a1 < 0) & (a2 < 0)].sum()
            res = np.array([win, hw, 0.0, hl, lose])
        return tuple(float(x) for x in res)
    return side(diff, line), side(-diff, -line)


def ah_ev(parts: tuple, odds: float) -> float:
    w, hw, pu, hl, lo = parts
    return w * (odds - 1) + hw * (odds - 1) / 2 - hl / 2 - lo
