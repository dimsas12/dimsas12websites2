"""Tennis match model: overall + surface Elo, then a point-level model for games.

Match winner: Elo with a K-factor that shrinks as a player accumulates matches
(FiveThirtyEight's tennis form), kept separately overall and per surface; the two
differences are blended. Derived markets (set betting, total games, game handicap,
tie-break, aces) come from a point-by-point Markov model whose serve probabilities
are solved so that the match-win probability equals the Elo one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import brentq


@dataclass
class EloConfig:
    k0: float = 200.0
    offset: float = 5.0
    shape: float = 0.4
    surface_weight: float = 0.3  # share of the surface rating in the blend (tuned 2021-23)
    scale: float = 0.9           # multiplies the blended difference (calibration)
    init: float = 1500.0
    new_surface_from_overall: bool = True
    absence_days: int = 120      # after a long break a player's K goes back up
    absence_boost: float = 1.5


@dataclass
class Elo:
    cfg: EloConfig = field(default_factory=EloConfig)
    r: dict = field(default_factory=dict)
    rs: dict = field(default_factory=dict)
    n: dict = field(default_factory=dict)
    ns: dict = field(default_factory=dict)
    last: dict = field(default_factory=dict)

    def get(self, pid, surface) -> tuple[float, float]:
        r = self.r.get(pid, self.cfg.init)
        return r, self.rs.get((pid, surface), r if self.cfg.new_surface_from_overall
                              else self.cfg.init)

    def diff(self, a, b, surface) -> float:
        ra, rsa = self.get(a, surface)
        rb, rsb = self.get(b, surface)
        w = self.cfg.surface_weight
        return self.cfg.scale * ((1 - w) * (ra - rb) + w * (rsa - rsb))

    def prob(self, a, b, surface) -> float:
        return 1.0 / (1.0 + 10 ** (-self.diff(a, b, surface) / 400.0))

    def _k(self, n: int) -> float:
        c = self.cfg
        return c.k0 / (n + c.offset) ** c.shape

    def update(self, winner, loser, surface, date=None) -> None:
        c = self.cfg
        rw, rsw = self.get(winner, surface)
        rl, rsl = self.get(loser, surface)
        e = 1.0 / (1.0 + 10 ** (-(rw - rl) / 400.0))
        es = 1.0 / (1.0 + 10 ** (-(rsw - rsl) / 400.0))
        for pid, sign, r0, rs0 in ((winner, 1.0, rw, rsw), (loser, -1.0, rl, rsl)):
            boost = 1.0
            if date is not None and pid in self.last and (
                    date - self.last[pid]).days > c.absence_days:
                boost = c.absence_boost
            n, ns = self.n.get(pid, 0), self.ns.get((pid, surface), 0)
            self.r[pid] = r0 + sign * boost * self._k(n) * (1 - e)
            self.rs[(pid, surface)] = rs0 + sign * boost * self._k(ns) * (1 - es)
            self.n[pid], self.ns[(pid, surface)] = n + 1, ns + 1
            if date is not None:
                self.last[pid] = date


def run(matches: pd.DataFrame, cfg: EloConfig | None = None) -> tuple[pd.DataFrame, dict]:
    """Pre-match P(player 1 wins) for every row, one Elo system per tour."""
    systems = {t: Elo(cfg or EloConfig()) for t in matches["tour"].unique()}
    p, n1, n2 = [], [], []
    for m in matches.itertuples(index=False):
        e = systems[m.tour]
        p.append(e.prob(m.p1_id, m.p2_id, m.surface))
        n1.append(e.n.get(m.p1_id, 0))
        n2.append(e.n.get(m.p2_id, 0))
        if m.winner in (1, 2) and m.status in ("STATUS_FINAL", "STATUS_RETIRED"):
            w, l = (m.p1_id, m.p2_id) if m.winner == 1 else (m.p2_id, m.p1_id)
            e.update(w, l, m.surface, m.date)
    out = matches.copy()
    out["p1_win3"], out["n1"], out["n2"] = p, n1, n2
    out["p1_win"] = out["p1_win3"]
    if "best_of" in out:
        five = (out["best_of"] == 5).to_numpy()
        out.loc[five, "p1_win"] = [best_of5(x) for x in out.loc[five, "p1_win3"]]
    return out, systems


# ---------------------------------------------------------------- point model

def hold_prob(p: float) -> float:
    """P(server holds) when he wins each service point with probability p."""
    q = 1 - p
    deuce = p * p / (p * p + q * q)
    return p ** 4 * (1 + 4 * q + 10 * q * q) + 20 * p ** 3 * q ** 3 * deuce


@lru_cache(maxsize=4096)
def _tiebreak(pa: float, pb: float) -> float:
    """P(A wins a 7-point tie-break); pa / pb = serve-point win prob of A / B."""
    from functools import lru_cache as _lc

    @_lc(maxsize=None)
    def f(a: int, b: int) -> float:
        if a >= 7 and a - b >= 2:
            return 1.0
        if b >= 7 and b - a >= 2:
            return 0.0
        if a + b >= 12 and a == b:  # from 6-6: win two in a row (one on each serve)
            w = pa * (1 - pb)
            l = (1 - pa) * pb
            return w / (w + l)
        k = a + b
        a_serves = (k == 0) or (((k - 1) // 2) % 2 == 1)
        pw = pa if a_serves else 1 - pb
        return pw * f(a + 1, b) + (1 - pw) * f(a, b + 1)
    return f(0, 0)


def set_distribution(pa: float, pb: float) -> dict:
    """{(games A, games B): prob} for a tie-break set. A serves first (the average
    over who serves first changes little)."""
    ha, hb = hold_prob(pa), hold_prob(pb)
    dist: dict = {}
    state = {(0, 0): 1.0}
    for _ in range(13):
        nxt: dict = {}
        for (a, b), pr in state.items():
            if (a >= 6 or b >= 6) and abs(a - b) >= 2 or (a, b) in ((7, 5), (5, 7)):
                dist[(a, b)] = dist.get((a, b), 0) + pr
                continue
            if a == 6 and b == 6:
                t = _tiebreak(round(pa, 4), round(pb, 4))
                dist[(7, 6)] = dist.get((7, 6), 0) + pr * t
                dist[(6, 7)] = dist.get((6, 7), 0) + pr * (1 - t)
                continue
            a_serves = (a + b) % 2 == 0
            pw = ha if a_serves else 1 - hb
            nxt[(a + 1, b)] = nxt.get((a + 1, b), 0) + pr * pw
            nxt[(a, b + 1)] = nxt.get((a, b + 1), 0) + pr * (1 - pw)
        state = nxt
    return dist


def match_markets(pa: float, pb: float, best_of: int = 3) -> dict:
    """Match probabilities from the serve-point probabilities (sets are independent)."""
    sd = set_distribution(pa, pb)
    ps = sum(v for (a, b), v in sd.items() if a > b)
    games = {}  # (games in a set) -> prob, split by who wins the set
    for (a, b), v in sd.items():
        games.setdefault(a > b, {}).setdefault((a, b), 0)
        games[a > b][(a, b)] += v
    need = best_of // 2 + 1
    scores: dict = {}

    def rec(sa, sb, pr):
        if sa == need or sb == need:
            scores[(sa, sb)] = scores.get((sa, sb), 0) + pr
            return
        rec(sa + 1, sb, pr * ps)
        rec(sa, sb + 1, pr * (1 - ps))
    rec(0, 0, 1.0)
    p_match = sum(v for (a, b), v in scores.items() if a > b)
    # distribution of total games and of (A games - B games) by convolution over sets
    won = {k: v / ps for k, v in games[True].items()}
    lost = {k: v / (1 - ps) for k, v in games[False].items()}

    def conv(d1, d2):
        out: dict = {}
        for (t1, m1), p1 in d1.items():
            for (t2, m2), p2 in d2.items():
                out[(t1 + t2, m1 + m2)] = out.get((t1 + t2, m1 + m2), 0) + p1 * p2
        return out
    w_d = {(a + b, a - b): v for (a, b), v in won.items()}
    l_d = {(a + b, a - b): v for (a, b), v in lost.items()}
    total: dict = {}
    for (sa, sb), pr in scores.items():
        d = {(0, 0): 1.0}
        for _ in range(sa):
            d = conv(d, w_d)
        for _ in range(sb):
            d = conv(d, l_d)
        for k, v in d.items():
            total[k] = total.get(k, 0) + pr * v
    tb = sd.get((7, 6), 0) + sd.get((6, 7), 0)
    n_sets = sum((a + b) * v for (a, b), v in scores.items())
    return {"p_match": p_match, "p_set": ps, "set_scores": scores, "games": total,
            "exp_games": sum(t * v for (t, _), v in total.items()),
            "p_tiebreak_set": tb, "p_any_tiebreak": 1 - (1 - tb) ** n_sets,
            "exp_sets": n_sets}


def solve_points(p_match: float, serve_sum: float, best_of: int = 3) -> tuple[float, float]:
    """Serve-point probabilities (pa, pb) with pa + pb = serve_sum that reproduce the
    match-win probability. serve_sum sets how serve-dominated the match is (ATP hard
    about 1.29, WTA about 1.13) and so the number of games and tie-breaks."""
    p_match = float(np.clip(p_match, 0.02, 0.98))

    def f(d):
        return match_markets(serve_sum / 2 + d, serve_sum / 2 - d, best_of)["p_match"] - p_match
    d = brentq(f, -0.25, 0.25, xtol=1e-4)
    return serve_sum / 2 + d, serve_sum / 2 - d


def best_of5(p3: float, serve_sum: float = 1.28) -> float:
    """Convert a best-of-3 win probability into best-of-5 with the same point skill."""
    pa, pb = solve_points(p3, serve_sum, 3)
    return match_markets(pa, pb, 5)["p_match"]


# sd (logit of match-win prob) of the true gap around the Elo estimate, chosen so the
# share of straight-set results and the average number of games match 2024-26.
SKILL_SD = {("atp", 3): 1.5, ("atp", 5): 1.9, ("wta", 3): 1.8}
_NODES = np.array([-1.645, -1.036, -0.674, -0.385, -0.126, 0.126, 0.385, 0.674, 1.036, 1.645])


def mixed_markets(p3: float, serve_sum: float, best_of: int = 3,
                  sd: float | None = None, tour: str = "atp") -> dict:
    """match_markets averaged over uncertainty in the true skill gap.

    Elo gives a noisy estimate of the gap on the day (form, fitness, match-up), so
    real matches are more one-sided than independent sets at the Elo probability
    would be: more straight-set results and fewer games. The shift distribution is
    centred so the average match-win probability stays the Elo one."""
    sd = SKILL_SD.get((tour, best_of), 1.6) if sd is None else sd
    p3 = float(np.clip(p3, 0.03, 0.97))
    base = np.log(p3 / (1 - p3))

    def mean_p(c):
        return float(np.mean(1 / (1 + np.exp(-(base + c + sd * _NODES))))) - p3
    c = brentq(mean_p, -3, 3) if sd > 0 else 0.0
    agg = {"p_match": 0.0, "p_set": 0.0, "exp_games": 0.0, "p_any_tiebreak": 0.0,
           "set_scores": {}, "games": {}}
    for z in _NODES:
        p = 1 / (1 + np.exp(-(base + c + sd * z)))
        pa, pb = solve_points(float(np.clip(p, 0.02, 0.98)), serve_sum, 3)
        mk = match_markets(round(pa, 4), round(pb, 4), best_of)
        w = 1 / len(_NODES)
        for k in ("p_match", "p_set", "exp_games", "p_any_tiebreak"):
            agg[k] += w * mk[k]
        for k in ("set_scores", "games"):
            for kk, v in mk[k].items():
                agg[k][kk] = agg[k].get(kk, 0.0) + w * v
    return agg
