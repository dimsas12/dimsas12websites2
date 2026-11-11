"""Points-based team ratings for NBA, NFL and college sports.

Stage 1 (online, strictly pre-game): offence and defence ratings in points,
    home = mu + hfa/2 + off[home] - dfn[away],  away = mu - hfa/2 + off[away] - dfn[home]
updated after every game toward the result (blowouts damped), regressed toward the
mean between seasons. Stage 2: a linear calibration of margin and total (plus rest)
fitted on earlier seasons; normal distributions give spread / total / win odds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class PtsConfig:
    k: float = 0.06            # learning rate
    mu0: float = 112.0         # starting points per team
    hfa0: float = 2.5          # starting home edge in points
    k_league: float = 0.004
    cap: float = 25.0          # margins beyond this count at `cap_slope`
    cap_slope: float = 0.4
    season_regress: float = 0.7
    new_team: float = 0.0      # starting off/def of unseen teams
    b2b_days: int = 1          # rest <= this many days counts as short rest


CONFIGS = {
    "nba": PtsConfig(k=0.055, mu0=112.0, hfa0=2.5, cap=25, season_regress=0.75),
    "ncaab": PtsConfig(k=0.14, mu0=71.0, hfa0=3.2, cap=22, season_regress=0.7,
                       new_team=-5.0),
    "nfl": PtsConfig(k=0.07, mu0=22.5, hfa0=1.8, cap=21, season_regress=0.65, b2b_days=5),
    "ncaaf": PtsConfig(k=0.22, mu0=27.0, hfa0=2.6, cap=28, season_regress=0.75,
                       new_team=-10.0, b2b_days=5),
}


@dataclass
class PtsRatings:
    cfg: PtsConfig
    off: dict = field(default_factory=dict)
    dfn: dict = field(default_factory=dict)
    n: dict = field(default_factory=dict)
    season: dict = field(default_factory=dict)
    last: dict = field(default_factory=dict)
    mu: float | None = None
    hfa: float | None = None

    def _enter(self, t, season):
        c = self.cfg
        if t not in self.off:
            self.off[t] = self.dfn[t] = c.new_team
            self.n[t] = 0
        elif self.season.get(t) != season:
            self.off[t] *= c.season_regress
            self.dfn[t] *= c.season_regress
        self.season[t] = season

    def expect(self, h, a, season, neutral=False) -> tuple[float, float]:
        if self.mu is None:
            self.mu, self.hfa = self.cfg.mu0, self.cfg.hfa0
        self._enter(h, season)
        self._enter(a, season)
        hf = 0.0 if neutral else self.hfa
        return (self.mu + hf / 2 + self.off[h] - self.dfn[a],
                self.mu - hf / 2 + self.off[a] - self.dfn[h])

    def update(self, h, a, eh, ea, ph, pa, date, neutral=False):
        c = self.cfg
        m, em = ph - pa, eh - ea
        over = abs(m - em) - c.cap
        if over > 0:  # damp garbage-time blowouts: shrink both errors proportionally
            shrink = (c.cap + c.cap_slope * over) / (c.cap + over)
        else:
            shrink = 1.0
        dh, da = (ph - eh) * shrink, (pa - ea) * shrink
        self.off[h] += c.k * dh
        self.dfn[a] -= c.k * dh
        self.off[a] += c.k * da
        self.dfn[h] -= c.k * da
        self.mu += c.k_league * (dh + da) / 2
        # Home edge: learn only from games between established teams (early-season
        # mismatches hosted by the stronger side would otherwise inflate it).
        if not neutral and self.n[h] >= 10 and self.n[a] >= 10:
            self.hfa += c.k_league * (dh - da) / 2
        for t in (h, a):
            self.n[t] += 1
            self.last[t] = date


def run(games: pd.DataFrame, cfg: PtsConfig) -> tuple[pd.DataFrame, PtsRatings]:
    """Adds pre-game columns eh0, ea0 (expected points), rest and games played."""
    r = PtsRatings(cfg)
    cols = {k: [] for k in ("eh0", "ea0", "rest_h", "rest_a", "n_h", "n_a")}
    for g in games.itertuples(index=False):
        neutral = bool(g.neutral)
        lh, la = r.last.get(g.h_id), r.last.get(g.a_id)
        eh, ea = r.expect(g.h_id, g.a_id, g.season, neutral)
        cols["eh0"].append(eh); cols["ea0"].append(ea)
        cols["rest_h"].append(min((g.date - lh).days, 10) if lh is not None else 5)
        cols["rest_a"].append(min((g.date - la).days, 10) if la is not None else 5)
        cols["n_h"].append(r.n[g.h_id]); cols["n_a"].append(r.n[g.a_id])
        if g.completed and g.h_pts == g.h_pts and g.a_pts == g.a_pts:
            r.update(g.h_id, g.a_id, eh, ea, g.h_pts, g.a_pts, g.date, neutral)
    out = games.copy()
    for k, v in cols.items():
        out[k] = v
    return out, r


def _X(df: pd.DataFrame, cfg: PtsConfig) -> np.ndarray:
    short_h = (df["rest_h"] <= cfg.b2b_days).astype(float)
    short_a = (df["rest_a"] <= cfg.b2b_days).astype(float)
    return np.column_stack([np.ones(len(df)), df["eh0"] - df["ea0"], df["eh0"] + df["ea0"],
                            short_h - short_a, short_h + short_a,
                            df["neutral"].astype(float)])


@dataclass
class GameModel:
    cfg: PtsConfig
    bm: np.ndarray | None = None
    bt: np.ndarray | None = None
    sd_m: float = 12.0
    sd_t: float = 18.0

    def fit(self, rated: pd.DataFrame, min_games: int = 5) -> "GameModel":
        d = rated[rated["completed"] & rated["h_pts"].notna() & (rated["n_h"] >= min_games)
                  & (rated["n_a"] >= min_games)]
        X = _X(d, self.cfg)
        m = (d["h_pts"] - d["a_pts"]).to_numpy(float)
        t = (d["h_pts"] + d["a_pts"]).to_numpy(float)
        ridge = 1e-3 * np.eye(X.shape[1])
        self.bm = np.linalg.solve(X.T @ X + ridge, X.T @ m)
        self.bt = np.linalg.solve(X.T @ X + ridge, X.T @ t)
        self.sd_m = float(np.std(m - X @ self.bm))
        self.sd_t = float(np.std(t - X @ self.bt))
        return self

    def predict(self, rated: pd.DataFrame) -> pd.DataFrame:
        X = _X(rated, self.cfg)
        out = rated.copy()
        out["pred_margin"], out["pred_total"] = X @ self.bm, X @ self.bt
        out["p_home"] = norm.cdf(out["pred_margin"] / self.sd_m)
        return out

    def p_cover(self, pred_margin, home_line) -> np.ndarray:
        """P(home covers) at a home spread (e.g. -6.5); half-point lines assumed."""
        return norm.cdf((np.asarray(pred_margin) + np.asarray(home_line)) / self.sd_m)

    def p_over(self, pred_total, line) -> np.ndarray:
        return norm.sf((np.asarray(line) - np.asarray(pred_total)) / self.sd_t)


def walk_forward(games: pd.DataFrame, cfg: PtsConfig, first_test: int) -> pd.DataFrame:
    rated, _ = run(games, cfg)
    preds = []
    for s in sorted(x for x in rated["season"].unique() if x >= first_test):
        train = rated[rated["season"] < s]
        test = rated[(rated["season"] == s) & rated["completed"]]
        if len(train) < 300 or test.empty:
            continue
        gm = GameModel(cfg).fit(train)
        preds.append(gm.predict(test).assign(sd_m=gm.sd_m, sd_t=gm.sd_t))
    return pd.concat(preds) if preds else pd.DataFrame()


def evaluate(p: pd.DataFrame, min_games: int = 5, edge_pts: float = 2.0,
             price: float = 1.909) -> dict:
    """Model vs the closing line: errors, and betting every game where the model
    disagrees with the line by `edge_pts` points at standard -110 pricing."""
    d = p[(p["n_h"] >= min_games) & (p["n_a"] >= min_games)].dropna(subset=["spread", "total"])
    m, t = d["h_pts"] - d["a_pts"], d["h_pts"] + d["a_pts"]
    out = {"n": len(d), "mae_margin_model": float((m - d["pred_margin"]).abs().mean()),
           "mae_margin_line": float((m + d["spread"]).abs().mean()),
           "mae_total_model": float((t - d["pred_total"]).abs().mean()),
           "mae_total_line": float((t - d["total"]).abs().mean()),
           "winner_acc_model": float(((d["pred_margin"] > 0) == (m > 0)).mean()),
           "winner_acc_line": float(((d["spread"] < 0) == (m > 0))[d["spread"] != 0].mean())}
    diff = d["pred_margin"] + d["spread"]           # >0: model likes the home side
    cover = m + d["spread"]
    bet = diff.abs() >= edge_pts
    win = ((diff > 0) & (cover > 0)) | ((diff < 0) & (cover < 0))
    push = cover == 0
    pnl = np.where(push, 0.0, np.where(win, price - 1, -1.0))[bet]
    out["ats_bets"], out["ats_roi"] = int(bet.sum()), float(pnl.mean()) if bet.any() else np.nan
    out["ats_win"] = float(win[bet & ~push].mean()) if bet.any() else np.nan
    dt = d["pred_total"] - d["total"]
    bet = dt.abs() >= edge_pts * 1.5
    win = ((dt > 0) & (t > d["total"])) | ((dt < 0) & (t < d["total"]))
    push = t == d["total"]
    pnl = np.where(push, 0.0, np.where(win, price - 1, -1.0))[bet]
    out["ou_bets"], out["ou_roi"] = int(bet.sum()), float(pnl.mean()) if bet.any() else np.nan
    out["ou_win"] = float(win[bet & ~push].mean()) if bet.any() else np.nan
    return out
