"""Build and use the football models on the unified ESPN match table."""

from __future__ import annotations

import pickle
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.stats import nbinom

from . import espn_data as E
from .model import GoalModel, RatingConfig, _fit_poisson, run_ratings

MODEL_PATH = E.ROOT / "data" / "soccer" / "model.pkl"
CORNER_CFG = RatingConfig(k=0.02, goal_weight=1.0, mu0=5.0, hfa0=0.12, new_team=0.0,
                          tier_gap=0.0, k_strength=0.004)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ESPN table -> the columns the rating code expects."""
    out = df.rename(columns={"h_id": "home", "a_id": "away", "league": "div", "as": "as_"}).copy()
    out["hxg"] = out["axg"] = np.nan
    # ESPN reports zeros when it has no match statistics (22% of games): treat as missing.
    stats = ["hs", "hst", "hc", "hf", "as_", "ast", "ac", "af"]
    empty = out[["hs", "as_", "hc", "ac"]].fillna(0).sum(axis=1) == 0
    bad = empty | (out["hc"] > 20) | (out["ac"] > 20) | (out["hs"] > 45) | (out["as_"] > 45)
    out.loc[bad, stats] = np.nan
    # Some league-years have shots but no corners (or the reverse) recorded.
    out.loc[out[["hc", "ac"]].sum(axis=1, min_count=1) == 0, ["hc", "ac"]] = np.nan
    out.loc[out[["hs", "as_"]].sum(axis=1, min_count=1) == 0, ["hs", "hst", "as_", "ast"]] = np.nan
    out["domestic"] = out["div"].isin(E.DOMESTIC)
    # ESPN seasons: use the starting year (Aug-May seasons span two calendar years).
    out["season"] = np.where(out["date"].dt.month >= 7, out["date"].dt.year,
                             out["date"].dt.year - 1)
    return out


def corner_perf(m, cfg):
    return m.hc, m.ac


def _corner_design(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(df)), np.log(df["ch0"]), np.log(df["ca0"]),
                            np.log(df["lh0"]), np.log(df["la0"])])


def rate_all(df: pd.DataFrame, cfg: RatingConfig | None = None) -> pd.DataFrame:
    """Pre-match goal and corner rates for every row."""
    m = prepare(df)
    rated, _ = run_ratings(m, cfg)
    cm = m.copy()
    cm["played"] = cm["played"] & cm["hc"].notna() & cm["ac"].notna()
    c, _ = run_ratings(cm, CORNER_CFG, corner_perf)
    rated["ch0"], rated["ca0"] = c["lh0"].to_numpy(), c["la0"].to_numpy()
    return rated


class SoccerModel:
    """Goal model + corner model fitted on rated matches."""

    def fit(self, rated: pd.DataFrame) -> "SoccerModel":
        self.goals = GoalModel().fit(rated)
        d = rated[rated["played"] & rated["hc"].notna() & (rated["n_h"] >= 6)
                  & (rated["n_a"] >= 6)]
        X = _corner_design(d)
        self.bc_h = _fit_poisson(X, d["hc"].to_numpy(float))
        self.bc_a = _fit_poisson(X[:, [0, 2, 1, 4, 3]], d["ac"].to_numpy(float))
        mu = np.exp(X @ self.bc_h) + np.exp(X[:, [0, 2, 1, 4, 3]] @ self.bc_a)
        y = (d["hc"] + d["ac"]).to_numpy(float)
        # NB dispersion of total corners by moments: var = mu + alpha mu^2
        self.corner_alpha = float(max(((y - mu) ** 2 - mu).sum() / (mu ** 2).sum(), 1e-4))
        return self

    def predict(self, rated: pd.DataFrame) -> pd.DataFrame:
        out = rated.copy()
        out["lh"], out["la"] = self.goals.rates(rated)
        X = _corner_design(rated)
        out["ch"] = np.exp(X @ self.bc_h)
        out["ca"] = np.exp(X[:, [0, 2, 1, 4, 3]] @ self.bc_a)
        return out

    def corner_probs(self, mean: float, line: float) -> tuple[float, float]:
        """(P over, P under) total corners (or a team's corners) at `line`."""
        n = 1.0 / self.corner_alpha
        p = n / (n + mean)
        k = int(np.floor(line))
        under = float(nbinom.cdf(k if line != k else k - 1, n, p))
        over = float(nbinom.sf(k, n, p))
        return over, under


def build(cfg: RatingConfig | None = None) -> dict:
    df = E.load()
    rated = rate_all(df, cfg)
    model = SoccerModel().fit(rated)
    MODEL_PATH.write_bytes(pickle.dumps(model))
    rated.to_pickle(MODEL_PATH.with_name("rated_espn.pkl"))
    return {"matches": int(rated["played"].sum()), "rho": model.goals.rho,
            "corner_alpha": model.corner_alpha}


def load_model() -> SoccerModel:
    return pickle.loads(MODEL_PATH.read_bytes())


def upcoming(days: float = 3.0, leagues=None) -> pd.DataFrame:
    rated = pd.read_pickle(MODEL_PATH.with_name("rated_espn.pkl"))
    now = pd.Timestamp.now(tz="UTC").tz_convert(None)
    up = rated[~rated["played"] & (rated["date"] > now)
               & (rated["date"] <= now + pd.Timedelta(days=days))
               & (rated["state"] == "pre")]
    if leagues:
        up = up[up["div"].isin(leagues)]
    return load_model().predict(up) if len(up) else up
