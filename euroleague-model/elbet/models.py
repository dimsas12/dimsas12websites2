"""Game model: predicts margin and total, and turns them into market probabilities.

Margin and total each start from the efficiency-rating projection (an offset)
and a ridge regression on the pre-game features corrects it; heavy shrinkage
means that with little evidence the model falls back to the rating projection.
Errors are normally distributed whose spread is the out-of-sample residual SD. The
normal is discretised onto integer scores so spreads and totals on whole
numbers get a proper push probability, and a margin of 0 (impossible in
basketball: overtime decides) is removed.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .features import MARGIN_FEATURES, TOTAL_FEATURES


@dataclass
class Ridge:
    alpha: float = 1.0
    coef_: np.ndarray | None = None
    intercept_: float = 0.0
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> "Ridge":
        w = np.ones(len(y)) if w is None else w
        self.mean_ = np.average(X, axis=0, weights=w)
        self.scale_ = np.sqrt(np.average((X - self.mean_) ** 2, axis=0, weights=w))
        self.scale_[self.scale_ == 0] = 1.0
        Z = (X - self.mean_) / self.scale_
        ym = np.average(y, weights=w)
        A = Z.T @ (Z * w[:, None]) + self.alpha * np.eye(Z.shape[1])
        self.coef_ = np.linalg.solve(A, Z.T @ (w * (y - ym)))
        self.intercept_ = ym
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.intercept_ + ((X - self.mean_) / self.scale_) @ self.coef_

    def raw_coefficients(self, names: list[str]) -> dict[str, float]:
        coefs = self.coef_ / self.scale_
        out = {"intercept": float(self.intercept_ - (self.mean_ * coefs).sum())}
        out.update({n: float(c) for n, c in zip(names, coefs)})
        return out


@dataclass
class GameModel:
    margin_features: list[str] = field(default_factory=lambda: list(MARGIN_FEATURES))
    total_features: list[str] = field(default_factory=lambda: list(TOTAL_FEATURES))
    alpha: float = 50.0
    margin_offset: str = "rating_margin"
    total_offset: str = "rating_total"
    season_decay: float = 0.8  # weight older training seasons less
    margin_model: Ridge | None = None
    total_model: Ridge | None = None
    margin_sd: float = 12.5
    total_sd: float = 17.0
    trained_through: int | None = None

    def fit(self, feats: pd.DataFrame) -> "GameModel":
        df = feats[feats["played"] & feats["has_ratings"]]
        if len(df) < 100:
            raise ValueError(f"not enough training games ({len(df)})")
        last = int(df["season"].max())
        w = self.season_decay ** (last - df["season"].to_numpy())
        self.margin_model = Ridge(self.alpha).fit(
            df[self.margin_features].to_numpy(float),
            (df["margin"] - df[self.margin_offset]).to_numpy(float), w)
        self.total_model = Ridge(self.alpha).fit(
            df[self.total_features].to_numpy(float),
            (df["total"] - df[self.total_offset]).to_numpy(float), w)
        pred = self.predict(df)
        res_m = df["margin"] - pred["pred_margin"]
        res_t = df["total"] - pred["pred_total"]
        # In-sample residuals understate out-of-sample error slightly; inflate a little.
        self.margin_sd = float(np.sqrt(np.average(res_m ** 2, weights=w)) * 1.02)
        self.total_sd = float(np.sqrt(np.average(res_t ** 2, weights=w)) * 1.02)
        self.trained_through = last
        return self

    def predict(self, feats: pd.DataFrame) -> pd.DataFrame:
        df = feats[feats["has_ratings"]]
        out = pd.DataFrame(index=df.index)
        out["pred_margin"] = df[self.margin_offset] + self.margin_model.predict(
            df[self.margin_features].to_numpy(float))
        out["pred_total"] = df[self.total_offset] + self.total_model.predict(
            df[self.total_features].to_numpy(float))
        out["p_home"] = [win_prob(m, self.margin_sd) for m in out["pred_margin"]]
        out["pred_home_pts"] = (out["pred_total"] + out["pred_margin"]) / 2
        out["pred_away_pts"] = (out["pred_total"] - out["pred_margin"]) / 2
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> "GameModel":
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run `elbet build` first")
        return pickle.loads(path.read_bytes())


# ---------------------------------------------------------------- market probabilities

def _margin_pmf(mu: float, sd: float) -> tuple[np.ndarray, np.ndarray]:
    k = np.arange(-90, 91)
    p = norm.cdf(k + 0.5, mu, sd) - norm.cdf(k - 0.5, mu, sd)
    p[k == 0] = 0.0
    return k, p / p.sum()


def _total_pmf(mu: float, sd: float) -> tuple[np.ndarray, np.ndarray]:
    k = np.arange(60, 301)
    p = norm.cdf(k + 0.5, mu, sd) - norm.cdf(k - 0.5, mu, sd)
    return k, p / p.sum()


def win_prob(mu_margin: float, sd: float) -> float:
    k, p = _margin_pmf(mu_margin, sd)
    return float(p[k > 0].sum())


def spread_probs(mu_margin: float, sd: float, home_line: float) -> tuple[float, float, float]:
    """(P home covers, P push, P away covers) for a home handicap such as -5.5."""
    k, p = _margin_pmf(mu_margin, sd)
    adj = k + home_line
    return float(p[adj > 0].sum()), float(p[adj == 0].sum()), float(p[adj < 0].sum())


def total_probs(mu_total: float, sd: float, line: float) -> tuple[float, float, float]:
    """(P over, P push, P under)."""
    k, p = _total_pmf(mu_total, sd)
    return float(p[k > line].sum()), float(p[k == line].sum()), float(p[k < line].sum())


def fair_spread(mu_margin: float) -> float:
    """Home handicap that makes both sides ~50%, rounded to the nearest half point."""
    return round(-mu_margin * 2) / 2


def fair_total(mu_total: float) -> float:
    return round(mu_total * 2) / 2
