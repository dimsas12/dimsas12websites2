"""Walk-forward backtests: each season is predicted by a model trained only on earlier seasons."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import betting
from .config import BetConfig, Settings
from .models import GameModel, spread_probs, total_probs
from .props import PROP_STATS, PropsModel, nb_probs


def walk_forward(feats: pd.DataFrame, start_season: int, settings: Settings) -> pd.DataFrame:
    feats = feats[feats["has_ratings"]]
    seasons = sorted(s for s in feats["season"].unique() if s >= start_season)
    preds = []
    for s in seasons:
        train = feats[(feats["season"] < s) & feats["played"]]
        test = feats[(feats["season"] == s) & feats["played"]]
        if len(train) < 200 or test.empty:
            continue
        model = GameModel().fit(train)
        p = model.predict(test)
        p["margin_sd"], p["total_sd"] = model.margin_sd, model.total_sd
        preds.append(test.join(p))
    return pd.concat(preds) if preds else pd.DataFrame()


def _logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def game_metrics(pred: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    rows = []
    hfa = settings.elo.hfa
    for season, d in list(pred.groupby("season")) + [("ALL", pred)]:
        y = d["home_win"].to_numpy()
        elo_p = 1 / (1 + 10 ** (-(d["elo_diff"] + hfa * (1 - d["neutral"])) / 400))
        base = np.full(len(d), pred["home_win"].mean())
        rows.append({
            "season": season, "games": len(d),
            "acc": float(((d["p_home"] > 0.5) == (y == 1)).mean()),
            "logloss": _logloss(d["p_home"].to_numpy(), y),
            "brier": float(np.mean((d["p_home"] - y) ** 2)),
            "logloss_elo": _logloss(elo_p.to_numpy(), y),
            "logloss_homerate": _logloss(base, y),
            "mae_margin": float((d["margin"] - d["pred_margin"]).abs().mean()),
            "mae_total": float((d["total"] - d["pred_total"]).abs().mean()),
            "mae_total_naive": float((d["total"] - d["mu_eff"] * d["mu_pace"] / 50).abs().mean()),
        })
    return pd.DataFrame(rows)


def calibration(pred: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    b = pd.cut(pred["p_home"], np.linspace(0, 1, bins + 1))
    return pred.groupby(b, observed=True).agg(games=("home_win", "size"),
                                              predicted=("p_home", "mean"),
                                              actual=("home_win", "mean")).reset_index()


# ---------------------------------------------------------------- betting simulation

ODDS_COLUMNS = ["game_id", "home_ml", "away_ml", "home_line", "home_spread_odds",
                "away_spread_odds", "total_line", "over_odds", "under_odds"]


def candidate_bets(row, bet: BetConfig) -> list[dict]:
    """All bets on one game with model probabilities and EV at the quoted prices."""
    out = []
    mu, sd = row.pred_margin, row.margin_sd
    if pd.notna(row.home_ml) and pd.notna(row.away_ml):
        p = row.p_home
        out += [
            {"market": "moneyline", "side": "home", "line": None, "odds": row.home_ml,
             "p_win": p, "p_push": 0.0, "won": row.margin > 0, "push": False},
            {"market": "moneyline", "side": "away", "line": None, "odds": row.away_ml,
             "p_win": 1 - p, "p_push": 0.0, "won": row.margin < 0, "push": False},
        ]
    if pd.notna(row.home_line) and pd.notna(row.home_spread_odds):
        ph, pp, pa = spread_probs(mu, sd, row.home_line)
        adj = row.margin + row.home_line
        out += [
            {"market": "spread", "side": "home", "line": row.home_line,
             "odds": row.home_spread_odds, "p_win": ph, "p_push": pp, "won": adj > 0,
             "push": adj == 0},
            {"market": "spread", "side": "away", "line": -row.home_line,
             "odds": row.away_spread_odds, "p_win": pa, "p_push": pp, "won": adj < 0,
             "push": adj == 0},
        ]
    if pd.notna(row.total_line) and pd.notna(row.over_odds):
        po, pp, pu = total_probs(row.pred_total, row.total_sd, row.total_line)
        out += [
            {"market": "total", "side": "over", "line": row.total_line, "odds": row.over_odds,
             "p_win": po, "p_push": pp, "won": row.total > row.total_line,
             "push": row.total == row.total_line},
            {"market": "total", "side": "under", "line": row.total_line,
             "odds": row.under_odds, "p_win": pu, "p_push": pp, "won": row.total < row.total_line,
             "push": row.total == row.total_line},
        ]
    for b in out:
        b["ev"] = betting.expected_value(b["p_win"], b["odds"], b["p_push"])
    return out


def simulate_bets(pred: pd.DataFrame, odds: pd.DataFrame, bet: BetConfig,
                  bankroll: float = 1000.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pred.merge(odds[ODDS_COLUMNS], on="game_id", how="inner").sort_values("date")
    bank = bankroll
    placed = []
    for row in df.itertuples(index=False):
        for b in candidate_bets(row, bet):
            if (b["ev"] < bet.min_edge or b["p_win"] < bet.min_prob or b["odds"] > bet.max_odds
                    or not math.isfinite(b["odds"])):
                continue
            st = min(betting.kelly(b["p_win"], b["odds"], b["p_push"]) * bet.kelly_fraction,
                     bet.max_stake_pct) * bank
            profit_unit = 0.0 if b["push"] else (b["odds"] - 1 if b["won"] else -1.0)
            bank += st * profit_unit
            placed.append({"game_id": row.game_id, "date": row.date, "season": row.season,
                           **b, "profit_flat": profit_unit, "stake": st, "bankroll": bank})
    bets = pd.DataFrame(placed)
    if bets.empty:
        return bets, pd.DataFrame()
    summary = bets.groupby("market").agg(
        bets=("ev", "size"), avg_odds=("odds", "mean"), avg_ev=("ev", "mean"),
        hit_rate=("won", "mean"), roi_flat=("profit_flat", "mean")).reset_index()
    total = pd.DataFrame([{"market": "ALL", "bets": len(bets), "avg_odds": bets["odds"].mean(),
                           "avg_ev": bets["ev"].mean(), "hit_rate": bets["won"].mean(),
                           "roi_flat": bets["profit_flat"].mean()}])
    return bets, pd.concat([summary, total], ignore_index=True)


# ---------------------------------------------------------------- props

def props_backtest(history: pd.DataFrame, start_season: int, model: PropsModel) -> pd.DataFrame:
    """Projection accuracy vs the player's naive recent average.

    Real historical prop lines are not available, so probabilities are scored
    against a synthetic half-point line at the naive average: `brier_model` vs
    `brier_naive` (same NB dispersion, naive mean) measures how much the
    pace/opponent/shrinkage adjustments add over a simple average.
    """
    h = history[(history["season"] >= start_season) & (history["prior_games"] >= 3)]
    rows = []
    for s in PROP_STATS:
        a = model.alpha[s]
        lines = np.round(h[f"naive_{s}"] - 0.5) + 0.5
        went_over = (h[s] > lines).to_numpy()
        p_model = np.array([nb_probs(m, a, L)[0] for m, L in zip(h[f"proj_{s}"], lines)])
        p_naive = np.array([nb_probs(max(m, 0.05), a, L)[0]
                            for m, L in zip(h[f"naive_{s}"], lines)])
        rows.append({
            "stat": s, "n": len(h),
            "mae_model": (h[s] - h[f"proj_{s}"]).abs().mean(),
            "mae_naive_avg": (h[s] - h[f"naive_{s}"]).abs().mean(),
            "alpha": a,
            "brier_model": float(np.mean((p_model - went_over) ** 2)),
            "brier_naive": float(np.mean((p_naive - went_over) ** 2)),
        })
    return pd.DataFrame(rows)
