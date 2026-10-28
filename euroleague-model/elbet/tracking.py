"""Bet log, closing-line value (CLV) and settlement.

CLV is the expected value of a bet priced at the *closing* no-vig market
probability: CLV = p_close x (odds - 1) - (1 - p_close - p_push). Beating the
close consistently is the fastest reliable sign of a real edge; profit alone
takes thousands of bets to separate skill from luck.

If a spread/total line moved between bet and close, the closing probability is
translated to the bet's line by finding the mean margin/total the closing
market implies (with the model's SD) and re-pricing the original line.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import config
from .models import spread_probs, total_probs

TRACK_DIR = config.DATA_DIR / "tracking"
LOG_PATH = TRACK_DIR / "bets.csv"

LOG_COLUMNS = [
    "logged_at", "game_id", "event_id", "commence", "game", "market", "selection", "pick",
    "line", "odds", "book", "model_p", "market_p", "p_push", "ev", "stake",
    "close_at", "close_line", "close_odds", "close_p", "clv",
    "result", "profit", "profit_flat",
]


def load_log() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(LOG_PATH, parse_dates=["logged_at", "commence", "close_at"])


def save_log(log: pd.DataFrame) -> None:
    TRACK_DIR.mkdir(parents=True, exist_ok=True)
    log[LOG_COLUMNS].to_csv(LOG_PATH, index=False)


def log_picks(picks: pd.DataFrame) -> int:
    """Append new picks; a pick already logged for the same game/market/selection is kept
    as first logged (the price you could actually have taken first)."""
    log = load_log()
    new = picks.copy()
    new["logged_at"] = pd.Timestamp.now(tz="UTC").tz_convert(None).floor("s")
    for c in LOG_COLUMNS:
        if c not in new:
            new[c] = np.nan
    key = ["game_id", "market", "selection"]
    if not log.empty:
        seen = set(map(tuple, log[key].astype(str).to_numpy()))
        new = new[[tuple(r) not in seen for r in new[key].astype(str).to_numpy()]]
    if new.empty:
        return 0
    log = pd.concat([log, new[LOG_COLUMNS]], ignore_index=True) if not log.empty \
        else new[LOG_COLUMNS]
    save_log(log)
    return len(new)


def needs_closing(log: pd.DataFrame, now: pd.Timestamp, window_min: float) -> pd.DataFrame:
    """Open bets without a closing price whose game tips off within the window."""
    if log.empty:
        return log
    mins = (log["commence"] - now).dt.total_seconds() / 60
    return log[log["close_p"].isna() & (mins <= window_min) & (mins > -5)]


def _implied_mean(p_side: float, line: float, sd: float, kind: str) -> float:
    """Mean margin (home) or total that makes the closing line price at p_side
    (p_side = P(home covers) or P(over), ignoring push)."""
    def f(mu):
        if kind == "spread":
            h, _, a = spread_probs(mu, sd, line)
            return h / (h + a) - p_side
        o, _, u = total_probs(mu, sd, line)
        return o / (o + u) - p_side
    lo, hi = (-60.0, 60.0) if kind == "spread" else (80.0, 280.0)
    return brentq(f, lo, hi)


def closing_prob(bet: pd.Series, lines: dict, margin_sd: float, total_sd: float):
    """(closing no-vig probability for the bet at the bet's line, push prob,
    closing line, best closing odds) or None if the market is missing."""
    sel = bet["selection"]
    if bet["market"] == "moneyline":
        m = lines.get("h2h") or {}
        if not m.get("fair_home"):
            return None
        p = m["fair_home"] if sel == "home" else 1 - m["fair_home"]
        return p, 0.0, np.nan, m[sel][0]
    if bet["market"] == "spread":
        m = lines.get("spreads") or {}
        if not m.get("fair_home"):
            return None
        mu = _implied_mean(m["fair_home"], m["line"], margin_sd, "spread")
        home_line = bet["line"] if sel == "home" else -bet["line"]
        ph, pp, pa = spread_probs(mu, margin_sd, home_line)
        close_line = m["line"] if sel == "home" else -m["line"]
        return (ph if sel == "home" else pa), pp, close_line, m[sel][0]
    m = lines.get("totals") or {}
    if not m.get("fair_over"):
        return None
    mu = _implied_mean(m["fair_over"], m["line"], total_sd, "total")
    po, pp, pu = total_probs(mu, total_sd, bet["line"])
    return (po if sel == "over" else pu), pp, m["line"], m[sel][0]


def clv(p_close: float, p_push: float, odds: float) -> float:
    return p_close * (odds - 1) - (1 - p_close - p_push)


def settle(log: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Fill result/profit for bets whose games have been played."""
    g = games[games["played"]].set_index("game_id")
    log = log.copy()
    log["result"] = log["result"].astype(object)  # all-NaN columns load as float
    for i, b in log[log["result"].isna()].iterrows():
        if b["game_id"] not in g.index:
            continue
        r = g.loc[b["game_id"]]
        margin, total = r["home_pts"] - r["away_pts"], r["home_pts"] + r["away_pts"]
        if b["market"] == "moneyline":
            won = margin > 0 if b["selection"] == "home" else margin < 0
            push = False
        elif b["market"] == "spread":
            adj = margin + (b["line"] if b["selection"] == "home" else -b["line"])
            won, push = (adj > 0 if b["selection"] == "home" else adj < 0), adj == 0
        else:
            won = total > b["line"] if b["selection"] == "over" else total < b["line"]
            push = total == b["line"]
        res = "push" if push else ("win" if won else "loss")
        unit = 0.0 if push else (b["odds"] - 1 if won else -1.0)
        log.at[i, "result"] = res
        log.at[i, "profit_flat"] = unit
        log.at[i, "profit"] = round(unit * (b["stake"] or 0), 2)
    return log


def report(log: pd.DataFrame) -> pd.DataFrame:
    if log.empty:
        return pd.DataFrame()

    def agg(d: pd.DataFrame, name: str) -> dict:
        s = d[d["result"].notna()]
        c = d[d["clv"].notna()]
        staked = s["stake"].sum()
        return {
            "market": name, "bets": len(d), "settled": len(s),
            "won": int((s["result"] == "win").sum()), "lost": int((s["result"] == "loss").sum()),
            "avg_odds": d["odds"].mean(), "avg_ev_at_bet": d["ev"].mean(),
            "roi_flat": s["profit_flat"].mean() if len(s) else np.nan,
            "profit": s["profit"].sum(), "roi_staked": s["profit"].sum() / staked
            if staked else np.nan,
            "with_close": len(c), "avg_clv": c["clv"].mean() if len(c) else np.nan,
            "beat_close_pct": (c["clv"] > 0).mean() if len(c) else np.nan,
        }
    rows = [agg(d, m) for m, d in log.groupby("market")] + [agg(log, "ALL")]
    return pd.DataFrame(rows)

