"""Backtest of the sharp-price value rule on football-data.co.uk odds.

Fair probability = Pinnacle's pre-match price with the margin removed (power
method). Bet the best price in the market ("Max" across the bookmakers the site
tracks) when it beats the fair price by `min_edge`. Flat 1-unit stakes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..vec import devig_power
from . import data


def _table(first_season: int) -> pd.DataFrame:
    d = data.load()
    return d[d["played"] & (d["season"] >= first_season)]


def run(min_edge: float = 0.02, first_season: int = 2015, max_odds: float = 8.0,
        max_edge: float = 0.25) -> pd.DataFrame:
    """One row per bet: season, div, market, selection, odds, edge, profit, clv."""
    d = _table(first_season)
    bets = []
    # ---- 1X2
    c = ["o_h", "o_d", "o_a", "max_h", "max_d", "max_a", "c_h", "c_d", "c_a"]
    x = d.dropna(subset=c)
    x = x[(x[c] > 1).all(axis=1)]
    fair = devig_power(x[["o_h", "o_d", "o_a"]].to_numpy())
    close = devig_power(x[["c_h", "c_d", "c_a"]].to_numpy())
    mx = x[["max_h", "max_d", "max_a"]].to_numpy()
    y = np.where(x["hg"] > x["ag"], 0, np.where(x["hg"] == x["ag"], 1, 2))
    edge = fair * mx - 1
    for i, sel in enumerate(["home", "draw", "away"]):
        m = (edge[:, i] >= min_edge) & (edge[:, i] <= max_edge) & (mx[:, i] <= max_odds)
        bets.append(pd.DataFrame({
            "season": x["season"].to_numpy()[m], "div": x["div"].to_numpy()[m],
            "date": x["date"].to_numpy()[m], "market": "1X2", "selection": sel,
            "odds": mx[m, i], "edge": edge[m, i],
            "profit": np.where(y[m] == i, mx[m, i] - 1, -1.0),
            "clv": close[m, i] * mx[m, i] - 1}))
    # ---- over/under 2.5
    c = ["o_over", "o_under", "max_over", "max_under", "c_over", "c_under"]
    x = d.dropna(subset=c)
    x = x[(x[c] > 1).all(axis=1)]
    fair = devig_power(x[["o_over", "o_under"]].to_numpy())
    close = devig_power(x[["c_over", "c_under"]].to_numpy())
    mx = x[["max_over", "max_under"]].to_numpy()
    over = (x["hg"] + x["ag"] > 2.5).to_numpy()
    edge = fair * mx - 1
    for i, sel in enumerate(["over 2.5", "under 2.5"]):
        won = over if i == 0 else ~over
        m = (edge[:, i] >= min_edge) & (edge[:, i] <= max_edge)
        bets.append(pd.DataFrame({
            "season": x["season"].to_numpy()[m], "div": x["div"].to_numpy()[m],
            "date": x["date"].to_numpy()[m], "market": "O/U 2.5", "selection": sel,
            "odds": mx[m, i], "edge": edge[m, i],
            "profit": np.where(won[m], mx[m, i] - 1, -1.0),
            "clv": close[m, i] * mx[m, i] - 1}))
    return pd.concat(bets, ignore_index=True)


def summarise(bets: pd.DataFrame, by: str) -> pd.DataFrame:
    g = bets.groupby(by)
    out = g.agg(bets=("profit", "size"), avg_odds=("odds", "mean"), profit=("profit", "sum"),
                roi=("profit", "mean"), clv=("clv", "mean"))
    out["roi_se"] = g["profit"].std() / np.sqrt(out["bets"])
    return out.reset_index()


def report(min_edge: float = 0.02) -> str:
    b = run(min_edge)
    hold = b[b["season"] >= 2020]
    lines = [f"# Football value backtest (edge >= {min_edge:.0%} over Pinnacle fair price)", ""]
    for title, tab in (("By market, 2020-26", summarise(hold, "market")),
                       ("By season, all markets", summarise(b, "season")),
                       ("By league, 2020-26", summarise(hold, "div"))):
        lines += [f"## {title}", "", tab.round(4).to_markdown(index=False), ""]
    tot = hold["profit"]
    lines.append(f"Total 2020-26: {len(tot)} bets, ROI {tot.mean():+.2%} "
                 f"(s.e. {tot.std() / np.sqrt(len(tot)):.2%}), CLV {hold['clv'].mean():+.2%}")
    return "\n".join(lines)
