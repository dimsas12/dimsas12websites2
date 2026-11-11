"""Build the tennis ratings and price a match: winner, sets, games, tie-break, aces."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from scipy.stats import nbinom

from ..window import now as _now
from . import data
from .model import EloConfig, mixed_markets, run, solve_points

STATE = data.DIR / "state.pkl"
# Sum of the two players' serve-point win rates by tour and surface (tour averages).
SERVE_SUM = {("atp", "Hard"): 1.285, ("atp", "Clay"): 1.245, ("atp", "Grass"): 1.325,
             ("wta", "Hard"): 1.135, ("wta", "Clay"): 1.115, ("wta", "Grass"): 1.165}
POINTS_PER_SERVICE_GAME = 6.4
ACE_ALPHA = 0.12  # NB dispersion of a player's aces around the expectation


def serve_profiles(window_days: int = 730) -> pd.DataFrame:
    """ATP players' recent serve/return numbers from TML: service points won,
    return points won, ace rate, aces conceded rate (per service point)."""
    t = data.load_tml()
    t = t[t["date"] >= t["date"].max() - pd.Timedelta(days=window_days)]
    t = t.dropna(subset=["w_svpt", "l_svpt"])
    t = t[(t["w_svpt"] > 0) & (t["l_svpt"] > 0)]
    rows = []
    for side, opp in (("w", "l"), ("l", "w")):
        name = "winner_name" if side == "w" else "loser_name"
        rows.append(pd.DataFrame({
            "pid": t[name].map(data.norm), "surface": t["surface"],
            "svpt": t[f"{side}_svpt"], "sv_won": t[f"{side}_1stWon"] + t[f"{side}_2ndWon"],
            "ace": t[f"{side}_ace"], "rtpt": t[f"{opp}_svpt"],
            "rt_won": t[f"{opp}_svpt"] - t[f"{opp}_1stWon"] - t[f"{opp}_2ndWon"],
            "ace_against": t[f"{opp}_ace"]}))
    g = pd.concat(rows).groupby("pid").sum(numeric_only=True)
    tot = g.sum()
    prior = 600.0  # points of tour-average play mixed in
    avg_spw, avg_rpw = tot["sv_won"] / tot["svpt"], tot["rt_won"] / tot["rtpt"]
    avg_ace = tot["ace"] / tot["svpt"]
    out = pd.DataFrame({
        "spw": (g["sv_won"] + prior * avg_spw) / (g["svpt"] + prior),
        "rpw": (g["rt_won"] + prior * avg_rpw) / (g["rtpt"] + prior),
        "ace_rate": (g["ace"] + prior * avg_ace) / (g["svpt"] + prior),
        "ace_against": (g["ace_against"] + prior * avg_ace) / (g["rtpt"] + prior),
        "svpt": g["svpt"]})
    out.attrs.update(avg_spw=float(avg_spw), avg_rpw=float(avg_rpw), avg_ace=float(avg_ace))
    return out


def build(cfg: EloConfig | None = None) -> dict:
    m = data.combined()
    played = m[m["winner"].isin([1, 2]) | (m["state"] == "pre")]
    rated, systems = run(played, cfg)
    prof = serve_profiles() if (data.DIR / "tml_matches.csv").exists() else pd.DataFrame()
    STATE.write_bytes(pickle.dumps({"systems": systems, "profiles": prof,
                                    "attrs": dict(prof.attrs)}))
    rated.to_pickle(data.DIR / "rated.pkl")
    return {"matches": len(rated), "players": {t: len(s.r) for t, s in systems.items()}}


def load_state() -> dict:
    return pickle.loads(STATE.read_bytes())


def serve_sum(state: dict, tour: str, surface: str, a: str, b: str) -> float:
    base = SERVE_SUM.get((tour, surface), 1.2)
    prof, at = state["profiles"], state["attrs"]
    if tour != "atp" or prof is None or len(prof) == 0:
        return base
    adj = 0.0
    for me, opp in ((a, b), (b, a)):
        if me in prof.index:
            adj += prof.at[me, "spw"] - at["avg_spw"]
        if opp in prof.index:
            adj -= prof.at[opp, "rpw"] - at["avg_rpw"]
    return float(np.clip(base + adj, base - 0.10, base + 0.10))


def price(state: dict, tour: str, surface: str, p1: str, p2: str, best_of: int = 3) -> dict:
    """All markets for one match (p1, p2 = player names in any spelling)."""
    a, b = data.norm(p1), data.norm(p2)
    elo = state["systems"][tour]
    p3 = elo.prob(a, b, surface)
    ssum = serve_sum(state, tour, surface, a, b)
    pa, pb = solve_points(p3, ssum, 3)
    mk = mixed_markets(p3, ssum, best_of, tour=tour)
    out = {"p1": p1, "p2": p2, "tour": tour, "surface": surface, "best_of": best_of,
           "n1": elo.n.get(a, 0), "n2": elo.n.get(b, 0),
           "elo1": elo.r.get(a, elo.cfg.init), "elo2": elo.r.get(b, elo.cfg.init),
           "p1_win": mk["p_match"], "p1_set1": mk["p_set"], "exp_games": mk["exp_games"],
           "p1_win_bo3": p3,
           "p_tiebreak": mk["p_any_tiebreak"], "set_scores": mk["set_scores"],
           "games": mk["games"], "serve_p1": pa, "serve_p2": pb}
    prof, at = state["profiles"], state["attrs"]
    if tour == "atp" and prof is not None and len(prof):
        svc_games = mk["exp_games"] / 2
        for tag, me, opp in (("1", a, b), ("2", b, a)):
            rate = prof.at[me, "ace_rate"] if me in prof.index else at["avg_ace"]
            against = prof.at[opp, "ace_against"] if opp in prof.index else at["avg_ace"]
            surf = {"Grass": 1.25, "Clay": 0.70}.get(surface, 1.0)
            out[f"aces{tag}"] = svc_games * POINTS_PER_SERVICE_GAME * rate * against / at[
                "avg_ace"] * surf
    return out


def games_over(games: dict, line: float) -> float:
    return float(sum(v for (t, _), v in games.items() if t > line))


def handicap_p1(games: dict, line: float) -> float:
    """P(player 1 covers a games handicap of `line`, e.g. -3.5)."""
    return float(sum(v for (_, m), v in games.items() if m + line > 0))


def aces_over(mean: float, line: float) -> float:
    n = 1.0 / ACE_ALPHA
    return float(nbinom.sf(int(np.floor(line)), n, n / (n + mean)))


def upcoming(days: float = 2.0) -> pd.DataFrame:
    m = data.combined()
    now = _now()
    return m[(m["state"] == "pre") & (m["date"] > now)
             & (m["date"] <= now + pd.Timedelta(days=days))]
