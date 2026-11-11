import numpy as np
import pandas as pd

from msbet import odds
from msbet.soccer.model import ah_ev, ah_value, market_probs, score_matrix
from msbet.tennis.model import Elo, hold_prob, match_markets, mixed_markets, solve_points
from msbet.us import props
from msbet.us.model import CONFIGS, GameModel, run
from msbet.vec import devig_power


def test_score_matrix_and_markets():
    M = score_matrix(1.6, 1.1, -0.05)
    assert abs(M.sum() - 1) < 1e-9
    p = market_probs(1.6, 1.1, -0.05)
    assert abs(p["p_h"] + p["p_d"] + p["p_a"] - 1) < 1e-9
    assert p["p_h"] > p["p_a"] and 0 < p["p_btts"] < 1


def test_asian_handicap_quarter_lines():
    M = score_matrix(1.5, 1.2)
    home, away = ah_value(M, -0.25)
    assert abs(sum(home) - 1) < 1e-9 and abs(sum(away) - 1) < 1e-9
    assert home[3] > 0 and home[1] == 0  # draw = half loss for the -0.25 side
    assert away[1] > 0
    # the same bet priced at fair odds on both sides cannot both be positive EV
    assert ah_ev(home, 1.9) + ah_ev(away, 1.9) < 0


def test_devig_power_rows_sum_to_one():
    p = devig_power(np.array([[2.0, 3.5, 3.8], [1.3, 5.5, 11.0]]))
    assert np.allclose(p.sum(axis=1), 1) and (p > 0).all()


def test_value_table_uses_sharp_book_and_flags_soft_price():
    rows = []
    for book, (h, a) in {"pinnacle": (1.95, 1.95), "softbook": (2.15, 1.75),
                         "other": (1.90, 1.90)}.items():
        for name, price in (("A", h), ("B", a)):
            rows.append({"event_id": "e1", "commence": pd.Timestamp("2026-01-01"),
                         "home_team": "A", "away_team": "B", "book": book, "market": "h2h",
                         "name": name, "description": None, "point": None, "price": price,
                         "net": price})
    t = odds.value_table(pd.DataFrame(rows))
    a = t[t["selection"] == "A"].iloc[0]
    assert a["fair_source"] == "pinnacle" and abs(a["fair_p"] - 0.5) < 1e-6
    assert a["book"] == "softbook" and abs(a["edge"] - 0.075) < 1e-6
    picks = odds.picks(t, 0.02)
    assert list(picks["selection"]) == ["A"]


def test_tennis_point_model_roundtrip():
    assert abs(hold_prob(0.5) - 0.5) < 1e-9 and hold_prob(0.65) > 0.8
    pa, pb = solve_points(0.7, 1.28, 3)
    mk = match_markets(pa, pb, 3)
    assert abs(mk["p_match"] - 0.7) < 2e-3
    assert abs(sum(mk["set_scores"].values()) - 1) < 1e-9
    assert abs(sum(mk["games"].values()) - 1) < 1e-6
    assert match_markets(pa, pb, 5)["p_match"] > mk["p_match"]  # best of 5 helps the favourite


def test_tennis_mixture_keeps_win_prob_and_adds_straight_sets():
    plain = match_markets(*solve_points(0.6, 1.28, 3), 3)
    mix = mixed_markets(0.6, 1.28, 3, sd=1.5)
    assert abs(mix["p_match"] - 0.6) < 0.01
    straight = lambda m: m["set_scores"][(2, 0)] + m["set_scores"][(0, 2)]  # noqa: E731
    assert straight(mix) > straight(plain) and mix["exp_games"] < plain["exp_games"]


def test_elo_moves_toward_winner():
    e = Elo()
    for _ in range(10):
        e.update("a", "b", "Hard")
    assert e.prob("a", "b", "Hard") > 0.7 and e.prob("a", "b", "Clay") > 0.5


def test_points_ratings_recover_strength():
    rng = np.random.default_rng(0)
    true = {"A": 6.0, "B": 2.0, "C": -2.0, "D": -6.0}
    rows, date = [], pd.Timestamp("2025-01-01")
    for i in range(600):
        h, a = rng.choice(list(true), 2, replace=False)
        m = true[h] - true[a] + 2.5 + rng.normal(0, 12)
        rows.append({"date": date + pd.Timedelta(days=i), "season": 2025, "h_id": h, "a_id": a,
                     "h_pts": 110 + m / 2, "a_pts": 110 - m / 2, "neutral": False,
                     "completed": True})
    rated, r = run(pd.DataFrame(rows), CONFIGS["nba"])
    net = {t: r.off[t] + r.dfn[t] for t in true}
    assert net["A"] > net["B"] > net["C"] > net["D"]
    gm = GameModel(CONFIGS["nba"]).fit(rated)
    assert 0.5 < gm.p_cover(5.0, 0.0) < 1 and gm.p_over(220, 210) > 0.5


def test_prop_distributions():
    assert props.p_over(20, 15.5, "pts", "nba") > props.p_over(20, 24.5, "pts", "nba")
    # yardage is right-skewed: less than half the time above the mean
    assert props.p_over(60, 60.0, "receiving_yards", "nfl") < 0.5
    ln = props.fair_line(22.3, "pts", "nba")
    assert ln % 1 == 0.5 and abs(props.p_over(22.3, ln, "pts", "nba") - 0.5) < 0.06


def test_lay_markets_ignored_and_single_exchange_not_sharp():
    ev = [{"id": "e", "commence_time": "2026-01-01T12:00:00Z", "home_team": "A",
           "away_team": "B", "bookmakers": [
               {"key": "betfair_ex_eu", "markets": [
                   {"key": "h2h", "outcomes": [{"name": "A", "price": 2.0},
                                               {"name": "B", "price": 2.0}]},
                   {"key": "h2h_lay", "outcomes": [{"name": "A", "price": 2.6},
                                                   {"name": "B", "price": 2.6}]}]},
               {"key": "soft", "markets": [{"key": "h2h", "outcomes": [
                   {"name": "A", "price": 2.2}, {"name": "B", "price": 1.7}]}]}]}]
    assert odds.flatten(ev).empty  # the game has started: live odds are never priced
    flat = odds.flatten(ev, future_only=False)
    assert not flat["market"].str.endswith("_lay").any()
    t = odds.value_table(flat, min_books=2)
    assert (t["fair_source"].str.startswith("consensus")).all()
    assert odds.picks(t, 0.02).empty  # consensus-only prices are not flagged by default
