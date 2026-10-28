import numpy as np
import pandas as pd
import pytest

from elbet import betting
from elbet.config import Settings
from elbet.features import build_features
from elbet.models import GameModel, spread_probs, total_probs, win_prob
from elbet.odds import market_lines, match_events, name_similarity
from elbet.props import nb_probs


def synthetic_games(seasons=(2020, 2021, 2022), seed=0) -> pd.DataFrame:
    """Round robins between 8 teams with known strengths."""
    rng = np.random.default_rng(seed)
    teams = list("ABCDEFGH")
    strength = dict(zip(teams, np.linspace(8, -8, len(teams))))
    rows = []
    for season in seasons:
        day = pd.Timestamp(f"{season}-10-01")
        code = 0
        for rnd in range(2):
            for h in teams:
                for a in teams:
                    if h == a:
                        continue
                    code += 1
                    margin = strength[h] - strength[a] + 3 + rng.normal(0, 11)
                    total = 155 + rng.normal(0, 15)
                    hp = round((total + margin) / 2)
                    ap = round((total - margin) / 2)
                    if hp == ap:
                        hp += 1
                    rows.append({
                        "game_id": f"E{season}_{code}", "season": season, "game_code": code,
                        "date": day + pd.Timedelta(days=code // 4 * 3), "phase": "RS",
                        "round": code // 4, "neutral": False, "home": h, "away": a,
                        "home_name": h, "away_name": a, "home_names": h, "away_names": a,
                        "played": True, "home_pts": hp, "away_pts": ap, "poss": 72.0,
                    })
    return pd.DataFrame(rows)


def test_devig_sums_to_one():
    for method in ("power", "multiplicative"):
        p = betting.devig([1.80, 2.10], method)
        assert sum(p) == pytest.approx(1.0)
        assert p[0] > p[1]


def test_kelly_and_ev():
    assert betting.kelly(0.5, 2.0) == 0.0
    assert betting.kelly(0.6, 2.0) == pytest.approx(0.2)
    assert betting.expected_value(0.6, 2.0) == pytest.approx(0.2)
    # A push returns the stake: EV only counts wins and losses.
    assert betting.expected_value(0.5, 2.0, p_push=0.1) == pytest.approx(0.1)


def test_market_probabilities_consistent():
    assert win_prob(0.0, 12) == pytest.approx(0.5, abs=1e-6)
    assert win_prob(5.0, 12) > 0.6
    ph, pp, pa = spread_probs(5.0, 12, -5.0)
    assert ph + pp + pa == pytest.approx(1.0)
    # Removing the impossible 0 margin takes mass from the "away covers" side.
    assert pp > 0.02 and ph > pa
    po, pp, pu = total_probs(150.3, 17, 150.5)
    assert pp == 0.0 and po < 0.5 < pu + 0.01


def test_nb_probs():
    po, pp, pu = nb_probs(12.0, 0.2, 11.5)
    assert po + pp + pu == pytest.approx(1.0)
    assert 0.4 < po < 0.5  # right-skewed: the median sits below the mean
    assert nb_probs(12.0, 0.2, 9.5)[0] > 0.6
    po, pp, pu = nb_probs(12.0, 0.2, 12.0)
    assert pp > 0


def test_ratings_recover_strength_and_model_fits():
    games = synthetic_games()
    feats = build_features(games, Settings())
    last = feats[feats["season"] == 2022]
    # Team A (strongest) should be favoured at home against H late in the season.
    row = last[(last["home"] == "A") & (last["away"] == "H")].iloc[-1]
    assert row["rating_margin"] > 8
    assert row["elo_home"] > row["elo_away"]
    model = GameModel().fit(feats[feats["season"] < 2022])
    pred = model.predict(last[last["has_ratings"]])
    assert ((pred["p_home"] > 0.5) == (last.loc[pred.index, "margin"] > 0)).mean() > 0.65
    assert 8 < model.margin_sd < 15


def test_no_lookahead():
    """Changing a future result must not change earlier features."""
    games = synthetic_games()
    a = build_features(games, Settings())
    games2 = games.copy()
    last = games2.index[-1]
    games2.loc[last, "home_pts"] += 40
    b = build_features(games2, Settings())
    cols = ["rating_margin", "elo_diff", "pace_exp"]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_team_name_matching():
    names = "Olympiacos Piraeus|Olympiacos"
    assert name_similarity("Olympiacos BC", names) > 0.9
    assert name_similarity("Real Madrid", names) < 0.5
    events = pd.DataFrame([{"event_id": "x", "commence": pd.Timestamp("2026-10-30T18:00Z"),
                            "home_team": "Real Madrid", "away_team": "Olympiacos BC"}])
    games = pd.DataFrame([{"game_id": "E2026_1", "date": pd.Timestamp("2026-10-30 18:00"),
                           "home_names": "Olympiacos Piraeus|Olympiacos",
                           "away_names": "Real Madrid"}])
    assert match_events(events, games) == {"x": ("E2026_1", True)}


def test_market_lines():
    rows = []
    for book, (h, a) in {"b1": (1.70, 2.20), "b2": (1.75, 2.10)}.items():
        rows += [{"market": "h2h", "name": "H", "price": h, "point": None, "book": book},
                 {"market": "h2h", "name": "A", "price": a, "point": None, "book": book},
                 {"market": "spreads", "name": "H", "price": 1.9, "point": -3.5, "book": book},
                 {"market": "spreads", "name": "A", "price": 1.9, "point": 3.5, "book": book},
                 {"market": "totals", "name": "Over", "price": 1.85, "point": 158.5,
                  "book": book},
                 {"market": "totals", "name": "Under", "price": 1.95, "point": 158.5,
                  "book": book}]
    df = pd.DataFrame(rows).assign(home_team="H", away_team="A")
    m = market_lines(df)
    assert m["h2h"]["home"] == (1.75, "b2") and m["h2h"]["away"] == (2.20, "b1")
    assert m["spreads"]["line"] == -3.5
    assert m["totals"]["line"] == 158.5 and 0.5 < m["totals"]["fair_over"] < 0.53
