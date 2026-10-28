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


def test_simulate_bets_one_per_game():
    from elbet.backtest import simulate_bets
    from elbet.config import BetConfig
    pred = pd.DataFrame([{
        "game_id": "g1", "date": pd.Timestamp("2025-01-01"), "season": 2025,
        "pred_margin": 10.0, "margin_sd": 12.0, "pred_total": 160.0, "total_sd": 17.0,
        "p_home": 0.8, "margin": 12.0, "total": 150.0}])
    odds = pd.DataFrame([{"game_id": "g1", "home_ml": 1.6, "away_ml": 2.4, "home_line": -3.5,
                          "home_spread_odds": 1.9, "away_spread_odds": 1.9,
                          "total_line": 165.5, "over_odds": 1.9, "under_odds": 1.9}])
    bets, summary = simulate_bets(pred, odds, BetConfig())
    assert len(bets) == 1 and bets.iloc[0]["won"]
    assert summary.set_index("market").loc["ALL", "bets"] == 1


def test_injury_feed_parser_handles_clean_clubs():
    from elbet.injuries import parse_feed, status_weight

    def block(team, rows):
        tbl = "" if not rows else '<table class="bs-inj"><tbody>' + "".join(
            f"<tr><td><a>{p}</a></td><td><span>{s}</span></td><td>x</td><td>&ndash;</td></tr>"
            for p, s in rows) + "</tbody></table>"
        return (f'<div class="bs-team-block bs-inj-club" id="{team}"><h3 class="bs-team-title">'
                f"<a>{team}</a></h3>{tbl}</div>")
    page = ("<p>updated 28 October 2026</p>" + block("Bayern", [])
            + block("Besiktas", [("Devon Dotson", "Out"), ("Lonnie Walker", "Day to day")]))
    df = parse_feed(page)
    assert list(df["team"]) == ["Besiktas", "Besiktas"]
    assert df["updated"].iloc[0] == "28 October 2026"
    assert status_weight("Out") == 1.0 and status_weight("Cleared") == 0.0
    assert status_weight("Day to day") == 0.5


def test_settle_and_clv():
    from elbet import tracking
    log = pd.DataFrame([
        {"game_id": "g1", "market": "spread", "selection": "home", "line": -3.5, "odds": 1.9,
         "stake": 10.0, "result": np.nan},
        {"game_id": "g1", "market": "total", "selection": "under", "line": 160.0, "odds": 1.9,
         "stake": 10.0, "result": np.nan},
        {"game_id": "g1", "market": "moneyline", "selection": "away", "line": np.nan,
         "odds": 2.5, "stake": 10.0, "result": np.nan},
    ])
    games = pd.DataFrame([{"game_id": "g1", "played": True, "home_pts": 85, "away_pts": 75}])
    out = tracking.settle(log, games)
    assert list(out["result"]) == ["win", "push", "loss"]
    assert list(out["profit"]) == [9.0, 0.0, -10.0]
    # Bet at 2.10 on a side that closed at a fair 50%: +5% CLV.
    assert tracking.clv(0.5, 0.0, 2.10) == pytest.approx(0.05)


def test_minutes_allocation_sums_to_200():
    from elbet.props import allocate_minutes
    base = np.array([30, 28, 25, 22, 20, 18, 15, 12, 10, 8.0])
    p_play = np.array([1, 0, 1, 0.5, 1, 1, 1, 1, 1, 0.8])  # one out, one 50/50
    m = allocate_minutes(base, p_play)
    assert (p_play * m).sum() == pytest.approx(200, abs=0.01)
    assert m.max() <= 40 and m[0] > 30  # starters absorb the missing minutes


def test_exchange_commission_and_offshore_books():
    from elbet.odds import net_price, usable_books
    assert net_price(2.10, "betfair_ex_eu") == pytest.approx(2.045)
    assert net_price(2.10, "williamhill") == 2.10
    assert usable_books({"onexbet", "pinnacle", "gtbets"}) == {"pinnacle"}
    assert usable_books({"onexbet", "pinnacle"}, {"onexbet"}) == {"onexbet"}


def test_strict_player_matching():
    from elbet.pipeline import match_player
    roster = ["GABRIEL, WENYEN", "WALKER IV, LONNIE", "TANASKOVIC, NIKOLA"]
    assert match_player("Gabriel Lundberg", roster) is None
    assert match_player("Nikola Milutinov", roster) is None
    assert match_player("Lonnie Walker", roster) == "WALKER IV, LONNIE"


def test_overtime_mass_goes_to_small_margins():
    # Normal approximation without the tie: P(M > 6.5) = 52.9%; OT adds only a little.
    ph, _, _ = spread_probs(7.36, 11.89, -6.5)
    assert 0.529 < ph < 0.535


def test_mixture_widens_distribution_and_keeps_probabilities_valid():
    from elbet.props import mixture_probs, nb_probs
    po, pp, pu = mixture_probs(12.0, 0.1, 11.5, [0.6, 0.9, 1.0, 1.1, 1.4], rate_sd=0.2)
    assert po + pp + pu == pytest.approx(1.0)
    # Minutes/rate uncertainty fattens the tails relative to the conditional NB.
    assert mixture_probs(12.0, 0.1, 24.5, [0.6, 1.0, 1.4], 0.2)[0] > nb_probs(12.0, 0.1, 24.5)[0]


def test_headroom_allocation_favours_bench():
    from elbet.props_backtest import allocate
    base = np.array([32.0, 28, 22, 18, 14, 10, 8])
    p_play = np.array([1, 1, 0, 1, 1, 1, 1.0])  # the 22-minute player is out
    m = allocate(base, p_play, "headroom", 1.0, 197.0)
    gain = m - base
    assert gain[5] > gain[0]  # a bench player gains more than the 32-minute starter
