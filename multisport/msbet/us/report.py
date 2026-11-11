"""Daily workbooks for NBA / NFL / NCAA: value bets, game lines, player props."""

from __future__ import annotations

import numpy as np
import pandas as pd
from openpyxl import Workbook

from .. import odds as O
from .. import xl
from ..window import clip
from . import data, pipeline, props

REPORTS = data.ROOT / "reports"
PROP_MARKETS = {
    "nba": "player_points,player_rebounds,player_assists,player_threes",
    "nfl": "player_pass_yds,player_rush_yds,player_reception_yds,player_receptions,"
           "player_anytime_td",
}


def games_table(up: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "date": up["date"], "away": up["a_name"], "home": up["h_name"],
        "model margin (home)": up["pred_margin"], "fair home spread": -(up["pred_margin"] * 2
                                                                        ).round() / 2,
        "P(home wins)": up["p_home"], "fair home ML": 1 / up["p_home"],
        "fair away ML": 1 / (1 - up["p_home"]), "model total": up["pred_total"],
        "home pts": up["home_pts"], "away pts": up["away_pts"],
        "book spread (home)": None, "book total": None,
        "games home": up["n_h"], "games away": up["n_a"]}).reset_index(drop=True)


def nba_props_table(pr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in pr.itertuples(index=False):
        row = {"team": r.team, "player": r.player, "status": r.status, "min": r.min}
        for s, lab in (("pts", "PTS"), ("reb", "REB"), ("ast", "AST"), ("fg3m", "3PM"),
                       ("pra", "PRA"), ("stl", "STL"), ("blk", "BLK"), ("tov", "TO")):
            m = getattr(r, s)
            row[lab] = m
            if s in ("pts", "reb", "ast", "fg3m", "pra"):
                ln = props.fair_line(m, s, "nba")
                row[f"{lab} line"] = ln
                row[f"{lab} P(over)"] = props.p_over(m, ln, s, "nba")
        rows.append(row)
    return pd.DataFrame(rows)


def nfl_props_table(pr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    spec = {"QB": [("passing_yards", "pass yds"), ("attempts", "pass att"),
                   ("completions", "comp"), ("passing_tds", "pass TD"),
                   ("passing_interceptions", "INT"), ("rushing_yards", "rush yds")],
            "RB": [("rushing_yards", "rush yds"), ("carries", "carries"),
                   ("receptions", "rec"), ("receiving_yards", "rec yds")],
            "WR": [("receptions", "rec"), ("receiving_yards", "rec yds"), ("targets", "targets")],
            "TE": [("receptions", "rec"), ("receiving_yards", "rec yds"), ("targets", "targets")]}
    for r in pr.sort_values(["team", "pos"]).itertuples(index=False):
        if r.pos not in spec:
            continue
        main = spec[r.pos][0][0]
        if getattr(r, main) < {"QB": 60, "RB": 12, "WR": 1.2, "TE": 1.0}[r.pos]:
            continue
        row = {"team": r.team, "player": r.player, "pos": r.pos, "games": r.games}
        for s, lab in spec[r.pos]:
            m = getattr(r, s)
            row[lab] = m
            if s in ("passing_yards", "rushing_yards", "receiving_yards", "receptions"):
                ln = props.fair_line(m, s, "nfl")
                row[f"{lab} line"] = ln
                row[f"{lab} P(over)"] = props.p_over(m, ln, s, "nfl")
        if r.pos != "QB":
            row["anytime TD"] = r.anytime_td
        rows.append(row)
    return pd.DataFrame(rows)


def model_edges(up: pd.DataFrame, table: pd.DataFrame, model) -> pd.DataFrame:
    """Model view of the market lines (information only: the model alone does not
    beat closing lines in the backtest)."""
    if table.empty or up.empty:
        return pd.DataFrame()
    rows = []
    for g in up.itertuples(index=False):
        t = table[(table["home_team"].fillna("").str.contains(str(g.h_name).split()[-1],
                                                              regex=False))
                  & (table["away_team"].fillna("").str.contains(str(g.a_name).split()[-1],
                                                                regex=False))]
        for r in t.itertuples(index=False):
            if r.market == "spreads" and r.point == r.point:
                home_line = r.point if r.selection == r.home_team else -r.point
                p = float(model.p_cover(g.pred_margin, home_line))
                p = p if r.selection == r.home_team else 1 - p
            elif r.market == "totals" and r.point == r.point:
                p = float(model.p_over(g.pred_total, r.point))
                p = p if r.selection == "Over" else 1 - p
            elif r.market == "h2h":
                p = g.p_home if r.selection == r.home_team else 1 - g.p_home
            else:
                continue
            rows.append({"date": g.date, "game": f"{g.a_name} @ {g.h_name}", "market": r.market,
                         "selection": r.selection, "line": r.point, "best_odds": r.best_odds,
                         "book": r.book, "market fair p": r.fair_p, "model p": p,
                         "market edge": r.edge, "model edge": p * r.best_odds - 1})
    return pd.DataFrame(rows)


def export(code: str, days: float = 2.0, with_odds: bool = False, with_props: bool = True,
           prop_odds: bool = False, min_edge: float = 0.02, regions: str = "eu", out=None):
    st = pipeline.load(code)
    up = pipeline.upcoming(code, days)
    wb = Workbook()
    ws = wb.active
    ws.title = "Value bets"
    row, full = 1, pd.DataFrame()
    if with_odds and len(up):
        events, quota = O.fetch(O.SPORT_KEYS[code], "h2h,spreads,totals", regions)
        full = clip(O.value_table(O.flatten(events)), "commence", days)
        picks = O.picks(full, min_edge) if len(full) else full
        cols = ["commence", "home_team", "away_team", "market", "selection", "point",
                "best_odds", "book", "fair_odds", "fair_p", "edge", "fair_source"]
        if len(picks):
            row = xl.write_df(ws, picks[cols], row, pct=("fair_p", "edge"),
                              title=f"Best price beats the sharp fair price by {min_edge:.0%}+")
        else:
            ws.cell(row=row, column=1, value="No value against the sharp price right now.")
            row += 2
        ws.cell(row=row, column=1, value=f"{len(events)} games priced, credits left "
                                         f"{quota['remaining']}")
        if len(full):
            w2 = wb.create_sheet("Model vs market")
            me = model_edges(up, full, st["model"])
            if len(me):
                xl.write_df(w2, me, pct=("market fair p", "model p", "market edge",
                                         "model edge"))
                xl.autofit(w2)
    elif not with_odds:
        ws.cell(row=1, column=1, value="Run with --odds (needs ODDS_API_KEY) to list value bets.")
    xl.autofit(ws)

    wg = wb.create_sheet("Games")
    if len(up):
        xl.write_df(wg, games_table(up), pct=("P(home wins)",),
                    input_cols=("book spread (home)", "book total"))
        xl.autofit(wg)
    else:
        wg.cell(row=1, column=1, value="No games in the window.")

    if with_props and len(up) and code in ("nba", "nfl"):
        wp = wb.create_sheet("Player props")
        table = props.nba_table() if code == "nba" else pd.read_csv(
            data.DIR / "nfl_players.csv", parse_dates=["date"])
        r = 1
        for g in up.itertuples(index=False):
            if code == "nba":
                pr = pipeline.nba_game_props(g, table, pipeline.nba_injuries(g.event_id))
                tb = nba_props_table(pr) if len(pr) else pr
            else:
                pr = pipeline.nfl_game_props(g, table)
                tb = nfl_props_table(pr) if len(pr) else pr
            if tb.empty:
                continue
            pct = tuple(c for c in tb.columns if "P(over)" in c or c == "anytime TD")
            r = xl.write_df(wp, tb, r, pct=pct,
                            title=f"{g.a_name} @ {g.h_name}  ({g.date:%a %d %b %H:%M} UTC)  "
                                  f"model {g.away_pts:.0f}-{g.home_pts:.0f}")
        xl.autofit(wp)

    wr = wb.create_sheet("Power ratings")
    names = dict(zip(st["rated"]["h_id"], st["rated"]["h_name"]))
    pw = pipeline.power(code, 400 if code.startswith("ncaa") else 40)
    pw["team"] = pw["team"].map(lambda t: names.get(t, t))
    xl.write_df(wr, pw.rename(columns={"net": "net pts vs average", "off": "offence",
                                       "def": "defence"}))
    xl.autofit(wr)
    bt = {"nba": "NBA 2023-26 (5.437 αγώνες): σφάλμα διαφοράς 11,0 έναντι 10,5 της γραμμής· "
                 "στοιχήματα κόντρα στη γραμμή κλεισίματος -6%.",
          "nfl": "NFL 2020-26 (1.824 αγώνες): σφάλμα διαφοράς 10,2 έναντι 9,7 της γραμμής· "
                 "στοιχήματα κόντρα στη γραμμή κλεισίματος -5%."}.get(code, "")
    xl.notes(wb, [
        "ΠΩΣ ΔΙΑΒΑΖΕΤΑΙ",
        "Games: το μοντέλο βαθμολογεί επίθεση και άμυνα κάθε ομάδας σε πόντους και δίνει "
        "διαφορά, σύνολο και πιθανότητα νίκης. Δεν ξέρει τραυματισμούς στο επίπεδο ομάδας.",
        "Το μοντέλο ομάδων ΔΕΝ κερδίζει τις γραμμές κλεισίματος. " + bt,
        "Value bets: εδώ είναι το πλεονέκτημα. Δίκαιη τιμή από Pinnacle/ανταλλακτήρια χωρίς "
        "γκανιότα έναντι της καλύτερης διαθέσιμης τιμής.",
        "Player props: πρόσφατοι ρυθμοί του παίκτη (πιο πρόσφατα ματς μετράνε περισσότερο), "
        "προσαρμογή για την άμυνα του αντιπάλου και το αναμενόμενο σκορ. Η 'line' είναι η "
        "γραμμή που το μοντέλο θεωρεί 50/50. Δεν έχουν ελεγχθεί απέναντι σε ιστορικές "
        "αποδόσεις props (δεν υπάρχουν δωρεάν).",
        "Στα κίτρινα κελιά βάζεις τη γραμμή της δικής σου εταιρείας για σύγκριση.",
    ], "Σημειώσεις")
    REPORTS.mkdir(exist_ok=True)
    out = out or REPORTS / f"{code}_{pd.Timestamp.now(tz='UTC'):%Y-%m-%d}.xlsx"
    wb.save(out)
    return out
