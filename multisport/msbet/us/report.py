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
        "games home": up["n_h"], "games away": up["n_a"],
        **({"missing value home": up["miss_h"], "missing value away": up["miss_a"]}
           if "miss_h" in up else {})}).reset_index(drop=True)


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


# How much of the model's disagreement with the closing line is real, by league and
# market: the weight a regression of the result on (line, model - line) gives the model
# in walk-forward tests (NBA 2023-27, NFL 2013-26, NCAAB 2026-27, NCAAF 2023-26).
# Negative or zero weights are set to 0: there the model adds nothing to the line.
MODEL_WEIGHT = {"nba": {"spreads": 0.10, "totals": 0.0, "h2h": 0.10},
                "nfl": {"spreads": 0.18, "totals": 0.0, "h2h": 0.18},
                "ncaab": {"spreads": 0.0, "totals": 0.0, "h2h": 0.0},
                "ncaaf": {"spreads": 0.0, "totals": 0.0, "h2h": 0.0}}

# NBA points props: bet the under when the model gives it 65%+ against the book line.
# Test against ESPN BET / DraftKings closing lines, 2024-26, 754 bets: 57.8% won,
# ROI +12.8% (overs at the same threshold: -7.5%, so they are not flagged).
PROP_RULE = {"stat": "pts", "min_p_under": 0.65}
LAST_PICKS: list = []  # picks of the last export (collected by msbet.daily for top picks)
LAST_PICKS_NBA: list = []


def model_edges(up: pd.DataFrame, table: pd.DataFrame, model, code: str = "nba"
                ) -> pd.DataFrame:
    """Model view of the market lines. The model's number is first pulled toward the
    market by the weight the backtest gives it (MODEL_WEIGHT); the raw number is shown
    for reference only: on its own it loses to closing lines in every US league."""
    if table.empty or up.empty:
        return pd.DataFrame()
    rows = []
    for g in up.itertuples(index=False):
        t = table[(table["home_team"].fillna("").str.contains(str(g.h_name).split()[-1],
                                                              regex=False))
                  & (table["away_team"].fillna("").str.contains(str(g.a_name).split()[-1],
                                                                regex=False))]
        for r in t.itertuples(index=False):
            w = MODEL_WEIGHT.get(code, {}).get(r.market, 0.0)
            if r.market == "spreads" and r.point == r.point:
                home_line = r.point if r.selection == r.home_team else -r.point
                p_raw = float(model.p_cover(g.pred_margin, home_line))
                adj = -home_line + w * (g.pred_margin + home_line)
                p = float(model.p_cover(adj, home_line))
                if r.selection != r.home_team:
                    p, p_raw = 1 - p, 1 - p_raw
            elif r.market == "totals" and r.point == r.point:
                p_raw = float(model.p_over(g.pred_total, r.point))
                p = float(model.p_over(r.point + w * (g.pred_total - r.point), r.point))
                if r.selection != "Over":
                    p, p_raw = 1 - p, 1 - p_raw
            elif r.market == "h2h":
                p_raw = g.p_home if r.selection == r.home_team else 1 - g.p_home
                p = r.fair_p + w * (p_raw - r.fair_p)
            else:
                continue
            rows.append({"date": g.date, "game": f"{g.a_name} @ {g.h_name}", "market": r.market,
                         "selection": r.selection, "line": r.point, "best_odds": r.best_odds,
                         "book": r.book, "market fair p": r.fair_p,
                         "model p (raw, not reliable)": p_raw, "model p (market-adjusted)": p,
                         "edge (market-adjusted)": p * r.best_odds - 1})
    return pd.DataFrame(rows)


def points_vs_book(g, pr: pd.DataFrame) -> list[dict]:
    """Model probabilities against today's DraftKings points lines for one game."""
    from . import prop_lines
    try:
        lines = prop_lines.current(g.event_id)
    except Exception:  # noqa: BLE001 - lines not published yet
        return []
    if lines.empty:
        return []
    lines = lines[lines["stat"] == "pts"]
    proj = pr.set_index("player_id")
    rows = []
    for x in lines.itertuples(index=False):
        if x.player_id not in proj.index:
            continue
        p = proj.loc[x.player_id]
        p = p.iloc[0] if isinstance(p, pd.DataFrame) else p
        po = props.p_over(p["pts"], x.line, "pts", "nba")
        pu = 1 - po
        rows.append({"date": g.date, "game": f"{g.a_name} @ {g.h_name}", "player": p["player"],
                     "player_id": x.player_id,
                     "status": p.get("status", ""), "proj min": p["min"], "model PTS": p["pts"],
                     "DK line": x.line, "DK prices": f"{x.price_lo:.2f} / {x.price_hi:.2f}",
                     "P(over)": po, "P(under)": pu,
                     "P(under) line-1": 1 - props.p_over(p["pts"], x.line - 1, "pts", "nba"),
                     "P(under) line+1": 1 - props.p_over(p["pts"], x.line + 1, "pts", "nba"),
                     "pick": f"UNDER {x.line:g}" if pu >= PROP_RULE["min_p_under"] else ""})
    return rows


def export(code: str, days: float = 2.0, with_odds: bool = False, with_props: bool = True,
           prop_odds: bool = False, min_edge: float = 0.02, regions: str = "eu", out=None):
    st = pipeline.load(code)
    up = pipeline.upcoming(code, days)
    LAST_PICKS.clear()
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
            me = model_edges(up, full, st["model"], code)
            if len(me):
                xl.write_df(w2, me, pct=("market fair p", "model p (raw, not reliable)",
                                         "model p (market-adjusted)",
                                         "edge (market-adjusted)"),
                            title="ΜΗΝ παίζεις τη διαφορά μοντέλου-γραμμής: σε 4 λίγκες "
                                  "χάνει 4-8% (backtest). Η στήλη market-adjusted δείχνει "
                                  "πόσο αξίζει πραγματικά.")
                xl.autofit(w2)
    elif not with_odds:
        ws.cell(row=1, column=1, value="Run with --odds (needs ODDS_API_KEY) to list value bets.")
    xl.autofit(ws)

    wg = wb.create_sheet("Games")
    if len(up):
        xl.write_df(wg, games_table(up), pct=("P(home wins)",),
                    input_cols=("book spread (home)", "book total"),
                    title="Προβλέψεις μοντέλου (ενημερωτικά). Όταν διαφέρουν από τη γραμμή, "
                          "συνήθως κάνει λάθος το μοντέλο - δεν είναι προτάσεις.")
        xl.autofit(wg)
    else:
        wg.cell(row=1, column=1, value="No games in the window.")

    if with_props and len(up) and code in ("nba", "nfl"):
        wp = wb.create_sheet("Player props")
        table = props.nba_table() if code == "nba" else pd.read_csv(
            data.DIR / "nfl_players.csv", parse_dates=["date"])
        r = 1
        nfl_cache: dict = {}
        dk_rows: list = []
        for g in up.itertuples(index=False):
            if code == "nba":
                pr = pipeline.nba_game_props(g, table, pipeline.nba_injuries(g.event_id))
                tb = nba_props_table(pr) if len(pr) else pr
                if len(pr):
                    dk_rows += points_vs_book(g, pr)
            else:
                from .avail_us import nfl_report
                rep = nfl_cache.setdefault((int(g.season), int(g.week)),
                                           nfl_report(int(g.season), int(g.week)))
                pr = pipeline.nfl_game_props(g, table, rep)
                tb = nfl_props_table(pr) if len(pr) else pr
            if tb.empty:
                continue
            pct = tuple(c for c in tb.columns if "P(over)" in c or c == "anytime TD")
            r = xl.write_df(wp, tb, r, pct=pct,
                            title=f"{g.a_name} @ {g.h_name}  ({g.date:%a %d %b %H:%M} UTC)  "
                                  f"model {g.away_pts:.0f}-{g.home_pts:.0f}")
        xl.autofit(wp)
        if code == "nba":
            wk = wb.create_sheet("Points vs DraftKings", 1)
            dk = pd.DataFrame(dk_rows)
            if len(dk):
                dk = dk.sort_values("P(under)", ascending=False)
                xl.write_df(wk, dk, pct=("P(over)", "P(under)", "P(under) line-1",
                                         "P(under) line+1"),
                            title="Πόντοι παικτών έναντι της γραμμής DraftKings. PICK = "
                                  "under με P>=65% (backtest 2024-26: 754 bets, 57,8%, "
                                  "ROI +12,8%). Τα overs δεν προτείνονται (-7,5%).")
                LAST_PICKS.extend(dk[dk["pick"] != ""].assign(sport="NBA props").to_dict(
                    "records"))
            else:
                wk.cell(row=1, column=1, value="No DraftKings player lines published yet.")
            xl.autofit(wk)

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
