"""Daily tennis workbook: value bets and model prices for ATP and WTA matches."""

from __future__ import annotations

import pandas as pd
from openpyxl import Workbook

from .. import odds as O
from .. import xl
from . import data
from . import pipeline as P

REPORTS = data.ROOT / "reports"


def match_row(state: dict, m) -> dict:
    x = P.price(state, m.tour, m.surface, m.p1, m.p2, int(m.best_of))
    need = int(m.best_of) // 2 + 1
    row = {"date": m.date, "tour": m.tour.upper(), "tournament": m.tournament,
           "round": m.round, "surface": m.surface, "player 1": m.p1, "player 2": m.p2,
           "P(1 wins)": x["p1_win"], "fair 1": 1 / x["p1_win"], "fair 2": 1 / (1 - x["p1_win"]),
           "your odds 1": None, "your odds 2": None,
           "matches 1": x["n1"], "matches 2": x["n2"]}
    for (a, b), v in sorted(x["set_scores"].items(), key=lambda kv: (-kv[0][0], kv[0][1])):
        row[f"sets {a}-{b}"] = v
    row["exp games"] = x["exp_games"]
    mid = round(x["exp_games"] - 0.5) + 0.5
    for ln in (mid - 1, mid, mid + 1):
        row[f"over {ln:g}"] = P.games_over(x["games"], ln)
    # games handicap where player 1 is closest to an even bet
    row["fair hcap 1"] = min((abs(P.handicap_p1(x["games"], h) - 0.5), h)
                             for h in [i + 0.5 for i in range(-9, 9)])[1]
    row["P(tie-break)"] = x["p_tiebreak"]
    if "aces1" in x:
        row["aces 1"], row["aces 2"] = float(x["aces1"]), float(x["aces2"])
    row["_need"] = need
    return row


def value_bets(min_edge: float) -> tuple[pd.DataFrame, list[str]]:
    tables, log = [], []
    for prefix in ("tennis_atp_", "tennis_wta_"):
        for key in O.active_sports(prefix):
            events, quota = O.fetch(key, "h2h")
            t = O.value_table(O.flatten(events))
            log.append(f"{key}: {len(events)} matches, credits left {quota['remaining']}")
            if not t.empty:
                tables.append(t.assign(league=key))
    full = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    return full, log


def export(days: float = 2.0, with_odds: bool = False, min_edge: float = 0.02, out=None):
    state = P.load_state()
    up = P.upcoming(days)
    up = up[~up["qualifying"]] if len(up) else up
    wb = Workbook()
    ws = wb.active
    ws.title = "Value bets"
    row = 1
    if with_odds:
        full, log = value_bets(min_edge)
        picks = O.picks(full, min_edge) if not full.empty else full
        cols = ["commence", "league", "home_team", "away_team", "selection", "best_odds",
                "book", "fair_odds", "fair_p", "edge", "fair_source"]
        if len(picks):
            row = xl.write_df(ws, picks[cols], row, pct=("fair_p", "edge"),
                              title=f"Best price beats the sharp fair price by {min_edge:.0%}+")
        else:
            ws.cell(row=row, column=1, value="No value against the sharp price right now.")
            row += 2
        for ln in log:
            ws.cell(row=row, column=1, value=ln)
            row += 1
        if len(full):
            w2 = wb.create_sheet("All prices")
            xl.write_df(w2, full[cols].sort_values(["commence", "home_team"]),
                        pct=("fair_p", "edge"))
            xl.autofit(w2)
    else:
        ws.cell(row=1, column=1, value="Run with --odds (needs ODDS_API_KEY) to list value bets.")
    xl.autofit(ws)
    wm = wb.create_sheet("Matches")
    if len(up):
        rows = [match_row(state, m) for m in up.itertuples(index=False)]
        mk = pd.DataFrame(rows).drop(columns="_need")
        pct = [c for c in mk.columns if c.startswith(("P(", "sets ", "over "))]
        xl.write_df(wm, mk, pct=tuple(pct), input_cols=("your odds 1", "your odds 2"))
        xl.autofit(wm)
    else:
        wm.cell(row=1, column=1, value="No scheduled matches in the window.")
    xl.notes(wb, [
        "ΠΩΣ ΔΙΑΒΑΖΕΤΑΙ",
        "Νικητής: βαθμολογία Elo συνολικά και ανά επιφάνεια (ATP από το 2015, WTA από το 2020). "
        "Σε αγώνες που δεν είχε δει (2024-26): ATP 64,6% σωστοί νικητές, WTA 64,0%.",
        "Σετ / games / tie-break: μοντέλο πόντο-πόντο, ρυθμισμένο ώστε να δίνει την ίδια "
        "πιθανότητα νίκης με το Elo και το πραγματικό ποσοστό αγώνων χωρίς χαμένο σετ.",
        "Άσοι (μόνο ATP): ρυθμός άσων του παίκτη x άσοι που δέχεται ο αντίπαλος x επιφάνεια.",
        "ΠΡΟΣΟΧΗ: δεν υπάρχουν δωρεάν ιστορικές αποδόσεις τένις προσβάσιμες από εδώ, άρα το "
        "μοντέλο ΔΕΝ έχει ελεγχθεί για κέρδος απέναντι στην αγορά. Τα Value bets βασίζονται "
        "στη σύγκριση καλύτερης τιμής με τη δίκαιη τιμή Pinnacle/ανταλλακτηρίων.",
        "Παίκτες με λίγους αγώνες (στήλες matches) έχουν αβέβαιη βαθμολογία.",
        "Στα κίτρινα κελιά βάζεις την τιμή της δικής σου εταιρείας.",
    ], "Σημειώσεις")
    REPORTS.mkdir(exist_ok=True)
    out = out or REPORTS / f"tennis_{pd.Timestamp.now(tz='UTC'):%Y-%m-%d}.xlsx"
    wb.save(out)
    return out
