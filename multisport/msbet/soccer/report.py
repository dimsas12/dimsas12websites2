"""Daily football workbook: value bets, match markets, corners, player props."""

from __future__ import annotations

import numpy as np
import pandas as pd
from openpyxl import Workbook

from .. import odds as O
from .. import xl
from ..window import clip
from . import espn_data as E
from . import pipeline as P
from . import props
from .model import ah_ev, ah_value, score_matrix

REPORTS = E.ROOT / "reports"
DEFAULT_LEAGUES = ["eng.1", "esp.1", "ger.1", "gre.1", "uefa.champions", "uefa.europa",
                   "uefa.europa.conf"] + list(E.NATIONAL)


def match_markets(g, model) -> dict:
    M = score_matrix(g.lh, g.la, model.goals.rho)
    i = np.arange(M.shape[0])
    diff, tot = i[:, None] - i[None, :], i[:, None] + i[None, :]
    row = {"date": g.date, "competition": E.ALL.get(g["div"], g["div"]), "home": g.h_name,
           "away": g.a_name, "xG home": g.lh, "xG away": g.la,
           "P(1)": M[diff > 0].sum(), "P(X)": M[diff == 0].sum(), "P(2)": M[diff < 0].sum()}
    for k in ("1", "X", "2"):
        row[f"fair {k}"] = 1 / row[f"P({k})"]
    for ln in (1.5, 2.5, 3.5):
        row[f"over {ln}"] = M[tot > ln].sum()
    row["BTTS"] = M[1:, 1:].sum()
    # Asian handicap line where the home side is closest to an even bet
    best = min((abs(ah_ev(ah_value(M, ln)[0], 2.0)), ln) for ln in np.arange(-3, 3.25, 0.25))
    row["fair AH home"] = best[1]
    top = np.dstack(np.unravel_index(np.argsort(M.ravel())[::-1][:3], M.shape))[0]
    row["likely scores"] = ", ".join(f"{a}-{b} ({M[a, b]:.0%})" for a, b in top)
    row["corners home"], row["corners away"] = g.ch, g.ca
    for ln in (8.5, 9.5, 10.5):
        row[f"corners over {ln}"] = model.corner_probs(g.ch + g.ca, ln)[0]
    return row


def value_bets(leagues: list[str], min_edge: float, markets: str = "h2h,totals"
               ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """(picks, full price table, log lines). One Odds API call per league."""
    tables, log = [], []
    for lg in leagues:
        key = O.SPORT_KEYS.get(lg)
        if not key:
            continue
        try:
            events, quota = O.fetch(key, markets)
        except Exception as exc:  # noqa: BLE001 - out-of-season leagues return 404
            log.append(f"{lg}: no odds ({exc.__class__.__name__})")
            continue
        t = O.value_table(O.flatten(events))
        log.append(f"{lg}: {len(events)} events, credits left {quota['remaining']}")
        if not t.empty:
            tables.append(t.assign(league=E.ALL.get(lg, lg)))
    full = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    return (O.picks(full, min_edge) if not full.empty else full), full, log


def export(days: float = 1.5, leagues: list[str] | None = None, with_odds: bool = False,
           with_props: bool = True, min_edge: float = 0.02, out=None):
    leagues = leagues or DEFAULT_LEAGUES
    model = P.load_model()
    up = P.upcoming(days, leagues)
    wb = Workbook()
    ws = wb.active
    ws.title = "Value bets"
    row = 1
    if with_odds:
        played = [lg for lg in leagues if lg in set(up["div"])] if len(up) else []
        picks, full, log = value_bets(played, min_edge)
        picks, full = clip(picks, "commence", days), clip(full, "commence", days)
        cols = ["commence", "league", "home_team", "away_team", "market", "selection", "point",
                "best_odds", "book", "fair_odds", "fair_p", "edge", "fair_source"]
        if not picks.empty:
            row = xl.write_df(ws, picks[cols], row, pct=("fair_p", "edge"),
                              title=f"Bets where the best price beats the sharp fair price by "
                                    f"{min_edge:.0%}+ (backtest 2020-26: +3.1% ROI)")
        else:
            ws.cell(row=row, column=1, value="No price beats the sharp fair price by "
                                             f"{min_edge:.0%} right now.")
            row += 2
        for ln in log:
            ws.cell(row=row, column=1, value=ln)
            row += 1
        if not full.empty:
            w2 = wb.create_sheet("All prices")
            xl.write_df(w2, full[cols].sort_values(["commence", "home_team", "market"]),
                        pct=("fair_p", "edge"))
            xl.autofit(w2)
    else:
        ws.cell(row=1, column=1, value="Run with --odds (needs ODDS_API_KEY) to list value bets.")
    xl.autofit(ws)

    wm = wb.create_sheet("Matches")
    if len(up):
        mk = pd.DataFrame([match_markets(g, model) for _, g in up.iterrows()])
        mk.insert(9, "your odds 1", None)
        mk.insert(10, "your odds X", None)
        mk.insert(11, "your odds 2", None)
        pct = [c for c in mk.columns if c.startswith(("P(", "over", "BTTS", "corners over"))]
        xl.write_df(wm, mk, pct=tuple(pct), input_cols=("your odds 1", "your odds X",
                                                        "your odds 2"))
        xl.autofit(wm)
    else:
        wm.cell(row=1, column=1, value="No matches in the window.")

    if with_props and len(up):
        wp = wb.create_sheet("Player props")
        rated = pd.read_pickle(P.MODEL_PATH.with_name("rated_espn.pkl"))
        r = 1
        for _, g in up.iterrows():
            for tid, name, lam in ((g.home, g.h_name, g.lh), (g.away, g.a_name, g.la)):
                hist = props.team_history(rated, tid, g.date)
                if hist.empty:
                    continue
                pr = props.project(hist, lam)
                pout = P.ABSENT.get(tid, {})
                if pout and len(pr):
                    pr["p_out"] = pr["player_id"].map(pout).fillna(0.0)
                    pr = pr[pr["p_out"] < 0.8].copy()
                    pr["p_start"] = pr["p_start"] * (1 - pr["p_out"])
                tb = props.prop_table(pr)
                if "p_out" in pr and len(tb):
                    tb.insert(4, "P(out)", pr["p_out"].to_numpy())
                if tb.empty:
                    continue
                pct = [c for c in tb.columns if c in ("p_start", "scorer", "2+ goals", "assist",
                                                      "card", "P(out)") or c.endswith("+")]
                r = xl.write_df(wp, tb, r, pct=tuple(pct),
                                title=f"{g.h_name} - {g.a_name}: {name} (team xG {lam:.2f}) - "
                                      f"probabilities if the player starts")
        xl.autofit(wp)

    if P.ABSENT:
        from .avail_soccer import absence_table
        rated_all = pd.read_pickle(P.MODEL_PATH.with_name("rated_espn.pkl"))
        names = dict(zip(rated_all["home"], rated_all["h_name"])) | dict(
            zip(rated_all["away"], rated_all["a_name"]))
        wa = wb.create_sheet("Absences")
        at = absence_table(P.ABSENT, names)
        xl.write_df(wa, at, pct=("P(out)",),
                    title="Players assumed absent (their chance of missing the match)")
        xl.autofit(wa)
    xl.notes(wb, [
        "ΠΩΣ ΔΙΑΒΑΖΕΤΑΙ",
        "Value bets: η δίκαιη πιθανότητα βγαίνει από Pinnacle/ανταλλακτήρια χωρίς τη γκανιότα. "
        "Στοίχημα προτείνεται όταν η καλύτερη διαθέσιμη τιμή την ξεπερνά κατά 2%+. "
        "Έλεγχος σε 6.660 στοιχήματα 2020-26: +3,1% απόδοση, θετική σε 9 από 11 σεζόν.",
        "Matches: πιθανότητες του στατιστικού μοντέλου (γκολ + σουτ, 2017-σήμερα). Το μοντέλο "
        "μόνο του ΔΕΝ κερδίζει την αγορά στα βασικά σημεία (log loss 0,985 έναντι 0,968). "
        "Χρησιμοποίησέ το για αγορές χωρίς αξιόπιστη τιμή αναφοράς (σκορ, κόρνερ, props) "
        "και ως έλεγχο λογικής, όχι για 1Χ2.",
        "Στα κίτρινα κελιά βάζεις την τιμή της δικής σου εταιρείας (π.χ. Stoiximan, Novibet).",
        "Player props: ρυθμοί ανά 90' από τα τελευταία 10 ματς, με σύγκλιση προς τον μέσο όρο "
        "της θέσης, προσαρμοσμένοι στα αναμενόμενα γκολ της ομάδας. Ισχύουν αν ο παίκτης "
        "ξεκινήσει βασικός. Δεν έχουν ελεγχθεί απέναντι σε ιστορικές αποδόσεις props.",
        "Κόρνερ: μέτρια προβλεψιμότητα (συσχέτιση 0,16 με το πραγματικό σύνολο).",
    ], "Σημειώσεις")
    REPORTS.mkdir(exist_ok=True)
    day = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    out = out or REPORTS / f"soccer_{day}.xlsx"
    wb.save(out)
    return out
