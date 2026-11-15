"""One short list of the day's picks that have backtest support, a running log of
every pick, and automatic grading of past picks.

Only two kinds of pick qualify:
 * Football: the best price beats Pinnacle's no-vig price by 2%+ (2020-26, 6,660 bets,
   ROI +3.1%).
 * NBA points props: under, when the model gives it 65%+ against the DraftKings line
   (2024-26, 754 bets, ROI +12.8% against closing lines).
The statistical models' disagreements with team lines are NOT picks: the backtests show
that the biggest disagreements are mostly model error (they lose 4-8%).
"""

from __future__ import annotations

import difflib
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook

from . import xl

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "history" / "picks_log.csv"
COLS = ["day", "sport", "date", "game", "pick", "book", "odds", "model_p", "edge",
        "event_id", "player_id", "market", "selection", "line", "home", "away",
        "result", "actual", "units"]


def collect(day: str, soccer: list, nba: list) -> pd.DataFrame:
    rows = []
    for r in soccer:
        if r.get("market") == "totals":
            pick = f"{r['selection']} {r['point']:g} goals"
        elif r.get("selection") == "Draw":
            pick = "Draw"
        else:
            pick = f"{r['selection']} to win"
        rows.append({"day": day, "sport": "Football", "date": r["commence"],
                     "game": f"{r['home_team']} - {r['away_team']}", "pick": pick,
                     "book": r["book"], "odds": r["best_odds"], "model_p": r["fair_p"],
                     "edge": r["edge"], "market": r["market"], "selection": r["selection"],
                     "line": r.get("point"), "home": r["home_team"], "away": r["away_team"]})
    for r in nba:
        price = float(str(r["DK prices"]).split("/")[0])  # under price unknown: the lower one
        rows.append({"day": day, "sport": "NBA props", "date": r["date"], "game": r["game"],
                     "pick": f"{r['player']} points {r['pick'].title()}", "book": "DraftKings line",
                     "odds": price, "model_p": r["P(under)"], "edge": r["P(under)"] * price - 1,
                     "player_id": r.get("player_id"), "market": "pts", "selection": "under",
                     "line": r["DK line"]})
    df = pd.DataFrame(rows, columns=[c for c in COLS if c not in ("result", "actual", "units")])
    return df.sort_values("edge", ascending=False) if len(df) else df


def append_log(picks: pd.DataFrame) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(LOG, dtype=str) if LOG.exists() else pd.DataFrame(columns=COLS)
    new = picks.reindex(columns=COLS).astype(str).replace("nan", "")
    if len(old):
        key = lambda d: d["day"] + "|" + d["game"] + "|" + d["pick"]  # noqa: E731
        new = new[~key(new).isin(set(key(old)))]
    pd.concat([old, new], ignore_index=True).to_csv(LOG, index=False)


# ---------------------------------------------------------------- grading

def _soccer_result(r, matches: pd.DataFrame):
    d = pd.Timestamp(r["date"])
    m = matches[(matches["date"] - d).abs() <= pd.Timedelta(hours=30)]

    def sim(a, b):
        return difflib.SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()
    best, score = None, 0.0
    for x in m.itertuples(index=False):
        s = sim(r["home"], x.h_name) + sim(r["away"], x.a_name)
        if s > score:
            best, score = x, s
    if best is None or score < 1.2 or best.hg != best.hg:
        return None
    hg, ag = int(best.hg), int(best.ag)
    sel, line = r["selection"], float(r["line"]) if r["line"] not in ("", None) else None
    if r["market"] == "totals":
        tot = hg + ag
        # quarter lines: half the stake on each neighbouring line
        parts = [line] if (line * 2) % 1 == 0 else [line - 0.25, line + 0.25]
        res = []
        for ln in parts:
            res.append(0 if tot == ln else (1 if (tot > ln) == (sel == "Over") else -1))
        return f"{hg}-{ag}", res
    won = (sel == r["home"] and hg > ag) or (sel == r["away"] and ag > hg) or (
        sel == "Draw" and hg == ag)
    return f"{hg}-{ag}", [1 if won else -1]


def grade() -> pd.DataFrame:
    if not LOG.exists():
        return pd.DataFrame(columns=COLS)
    log = pd.read_csv(LOG, dtype=str).fillna("")
    todo = log["result"] == ""
    if not todo.any():
        return log
    from .soccer import espn_data
    from .us import data as udata
    matches = espn_data.load()
    try:
        box = udata.load_players("nba")
    except FileNotFoundError:
        box = pd.DataFrame()
    for i in log.index[todo]:
        r = log.loc[i]
        if pd.Timestamp(r["date"]) > pd.Timestamp.now(tz="UTC").tz_convert(None) - pd.Timedelta(
                hours=3):
            continue
        odds = float(r["odds"])
        if r["sport"] == "Football":
            out = _soccer_result(r, matches)
            if out is None:
                continue
            score, parts = out
            units = float(np.mean([(odds - 1) if p > 0 else (0 if p == 0 else -1) for p in parts]))
            log.loc[i, ["actual", "units"]] = [score, f"{units:.3f}"]
            log.loc[i, "result"] = "W" if units > 0 else ("P" if units == 0 else "L")
        elif r["sport"] == "NBA props" and len(box):
            pid = r["player_id"]
            b = box[(box["player_id"] == pid) & (box["date"] >= pd.Timestamp(r["date"]) -
                                                 pd.Timedelta(hours=6))
                    & (box["date"] <= pd.Timestamp(r["date"]) + pd.Timedelta(hours=6))]
            if b.empty:
                g_done = (box["date"] >= pd.Timestamp(r["date"]) - pd.Timedelta(hours=6)).any()
                if g_done:
                    log.loc[i, ["result", "actual", "units"]] = ["V", "did not play", "0"]
                continue
            b = b.iloc[0]
            if b.get("dnp") is True or not (b["min"] > 0):
                log.loc[i, ["result", "actual", "units"]] = ["V", "did not play", "0"]
                continue
            pts, line = float(b["pts"]), float(r["line"])
            won = pts < line
            log.loc[i, ["result", "actual", "units"]] = ["W" if won else "L", f"{pts:g} pts",
                                                         f"{(odds - 1) if won else -1:.3f}"]
    log.to_csv(LOG, index=False)
    return log


def summary(log: pd.DataFrame) -> pd.DataFrame:
    d = log[log["result"].isin(["W", "L", "P"])].copy()
    if d.empty:
        return pd.DataFrame()
    d["units"] = d["units"].astype(float)
    g = d.groupby("sport").agg(bets=("units", "size"), won=("result", lambda s: (s == "W").sum()),
                               units=("units", "sum"))
    g.loc["ALL"] = [len(d), int((d["result"] == "W").sum()), d["units"].sum()]
    g["ROI"] = g["units"] / g["bets"]
    return g.reset_index()


def export(day: str, picks: pd.DataFrame, out: Path) -> Path:
    log = grade()
    wb = Workbook()
    ws = wb.active
    ws.title = "Top picks"
    cols = ["sport", "date", "game", "pick", "book", "odds", "model_p", "edge"]
    if len(picks):
        xl.write_df(ws, picks[cols], pct=("model_p", "edge"),
                    title=f"Picks με ιστορική τεκμηρίωση για {day} (μόνο αυτά προτείνονται)")
    else:
        ws.cell(row=1, column=1, value="Καμία πρόταση με ιστορική τεκμηρίωση σήμερα. "
                                       "Αυτό είναι φυσιολογικό: πολλές μέρες δεν υπάρχει αξία.")
    xl.autofit(ws)
    wr = wb.create_sheet("Results")
    sm = summary(log)
    r = 1
    if len(sm):
        r = xl.write_df(wr, sm, pct=("ROI",), title="Απολογισμός (1 μονάδα ανά pick)")
    done = log[log["result"] != ""]
    if len(done):
        xl.write_df(wr, done[["day", "sport", "game", "pick", "odds", "result", "actual",
                              "units"]].iloc[::-1], r, title="Αποτελέσματα")
    xl.autofit(wr)
    xl.notes(wb, [
        "ΓΙΑΤΙ ΜΟΝΟ ΑΥΤΑ",
        "Ελέγξαμε κάθε μοντέλο σε χιλιάδες παλιούς αγώνες απέναντι στις γραμμές κλεισίματος.",
        "Η διαφορά μοντέλου-γραμμής στις ομάδες (spread, totals) ΧΑΝΕΙ: NBA -4%, NFL -3%, "
        "NCAAB -4%, NCAAF -8%. Όσο πιο μεγάλη η διαφορά, τόσο πιο συχνά κάνει λάθος το μοντέλο "
        "(π.χ. NCAAB διαφορές 8+ πόντων: -9,5%). Γι' αυτό δεν μπαίνουν εδώ.",
        "Football value: η καλύτερη τιμή νικά τη δίκαιη τιμή της Pinnacle κατά 2%+ "
        "(6.660 bets 2020-26, +3,1%).",
        "NBA points under: P(under) >= 65% απέναντι στη γραμμή DraftKings (754 bets 2024-26, "
        "57,8%, +12,8%). Ασίστ/ριμπάουντ props ΔΕΝ προτείνονται: δεν υπάρχει τεκμηρίωση "
        "(και εκεί ήταν οι περισσότερες χαμένες προτάσεις).",
        "Η απόδοση στο NBA είναι η χαμηλότερη από τις δύο τιμές του DraftKings (συντηρητικά). "
        "Αν η εταιρεία σου έχει άλλη γραμμή, δες το φύλλο 'Points vs DraftKings' του NBA αρχείου "
        "(P(under) σε γραμμή -1 / +1).",
    ], "Σημειώσεις")
    wb.save(out)
    return out
