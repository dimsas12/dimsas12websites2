"""Load football-data.co.uk match files into one table.

Main-league files (E0, SP1, ...) carry results, shots, corners, cards, opening and
closing odds (1X2, over/under 2.5, Asian handicap) and, in recent seasons, xG.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "soccer" / "raw"
BASE = "https://www.football-data.co.uk/mmz4281"

LEAGUES = {
    "E0": "Premier League", "E1": "Championship", "SP1": "La Liga", "D1": "Bundesliga",
    "I1": "Serie A", "F1": "Ligue 1", "G1": "Super League Greece", "N1": "Eredivisie",
    "P1": "Primeira Liga", "B1": "Belgian Pro League", "T1": "Super Lig", "SC0": "Scottish Prem",
}

RENAME = {
    "FTHG": "hg", "FTAG": "ag", "HS": "hs", "AS": "as_", "HST": "hst", "AST": "ast",
    "HC": "hc", "AC": "ac", "HY": "hy", "AY": "ay", "HR": "hr", "AR": "ar",
    "HF": "hf", "AF": "af", "HxG": "hxg", "AxG": "axg", "Referee": "referee",
}
# (our name, candidates in order of preference)
ODDS = {
    "o_h": ["PSH", "AvgH", "BbAvH", "B365H"], "o_d": ["PSD", "AvgD", "BbAvD", "B365D"],
    "o_a": ["PSA", "AvgA", "BbAvA", "B365A"],
    "max_h": ["MaxH", "BbMxH"], "max_d": ["MaxD", "BbMxD"], "max_a": ["MaxA", "BbMxA"],
    "avg_h": ["AvgH", "BbAvH"], "avg_d": ["AvgD", "BbAvD"], "avg_a": ["AvgA", "BbAvA"],
    "c_h": ["PSCH", "AvgCH"], "c_d": ["PSCD", "AvgCD"], "c_a": ["PSCA", "AvgCA"],
    "cmax_h": ["MaxCH"], "cmax_d": ["MaxCD"], "cmax_a": ["MaxCA"],
    "o_over": ["P>2.5", "Avg>2.5", "BbAv>2.5", "B365>2.5"],
    "o_under": ["P<2.5", "Avg<2.5", "BbAv<2.5", "B365<2.5"],
    "max_over": ["Max>2.5", "BbMx>2.5"], "max_under": ["Max<2.5", "BbMx<2.5"],
    "c_over": ["PC>2.5", "AvgC>2.5"], "c_under": ["PC<2.5", "AvgC<2.5"],
    "ah_line": ["AHh", "BbAHh"], "ah_h": ["PAHH", "AvgAHH", "BbAvAHH"],
    "ah_a": ["PAHA", "AvgAHA", "BbAvAHA"], "max_ah_h": ["MaxAHH", "BbMxAHH"],
    "max_ah_a": ["MaxAHA", "BbMxAHA"],
    "c_ah_line": ["AHCh"], "c_ah_h": ["PCAHH", "AvgCAHH"], "c_ah_a": ["PCAHA", "AvgCAHA"],
}


def season_codes(first: int = 2012, last: int | None = None) -> list[str]:
    last = last or (pd.Timestamp.now().year if pd.Timestamp.now().month >= 7
                    else pd.Timestamp.now().year - 1)
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(first, last + 1)]


def download(leagues=None, seasons=None, refresh_last: int = 1) -> None:
    import requests

    RAW.mkdir(parents=True, exist_ok=True)
    seasons = seasons or season_codes()
    for i, s in enumerate(seasons):
        for lg in leagues or LEAGUES:
            f = RAW / f"{lg}_{s}.csv"
            if f.exists() and f.stat().st_size > 500 and i < len(seasons) - refresh_last:
                continue
            r = requests.get(f"{BASE}/{s}/{lg}.csv", headers={"User-Agent": "Mozilla/5.0"},
                             timeout=40)
            if r.ok and len(r.content) > 500:
                f.write_bytes(r.content)


def _read(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "Date"])
    out = pd.DataFrame({
        "div": df["Div"], "home": df["HomeTeam"].str.strip(), "away": df["AwayTeam"].str.strip(),
        "date": pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce"),
    })
    for src, dst in RENAME.items():
        out[dst] = df[src] if src in df else np.nan
    for dst, cands in ODDS.items():
        col = pd.Series(np.nan, index=df.index, dtype=float)
        for c in cands:
            if c in df:
                col = col.fillna(pd.to_numeric(df[c], errors="coerce"))
        out[dst] = col
    code = path.stem.split("_")[1]
    out["season"] = 2000 + int(code[:2])
    return out


def load(leagues=None) -> pd.DataFrame:
    frames = [_read(p) for p in sorted(RAW.glob("*_*.csv"))
              if not p.name.startswith("new_") and p.stat().st_size > 500
              and (leagues is None or p.stem.split("_")[0] in leagues)]
    df = pd.concat(frames, ignore_index=True).dropna(subset=["date"])
    num = [c for c in df.columns if c not in ("div", "home", "away", "date", "referee")]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df["played"] = df["hg"].notna() & df["ag"].notna()
    return df.sort_values(["date", "div", "home"]).reset_index(drop=True)
