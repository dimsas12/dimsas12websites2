"""Optional start of the prediction window: MSBET_FROM=2026-11-11T22:00 (UTC)."""

import os

import pandas as pd


def now() -> pd.Timestamp:
    n = pd.Timestamp.now(tz="UTC").tz_convert(None)
    f = os.environ.get("MSBET_FROM")
    return max(n, pd.Timestamp(f)) if f else n


def clip(df: pd.DataFrame, col: str, days: float) -> pd.DataFrame:
    """Keep rows whose `col` time lies inside the window [now(), now() + days]."""
    if df is None or df.empty:
        return df
    lo = now()
    return df[(df[col] > lo) & (df[col] <= lo + pd.Timedelta(days=days))]
