"""Small Excel helpers shared by the daily workbooks."""

from __future__ import annotations

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

HEAD = PatternFill("solid", fgColor="1F3864")
INPUT = PatternFill("solid", fgColor="FFF2CC")
GOOD = PatternFill("solid", fgColor="C6EFCE")


def write_df(ws, df: pd.DataFrame, row: int = 1, pct: tuple = (), dec2: tuple = (),
             title: str | None = None, input_cols: tuple = ()) -> int:
    """Write a table at `row`; returns the next free row."""
    if title:
        ws.cell(row=row, column=1, value=title).font = Font(bold=True, size=12)
        row += 1
    for j, c in enumerate(df.columns, 1):
        cell = ws.cell(row=row, column=j, value=str(c))
        cell.font, cell.fill = Font(bold=True, color="FFFFFF"), HEAD
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
    for i, rec in enumerate(df.itertuples(index=False), row + 1):
        for j, (c, v) in enumerate(zip(df.columns, rec), 1):
            if isinstance(v, float) and v != v:
                v = None
            elif hasattr(v, "item"):
                v = v.item()
            if isinstance(v, pd.Timestamp):
                v = v.strftime("%a %d %b %H:%M")
            cell = ws.cell(row=i, column=j, value=v)
            if c in pct:
                cell.number_format = "0.0%"
            elif c in dec2 or isinstance(v, float):
                cell.number_format = "0.00"
            if c in input_cols:
                cell.fill = INPUT
    return row + len(df) + 2


def autofit(ws, max_width: int = 34) -> None:
    widths: dict[int, int] = {}
    for r in ws.iter_rows():
        for c in r:
            if c.value is not None:
                n = len(f"{c.value:.2f}") if isinstance(c.value, float) else len(str(c.value))
                widths[c.column] = max(widths.get(c.column, 0), n)
    for col, w in widths.items():
        ws.column_dimensions[get_column_letter(col)].width = min(max(w + 2, 7), max_width)


def notes(wb: Workbook, lines: list[str], title: str = "Notes") -> None:
    ws = wb.create_sheet(title)
    for i, ln in enumerate(lines, 1):
        ws.cell(row=i, column=1, value=ln).alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = 120
