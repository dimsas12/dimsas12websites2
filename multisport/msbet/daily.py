"""Daily run: projections for every sport for the next local day, one Excel per sport
plus one zip with everything.

    msbet daily                    # target = tomorrow in Europe/Athens
    msbet daily --date 2026-11-13  # a specific local day

The window is the local day (00:00-24:00) plus 8 hours, so US evening games of that
day (early next morning in UTC) are included. A failing sport never stops the others;
failures are listed in summary.txt inside the zip.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EURO = ROOT.parent / "euroleague-model"
WINDOW_DAYS = 1.34  # 24 h + 8 h


def _recalc(path: Path) -> None:
    for script in glob.glob("/root/.claude/skills/synced/*/xlsx/scripts/recalc.py"):
        subprocess.run([sys.executable, script, str(path)], capture_output=True, timeout=300)
        return


def _run(cmd: list[str], cwd: Path, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=3400)


def run(date: str | None = None, tz: str = "Europe/Athens", odds: bool = True,
        only: list[str] | None = None) -> Path:
    from .soccer import data as sdata
    from .soccer import espn_data, pipeline as spipe, props as sprops, report as sreport
    from .tennis import data as tdata, pipeline as tpipe, report as treport
    from .us import data as udata, pipeline as upipe, props as uprops, report as ureport

    local = pd.Timestamp(date) if date else (
        pd.Timestamp.now(tz=tz).normalize().tz_localize(None) + pd.Timedelta(days=1))
    start = pd.Timestamp(local.date(), tz=tz).tz_convert("UTC").tz_localize(None)
    os.environ["MSBET_FROM"] = f"{start:%Y-%m-%dT%H:%M}"
    have_key = bool(os.environ.get("ODDS_API_KEY"))
    odds = odds and have_key
    out_dir = ROOT / "reports" / "daily" / f"{local:%Y-%m-%d}"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    log: list[str] = [f"Projections for {local:%A %d %B %Y} ({tz}); window "
                      f"{start:%d %b %H:%M} UTC + {WINDOW_DAYS * 24:.0f} h",
                      "Live odds: " + ("on" if odds else "OFF (no ODDS_API_KEY)")]
    files: list[Path] = []

    def step(name: str, fn) -> None:
        if only and name not in only:
            return
        try:
            res = fn()
            files.append(Path(res))
            log.append(f"OK    {name}")
        except Exception as exc:  # noqa: BLE001 - one sport must not stop the rest
            log.append(f"FAIL  {name}: {exc.__class__.__name__}: {exc}")
            traceback.print_exc()

    def with_fallback(fn_odds, fn_plain, name):
        def go():
            if odds:
                try:
                    return fn_odds()
                except Exception as exc:  # noqa: BLE001 - e.g. sport out of season on the API
                    log.append(f"note  {name}: odds failed ({exc.__class__.__name__}), "
                               "projections without odds")
            return fn_plain()
        return go

    # ---- football
    def soccer():
        sdata.download()
        matches = espn_data.download()
        sprops.download_lineups(matches, "2022-08-01", sprops.LINEUP_LEAGUES)
        spipe.build()
        leagues = list(espn_data.ALL)
        out = out_dir / f"soccer_{local:%Y-%m-%d}.xlsx"
        return with_fallback(
            lambda: sreport.export(WINDOW_DAYS, leagues, True, True, 0.02, out),
            lambda: sreport.export(WINDOW_DAYS, leagues, False, True, 0.02, out), "soccer")()

    # ---- tennis
    def tennis():
        tdata.download()
        tdata.download_tml()
        tpipe.build()
        out = out_dir / f"tennis_{local:%Y-%m-%d}.xlsx"
        return with_fallback(lambda: treport.export(WINDOW_DAYS, True, 0.02, out),
                             lambda: treport.export(WINDOW_DAYS, False, 0.02, out), "tennis")()

    # ---- US sports
    def us(code):
        def go():
            if code == "nfl":
                udata.download_nfl()
                uprops.download_nfl_players()
            else:
                udata.download_schedule(code)
                if code == "nba":
                    udata.download_boxes("nba", 2022)
            upipe.build(code)
            out = out_dir / f"{code}_{local:%Y-%m-%d}.xlsx"
            return with_fallback(
                lambda: ureport.export(code, WINDOW_DAYS, True, True, False, 0.02, "eu", out),
                lambda: ureport.export(code, WINDOW_DAYS, False, True, False, 0.02, "eu", out),
                code)()
        return go

    # ---- EuroLeague (separate project)
    def euroleague():
        env = {**os.environ, "ELBET_HOME": str(EURO)}
        out = out_dir / f"euroleague_{local:%Y-%m-%d}.xlsx"
        for cmd in (["update"], ["picks", "--days", "3", "--no-log"] if odds else None,
                    ["excel", "--date", f"{local:%Y-%m-%d}", "--out", str(out)]):
            if cmd is None:
                continue
            p = _run([sys.executable, "-m", "elbet.cli", *cmd], EURO, env)
            if p.returncode and cmd[0] != "picks":
                raise RuntimeError(f"elbet {cmd[0]}: {p.stderr.strip().splitlines()[-1:]}")
            if p.returncode:
                log.append(f"note  euroleague picks failed: {p.stderr.strip().splitlines()[-1:]}")
        _recalc(out)
        return out

    step("soccer", soccer)
    step("tennis", tennis)
    for code in ("nba", "nfl", "ncaab", "ncaaf"):
        step(code, us(code))
    step("euroleague", euroleague)

    (out_dir / "summary.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    zip_path = out_dir / f"projections_{local:%Y-%m-%d}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files + [out_dir / "summary.txt"]:
            if f.exists():
                z.write(f, f.name)
    print("\n".join(log))
    print(f"zip: {zip_path}")
    return out_dir
