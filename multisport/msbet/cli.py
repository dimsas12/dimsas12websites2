"""msbet: multi-sport betting models. Usage: msbet <sport> <command> [options]."""

from __future__ import annotations

import argparse
import warnings


def _soccer(args) -> None:
    from .soccer import data, espn_data, pipeline, report, value

    if args.command in ("update", "build"):
        if args.command == "update":
            data.download()
            df = espn_data.download()
            print(f"ESPN matches: {len(df)} ({int(df['played'].sum())} played)")
            from .soccer import props as sprops
            sprops.download_lineups(df, "2022-08-01", sprops.LINEUP_LEAGUES)
        print(pipeline.build())
    elif args.command == "backtest":
        print(value.report(args.min_edge))
    elif args.command == "predict":
        up = pipeline.upcoming(args.days, args.leagues)
        model = pipeline.load_model()
        import pandas as pd
        rows = [report.match_markets(g, model) for _, g in up.iterrows()]
        cols = ["date", "competition", "home", "away", "xG home", "xG away", "P(1)", "P(X)",
                "P(2)", "over 2.5", "BTTS", "fair AH home"]
        print(pd.DataFrame(rows)[cols].round(2).to_string(index=False) if rows
              else "no matches in the window")
    elif args.command == "excel":
        out = report.export(args.days, args.leagues, args.odds, not args.no_props,
                            args.min_edge)
        print(f"wrote {out}")


def _tennis(args) -> None:
    from .tennis import data, pipeline, report

    if args.command in ("update", "build"):
        if args.command == "update":
            df = data.download()
            data.download_tml()
            print(f"matches: {len(df)}")
        print(pipeline.build())
    elif args.command == "predict":
        import pandas as pd
        st = pipeline.load_state()
        up = pipeline.upcoming(args.days)
        rows = [report.match_row(st, m) for m in up[~up["qualifying"]].itertuples(index=False)]
        cols = ["date", "tour", "tournament", "player 1", "player 2", "P(1 wins)", "fair 1",
                "fair 2", "exp games", "P(tie-break)"]
        print(pd.DataFrame(rows)[cols].round(2).to_string(index=False) if rows
              else "no scheduled matches in the window")
    elif args.command == "excel":
        print(f"wrote {report.export(args.days, args.odds, args.min_edge)}")


def _us(args) -> None:
    from .us import data, pipeline, props, report

    code = args.sport
    if args.command in ("update", "build"):
        if args.command == "update":
            if code == "nfl":
                data.download_nfl()
                props.download_nfl_players()
            else:
                data.download_schedule(code)
                if code == "nba":
                    data.download_boxes("nba", args.first_season or 2022)
        print(pipeline.build(code))
    elif args.command == "lines":
        df = data.download_lines(code, args.first_season)
        print(f"lines for {df['spread'].notna().sum()} games")
    elif args.command == "backtest":
        from .us import model
        g = data.games_with_lines(code)
        p = model.walk_forward(g, model.CONFIGS[code], args.first_season or 2023)
        for k, v in model.evaluate(p).items():
            print(f"{k:22s} {v:.3f}" if isinstance(v, float) else f"{k:22s} {v}")
    elif args.command == "predict":
        up = pipeline.upcoming(code, args.days)
        print(report.games_table(up).drop(columns=["book spread (home)", "book total"]).round(
            2).to_string(index=False) if len(up) else "no games in the window")
    elif args.command == "excel":
        print(f"wrote {report.export(code, args.days, args.odds, not args.no_props, False, args.min_edge, args.regions)}")


def _daily(args) -> None:
    from . import daily
    daily.run(args.date, args.tz, not args.no_odds, args.only)


def main(argv=None) -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser(prog="msbet", description=__doc__)
    sports = ap.add_subparsers(dest="sport", required=True)

    s = sports.add_parser("soccer", help="football: leagues, European cups, national teams")
    s.add_argument("command", choices=["update", "build", "backtest", "predict", "excel"])
    s.add_argument("--days", type=float, default=1.5)
    s.add_argument("--leagues", nargs="*", help="ESPN league codes, e.g. eng.1 gre.1")
    s.add_argument("--odds", action="store_true", help="fetch live odds (uses API credits)")
    s.add_argument("--no-props", action="store_true")
    s.add_argument("--min-edge", type=float, default=0.02)
    s.set_defaults(func=_soccer)

    t = sports.add_parser("tennis", help="ATP and WTA tour")
    t.add_argument("command", choices=["update", "build", "predict", "excel"])
    t.add_argument("--days", type=float, default=2.0)
    t.add_argument("--odds", action="store_true")
    t.add_argument("--min-edge", type=float, default=0.02)
    t.set_defaults(func=_tennis)

    for code, helptext in (("nba", "NBA: lines and player props"),
                           ("nfl", "NFL: lines and player props"),
                           ("ncaab", "NCAA basketball: main lines"),
                           ("ncaaf", "NCAA football: main lines")):
        u = sports.add_parser(code, help=helptext)
        u.add_argument("command", choices=["update", "build", "lines", "backtest", "predict",
                                           "excel"])
        u.add_argument("--days", type=float, default=2.0 if code != "nfl" else 7.0)
        u.add_argument("--odds", action="store_true")
        u.add_argument("--no-props", action="store_true")
        u.add_argument("--min-edge", type=float, default=0.02)
        u.add_argument("--regions", default="eu")
        u.add_argument("--first-season", type=int)
        u.set_defaults(func=_us)

    d = sports.add_parser("daily", help="every sport for the next day: Excel files + one zip")
    d.add_argument("--date", help="local day to project, YYYY-MM-DD (default: tomorrow)")
    d.add_argument("--tz", default="Europe/Athens")
    d.add_argument("--no-odds", action="store_true")
    d.add_argument("--only", nargs="*", help="sports to run, e.g. nba tennis")
    d.set_defaults(func=_daily)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
