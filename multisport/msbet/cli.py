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

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
