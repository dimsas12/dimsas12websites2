"""Command line interface: `elbet <command>`."""

from __future__ import annotations

import argparse
import logging
import signal
import sys

import numpy as np
import pandas as pd

from . import backtest, betting, config, dataset, fetch, odds, pipeline
from .config import SETTINGS
from .models import GameModel, spread_probs, total_probs
from .props import PROP_STATS, nb_median, nb_probs

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def _seasons(args) -> list[int]:
    return list(range(args.start, args.end + 1))


def _markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


def _write_report(name: str, text: str) -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORTS_DIR / name
    path.write_text(text)
    print(f"\nreport written to {path}")


# ---------------------------------------------------------------- commands

def cmd_fetch(args):
    fetch.fetch_seasons(_seasons(args), workers=args.workers)


def cmd_build(args):
    info = pipeline.build(_seasons(args))
    print(f"games: {info['games']} ({info['played']} played, {info['with_box']} with box score)")
    print(f"player box rows: {info['player_rows']}")
    print(f"margin SD {info['margin_sd']:.2f}, total SD {info['total_sd']:.2f}")
    print("props NB dispersion:", {k: round(v, 3) for k, v in info["props_alpha"].items()})


def cmd_update(args):
    cur = config.current_season()
    fetch.fetch_seasons([cur], workers=args.workers)
    args.end = max(args.end, cur)
    cmd_build(args)


def cmd_backtest(args):
    feats = pipeline.load_features()
    pred = backtest.walk_forward(feats, args.from_season, SETTINGS)
    metrics = backtest.game_metrics(pred, SETTINGS)
    calib = backtest.calibration(pred)
    lines = ["# EuroLeague model backtest (walk-forward)\n",
             "Each season predicted by a model trained only on earlier seasons; every rating uses",
             "only games before tip-off.\n", "## Games\n", metrics.round(4).pipe(_markdown),
             "\n## Win probability calibration\n", calib.round(3).pipe(_markdown)]
    print(metrics.round(4).to_string(index=False))
    print("\ncalibration\n", calib.round(3).to_string(index=False))

    if args.odds_csv:
        odds_hist = pd.read_csv(args.odds_csv)
        bets, summary = backtest.simulate_bets(pred, odds_hist, SETTINGS.bet)
        if summary.empty:
            print("\nno qualifying bets against the supplied odds")
        else:
            print("\nbetting simulation\n", summary.round(4).to_string(index=False))
            print(f"bankroll 1000 -> {bets['bankroll'].iloc[-1]:.2f} "
                  f"({SETTINGS.bet.kelly_fraction} Kelly, {SETTINGS.bet.max_stake_pct:.0%} cap)")
            lines += ["\n## Betting simulation\n", summary.round(4).pipe(_markdown)]

    hist_path = pipeline.PROPS_HISTORY_PATH
    if hist_path.exists():
        props = pipeline.load_props_model()
        pb = backtest.props_backtest(pd.read_csv(hist_path), args.from_season, props)
        print("\nplayer props\n", pb.round(3).to_string(index=False))
        lines += ["\n## Player props\n",
                  "Synthetic line = player's recent average rounded to .5 (real prop lines are "
                  "not available historically).\n", pb.round(3).pipe(_markdown)]
    _write_report("backtest.md", "\n".join(lines) + "\n")


def cmd_ratings(args):
    feats = pipeline.load_features()
    r = pipeline.power_ratings(feats)
    print(r.round(1).to_string())


def cmd_predict(args):
    feats = pipeline.load_features()
    model = GameModel.load(pipeline.GAME_MODEL_PATH)
    up = pipeline.upcoming(feats, args.days)
    if up.empty:
        print(f"no games in the next {args.days} days (run `elbet update` to refresh)")
        return
    pred = pipeline.predict_games(up, model)
    show = pred[["game_id", "date", "home", "away", "p_home", "fair_home_ml", "fair_away_ml",
                 "pred_margin", "fair_home_line", "pred_total", "pred_home_pts",
                 "pred_away_pts"]].copy()
    show["date"] = show["date"].dt.strftime("%a %d %b %H:%M")
    print(show.round(2).to_string(index=False))
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    show.to_csv(config.REPORTS_DIR / "predictions.csv", index=False)


def _blend(p_model: float, p_market: float | None, w: float) -> float:
    if p_market is None or w <= 0:
        return p_model
    return (1 - w) * p_model + w * p_market


def cmd_picks(args):
    feats = pipeline.load_features()
    model = GameModel.load(pipeline.GAME_MODEL_PATH)
    up = pipeline.upcoming(feats, args.days)
    if up.empty:
        print(f"no games in the next {args.days} days")
        return
    pred = pipeline.predict_games(up, model).set_index("game_id")
    events, quota = odds.fetch_odds(regions=args.regions, bookmakers=args.bookmakers)
    print(f"Odds API: {len(events)} events, credits remaining {quota['x-requests-remaining']}")
    flat = odds.flatten(events)
    if flat.empty:
        print("no odds returned")
        return
    matches = odds.match_events(flat, pred.reset_index())
    bet_cfg = SETTINGS.bet
    min_edge = args.min_edge if args.min_edge is not None else bet_cfg.min_edge
    rows = []
    for event_id, (gid, swapped) in matches.items():
        ev = flat[flat["event_id"] == event_id]
        if swapped:  # Odds API lists the teams the other way round
            ev = ev.assign(home_team=ev["away_team"].iloc[0], away_team=ev["home_team"].iloc[0])
        m = odds.market_lines(ev)
        g = pred.loc[gid]
        cands = []
        if m.get("h2h"):
            ph = _blend(g.p_home, m["h2h"]["fair_home"], args.market_weight)
            cands += [("moneyline", g.home, None, *m["h2h"]["home"], ph, 0.0,
                       m["h2h"]["fair_home"]),
                      ("moneyline", g.away, None, *m["h2h"]["away"], 1 - ph, 0.0,
                       None if m["h2h"]["fair_home"] is None else 1 - m["h2h"]["fair_home"])]
        if m.get("spreads"):
            L = m["spreads"]["line"]
            ph, pp, pa = spread_probs(g.pred_margin, g.margin_sd, L)
            fh = m["spreads"]["fair_home"]
            ph_b = _blend(ph / (ph + pa), fh, args.market_weight) * (1 - pp)
            cands += [("spread", g.home, L, *m["spreads"]["home"], ph_b, pp, fh),
                      ("spread", g.away, -L, *m["spreads"]["away"], 1 - pp - ph_b, pp,
                       None if fh is None else 1 - fh)]
        if m.get("totals"):
            L = m["totals"]["line"]
            po, pp, pu = total_probs(g.pred_total, g.total_sd, L)
            fo = m["totals"]["fair_over"]
            po_b = _blend(po / (po + pu), fo, args.market_weight) * (1 - pp)
            cands += [("total", "over", L, *m["totals"]["over"], po_b, pp, fo),
                      ("total", "under", L, *m["totals"]["under"], 1 - pp - po_b, pp,
                       None if fo is None else 1 - fo)]
        for market, side, line, price, book, p, push, p_mkt in cands:
            ev_ = betting.expected_value(p, price, push)
            rows.append({
                "date": g.date.strftime("%a %d %b %H:%M"), "game": f"{g.home}-{g.away}",
                "market": market, "pick": side, "line": line, "best_odds": price, "book": book,
                "model_p": p, "market_p": p_mkt, "fair_odds": 1 / p if p > 0 else np.inf,
                "ev": ev_, "stake": betting.stake(p, price, args.bankroll, args.kelly,
                                                  bet_cfg.max_stake_pct, push),
            })
    unmatched = len(flat["event_id"].unique()) - len(matches)
    if unmatched:
        print(f"warning: {unmatched} Odds API events could not be matched to the schedule")
    allbets = pd.DataFrame(rows)
    if allbets.empty:
        print("no markets matched")
        return
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    allbets.to_csv(config.REPORTS_DIR / f"markets_{stamp}.csv", index=False)
    picks = allbets[(allbets["ev"] >= min_edge) & (allbets["model_p"] >= bet_cfg.min_prob)
                    & (allbets["best_odds"] <= bet_cfg.max_odds)]
    picks = picks.sort_values("ev", ascending=False)
    if bet_cfg.one_bet_per_game and not args.all_markets:
        picks = picks.drop_duplicates("game")
    cap = bet_cfg.max_total_exposure * args.bankroll
    if picks["stake"].sum() > cap:
        picks = picks.assign(stake=(picks["stake"] * cap / picks["stake"].sum()).round(2))
    if picks.empty:
        print(f"no bets with EV >= {min_edge:.1%}. Full board in reports/markets_{stamp}.csv")
        return
    print(picks.round(3).to_string(index=False))
    picks.to_csv(config.REPORTS_DIR / f"picks_{stamp}.csv", index=False)
    print(f"\n{len(picks)} picks, total stake {picks['stake'].sum():.2f} "
          f"of bankroll {args.bankroll:.0f}; saved reports/picks_{stamp}.csv")


def _match_player(name: str, players: pd.Series) -> str | None:
    """Match 'Sasha Vezenkov' / 'VEZENKOV, SASHA' / 'Vezenkov' to a projection row."""
    toks = {t for t in odds._norm(name).split() if len(t) > 1}
    best, score = None, 0.0
    for p in players:
        ptoks = set(odds._norm(p).split())
        s = len(toks & ptoks) / max(1, len(toks))
        if s > score:
            best, score = p, s
    return best if score >= 0.5 else None


def cmd_props(args):
    feats = pipeline.load_features().set_index("game_id")
    if args.game not in feats.index:
        up = pipeline.upcoming(feats.reset_index(), 10)
        print(f"unknown game {args.game}. Upcoming games:")
        print(up[["game_id", "date", "home", "away"]].to_string(index=False))
        return
    game = feats.loc[args.game]
    _, players = dataset.load_tables()
    props = pipeline.load_props_model()
    proj = pipeline.game_props(game, players, props, set(args.exclude or []))
    if proj.empty:
        print("no player history for these teams yet")
        return
    print(f"{game['home']} vs {game['away']}  ({game['date']:%a %d %b %H:%M})  "
          f"expected pace {game['pace_exp']:.1f}")
    show = proj[["team", "player", "exp_min", *PROP_STATS]].copy()
    for s in PROP_STATS:
        show[f"{s}_med"] = [nb_median(m, props.alpha[s]) for m in proj[s]]
    print(show.round(1).to_string(index=False))

    lines = None
    if args.lines:
        lines = pd.read_csv(args.lines)
    elif args.odds_api:
        lines = _odds_api_prop_lines(game, args.regions)
    if lines is None or lines.empty:
        return
    rows = []
    for ln in lines.itertuples(index=False):
        who = _match_player(ln.player, proj["player"])
        if who is None or ln.stat not in PROP_STATS:
            print(f"skip: {ln.player} {ln.stat} (no projection)")
            continue
        mean = float(proj.loc[proj["player"] == who, ln.stat].iloc[0])
        po, pp, pu = nb_probs(mean, props.alpha[ln.stat], float(ln.line))
        for side, p, price in (("over", po, getattr(ln, "over_odds", np.nan)),
                               ("under", pu, getattr(ln, "under_odds", np.nan))):
            ev_ = betting.expected_value(p, price, pp) if pd.notna(price) else np.nan
            rows.append({"player": who, "stat": ln.stat, "line": ln.line, "side": side,
                         "proj": mean, "model_p": p, "fair_odds": 1 / p if p > 0 else np.inf,
                         "odds": price, "ev": ev_,
                         "stake": betting.stake(p, price, args.bankroll, args.kelly,
                                                SETTINGS.bet.max_stake_pct, pp)
                         if pd.notna(price) else np.nan})
    res = pd.DataFrame(rows)
    print("\nprop lines\n", res.round(3).to_string(index=False))
    good = res[res["ev"] >= SETTINGS.bet.min_edge]
    if not good.empty:
        print("\nvalue props\n", good.sort_values("ev", ascending=False).round(3)
              .to_string(index=False))


def _odds_api_prop_lines(game: pd.Series, regions: str) -> pd.DataFrame | None:
    events, _ = odds.fetch_odds(regions=regions, markets="h2h")
    flat = odds.flatten(events)
    if flat.empty:
        print("no Odds API events")
        return None
    g = game.to_frame().T.reset_index(names="game_id")
    g["date"] = pd.to_datetime(g["date"])
    matches = {e: v for e, v in odds.match_events(flat, g).items() if v[0] == game.name}
    if not matches:
        print("game not found on The Odds API")
        return None
    event_id = next(iter(matches))
    try:
        data, _ = odds.fetch_event_props(event_id, regions)
    except odds.OddsApiError as exc:
        print(f"props not available from The Odds API for EuroLeague: {exc}")
        return None
    pf = odds.flatten([data])
    if pf.empty:
        print("The Odds API returned no player props for this game; use --lines CSV instead")
        return None
    rows = []
    for (player, market, point), d in pf.groupby(["description", "market", "point"]):
        over, under = d[d["name"] == "Over"], d[d["name"] == "Under"]
        rows.append({"player": player, "stat": odds.PROP_STAT.get(market, market), "line": point,
                     "over_odds": over["price"].max() if not over.empty else np.nan,
                     "under_odds": under["price"].max() if not under.empty else np.nan})
    return pd.DataFrame(rows)


def cmd_odds_history(args):
    """Fetch historical pre-game odds (paid Odds API plan) into a backtest CSV."""
    feats = pipeline.load_features()
    season = feats[(feats["season"] == args.season) & feats["played"]]
    days = sorted(season["date"].dt.normalize().unique())
    print(f"{len(days)} game days; each snapshot costs ~30 credits (3 markets, 1 region)")
    rows = []
    for day in days:
        first_tip = season[season["date"].dt.normalize() == day]["date"].min()
        snap_time = (first_tip - pd.Timedelta(hours=args.hours_before)).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        try:
            events, quota = odds.fetch_historical(snap_time, regions=args.regions)
        except odds.OddsApiError as exc:
            print(f"{snap_time}: {exc}")
            break
        flat = odds.flatten(events)
        if flat.empty:
            continue
        for event_id, (gid, swapped) in odds.match_events(flat, season).items():
            ev = flat[flat["event_id"] == event_id]
            if swapped:
                ev = ev.assign(home_team=ev["away_team"].iloc[0],
                               away_team=ev["home_team"].iloc[0])
            m = odds.market_lines(ev)
            rows.append({
                "game_id": gid,
                "home_ml": m.get("h2h", {}).get("home", (np.nan,))[0],
                "away_ml": m.get("h2h", {}).get("away", (np.nan,))[0],
                "home_line": m.get("spreads", {}).get("line", np.nan),
                "home_spread_odds": m.get("spreads", {}).get("home", (np.nan,))[0],
                "away_spread_odds": m.get("spreads", {}).get("away", (np.nan,))[0],
                "total_line": m.get("totals", {}).get("line", np.nan),
                "over_odds": m.get("totals", {}).get("over", (np.nan,))[0],
                "under_odds": m.get("totals", {}).get("under", (np.nan,))[0],
            })
        print(f"{snap_time}: {len(rows)} games so far, credits left "
              f"{quota['x-requests-remaining']}")
    out = config.ODDS_DIR / f"odds_history_{args.season}.csv"
    config.ODDS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=backtest.ODDS_COLUMNS).to_csv(out, index=False)
    print(f"wrote {out}; run `elbet backtest --odds-csv {out}`")


# ---------------------------------------------------------------- parser

def main(argv: list[str] | None = None) -> int:
    cur = config.current_season()
    p = argparse.ArgumentParser(prog="elbet", description="EuroLeague betting model")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def seasons(sp):
        sp.add_argument("--start", type=int, default=config.FIRST_SEASON,
                        help="first season (2012 = 2012-13)")
        sp.add_argument("--end", type=int, default=cur, help="last season")

    def staking(sp):
        sp.add_argument("--bankroll", type=float, default=1000.0)
        sp.add_argument("--kelly", type=float, default=SETTINGS.bet.kelly_fraction,
                        help="Kelly fraction (0.25 = quarter Kelly)")
        sp.add_argument("--regions", default="eu", help="Odds API regions, e.g. eu,uk")

    sp = sub.add_parser("fetch", help="download schedules and box scores")
    seasons(sp)
    sp.add_argument("--workers", type=int, default=6)
    sp.set_defaults(func=cmd_fetch)

    sp = sub.add_parser("build", help="build tables and features, train models")
    seasons(sp)
    sp.set_defaults(func=cmd_build)

    sp = sub.add_parser("update", help="refresh the current season and rebuild")
    seasons(sp)
    sp.add_argument("--workers", type=int, default=6)
    sp.set_defaults(func=cmd_update)

    sp = sub.add_parser("backtest", help="walk-forward evaluation")
    sp.add_argument("--from-season", type=int, default=config.FIRST_SEASON + 3)
    sp.add_argument("--odds-csv", help="historical odds CSV for an ROI simulation")
    sp.set_defaults(func=cmd_backtest)

    sp = sub.add_parser("ratings", help="current team power ratings")
    sp.set_defaults(func=cmd_ratings)

    sp = sub.add_parser("predict", help="model prices for upcoming games")
    sp.add_argument("--days", type=float, default=7)
    sp.set_defaults(func=cmd_predict)

    sp = sub.add_parser("picks", help="compare model to live bookmaker odds (needs ODDS_API_KEY)")
    sp.add_argument("--days", type=float, default=3)
    sp.add_argument("--min-edge", type=float, default=None, help="minimum EV, e.g. 0.03")
    sp.add_argument("--market-weight", type=float, default=0.0,
                    help="blend model with no-vig market probability (0 = pure model)")
    sp.add_argument("--bookmakers", help="comma-separated Odds API bookmaker keys")
    sp.add_argument("--all-markets", action="store_true",
                    help="allow several bets on the same game (default: best EV only)")
    staking(sp)
    sp.set_defaults(func=cmd_picks)

    sp = sub.add_parser("props", help="player prop projections for one game")
    sp.add_argument("game", help="game id, e.g. E2026_85 (see `elbet predict`)")
    sp.add_argument("--lines", help="CSV: player,stat,line,over_odds,under_odds")
    sp.add_argument("--odds-api", action="store_true", help="try The Odds API for prop lines")
    sp.add_argument("--exclude", nargs="*", help="players ruled out (name as in box score)")
    staking(sp)
    sp.set_defaults(func=cmd_props)

    sp = sub.add_parser("odds-history", help="download historical odds (paid Odds API plan)")
    sp.add_argument("--season", type=int, required=True)
    sp.add_argument("--hours-before", type=float, default=2.0)
    sp.add_argument("--regions", default="eu")
    sp.set_defaults(func=cmd_odds_history)

    signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # quiet exit when piped into `head`
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    try:
        args.func(args)
    except (FileNotFoundError, odds.OddsApiError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
