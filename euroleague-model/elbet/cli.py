"""Command line interface: `elbet <command>`."""

from __future__ import annotations

import argparse
import logging
import signal
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from . import (backtest, betting, config, dataset, fetch, injuries, odds, pipeline,
               tracking)
from .config import SETTINGS
from .models import GameModel, spread_probs, total_probs
from .props import PROP_STATS, nb_probs

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


def _upcoming_with_injuries(args, days: float) -> tuple[pd.DataFrame, GameModel]:
    """Upcoming games with model predictions, adjusted for injuries and `--out`."""
    feats = pipeline.load_features()
    model = GameModel.load(pipeline.GAME_MODEL_PATH)
    up = pipeline.upcoming(feats, days)
    if up.empty:
        return up, model
    inj = None if args.no_injuries else injuries.load_injuries()
    up, absent = pipeline.apply_absences(up, inj, args.out)
    pred = pipeline.predict_games(up, model)
    if not absent.empty:
        coef = model.margin_model.raw_coefficients(model.margin_features)["avail_diff"]
        absent["impact_pts"] = absent["value"] * absent["p_out"] * coef
        print("absences applied (impact = points of margin lost):")
        print(absent[["game_id", "team", "player", "p_out", "exp_min", "impact_pts"]]
              .round(2).to_string(index=False))
        print()
    elif inj is not None:
        print("no rotation players listed as absent for these games\n")
    return pred, model


def cmd_predict(args):
    pred, _ = _upcoming_with_injuries(args, args.days)
    if pred.empty:
        print(f"no games in the next {args.days} days (run `elbet update` to refresh)")
        return
    show = pred[["game_id", "date", "home", "away", "p_home", "fair_home_ml", "fair_away_ml",
                 "pred_margin", "fair_home_line", "pred_total", "avail_diff"]].copy()
    show["date"] = show["date"].dt.strftime("%a %d %b %H:%M")
    print(show.round(2).to_string(index=False))
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    show.to_csv(config.REPORTS_DIR / "predictions.csv", index=False)


def _blend(p_model: float, p_market: float | None, w: float) -> float:
    if p_market is None or w <= 0:
        return p_model
    return (1 - w) * p_model + w * p_market


def _allowed(args) -> set[str] | None:
    """Books you can bet with (`--bookmakers`); None = all except offshore books."""
    b = getattr(args, "bookmakers", None)
    return {x.strip() for x in b.split(",") if x.strip()} if b else None


def _event_lines(flat: pd.DataFrame, event_id: str, swapped: bool,
                 allowed: set[str] | None = None) -> dict:
    ev = flat[flat["event_id"] == event_id]
    if swapped:  # Odds API lists the teams the other way round
        ev = ev.assign(home_team=ev["away_team"].iloc[0], away_team=ev["home_team"].iloc[0])
    return odds.market_lines(ev, allowed)


def build_board(pred: pd.DataFrame, flat: pd.DataFrame, args) -> pd.DataFrame:
    """Every market side with model probability, best price, EV and stake."""
    bet_cfg = SETTINGS.bet
    matches = odds.match_events(flat, pred.reset_index())
    unmatched = flat["event_id"].nunique() - len(matches)
    if unmatched:
        print(f"warning: {unmatched} Odds API events could not be matched to the schedule")
    mw = dict(bet_cfg.market_weight)
    if args.market_weight is not None:  # one weight for sides; totals keep theirs unless set
        mw["moneyline"] = mw["spread"] = args.market_weight
    if getattr(args, "total_market_weight", None) is not None:
        mw["total"] = args.total_market_weight
    w = mw["moneyline"]
    allowed = _allowed(args)
    rows = []
    for event_id, (gid, swapped) in matches.items():
        m = _event_lines(flat, event_id, swapped, allowed)
        g = pred.loc[gid]
        cands = []
        if m.get("h2h"):
            fh = m["h2h"]["fair_home"]
            ph = _blend(g.p_home, fh, w)
            cands += [("moneyline", "home", g.home, None, *m["h2h"]["home"], ph, 0.0, fh),
                      ("moneyline", "away", g.away, None, *m["h2h"]["away"], 1 - ph, 0.0,
                       None if fh is None else 1 - fh)]
        if m.get("spreads"):
            L = m["spreads"]["line"]
            ph, pp, pa = spread_probs(g.pred_margin, g.margin_sd, L)
            fh = m["spreads"]["fair_home"]
            ph_b = _blend(ph / (ph + pa), fh, mw["spread"]) * (1 - pp)
            cands += [("spread", "home", g.home, L, *m["spreads"]["home"], ph_b, pp, fh),
                      ("spread", "away", g.away, -L, *m["spreads"]["away"], 1 - pp - ph_b, pp,
                       None if fh is None else 1 - fh)]
        if m.get("totals"):
            L = m["totals"]["line"]
            po, pp, pu = total_probs(g.pred_total, g.total_sd, L)
            fo = m["totals"]["fair_over"]
            po_b = _blend(po / (po + pu), fo, mw["total"]) * (1 - pp)
            cands += [("total", "over", "over", L, *m["totals"]["over"], po_b, pp, fo),
                      ("total", "under", "under", L, *m["totals"]["under"], 1 - pp - po_b, pp,
                       None if fo is None else 1 - fo)]
        for market, sel, pick, line, price, book, p, push, p_mkt in cands:
            rows.append({
                "game_id": gid, "event_id": event_id, "commence": g.date,
                "date": g.date.strftime("%a %d %b %H:%M"), "game": f"{g.home}-{g.away}",
                "market": market, "selection": sel, "pick": pick, "line": line, "odds": price,
                "book": book, "model_p": p, "market_p": p_mkt, "p_push": push,
                "fair_odds": 1 / p if p > 0 else np.inf,
                "ev": betting.expected_value(p, price, push),
                "stake": betting.stake(p, price, args.bankroll, args.kelly,
                                       bet_cfg.max_stake_pct, push),
            })
    return pd.DataFrame(rows)


def cmd_picks(args):
    pred, _ = _upcoming_with_injuries(args, args.days)
    if pred.empty:
        print(f"no games in the next {args.days} days")
        return
    # Never price games that have started: the Odds API also returns in-play odds.
    now = pd.Timestamp.now(tz="UTC").tz_convert(None)
    pred = pred[pred["date"] > now].set_index("game_id")
    if pred.empty:
        print("all games in the window have started")
        return
    events, quota = odds.fetch_odds(regions=args.regions)
    print(f"Odds API: {len(events)} events, credits remaining {quota['x-requests-remaining']}")
    flat = odds.flatten(events)
    if flat.empty:
        print("no odds returned")
        return
    # Only events inside the requested window (others are other days' games).
    last = pred["date"].max() + pd.Timedelta(hours=2)
    flat = flat[flat["commence"].dt.tz_convert(None) <= last]
    allowed = odds.usable_books(flat["book"].unique(), _allowed(args))
    print(f"best prices from {len(allowed)} books: {', '.join(sorted(allowed))} "
          "(exchange prices net of commission)")
    board = build_board(pred, flat, args)
    if board.empty:
        print("no markets matched")
        return
    bet_cfg = SETTINGS.bet
    min_edge = args.min_edge if args.min_edge is not None else bet_cfg.min_edge
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    board.to_csv(config.REPORTS_DIR / f"markets_{stamp}.csv", index=False)
    picks = board[(board["ev"] >= min_edge) & (board["model_p"] >= bet_cfg.min_prob)
                  & (board["odds"] <= bet_cfg.max_odds)]
    picks = picks.sort_values("ev", ascending=False)
    if bet_cfg.one_bet_per_game and not args.all_markets:
        picks = picks.drop_duplicates("game")
    cap = bet_cfg.max_total_exposure * args.bankroll
    if picks["stake"].sum() > cap:
        picks = picks.assign(stake=(picks["stake"] * cap / picks["stake"].sum()).round(2))
    if picks.empty:
        print(f"no bets with EV >= {min_edge:.1%}. Full board in reports/markets_{stamp}.csv")
        return
    cols = ["date", "game", "market", "pick", "line", "odds", "book", "model_p", "market_p",
            "fair_odds", "ev", "stake"]
    print(picks[cols].round(3).to_string(index=False))
    picks.to_csv(config.REPORTS_DIR / f"picks_{stamp}.csv", index=False)
    print(f"\n{len(picks)} picks, total stake {picks['stake'].sum():.2f} "
          f"of bankroll {args.bankroll:.0f}; saved reports/picks_{stamp}.csv")
    if not args.no_log:
        tips = sorted(pd.to_datetime(picks["commence"]).dt.floor("min").unique())
        print("closing prices: run `elbet closing --window 45` at "
              + ", ".join(f"{(t - pd.Timedelta(minutes=30)):%a %H:%M} UTC" for t in tips))
        n = tracking.log_picks(picks)
        print(f"logged {n} new picks to {tracking.LOG_PATH} "
              f"(run `elbet closing` shortly before tip-off, `elbet track` after)")


def cmd_closing(args):
    """Record closing prices for logged bets whose games start soon."""
    log = tracking.load_log()
    now = pd.Timestamp.now(tz="UTC").tz_convert(None)
    todo = tracking.needs_closing(log, now, args.window)
    if todo.empty:
        nxt = log[log["close_p"].isna() & (log["commence"] > now)]["commence"].min() \
            if not log.empty else None
        print("no logged bets tip off within the window; nothing fetched (0 credits used)")
        if nxt is not None and pd.notna(nxt):
            print(f"next bet without a closing price: {nxt:%a %d %b %H:%M} UTC")
        return
    model = GameModel.load(pipeline.GAME_MODEL_PATH)
    events, quota = odds.fetch_odds(regions=args.regions)
    print(f"Odds API: {len(events)} events, credits remaining {quota['x-requests-remaining']}")
    flat = odds.flatten(events)
    feats = pipeline.load_features()
    games = feats[feats["game_id"].isin(todo["game_id"])]
    matches = {gid: (eid, sw) for eid, (gid, sw) in odds.match_events(flat, games).items()}
    done = 0
    for i, b in todo.iterrows():
        if b["game_id"] not in matches:
            print(f"{b['game']}: not on the odds board any more")
            continue
        lines = _event_lines(flat, *matches[b["game_id"]], _allowed(args))
        res = tracking.closing_prob(b, lines, model.margin_sd, model.total_sd)
        if res is None:
            continue
        p, push, close_line, close_odds = res
        log.loc[i, ["close_at", "close_p", "close_line", "close_odds"]] = [
            now.floor("s"), p, close_line, close_odds]
        log.loc[i, "clv"] = tracking.clv(p, push, b["odds"])
        done += 1
        line = "" if pd.isna(b["line"]) else f" {b['line']:+g}"
        print(f"{b['game']} {b['market']} {b['pick']}{line} @ {b['odds']:.2f}: "
              f"closing p {p:.3f}, CLV {log.loc[i, 'clv']:+.1%}")
    tracking.save_log(log)
    print(f"recorded {done} closing prices")


def cmd_track(args):
    """Settle finished bets and report profit and closing-line value."""
    log = tracking.load_log()
    if log.empty:
        print("no bets logged yet; `elbet picks` logs its picks automatically")
        return
    feats = pipeline.load_features()
    log = tracking.settle(log, feats)
    tracking.save_log(log)
    rep = tracking.report(log)
    print(rep.round(3).to_string(index=False))
    c = log[log["clv"].notna()]
    if len(c) >= 2:
        se = c["clv"].std() / np.sqrt(len(c))
        print(f"\naverage CLV {c['clv'].mean():+.2%} +/- {se:.2%} (1 s.e.) over {len(c)} bets."
              " Positive and well clear of zero after ~100+ bets is the sign of a real edge.")
    if args.show:
        cols = ["commence", "game", "market", "pick", "line", "odds", "ev", "close_line",
                "close_p", "clv", "result", "profit"]
        tail = log[cols].tail(args.show)
        num = tail.select_dtypes("number").columns
        print("\n" + tail.assign(**{c: tail[c].round(3) for c in num}).to_string(index=False))


def cmd_excel(args):
    from . import excel

    day = pd.Timestamp(args.date) if args.date else pd.Timestamp.now(tz="UTC").tz_convert(None)
    path = excel.export(day, Path(args.out) if args.out else None,
                        fetch_injuries=not args.cached_injuries)
    print(f"wrote {path}")


def cmd_injuries(args):
    inj = injuries.load_injuries(fetch=not args.cached)
    if inj.empty:
        print("no injury data")
        return
    upd = inj["updated"].dropna()
    upd = upd[upd != ""].iloc[0] if (upd != "").any() else "?"
    print(f"{len(inj)} entries (feed updated {upd}); manual overrides: {injuries.MANUAL_PATH}")
    print(inj[["player", "team", "status", "weight", "injury", "back", "source"]]
          .fillna("").to_string(index=False))


def _project_game_props(game_row: pd.DataFrame, inj, extra_out, players, props):
    """Injury-adjusted game prediction + team-consistent prop projections for one game."""
    model = GameModel.load(pipeline.GAME_MODEL_PATH)
    up, _ = pipeline.apply_absences(game_row, inj, extra_out)
    game = pipeline.predict_games(up, model).iloc[0]
    absences = pipeline.game_absences(game, inj, extra_out)
    learned = None
    if pipeline.MINUTES_MODEL_PATH.exists():
        g1 = game.to_frame().T
        learned = pipeline.predict_minutes(
            g1, players, pipeline.absences_by_pid(g1, players, inj, extra_out))
    proj = pipeline.game_props(game, players, props, absences,
                               {"home": game["pred_home_pts"], "away": game["pred_away_pts"]},
                               learned)
    return proj, game


def cmd_props(args):
    feats = pipeline.load_features().set_index("game_id")
    if args.game not in feats.index:
        up = pipeline.upcoming(feats.reset_index(), 10)
        print(f"unknown game {args.game}. Upcoming games:")
        print(up[["game_id", "date", "home", "away"]].to_string(index=False))
        return
    game_row = feats.loc[[args.game]].reset_index()
    _, players = dataset.load_tables()
    props = pipeline.load_props_model()
    inj = None if args.no_injuries else injuries.load_injuries()
    proj, game = _project_game_props(game_row, inj, args.exclude, players, props)
    if proj.empty:
        print("no player history for these teams yet")
        return
    print(f"{game['home']} vs {game['away']}  ({game['date']:%a %d %b %H:%M})  "
          f"predicted {game['pred_home_pts']:.1f}-{game['pred_away_pts']:.1f}, "
          f"pace {game['pace_exp']:.1f}")
    out_players = proj.loc[proj["p_play"] <= 0, "player"].tolist()
    if out_players:
        print("ruled out (not projected):", ", ".join(out_players))
    proj = proj[proj["p_play"] > 0]
    show = proj[["team", "player", "p_play", "exp_min"]].copy()
    for s in PROP_STATS:
        show[s] = proj[s]
        show[f"{s}_line"] = proj[f"{s}_line"]
    print(show.round(2).to_string(index=False))

    lines = None
    if args.lines:
        lines = pd.read_csv(args.lines)
    elif args.odds_api:
        lines = _odds_api_prop_lines(game, args.regions)
    if lines is None or lines.empty:
        return
    rows = []
    for ln in lines.itertuples(index=False):
        who = pipeline.match_player(ln.player, proj["player"])
        if who is None or ln.stat not in PROP_STATS:
            print(f"skip: {ln.player} {ln.stat} (no projection)")
            continue
        row = proj.loc[proj["player"] == who].iloc[0]
        mean = float(row[ln.stat])  # if the player plays
        po, pp, pu = pipeline.prop_probs(props, mean, ln.stat, float(row["exp_min"]),
                                         float(ln.line))
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

    def availability(sp):
        sp.add_argument("--out", nargs="*", default=[],
                        help='extra players ruled out, e.g. --out "Vezenkov" "Nunn"')
        sp.add_argument("--no-injuries", action="store_true", help="ignore the injury feed")

    sp = sub.add_parser("predict", help="model prices for upcoming games")
    sp.add_argument("--days", type=float, default=7)
    availability(sp)
    sp.set_defaults(func=cmd_predict)

    sp = sub.add_parser("picks", help="compare model to live bookmaker odds (needs ODDS_API_KEY)")
    sp.add_argument("--days", type=float, default=3)
    sp.add_argument("--min-edge", type=float, default=None, help="minimum EV, e.g. 0.03")
    sp.add_argument("--market-weight", type=float, default=None,
                    help="market weight for moneyline/spread (default 0.5)")
    sp.add_argument("--total-market-weight", type=float, default=None,
                    help="market weight for totals (default 0.8: the totals model is weak)")
    sp.add_argument("--bookmakers",
                    help="only take prices from these Odds API bookmaker keys, e.g. "
                         "pinnacle,williamhill,betsson (default: all but offshore books)")
    sp.add_argument("--all-markets", action="store_true",
                    help="allow several bets on the same game (default: best EV only)")
    sp.add_argument("--no-log", action="store_true", help="don't record picks in the bet log")
    staking(sp)
    availability(sp)
    sp.set_defaults(func=cmd_picks)

    sp = sub.add_parser("closing", help="record closing odds for logged bets (CLV)")
    sp.add_argument("--window", type=float, default=120,
                    help="capture bets whose game starts within this many minutes")
    sp.add_argument("--regions", default="eu")
    sp.add_argument("--bookmakers", help="same as for picks")
    sp.set_defaults(func=cmd_closing)

    sp = sub.add_parser("track", help="settle logged bets; profit and CLV report")
    sp.add_argument("--show", type=int, default=0, help="also list the last N bets")
    sp.set_defaults(func=cmd_track)

    sp = sub.add_parser("excel", help="Excel report for one game day (uses the last picks run)")
    sp.add_argument("--date", help="YYYY-MM-DD (default: today, UTC)")
    sp.add_argument("--out", help="output .xlsx path")
    sp.add_argument("--cached-injuries", action="store_true")
    sp.set_defaults(func=cmd_excel)

    sp = sub.add_parser("injuries", help="show the injury list used by the model")
    sp.add_argument("--cached", action="store_true", help="don't re-download")
    sp.set_defaults(func=cmd_injuries)

    sp = sub.add_parser("props", help="player prop projections for one game")
    sp.add_argument("game", help="game id, e.g. E2026_85 (see `elbet predict`)")
    sp.add_argument("--lines", help="CSV: player,stat,line,over_odds,under_odds")
    sp.add_argument("--odds-api", action="store_true", help="try The Odds API for prop lines")
    sp.add_argument("--exclude", nargs="*", help="extra players ruled out")
    sp.add_argument("--no-injuries", action="store_true", help="ignore the injury feed")
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
