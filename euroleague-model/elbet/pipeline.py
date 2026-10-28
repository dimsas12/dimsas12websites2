"""High-level steps shared by the CLI: build data, train models, predict upcoming games."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from . import config, dataset
from .availability import AvailabilityModel
from .config import SETTINGS, Settings
from .features import build_features, set_availability, set_bottom_up
from .models import GameModel, fair_spread, fair_total
from .props import PROP_STATS, PropsModel, recent_roster

FEATURES_PATH = config.PROCESSED_DIR / "features.csv"
GAME_MODEL_PATH = config.MODEL_DIR / "game_model.pkl"
PROPS_MODEL_PATH = config.MODEL_DIR / "props_model.pkl"
PROPS_HISTORY_PATH = config.PROCESSED_DIR / "props_history.csv"
AVAIL_MODEL_PATH = config.MODEL_DIR / "availability.pkl"
MINUTES_MODEL_PATH = config.MODEL_DIR / "minutes_gbm.pkl"
PLAY_MODEL_PATH = config.MODEL_DIR / "play_clf.pkl"
PEOPLE_PATH = config.PROCESSED_DIR / "people.csv"


def load_people() -> pd.DataFrame:
    """Season rosters with positions (fetched by `elbet fetch`)."""
    return pd.read_csv(PEOPLE_PATH) if PEOPLE_PATH.exists() else pd.DataFrame()


def predict_minutes(games: pd.DataFrame, players: pd.DataFrame,
                    absences: dict[tuple, dict[str, float]]) -> pd.DataFrame:
    """Learned minutes (if playing) for every recent-roster player of the given upcoming
    games, in one pass over history. absences: {(game_id, team): {player_id: P(out)}}."""
    from . import minutes

    model = pickle.loads(MINUTES_MODEL_PATH.read_bytes())
    up = []
    for g in games.itertuples(index=False):
        up += [(g.game_id, g.home, int(g.season), True), (g.game_id, g.away, int(g.season), False)]
    ds = minutes.build_dataset(players, absences, up, load_people())
    ds = ds[ds["played"].isna()].copy()
    ds["pred_min"] = np.clip(model.predict(ds[minutes.FEATURES]), 0, 40)
    ds["played_last"] = (ds["last1"] > 0).astype(float)
    if PLAY_MODEL_PATH.exists():
        clf = pickle.loads(PLAY_MODEL_PATH.read_bytes())
        ds["p_appear"] = clf.predict_proba(ds[minutes.PLAY_FEATURES])[:, 1]
    else:
        ds["p_appear"] = ds["appear"]
    return ds


def build(seasons: list[int], settings: Settings = SETTINGS) -> dict:
    games, players = dataset.build_tables(seasons)
    # Drop fixtures that were never played (e.g. the cancelled end of 2019-20).
    cutoff = pd.Timestamp.now(tz="UTC").tz_convert(None) - pd.Timedelta(days=2)
    games = games[games["played"] | (games["date"] > cutoff)].reset_index(drop=True)
    dataset.save_tables(games, players)
    avail = AvailabilityModel()
    feats = build_features(games, settings, players, avail)
    feats.to_csv(FEATURES_PATH, index=False)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    AVAIL_MODEL_PATH.write_bytes(pickle.dumps(avail))

    props = PropsModel(settings.props)
    history = props.run(players, feats)
    props.fit_dispersion(history)
    from .features import add_bottom_up
    from .props_backtest import fit_distribution
    props.dist, team_rows = fit_distribution(players, feats, props)
    feats = add_bottom_up(feats, team_rows)
    feats.to_csv(FEATURES_PATH, index=False)

    game_model = GameModel().fit(feats)
    game_model.save(GAME_MODEL_PATH)
    from . import minutes
    people = load_people()
    mds = minutes.build_dataset(players, people=people)
    MINUTES_MODEL_PATH.write_bytes(pickle.dumps(minutes.train(mds)))
    PLAY_MODEL_PATH.write_bytes(pickle.dumps(minutes.train_play(mds)))
    history.to_csv(PROPS_HISTORY_PATH, index=False)
    PROPS_MODEL_PATH.write_bytes(pickle.dumps(props))
    return {"games": len(games), "played": int(games["played"].sum()),
            "with_box": int(games["poss"].notna().sum()), "player_rows": len(players),
            "margin_sd": game_model.margin_sd, "total_sd": game_model.total_sd,
            "props_alpha": props.alpha}


def load_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError("features missing; run `elbet build` first")
    return pd.read_csv(FEATURES_PATH, parse_dates=["date"])


def load_props_model() -> PropsModel:
    if not PROPS_MODEL_PATH.exists():
        raise FileNotFoundError("props model missing; run `elbet build` first")
    return pickle.loads(PROPS_MODEL_PATH.read_bytes())


def load_availability() -> AvailabilityModel:
    if not AVAIL_MODEL_PATH.exists():
        raise FileNotFoundError("availability model missing; run `elbet build` first")
    return pickle.loads(AVAIL_MODEL_PATH.read_bytes())


_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _name_parts(name: str) -> tuple[set[str], set[str]]:
    """(surname tokens, first-name tokens) for 'LAST, FIRST' or 'First Last' spellings."""
    from .odds import _norm

    def toks(x: str) -> list[str]:
        return [t for t in _norm(x).split() if t not in _SUFFIXES]

    if "," in name:
        last, first = name.split(",", 1)
        return set(toks(last)), set(toks(first))
    t = toks(name)
    if not t:
        return set(), set()
    return {t[-1]}, set(t[:-1])


def match_player(name: str, candidates) -> str | None:
    """Match a player name in any spelling ('Sasha Vezenkov', 'VEZENKOV, SASHA',
    'Vezenkov') to one of `candidates`. The surname must match; first names only
    break ties. Call it with one team's roster so a shared surname or first name
    on another team can never match ("Gabriel Lundberg" is not "GABRIEL, WENYEN")."""
    q_last, q_first = _name_parts(name)
    if not q_last:
        return None
    best, best_score = None, -1.0
    for c in candidates:
        c_last, c_first = _name_parts(c)
        c_all = c_last | c_first
        if not q_last <= c_all:
            continue
        score = 2.0 * (q_last <= c_last) + len(q_first & c_all)
        if score > best_score:
            best, best_score = c, score
    return best


def apply_absences(up: pd.DataFrame, injuries: pd.DataFrame | None = None,
                   out_names: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Set availability features of upcoming games from the injury feed and `--out`.

    Returns (updated features, table of matched absent players)."""
    from .injuries import absences_for_team

    up = up.copy()
    extra = {n: 1.0 for n in (out_names or [])}
    if (injuries is None or injuries.empty) and not extra:
        live_bottom_up(up, injuries, out_names)
        return up, pd.DataFrame()
    avail = load_availability()
    report = []
    for i, g in up.iterrows():
        for side in ("home", "away"):
            absent = {}
            if injuries is not None and not injuries.empty:
                absent.update(absences_for_team(injuries, g[f"{side}_names"]))
            absent.update(extra)
            value, pts_value, miss = avail.missing_for(g[side], int(g["season"]), absent,
                                                       match_player)
            up.at[i, f"missing_{side}"] = value
            up.at[i, f"missing_pts_{side}"] = pts_value
            if not miss.empty:
                report.append(miss.assign(team=g[side], game_id=g["game_id"]))
    set_availability(up)
    live_bottom_up(up, injuries, out_names)
    return up, (pd.concat(report, ignore_index=True) if report else pd.DataFrame())


def live_bottom_up(up: pd.DataFrame, injuries: pd.DataFrame | None,
                   out_names: list[str] | None) -> None:
    """Bottom-up features for upcoming games, defined exactly as in training
    (props_backtest.replay): recent roster, chance of playing = appearance rate x
    (1 - P(out)), projections at usual minutes."""
    from .props import appearance_rates

    if not PROPS_MODEL_PATH.exists():
        set_bottom_up(up)
        return
    props = load_props_model()
    _, players = dataset.load_tables()
    for i, g in up.iterrows():
        named = game_absences(g, injuries, out_names)
        season = int(g["season"])
        pace_f = g["pace_exp"] / g["mu_pace"] if g["mu_pace"] else 1.0
        for side, opp_def in (("home", g["def_away"]), ("away", g["def_home"])):
            roster = recent_roster(players, g[side], season, props.cfg.recent_team_games)
            proj = props.project(g[side], opp_def, g["mu_eff"], pace_f, season, roster)
            if proj.empty:
                continue
            appear = appearance_rates(players, g[side], season)
            p_out = np.zeros(len(proj))
            for name, w in named[side].items():
                hit = match_player(name, proj["player"])
                if hit is not None:
                    p_out = np.where(proj["player"].to_numpy() == hit, np.maximum(p_out, w), p_out)
            p_play = proj["player_id"].map(appear).fillna(0.5).to_numpy() * (1 - p_out)
            up.at[i, f"bu_{side}"] = float((p_play * proj["pts"]).sum())
            up.at[i, f"bu_out_{side}"] = float(((1 - p_play) * proj["pts"] * (p_out > 0)).sum())
    set_bottom_up(up)


def questionable(up: pd.DataFrame) -> pd.DataFrame:
    """Rotation players who missed their team's most recent game (check their status)."""
    avail = load_availability()
    rows = []
    for team in pd.unique(pd.concat([up["home"], up["away"]])):
        season = int(up["season"].max())
        rot = avail.rotation(team, season)
        rows.append(rot[rot["games_since"] >= 1].assign(team=team))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def upcoming(feats: pd.DataFrame, days: float, now: pd.Timestamp | None = None) -> pd.DataFrame:
    now = now or pd.Timestamp.now(tz="UTC").tz_convert(None)
    up = feats[~feats["played"] & (feats["date"] >= now - pd.Timedelta(hours=3))
               & (feats["date"] <= now + pd.Timedelta(days=days))]
    return up.sort_values("date")


def predict_games(feats: pd.DataFrame, model: GameModel) -> pd.DataFrame:
    df = feats[feats["has_ratings"]]
    if df.empty:
        return df.assign(pred_margin=[], pred_total=[], p_home=[])
    p = model.predict(df)
    out = df.join(p)
    out["fair_home_ml"] = 1 / out["p_home"]
    out["fair_away_ml"] = 1 / (1 - out["p_home"])
    out["fair_home_line"] = out["pred_margin"].map(fair_spread)
    out["fair_total"] = out["pred_total"].map(fair_total)
    out["margin_sd"], out["total_sd"] = model.margin_sd, model.total_sd
    return out


def power_ratings(feats: pd.DataFrame, settings: Settings = SETTINGS) -> pd.DataFrame:
    """Current team ratings: net points per 100 possessions on a neutral floor."""
    from .ratings import latest_snapshot, run_elo

    season = int(feats["season"].max())
    snap = latest_snapshot(feats, settings.ratings)
    _, elo = run_elo(feats.sort_values("date"), settings.elo)
    teams = pd.unique(pd.concat([feats.loc[feats["season"] == season, "home"],
                                 feats.loc[feats["season"] == season, "away"]]))
    names = dict(zip(feats["home"], feats["home_name"]))
    rows = []
    for t in teams:
        off, de = snap["off"].get(t, 0.0), snap["def"].get(t, 0.0)
        rows.append({"team": t, "name": names.get(t, t), "net_rtg": off + de,
                     "off_rtg": snap["mu"] + off, "def_rtg": snap["mu"] - de,
                     "pace": snap["mu_pace"] + 2 * snap["pace"].get(t, 0.0),
                     "elo": elo.get(t, np.nan)})
    return pd.DataFrame(rows).sort_values("net_rtg", ascending=False).reset_index(drop=True)


def game_props(game: pd.Series, players: pd.DataFrame, props: PropsModel,
               absences: dict[str, dict[str, float]] | None = None,
               team_pts: dict[str, float] | None = None,
               learned_minutes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Prop projections for both teams of one game (game = a features row).

    absences: {"home": {name: P(out)}, "away": {...}} from the injury feed / --out.
    team_pts: {"home": points, "away": points} from the game model (reported only).

    Chance of playing = recent appearance rate x (1 - P(out)). Minutes if playing:
    minutes freed by absences go mostly to players with room to play more, at half
    strength, toward 197 expected minutes for the recent roster (the other ~2% go
    to returning players / new signings). Both choices were picked by replaying
    15 seasons (props_backtest.replay). Projections are conditional on playing,
    which is how bookmakers settle props. Distributions include minutes and
    per-minute-rate uncertainty (PropDistribution).
    """
    from . import props_backtest as pb
    from .props import appearance_rates

    absences = absences or {}
    season = int(game["season"])
    pace_f = game["pace_exp"] / game["mu_pace"]
    dist = getattr(props, "dist", None)
    out = []
    for side, opp_def in (("home", game["def_away"]), ("away", game["def_home"])):
        team = game[side]
        roster = recent_roster(players, team, season, props.cfg.recent_team_games)
        proj = props.project(team, opp_def, game["mu_eff"], pace_f, season, roster)
        if proj.empty:
            continue
        appear = appearance_rates(players, team, season)
        p_out = pd.Series(0.0, index=proj.index)
        for name, w in (absences.get(side) or {}).items():
            hit = match_player(name, proj["player"])
            if hit is not None:
                p_out[proj["player"] == hit] = np.maximum(p_out[proj["player"] == hit], w)
        proj["p_out"] = p_out
        proj["p_play"] = proj["player_id"].map(appear).fillna(0.5) * (1 - p_out)
        base = proj["exp_min"].to_numpy()
        # Team-specific target: 201.3 minutes (incl. overtime) minus what the team's
        # spare players (played this season, not in the last few games, not injured)
        # are expected to play, minus a small allowance for call-ups.
        spare = 0.0
        season_ids = players.loc[(players["team"] == team) & (players["season"] == season)
                                 & (players["min"] > 0), "player_id"].unique()
        out_names = [n for n, w in (absences.get(side) or {}).items() if w >= 0.75]
        for pid in set(season_ids) - set(proj["player_id"]):
            st = props.states.get(pid)
            if st is None or st.s_n <= 0 or any(match_player(n, [st.player]) for n in out_names):
                continue
            spare += appear.get(pid, 0.0) * st.exp_minutes()
        target = pb.TEAM_MINUTES_AVG - spare - pb.CALLUP_ALLOWANCE
        lm = None
        if learned_minutes is not None and not learned_minutes.empty:
            lm = learned_minutes[(learned_minutes["game_id"] == game["game_id"])
                                 & (learned_minutes["team"] == team)]
        if lm is not None and not lm.empty:
            # Learned model (MAE 4.41 vs 4.50 for the rule on 2023-26 player-games).
            pm = dict(zip(lm["player_id"], lm["pred_min"]))
            minutes = np.array([pm.get(pid, b) for pid, b in zip(proj["player_id"], base)])
            # Learned chance of playing (log loss 0.174 vs 0.294 for the appearance rate).
            pa = dict(zip(lm["player_id"], lm["p_appear"]))
            proj["p_play"] = [pa.get(pid, a) * (1 - po) for pid, a, po in
                              zip(proj["player_id"], proj["p_play"] / (1 - proj["p_out"]).clip(
                                  lower=1e-9), proj["p_out"])]
            proj.loc[proj["p_out"] >= 1, "p_play"] = 0.0
        else:
            minutes = pb.allocate(base, proj["p_play"].to_numpy(), pb.ALLOC_MODE,
                                  pb.ALLOC_STRENGTH, target)
        # Ruled-out players: show their usual role, not a reallocated one.
        minutes = np.where(proj["p_play"].to_numpy() > 0, minutes, base)
        scale = np.divide(minutes, base, out=np.ones_like(base), where=base > 0)
        proj["base_min"] = base
        proj["exp_min"] = minutes
        for st in PROP_STATS:
            proj[st] = proj[st] * scale
        proj["pra"] = proj["pts"] + proj["reb"] + proj["ast"]
        proj["team_pts"] = (team_pts or {}).get(side, np.nan)
        proj["spare_min"] = spare
        proj["target_min"] = target
        # Minutes the listed players leave to everyone else (history: 3 with no
        # absences up to ~17 with four rotation players out).
        proj["outsider_min"] = max(0.0, pb.TEAM_MINUTES_AVG
                                   - float((proj["p_play"] * minutes).sum()))
        for st in PROP_STATS:
            if dist is not None:
                proj[f"{st}_sd"] = [dist.sd(m, st, mn) for m, mn in zip(proj[st], minutes)]
                fl = [dist.fair_line(m, st, mn) for m, mn in zip(proj[st], minutes)]
            else:
                from .props import fair_line, nb_sd
                proj[f"{st}_sd"] = [nb_sd(m, props.alpha[st]) for m in proj[st]]
                fl = [fair_line(m, props.alpha[st]) for m in proj[st]]
            proj[f"{st}_line"] = [x[0] for x in fl]
            proj[f"{st}_p_over"] = [x[1] for x in fl]
        proj["side"] = side
        out.append(proj)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def prop_probs(props: PropsModel, mean: float, stat: str, minutes: float, line: float):
    """(P over, P push, P under) with the calibrated distribution when available."""
    from .props import nb_probs

    dist = getattr(props, "dist", None)
    return dist.probs(mean, stat, minutes, line) if dist is not None \
        else nb_probs(mean, props.alpha[stat], line)


def game_absences(game: pd.Series, injuries: pd.DataFrame | None,
                  extra_out: list[str] | None = None) -> dict[str, dict[str, float]]:
    """{"home": {name: P(out)}, "away": {...}} for one game from the feed and --out names.
    --out names are matched against both rosters; only a true surname match counts."""
    from .injuries import absences_for_team

    res = {}
    for side in ("home", "away"):
        d = {}
        if injuries is not None and not injuries.empty:
            d.update(absences_for_team(injuries, game[f"{side}_names"]))
        d.update({n: 1.0 for n in (extra_out or [])})
        res[side] = d
    return res


def absences_by_pid(games: pd.DataFrame, players: pd.DataFrame, injuries: pd.DataFrame | None,
                    extra_out: list[str] | None = None) -> dict[tuple, dict[str, float]]:
    """{(game_id, team): {player_id: P(out)}} matched against each team's recent roster."""
    out = {}
    for _, g in games.iterrows():
        named = game_absences(g, injuries, extra_out)
        for side in ("home", "away"):
            team = g[side]
            t = players[(players["team"] == team) & (players["season"] == int(g["season"]))]
            names = t.drop_duplicates("player_id").set_index("player")["player_id"]
            d = {}
            for name, w in named[side].items():
                hit = match_player(name, list(names.index))
                if hit is not None:
                    d[names[hit]] = max(w, d.get(names[hit], 0.0))
            out[(g["game_id"], team)] = d
    return out
