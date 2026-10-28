"""High-level steps shared by the CLI: build data, train models, predict upcoming games."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from . import config, dataset
from .availability import AvailabilityModel
from .config import SETTINGS, Settings
from .features import build_features, set_availability
from .models import GameModel, fair_spread, fair_total
from .props import PROP_STATS, PropsModel, recent_roster

FEATURES_PATH = config.PROCESSED_DIR / "features.csv"
GAME_MODEL_PATH = config.MODEL_DIR / "game_model.pkl"
PROPS_MODEL_PATH = config.MODEL_DIR / "props_model.pkl"
PROPS_HISTORY_PATH = config.PROCESSED_DIR / "props_history.csv"
AVAIL_MODEL_PATH = config.MODEL_DIR / "availability.pkl"


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

    game_model = GameModel().fit(feats)
    game_model.save(GAME_MODEL_PATH)

    props = PropsModel(settings.props)
    history = props.run(players, feats)
    props.fit_dispersion(history)
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


def match_player(name: str, candidates: pd.Series) -> str | None:
    """Match 'Sasha Vezenkov' / 'VEZENKOV, SASHA' / 'Vezenkov' to one of `candidates`."""
    from .odds import _norm

    toks = {t for t in _norm(name).split() if len(t) > 1}
    best, score = None, 0.0
    for c in candidates:
        ctoks = set(_norm(c).split())
        s = len(toks & ctoks) / max(1, len(toks))
        if s > score:
            best, score = c, s
    return best if score >= 0.5 else None


def apply_absences(up: pd.DataFrame, injuries: pd.DataFrame | None = None,
                   out_names: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Set availability features of upcoming games from the injury feed and `--out`.

    Returns (updated features, table of matched absent players)."""
    from .injuries import absences_for_team

    up = up.copy()
    extra = {n: 1.0 for n in (out_names or [])}
    if (injuries is None or injuries.empty) and not extra:
        return up, pd.DataFrame()
    avail = load_availability()
    report = []
    for i, g in up.iterrows():
        for side in ("home", "away"):
            absent = {}
            if injuries is not None and not injuries.empty:
                absent.update(absences_for_team(injuries, g[f"{side}_names"]))
            absent.update(extra)
            value, miss = avail.missing_for(g[side], int(g["season"]), absent, match_player)
            up.at[i, f"missing_{side}"] = value
            if not miss.empty:
                report.append(miss.assign(team=g[side], game_id=g["game_id"]))
    set_availability(up)
    return up, (pd.concat(report, ignore_index=True) if report else pd.DataFrame())


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
               exclude: set[str] | None = None) -> pd.DataFrame:
    """Prop projections for both teams of one game (game = a features row)."""
    exclude = {e.lower() for e in (exclude or set())}
    season = int(game["season"])
    pace_f = game["pace_exp"] / game["mu_pace"]
    out = []
    for team, opp_def in ((game["home"], game["def_away"]), (game["away"], game["def_home"])):
        roster = recent_roster(players, team, season, props.cfg.recent_team_games)
        proj = props.project(team, opp_def, game["mu_eff"], pace_f, season, roster)
        if proj.empty:
            continue
        proj = proj[~proj["player"].str.lower().isin(exclude)
                    & ~proj["player_id"].str.lower().isin(exclude)].copy()
        # Hand minutes of excluded/absent players to the rest so the team plays ~200.
        scale = float(np.clip(200.0 / proj["exp_min"].sum(), 0.85, 1.2))
        cols = ["exp_min", *PROP_STATS]
        proj[cols] = proj[cols] * scale
        out.append(proj)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)
