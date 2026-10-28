"""Historical test of the team-level props layer and of the minutes-uncertainty
distribution.

`replay` walks through every past game in date order. Before each game it builds,
for both teams, exactly what `pipeline.game_props` builds for an upcoming game:

* roster = players who appeared in one of the team's last 3 games this season;
* absences = rotation players (>= 15 expected minutes, played in one of the last
  4 games) who then did not play: the information an injury report gives. Other
  players keep their recent appearance rate as chance of playing; whether a
  bench player actually got minutes is not used;
* minutes scaled so expected team minutes are 200, points scaled to the game
  model's walk-forward team score.

Then it records the base projection (no team layer) and the allocated one for
every player who actually played, and finally updates player histories with the
game's box score, so nothing from the game leaks into its own projection.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .props import (OPP_ADJUSTED, PROP_STATS, PlayerState, PropsModel, _context,
                    _league_rates, allocate_minutes)

ROTATION_MIN = 15.0

# Live allocation settings, chosen with `replay` on 2015-24 and confirmed on 2025-27:
# headroom at half strength cut minutes error 4.75 -> 4.69 (5.10 -> 4.83 with 2+
# absences); full proportional redistribution and points scaling made things worse.
ALLOC_MODE = "headroom"
ALLOC_STRENGTH = 0.5
SCALE_POINTS = False  # scaling to the team score made projections worse
# Share of a team's minutes / points that its recent roster (players from the last 3
# games) actually supplies; the rest goes to returning players, new signings and
# overtime. Measured with `replay` (see shares()).
ROSTER_MIN_SHARE = 0.985  # x 200 = 197 minutes
ROSTER_PTS_SHARE = 0.981


def allocate(base: np.ndarray, p_play: np.ndarray, mode: str, strength: float,
             target: float = 200.0) -> np.ndarray:
    """Minutes if playing.

    proportional: everyone scaled by one factor so expected minutes = 200.
    headroom: minutes freed by absences (or cut when the roster over-fills) are
              shared in proportion to each player's room below 36 minutes, so
              bench players absorb more than starters.
    strength in [0, 1] shrinks the change toward the players' usual minutes."""
    base = np.asarray(base, float)
    p_play = np.asarray(p_play, float)
    if strength <= 0 or base.sum() <= 0:
        return base
    if mode == "proportional":
        full = allocate_minutes(base, p_play, total=target)
    else:
        gap = target - float((p_play * base).sum())
        room = np.clip(36.0 - base, 1.0, None) if gap > 0 else base.copy()
        w = p_play * room
        full = base + gap * room / w.sum() if w.sum() > 0 else base
        full = np.clip(full, 0, 40)
    return base + strength * (full - base)


def replay(players: pd.DataFrame, feats: pd.DataFrame, team_pts: dict,
           model: PropsModel, variants: dict | None = None) -> pd.DataFrame:
    """team_pts: {(game_id, team): predicted points} from the walk-forward game model.

    variants: {name: (mode, strength, scale_points)} minutes-allocation variants to
    record side by side (see `allocate`); default = the live method."""
    variants = variants or {"alloc": (ALLOC_MODE, ALLOC_STRENGTH, SCALE_POINTS)}
    cfg = model.cfg
    players = players.copy()
    players["pra"] = players["pts"] + players["reb"] + players["ast"]
    players = players.sort_values(["date", "game_id"])
    league_by_season = _league_rates(players)
    ctx = _context(feats)
    states: dict[str, PlayerState] = {}
    # (team, season) -> list of sets of player_ids who played, one per team game
    history: dict[tuple, list[set]] = defaultdict(list)
    first_game: dict[tuple, int] = {}  # (team, season, pid) -> index of first appearance
    out = []
    team_rows = []
    for gid, box in players.groupby("game_id", sort=False):
        season = int(box["season"].iloc[0])
        league = league_by_season.get(season) or next(iter(league_by_season.values()))
        for team, tb in box.groupby("team"):
            is_home = bool(tb["is_home"].iloc[0])
            pace_f, opp_f = ctx.get(gid, {}).get(is_home, (1.0, 1.0))
            hist = history[(team, season)]
            roster = set().union(*hist[-cfg.recent_team_games:]) if hist else set()
            played_now = set(tb.loc[tb["min"] > 0, "player_id"])
            rows = []
            for pid in roster:
                st = states.get(pid)
                if st is None or st.s_n <= 0:
                    continue
                decay = cfg.season_decay if st.season != season else 1.0
                base_min = st.exp_minutes()
                proj = {}
                for s in PROP_STATS:
                    f = pace_f * (opp_f if s in OPP_ADJUSTED else 1.0)
                    prior = cfg.prior_minutes[s]
                    rate = (st.s_stat[s] * decay + prior * league[s]) / (st.s_min * decay + prior)
                    proj[s] = base_min * rate * f
                n = len(hist)
                first = max(first_game.get((team, season, pid), n), n - 6)
                window = hist[first:]
                appear = (sum(pid in g for g in window) / len(window)) if window else 0.5
                rotation = base_min >= ROTATION_MIN and any(pid in g for g in hist[-4:])
                p_out = 1.0 if (rotation and pid not in played_now) else 0.0
                rows.append({"player_id": pid, "base_min": base_min, "appear": appear,
                             "p_out": p_out, "prior_games": st.games, **proj})
            in_roster = tb["player_id"].isin([x["player_id"] for x in rows])
            team_rows.append({"game_id": gid, "season": season, "team": team,
                              "team_min": tb["min"].sum(), "team_pts": tb["pts"].sum(),
                              "roster_min": tb.loc[in_roster, "min"].sum(),
                              "roster_pts": tb.loc[in_roster, "pts"].sum()})
            if rows:
                r = pd.DataFrame(rows)
                r["p_play"] = r["appear"] * (1 - r["p_out"])
                actual = tb.set_index("player_id")
                pts_target = team_pts.get((gid, team))
                var_out = {}
                for name, (mode, strength, scale_pts) in variants.items():
                    alloc = allocate(r["base_min"].to_numpy(), r["p_play"].to_numpy(),
                                     mode, strength, ROSTER_MIN_SHARE * 200.0)
                    scale = np.divide(alloc, r["base_min"], out=np.ones(len(r)),
                                      where=r["base_min"] > 0)
                    exp_pts = float((r["p_play"] * r["pts"] * scale).sum())
                    k = ROSTER_PTS_SHARE * pts_target / exp_pts \
                        if (scale_pts and pts_target and exp_pts > 0) else 1.0
                    var_out[name] = (alloc, scale, k)
                for i, row in r.iterrows():
                    pid = row["player_id"]
                    if pid not in played_now or row["prior_games"] < 3:
                        continue
                    a = actual.loc[pid]
                    rec = {"game_id": gid, "season": season, "team": team, "player_id": pid,
                           "min": a["min"], "base_min": row["base_min"],
                           "n_out_team": int(r["p_out"].sum())}
                    for s in PROP_STATS:
                        rec[s] = a[s]
                        rec[f"base_{s}"] = row[s]
                    for name, (alloc, scale, k) in var_out.items():
                        rec[f"{name}_min"] = alloc[i]
                        for s in ("pts", "reb", "ast", "fgm3"):
                            rec[f"{name}_{s}"] = row[s] * scale[i] * (k if s in ("pts", "fgm3")
                                                                     else 1.0)
                        rec[f"{name}_pra"] = (rec[f"{name}_pts"] + rec[f"{name}_reb"]
                                              + rec[f"{name}_ast"])
                    out.append(rec)
            # update history and player states with this game's box score
            hist.append(played_now)
            for pid in played_now:
                first_game.setdefault((team, season, pid), len(hist) - 1)
        for r in box.itertuples(index=False):
            st = states.get(r.player_id)
            if st is None:
                st = states[r.player_id] = PlayerState(r.player, r.team, r.season)
            if r.season != st.season:
                st.decay(cfg.season_decay)
                st.season = r.season
            st.team, st.player = r.team, r.player
            if r.min <= 0:
                continue
            st.decay(cfg.decay)
            st.s_min += r.min
            st.s_n += 1
            for s in PROP_STATS:
                st.s_stat[s] += getattr(r, s)
            st.games += 1
    res = pd.DataFrame(out)
    res.attrs["team"] = pd.DataFrame(team_rows)
    return res


# ---------------------------------------------------------------- minutes uncertainty

MIN_BUCKETS = [0, 12, 20, 26, 60]
N_QUANTILES = 20


def minutes_bucket(exp_min: float) -> int:
    return int(np.searchsorted(MIN_BUCKETS, exp_min, side="right") - 1)


def fit_minutes_mixture(rep: pd.DataFrame, min_col: str = "alloc_min") -> dict[int, list[float]]:
    """Quantiles of actual/projected minutes (given the player plays), per role bucket,
    normalised to mean 1 so the mixture keeps the projection's mean."""
    out = {}
    q = (np.arange(N_QUANTILES) + 0.5) / N_QUANTILES
    for b in range(len(MIN_BUCKETS) - 1):
        d = rep[(rep[min_col] >= MIN_BUCKETS[b]) & (rep[min_col] < MIN_BUCKETS[b + 1])]
        if len(d) < 200:
            out[b] = [1.0]
            continue
        ratio = (d["min"] / d[min_col]).clip(0, 3).to_numpy()
        vals = np.quantile(ratio, q)
        out[b] = list(vals / vals.mean())
    return out


def fit_conditional_alpha(rep: pd.DataFrame, prefix: str = "alloc_",
                          min_col: str = "alloc_min") -> dict[str, float]:
    """NB dispersion of each stat *given the actual minutes played* (minutes
    uncertainty is handled separately by the mixture)."""
    out = {}
    ratio = (rep["min"] / rep[min_col]).clip(0, 3)
    for s in PROP_STATS:
        m = (rep[f"{prefix}{s}"] * ratio).to_numpy()
        y = rep[s].to_numpy()
        ok = m > 0.3
        a = ((y[ok] - m[ok]) ** 2 - m[ok]).sum() / (m[ok] ** 2).sum()
        out[s] = float(max(a, 0.01))
    return out


RATE_SD = 0.20  # per-minute rate uncertainty (lognormal sd), tuned on 2015-24


def fit_distribution(players: pd.DataFrame, feats: pd.DataFrame, model: PropsModel):
    """Replay history with the live allocation and fit the calibrated prop distribution."""
    from .props import PropDistribution

    rep = replay(players, feats, {}, model)
    rep = rep[rep["season"] < rep["season"].max()]  # current season stays out of the fit
    return PropDistribution(fit_minutes_mixture(rep), fit_conditional_alpha(rep), RATE_SD,
                            dict(model.alpha))
