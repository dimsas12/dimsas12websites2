"""Player availability: how much weaker is a team when rotation players are missing?

Each player gets a pre-game value: decayed PIR (EuroLeague "valuation") per
minute above a replacement-level rate, times the player's expected minutes. For every
game, a team's `missing_value` is the summed value of rotation players (played
in one of the team's last few games, averaging real minutes) who did not play.
The game model learns how many points of margin and total one unit is worth.

In backtests absences come from the box score (who actually played), which is
the information a bettor has once line-ups are announced. For upcoming games
you list players ruled out with `--out`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class AvailabilityConfig:
    decay: float = 0.90  # per-game decay of a player's history
    season_decay: float = 0.5
    prior_minutes: float = 60.0  # shrink PIR/min toward the league rate
    replacement: float = 0.6  # replacement level as a fraction of league PIR/min
    rotation_games: int = 4  # must have played in one of the team's last N games
    rotation_min: float = 15.0  # ...and average at least this many minutes (tuned)


@dataclass
class _Player:
    name: str
    team: str
    season: int
    s_min: float = 0.0
    s_pir: float = 0.0
    s_n: float = 0.0
    last_team_game: int = -1  # index of the team's last game this player played in


@dataclass
class AvailabilityModel:
    cfg: AvailabilityConfig = field(default_factory=AvailabilityConfig)
    players: dict = field(default_factory=dict)
    team_games: dict = field(default_factory=dict)  # team -> number of games processed
    league_rate: float = 0.43

    def _value(self, p: _Player) -> tuple[float, float]:
        """(expected minutes, value above replacement) for one player."""
        cfg = self.cfg
        exp_min = p.s_min / p.s_n if p.s_n > 0 else 0.0
        rate = (p.s_pir + cfg.prior_minutes * self.league_rate) / (p.s_min + cfg.prior_minutes)
        return exp_min, exp_min * (rate - cfg.replacement * self.league_rate)

    def rotation(self, team: str, season: int) -> pd.DataFrame:
        n = self.team_games.get(team, 0)
        rows = []
        for pid, p in self.players.items():
            if p.team != team or p.season != season:
                continue
            if n - p.last_team_game > self.cfg.rotation_games:
                continue
            exp_min, value = self._value(p)
            if exp_min >= self.cfg.rotation_min:
                rows.append({"player_id": pid, "player": p.name, "exp_min": exp_min,
                             "value": value, "games_since": n - 1 - p.last_team_game})
        cols = ["player_id", "player", "exp_min", "value", "games_since"]
        return pd.DataFrame(rows, columns=cols).sort_values("value", ascending=False)

    def run(self, games: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
        """Walk through history; return missing value per side for every played game."""
        cfg = self.cfg
        p_ok = box[box["min"] > 0]
        tot = p_ok.groupby("season")[["pir", "min"]].sum()
        rates = (tot["pir"] / tot["min"]).to_dict()
        by_game = {gid: d for gid, d in box.groupby("game_id")}
        self.players, self.team_games = {}, {}
        out = []
        for g in games[games["played"]].sort_values("date").itertuples(index=False):
            # League rate from earlier seasons only (fallback: first season's own rate).
            prev = [rates[s] for s in rates if s < g.season]
            self.league_rate = float(np.mean(prev[-3:])) if prev else rates.get(g.season, 0.43)
            d = by_game.get(g.game_id)
            row = {"game_id": g.game_id, "missing_home": 0.0, "missing_away": 0.0,
                   "n_missing_home": 0, "n_missing_away": 0}
            if d is None:
                out.append(row)
                continue
            for side, team in (("home", g.home), ("away", g.away)):
                rot = self.rotation(team, g.season)
                played = set(d.loc[(d["team"] == team) & (d["min"] > 0), "player_id"])
                miss = rot[~rot["player_id"].isin(played)]
                row[f"missing_{side}"] = float(miss["value"].clip(lower=0).sum())
                row[f"n_missing_{side}"] = len(miss)
            out.append(row)
            self._update(g, d)
        return pd.DataFrame(out)

    def _update(self, g, d: pd.DataFrame) -> None:
        cfg = self.cfg
        for team in (g.home, g.away):
            idx = self.team_games.get(team, 0)
            for r in d[d["team"] == team].itertuples(index=False):
                p = self.players.get(r.player_id)
                if p is None:
                    p = self.players[r.player_id] = _Player(r.player, team, g.season)
                if p.season != g.season:
                    p.s_min *= cfg.season_decay
                    p.s_pir *= cfg.season_decay
                    p.s_n *= cfg.season_decay
                    p.season = g.season
                p.team, p.name = team, r.player
                if r.min <= 0:
                    continue
                p.s_min = p.s_min * cfg.decay + r.min
                p.s_pir = p.s_pir * cfg.decay + r.pir
                p.s_n = p.s_n * cfg.decay + 1
                p.last_team_game = idx
            self.team_games[team] = idx + 1

    def missing_for(self, team: str, season: int, absences: dict[str, float],
                    matcher) -> tuple[float, pd.DataFrame]:
        """Expected missing value for a team.

        `absences` maps player names (any spelling) to the probability they miss
        the game; names are fuzzy-matched to the team's rotation, so players from
        other teams or outside the rotation are ignored."""
        rot = self.rotation(team, season)
        if rot.empty or not absences:
            return 0.0, rot.iloc[0:0].assign(p_out=[])
        p_out = {}
        for name, w in absences.items():
            hit = matcher(name, rot["player"])
            if hit:
                p_out[hit] = max(w, p_out.get(hit, 0.0))
        miss = rot[rot["player"].isin(p_out)].copy()
        miss["p_out"] = miss["player"].map(p_out)
        return float((miss["value"].clip(lower=0) * miss["p_out"]).sum()), miss
