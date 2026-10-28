"""Paths, constants and tunable model settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

COMPETITION = "E"  # E = EuroLeague, U = EuroCup

ROOT = Path(os.environ.get("ELBET_HOME", Path.cwd()))
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"
ODDS_DIR = DATA_DIR / "odds"
REPORTS_DIR = ROOT / "reports"

FIRST_SEASON = 2012  # earliest season fetched by default


def current_season(now: datetime | None = None) -> int:
    """EuroLeague seasons start in autumn; season 2026 = 2026-27."""
    now = now or datetime.now(timezone.utc)
    return now.year if now.month >= 7 else now.year - 1


def season_code(year: int) -> str:
    return f"{COMPETITION}{year}"


@dataclass
class EloConfig:
    k: float = 20.0
    hfa: float = 65.0  # home-court edge in Elo points
    carryover: float = 0.70  # fraction of rating kept between seasons
    init: float = 1500.0
    new_team: float = 1450.0


@dataclass
class RatingConfig:
    half_life_days: float = 70.0  # recency decay of past games
    prev_season_weight: float = 0.5  # extra discount for last season's games (roster churn)
    window_days: int = 420
    ridge_team: float = 6.0  # shrinkage of team ratings toward league average
    ridge_pace: float = 6.0
    default_pace: float = 72.0


@dataclass
class PropsConfig:
    decay: float = 0.88  # per-game decay of a player's history
    season_decay: float = 0.4  # extra decay applied across a season break
    prior_minutes: float = 40.0  # shrink per-minute rates toward the league rate
    recent_team_games: int = 3  # player must have played in one of the team's last N games


@dataclass
class BetConfig:
    kelly_fraction: float = 0.25
    max_stake_pct: float = 0.03
    min_edge: float = 0.03  # minimum expected value per unit staked
    min_prob: float = 0.10
    max_odds: float = 6.0


@dataclass
class Settings:
    elo: EloConfig = field(default_factory=EloConfig)
    ratings: RatingConfig = field(default_factory=RatingConfig)
    props: PropsConfig = field(default_factory=PropsConfig)
    bet: BetConfig = field(default_factory=BetConfig)


SETTINGS = Settings()
