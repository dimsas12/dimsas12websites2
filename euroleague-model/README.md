# elbet: EuroLeague betting model

A EuroLeague basketball model that prices **moneyline, point spread, totals and
player props**, compares the prices with live bookmaker odds from
[The Odds API](https://the-odds-api.com) and sizes bets with fractional Kelly.

Data comes from the official EuroLeague APIs (schedules and full box scores),
cached locally under `data/`.

## Quick start

```bash
cd euroleague-model
pip install -e .
```

```bash
elbet fetch            # download 2012-13 .. current season (first run takes ~45 min)
```

```bash
elbet build            # build tables and features, train game + props models
```

```bash
elbet backtest         # walk-forward evaluation, writes reports/backtest.md
```

```bash
elbet predict --days 7 # model prices for upcoming games
```

```bash
export ODDS_API_KEY=your_key
elbet picks --bankroll 1000      # value bets against live odds
```

```bash
elbet props E2026_85 --lines my_lines.csv --exclude "LESSORT, MATHIAS"
```

During the season run `elbet update` (re-downloads the current season and
rebuilds) before `predict` or `picks`.

### Weekly routine (paper trading)

1. `elbet update`, then `elbet picks --market-weight 0.5`: picks are logged to
   `data/tracking/bets.csv` (about 3 Odds API credits).
2. 30-90 minutes before each game day's first tip: `elbet closing` records the
   closing price of every logged bet (3 credits; free if nothing is due).
3. After the games: `elbet update && elbet track`.

**Closing-line value (CLV)** = the expected value of your bet at the closing
no-vig price. A model with a real edge beats the close on average; after 100+
bets, an average CLV clearly above zero is far stronger evidence than profit.
About 50 credits a month covers this routine; the free plan has 500.

### Injuries

`predict`, `picks` and `props` download the current EuroLeague injury list
(basketballsphere.com, built from club reports) before every run. Out = 100%
absent, Doubtful 75%, Questionable / Day to day 50%, Cleared 0%. Rotation
players who are absent lower their team's rating by their value; the table
shows the impact in points. Add news the feed is missing in
`data/injuries/manual.csv`:

```
player,team,status
Sasha Vezenkov,Olympiacos,Out
Kendrick Nunn,Panathinaikos,Questionable
```

or pass `--out "Name" ...` for a one-off. `--no-injuries` switches it off.

## Commands

| command | what it does |
|---|---|
| `fetch [--start Y --end Y]` | download schedules + box scores (cached; only new games are fetched) |
| `build` | parse data, compute features, train and save models |
| `update` | `fetch` current season + `build` |
| `backtest [--from-season Y] [--odds-csv F]` | walk-forward metrics; with an odds CSV, an ROI simulation |
| `ratings` | current power ratings (net/off/def rating per 100 possessions, pace, Elo) |
| `predict [--days N]` | win probability, fair odds, fair spread and total for upcoming games |
| `picks` | live odds from The Odds API, edge (EV) per market, Kelly stakes; saves `reports/picks_*.csv` and the full board |
| `props GAME_ID` | player projections (minutes, PTS, REB, AST, 3PM, PRA), probabilities for lines you give |
| `excel [--date D]` | Excel report for one game day: picks + tracker, games vs market, odds board, injuries, props with SD/fair lines, over/under ladder and a line checker |
| `injuries` | the injury list the model uses (auto-downloaded; manual overrides in `data/injuries/manual.csv`) |
| `closing [--window MIN]` | record closing odds for logged bets tipping off soon; computes CLV |
| `track [--show N]` | settle finished bets; profit, ROI and closing-line value report |
| `odds-history --season Y` | historical pre-game odds for a backtest (paid Odds API plan) |

Useful options for `picks`: `--min-edge 0.04`, `--kelly 0.25`,
`--regions eu,uk`, `--bookmakers pinnacle,bet365`, and `--market-weight 0.5`
to blend the model with the no-vig market probability.

### Prop lines CSV

```
player,stat,line,over_odds,under_odds
Vezenkov,pts,15.5,1.87,1.93
"MILUTINOV, NIKOLA",reb,6.5,1.80,2.00
```

`stat` is one of `pts reb ast fgm3 pra`. Names are matched loosely. The Odds API
does not currently list EuroLeague player props for most bookmakers; `--odds-api`
tries anyway and tells you if nothing comes back.

## How the model works

**Team ratings (all strictly pre-game, so backtests can't see the future)**

* *Adjusted efficiency*: for every game day, a recency-weighted ridge
  regression on the last ~14 months of games fits each team's offensive and
  defensive rating (points per 100 possessions, adjusted for opponent and home
  court) and pace. Half-life 250 days (tuned on 2014-17 and checked on
  later seasons). Possessions = FGA + 0.44·FTA − OREB + TOV.
* *Elo* with a margin-of-victory multiplier, 70% carry-over between seasons.

**Game model**: projected margin and total start from the efficiency
projection (`pts = efficiency × pace / 100`); a heavily shrunk ridge regression
then corrects them with Elo, rest (EuroLeague double-round weeks), neutral venue,
playoff phase, early-season flags and **player availability** (the summed
value of absent rotation players, from PIR per minute above replacement level
times expected minutes; learned from 15 seasons of who actually played). It is retrained each season on earlier
seasons only. Outcomes are normal with the residual SD (≈12 points for margin,
≈17 for total), discretised to whole points so whole-number lines get a correct
push probability and a tied margin is impossible (overtime).

**Player props** (projections are *if the player plays*, as books settle them):
each player gets a chance of playing (recent appearance rate × (1 − injury
probability)); minutes if playing are scaled so the team's expected minutes are
exactly 200 (max 40 each), and points are scaled so expected player points equal
the game model's team score. Per player: expected minutes × per-minute rate (decayed averages of the
player's previous games, shrunk toward the league rate; little shrinkage for
assists and threes, which are very player-specific) × pace factor ×
opponent-defence factor. Negative binomial outcomes with dispersion fitted per
stat. Minutes of players you `--exclude` are given to the rest of the roster.

**Betting**: no-vig market probabilities (power method, averaged across all
books), EV at the best price among usable books (offshore books such as 1xBet,
GTbets, BetOnline are excluded; `--bookmakers` limits it to your own books) and
after exchange commission (Betfair 5%, Matchbook 2%), fractional Kelly capped at 3% of bankroll per
bet and 20% per slate, one bet per game (use `--all-markets` to allow more),
and filters on minimum EV (3%), minimum probability and maximum odds.

## Backtest results

Walk-forward, 2015-16 to 2026-27 (first 76 games), 3,375 games. Every season
is predicted by a model trained only on earlier seasons. Rating settings were
tuned on 2014-17, so 2018 onward is a genuine out-of-sample check.

| metric | model | Elo only | always "home team p" |
|---|---|---|---|
| winner accuracy | 66.1% | | |
| log loss (lower is better) | **0.6097** | 0.621 | 0.662 |
| Brier score | 0.212 | | |
| margin MAE (points) | 9.27 | | |
| total MAE (points) | **13.20** | | 13.84 (recent league average) |

Win probabilities are well calibrated: games priced 60-70% were won 65.3% of
the time, 80-90% won 88.4%.

Player props (69,912 player-games): MAE PTS 3.99, REB 1.57, AST 1.09, 3PM 0.72,
PRA 5.16, slightly better than the player's recent average on every stat
except assists (tied). Props depend heavily on knowing who plays, so use
`--exclude` for absent players; that is where most of the value is.

None of this shows the model beats bookmakers: that needs odds. See below.

## Limitations and honest caveats

* The model knows only EuroLeague games: no domestic-league fatigue or travel.
  Injuries come from a news feed and can lag late scratches; check line-ups.
* Totals are the weakest market: the model disagrees with bookmakers by 6-10
  points on some games. Use `--market-weight 0.5` or higher until CLV says otherwise.
* Season starts are the weakest period: ratings come from last season while
  rosters have changed.
* A good backtest log loss does not mean the model beats bookmakers. Prove that
  with `odds-history` + `backtest --odds-csv`, or by logging `picks` and
  checking closing-line value, before staking real money.
* Bet responsibly and only where it is legal for you.

## Layout

```
elbet/fetch.py      EuroLeague API download + cache
elbet/dataset.py    raw JSON -> games / player tables
elbet/ratings.py    Elo + adjusted efficiency ratings
elbet/features.py   pre-game feature table
elbet/models.py     margin/total model, market probabilities
elbet/props.py      player prop projections
elbet/odds.py       The Odds API client, team matching, consensus lines
elbet/betting.py    devig, EV, Kelly
elbet/backtest.py   walk-forward evaluation and bet simulation
elbet/cli.py        command line
```
