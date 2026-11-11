# msbet: multi-sport betting models

Football (leagues, European cups, national teams), tennis (ATP, WTA), NBA, NFL and
NCAA. One command line, one Excel workbook per sport per day. The EuroLeague model
lives next door in `../euroleague-model`.

```bash
pip install -e .
msbet soccer update        # download + build (also: tennis, nba, nfl, ncaab, ncaaf)
msbet soccer excel --odds  # daily workbook; --odds needs ODDS_API_KEY in the environment
msbet soccer backtest      # the value-rule backtest below
msbet nba backtest         # model vs closing lines
```

## What is and is not profitable (honest results)

Every number is out of sample: ratings are strictly pre-game and calibrations are
fitted on earlier seasons only.

| Sport | Test | Model alone | Market | Verdict |
|---|---|---|---|---|
| Football 1X2 | 25,335 matches 2020-26, log loss | 0.985 | 0.968 (close) | model loses; betting it: -6% |
| Football value rule | 6,660 bets 2020-26 | **+3.1% ROI** (s.e. 1.75%), CLV +2.75% | | profitable in 9 of 11 seasons since 2015 |
| NBA | 5,764 games 2023-26, margin error | 11.07 | 10.55 (closing spread) | -5.5% against the spread |
| NFL | 1,824 games 2020-26, margin error | 10.17 | 9.75 | -6.6% against the spread |
| NCAA basketball | 11,851 games 2025-26 | 9.64 | 8.87 | -3.9% |
| NCAA football | 2,180 games 2023-26 | 13.38 | 11.77 | -8.0% |
| Tennis winner | 2024-26, accuracy | ATP 64.6%, WTA 64.0% | no free odds history | not tested for profit |

**The value rule** is the only strategy here with evidence of profit: take the fair
probability from the sharp books (Pinnacle, exchanges) with the margin removed, and
bet when the best available price beats it by 2% or more. `msbet/odds.py` applies it
to every sport. Caveats: the best price often sits at books that limit winning
customers, and the number of qualifying bets has fallen every year since 2021.

The statistical models are used where no sharp reference price exists or to derive
markets from it: correct score, both teams to score, Asian handicap, corners, set and
game markets in tennis, and player props. Player props have calibration checks (below)
but no profit backtest, because free historical prop odds do not exist.

## Models

- **Football** (`msbet/soccer`): online attack/defence ratings learning from goals
  (35%) and shots on target (65%), league strengths linked through European cup games,
  Poisson calibration + Dixon-Coles. Corners: separate ratings, negative binomial.
  Props: per-90 rates from ESPN line-ups, shrunk to position averages, scaled to the
  team's expected goals. Data: football-data.co.uk (odds), ESPN (72,000 matches, 45
  competitions incl. national teams).
- **Tennis** (`msbet/tennis`): overall + surface Elo (K shrinking with experience);
  a point-by-point Markov model, mixed over uncertainty in the true skill gap, prices
  sets, total games, handicaps and tie-breaks; aces from serve statistics (ATP).
- **NBA / NFL / NCAA** (`msbet/us`): offence/defence ratings in points with blowout
  damping and rest, normal margin/total distributions. NBA props: per-minute rates x
  projected minutes, ESPN injury report, negative binomial (minutes MAE 4.5, points
  MAE 4.9; P(over) calibrated within 2-4 points). NFL props: recency-weighted game
  averages, opponent adjustment, gamma yardage (skewed) and anytime touchdown.

## Odds API credits

One call costs markets x regions. Per run: football 2 per league, NBA/NFL/NCAA 3,
tennis 1 per active tournament. A full daily run of everything is roughly 20-30
credits, so the free 500/month covers about two weeks of daily use; player-prop odds
cost extra per game and are not fetched by default. `MSBET_REPLAY=1` re-uses the last
saved snapshot without spending credits.


## Absences (injuries, suspensions)

| Sport | Source of absences | Effect tested (out of sample) |
|---|---|---|
| EuroLeague | injury feed + inference (see `../euroleague-model`) | in the main model |
| NBA | ESPN injury report per game | team margin error 11.27 -> 11.18 (2024-26); props drop / redistribute minutes |
| NFL | nflverse injury report (game status, else practice, else last week's Outs) | margin error 10.17 -> 10.12 (2020-26); one projected QB, workload of absent players moved to teammates |
| Football | Premier League: official FPL availability list; every club: regulars who missed recent matches (P(out) learned by streak length) and red cards | log loss 0.9684 -> 0.9676, goals deviance 1.679 -> 1.671 (2023-26); props drop / scale absent players; `Absences` sheet lists them |
| Tennis | none: withdrawals leave the draw; recent retirements and long layoffs were tested and did not help | - |
| NCAA basketball | tested with inferred absences; no gain (9.641 -> 9.654), so not used | - |
| NCAA football | no player data | - |

Mechanics: `msbet/avail.py` computes, for each team-game, the value (exposure x
production above replacement) of regulars who did not play; the game model learns the
cost of one unit of missing value from earlier seasons. For upcoming games the same
sum is weighted by each player's chance of being out. The gains are small: the market
already prices absences, and the statistical models still lose to closing lines.
