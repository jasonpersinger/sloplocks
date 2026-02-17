# NCAAM Support — Design Document

**Date:** 2026-02-17
**Status:** Approved

## Overview

Add NCAA Men's Basketball (NCAAM) as the third sport on SLOP LOCKS, featuring a 3-model ensemble: Elo ratings, Adjusted Efficiency (KenPom-style), and Four Factors logistic regression. Regular season predictions ship first; tournament-specific adjustments (neutral site, single elimination) follow later.

## Decisions

- **Scope:** Regular season now, enhanced tournament mode later
- **Data source:** ESPN hidden API (no key needed) + The Odds API (`basketball_ncaab`)
- **History depth:** Current season only (2025-26)
- **Architecture:** Three independent models, softmax-weighted ensemble (Approach A)
- **Four Factors approach:** Logistic regression (scikit-learn)

## Data Layer — `pipeline/fetch_ncaam.py`

ESPN API client with two main functions:

### `fetch_ncaam_games(season) -> (DataFrame, DataFrame)`

Returns two DataFrames:

1. **Scores** (standard schema): `date, home_team, away_team, home_goals, away_goals`
2. **Box scores**: `game_id, team, fgm, fga, fg3m, fg3a, ftm, fta, orb, drb, to, pts, possessions`

Possessions estimated as: `FGA - ORB + TO + 0.44 * FTA`

Endpoints:
- Scoreboard: `site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates=YYYYMMDD`
- Game summary: `site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary?event={id}`

Paginates by date range (season start -> today). Polite 0.5s delay between requests.

### `fetch_ncaam_schedule() -> list[dict]`

Upcoming games (today + 7 days). Returns `[{home_team, away_team, date}]`.

### Team name normalization

ESPN uses full names ("Duke Blue Devils"). Normalize to display names ("Duke"). Map built from ESPN teams endpoint (~360 D1 teams).

## Models — `pipeline/models.py` additions

### Model 1: Elo Ratings (existing)

Reuses existing `EloRatings` class with NCAAM-specific parameters:
- K-factor: 32 (higher variance in college)
- Home advantage: 125 Elo points (stronger home court in college)

### Model 2: Adjusted Efficiency (new)

`AdjustedEfficiency` class — KenPom-style tempo-adjusted ratings:

**Attributes:**
- `off_efficiency` — {team: adjusted points per 100 possessions}
- `def_efficiency` — {team: adjusted points allowed per 100 possessions}
- `tempo` — {team: average possessions per game}

**Algorithm:**
1. Compute raw offensive/defensive efficiency per team
2. Opponent-adjust iteratively (5-10 iterations until convergence)
3. Adjustment formula: `raw_eff * (league_avg / opp_avg_allowed)`

**Prediction (`efficiency_predict`):**
1. Expected tempo = average of both teams' tempos
2. Expected points = `off_eff * opp_def_eff / league_avg * tempo / 100`
3. Add home court bonus (+3.5 points equivalent)
4. Convert point spread to win probability via logistic function

### Model 3: Four Factors Logistic Regression (new)

`FourFactorsModel` class — Dean Oliver's four factors:

**Features (16 per matchup):**
- Home team: off_efg, off_to_rate, off_orb_pct, off_ft_rate, def_efg, def_to_rate, def_orb_pct, def_ft_rate
- Away team: same 8 features

**Four Factors definitions:**
- eFG% = `(FGM + 0.5 * FG3M) / FGA`
- TO rate = `TO / possessions`
- ORB% = `ORB / (ORB + opp_DRB)`
- FT rate = `FTA / FGA`

**Training:** Fit `sklearn.LogisticRegression` on all completed games this season.
**Prediction:** `predict_proba` on the 16-feature matchup vector.

## Config — `pipeline/config.py`

New entry in `SPORTS` dict:

```python
"ncaam": {
    "name": "NCAAM",
    "display_name": "NCAAM",
    "odds_sport": "basketball_ncaab",
    "outcomes": ["home", "away"],
    "models": ["elo", "efficiency", "four_factors"],
    "elo_k_factor": 32,
    "elo_home_advantage": 125,
    "efficiency_home_bonus": 3.5,
    "data_dir": os.path.join(DATA_DIR, "ncaam"),
}
```

## Pipeline Integration — `pipeline/run.py`

New `elif sport_key == "ncaam"` branch in `run_sport_pipeline()`:
- Calls `fetch_ncaam_games()` for scores + box scores
- Calls `fetch_ncaam_schedule()` for fixtures
- Fits all three models
- Feeds into existing ensemble/edge/pick machinery unchanged

New model fitting conditionals:
- `if "efficiency" in sport["models"]` -> fit AdjustedEfficiency
- `if "four_factors" in sport["models"]` -> fit FourFactorsModel

## Frontend — `index.html`

- Add "NCAAM" sport pill to toggle bar
- No new UI components (2-way outcomes like NBA, existing renderer handles it)

## Dependencies

- Add `scikit-learn` to `pipeline/requirements.txt`

## GitHub Actions

- No new secrets needed
- NCAAM runs alongside EPL and NBA in daily cron
- First run fetches full season (~4000 game summaries, ~5-10 min)

## Future: Tournament Mode

Not in scope for this implementation. Future additions:
- Neutral site detection (home advantage -> 0)
- Tournament-specific Elo adjustments
- Bracket prediction mode
