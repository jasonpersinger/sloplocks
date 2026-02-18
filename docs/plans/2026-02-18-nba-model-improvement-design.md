# NBA Model Improvement Design

**Date:** 2026-02-18
**Goal:** Improve NBA prediction edge quality by fixing home advantage bias and adding two additional models to match the EPL/NCAAM ensemble structure.

## Problem

The current NBA model is Elo-only with a 100-point home advantage. This produces heavily home-biased picks — all five of today's slop locks were home teams, including terrible teams like the Hornets and Wizards. The 100-point value is uncalibrated and the single-model approach has no cross-validation.

## Design

### Data Layer

Switch NBA game and box score fetching from balldontlie.io to the ESPN API (same source already powering NCAAM). ESPN provides pre-aggregated team box score totals via the scoreboard + summary endpoints.

- Update `fetch_nba.py` with ESPN-based `fetch_nba_espn_games()` and `fetch_nba_espn_schedule()` functions
- `fetch_nba_espn_games()` returns `(games_df, box_scores_df)` matching the NCAAM schema, including `game_id`
- Season range: Oct 1 through Apr 20 (or today, whichever is earlier)
- Existing balldontlie.io functions stay in the file but are no longer called from `run.py`
- ESPN base: `https://site.api.espn.com/apis/site/v2/sports/basketball/nba`

### Model Changes

**1. Elo recalibration**
- Change `elo_home_advantage` for NBA in `config.py`: `100 → 65`
- 65 points is the empirically calibrated NBA standard, mapping to roughly a 3–4 point home court edge
- One-line config change; no model code changes

**2. Adjusted Net Rating (AdjustedEfficiency)**
- Reuse existing `AdjustedEfficiency` class from `models.py` (currently used by NCAAM)
- Computes offensive and defensive efficiency per 100 possessions, adjusted for opponent strength via iterative rating
- Requires box scores: pts, fgm, fga, fg3m, fg3a, ftm, fta, orb, drb, to, possessions

**3. Four Factors (FourFactorsModel)**
- Reuse existing `FourFactorsModel` class from `models.py` (currently used by NCAAM)
- Models eFG%, turnover rate, offensive rebounding rate, free throw rate via logistic regression
- Captures *how* a team wins, not just win/loss record — useful for teams on unsustainable streaks

**4. Rest/Back-to-Back Adjustment**
- Not a standalone model; a correction applied at prediction time to the Elo step
- Compute `days_since_last_game` for each team using the historical `matches` DataFrame
- If a team played yesterday (B2B): apply −30 Elo point penalty to their effective rating for that prediction
- Stored Elo ratings are unchanged; the adjustment is ephemeral per prediction
- Maps to ~4–5% win probability swing, consistent with published B2B research
- Applied only to Elo; efficiency/four factors are season averages and absorb fatigue naturally

### Pipeline Integration

`run.py` NBA branch changes:
- Replace `fetch_nba_games()` / `fetch_nba_schedule()` calls with `fetch_nba_espn_games()` / `fetch_nba_espn_schedule()`
- Add `box_scores_df` handling (same as NCAAM branch)
- Build `AdjustedEfficiency` and `FourFactorsModel` for NBA
- Add rest adjustment logic in prediction loop (compute B2B status, adjust Elo ratings per prediction)
- Model weight list: `["elo"] → ["elo", "efficiency", "four_factors"]`

`config.py` changes:
- `elo_home_advantage: 100 → 65` for NBA
- `models: ["elo"] → ["elo", "efficiency", "four_factors"]` for NBA

### Testing

- `TestFetchNbaEspn`: mirrors NCAAM test structure, covers `fetch_nba_espn_games()` and `fetch_nba_espn_schedule()` with mocked ESPN responses
- `TestRestAdjustment`: unit tests for B2B detection logic
- Existing `TestAdjustedEfficiency` and `TestFourFactorsModel` tests already cover the reused model classes

### Files Touched

| File | Change |
|------|--------|
| `pipeline/config.py` | NBA `elo_home_advantage` 100→65, `models` list |
| `pipeline/fetch_nba.py` | Add ESPN-based game/box score/schedule fetchers |
| `pipeline/run.py` | Wire ESPN fetch + efficiency models + rest adjustment for NBA |
| `tests/test_fetch_nba.py` | Add ESPN fetch tests + rest adjustment tests |

## Non-Goals

- Removing `BALLDONTLIE_API_KEY` from GitHub Actions secrets (separate cleanup)
- Injury data integration (requires a different data source)
- Pace-adjusted point spread modeling (future improvement)
