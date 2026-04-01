# Model Analysis

## Current Model Architecture

### Data flow

The live pipeline runs through `pipeline/run.py:647-1169`.

1. Historical results and schedules are fetched per sport from ESPN in the sport-specific fetchers:
   - NBA: `pipeline/fetch_nba.py:300-474`
   - NCAAM: `pipeline/fetch_ncaam.py:240-440`
   - MLB: `pipeline/fetch_mlb.py:191-243`
   - MMA: `pipeline/fetch_mma.py:97-195`
2. Market odds are fetched from The Odds API and reduced to best available decimal prices per outcome in `pipeline/fetch_data.py:15-102`.
3. Sport-specific models generate per-outcome probabilities:
   - Elo for all sports: `pipeline/models.py`
   - Results-feature logistic model for NBA, NCAAM, MLB, and MMA: `pipeline/models.py`
   - Recent box-score matchup model for NBA and NCAAM: `pipeline/models.py`
   - NBA matchup-context model for venue splits, rest, and pace/style interaction: `pipeline/models.py`
   - NBA totals model for projected game totals and over/under pricing: `pipeline/models.py`
   - Adjusted Efficiency for NBA and NCAAM: `pipeline/models.py`
   - Four Factors logistic regression for NBA and NCAAM: `pipeline/models.py`
   - Pitcher matchup, bullpen matchup, handedness matchup, run-environment, and totals models for MLB: `pipeline/models.py`
4. Historical model accuracy is converted into ensemble weights in `pipeline/backtest.py:72-87`, then blended in `pipeline/ensemble.py:55-69`.
5. Historical isotonic calibration is fit from resolved `history.json` entries in `pipeline/ensemble.py:72-136` and applied in `pipeline/run.py:783-797`, `pipeline/run.py:891-897`.
6. Odds-aware edges, EV, Kelly fractions, and confidence scores are computed in `pipeline/ensemble.py:210-276`.
7. Picks are filtered into `slop_locks`, `longslop`, and `slimegrinder` in `pipeline/run.py:445-605`.
8. Resolved predictions and picks feed `history.json`, `pick_history.json`, `model_accuracy.json`, and the persistent CSV log at `data/tracking/results_log.csv` through `pipeline/run.py:87-183` and `pipeline/run.py:981-1128`.

### Feature inventory by sport

#### NBA

Configured models live in `pipeline/config.py:62-86`.

- Elo features:
  - Base team Elo and home-court advantage from historical win/loss results in `pipeline/models.py:458-469`.
  - Rest/fatigue adjustments from recent schedule density in `pipeline/run.py:286-319`.
  - Recent-form Elo adjustment from the last `6` games, capped at `35` Elo points, configured in `pipeline/config.py:74-83` and computed in `pipeline/run.py:245-284`.
- Results-feature model:
  - Walk-forward season win%, recent win%, season margin, recent margin, venue win%, rest-day differential, and games-played differential from historical results in `pipeline/models.py`.
- Recent box-score model:
  - Walk-forward season/recent net rating, eFG%, turnover rate, rebound rate, free-throw rate, and pace differentials from ESPN box scores in `pipeline/models.py`.
- NBA matchup model:
  - Walk-forward season and recent net-rating differentials, home-versus-road split strength, offense-vs-defense interaction, pace mismatch, recent margin, and rest-day differential from ESPN box scores plus game dates in `pipeline/models.py`.
- NBA totals model:
  - Walk-forward recent scoring, recent prevention, pace, and home/road scoring form from ESPN box scores to project game totals and price over/under markets in `pipeline/models.py` and `pipeline/run.py`.
- Live availability features:
  - Current roster injury burden, available core-player count, and weighted absence of current scoring/playmaking/rebounding leaders from ESPN scoreboard plus roster data in `pipeline/fetch_nba.py` and `pipeline/run.py`.
- Adjusted Efficiency features:
  - Weighted points scored, points allowed, possessions, and opponent-adjusted offense/defense from ESPN box scores in `pipeline/models.py:510-605`.
  - Tempo is derived from weighted possessions and fed into expected pace in `pipeline/models.py:635-647`.
  - Recency weighting uses exponential decay plus a boost on the most recent `FORM_WINDOW=6` games in `pipeline/models.py:59-89`.
- Four Factors features:
  - Offensive and defensive eFG%, turnover rate, offensive rebound rate, and free-throw rate from ESPN box scores in `pipeline/models.py:729-819`.
  - These are season-to-date weighted aggregates, not fixture-specific snapshots.
- Fetched but unused inputs:
  - Raw ESPN box-score fields like `fg3a` are only used indirectly through possessions and factor calculations.
  - No travel, referee, or confirmed-lineup minute-allocation features are currently consumed.

#### NCAAM

Configured models live in `pipeline/config.py:87-111`.

- Elo features:
  - Same core Elo path as NBA, with larger home advantage and NCAAM-specific rest/form parameters in `pipeline/config.py:96-108`.
  - Recent-form adjustment uses the last `8` games, capped at `25` Elo points.
  - Rest logic now applies to NCAAM as well through `pipeline/run.py:286-319`.
- Results-feature model:
  - Same walk-forward results features as NBA, with a longer `10`-game recent window configured in `pipeline/config.py`.
- Recent box-score model:
  - Same walk-forward box-score form features as NBA, tuned with a `10`-game recent window in `pipeline/config.py`.
- Adjusted Efficiency features:
  - Same weighted offense/defense/tempo machinery as NBA, sourced from ESPN box scores in `pipeline/models.py:487-647`.
  - Opponent adjustment is iterative and based on season-to-date weighted stats.
- Four Factors features:
  - Same eight offensive/defensive factor features as NBA in `pipeline/models.py:751-819`.
- Fetched but unused inputs:
  - Dual ESPN scoreboard groups (`50` and `100`) improve coverage in `pipeline/fetch_ncaam.py:279-304`, but no tournament- or conference-specific feature is derived from them.
  - No roster, injury, travel, or coaching-style features are used.

#### MLB

Configured models live in `pipeline/config.py:112-137`.

- Elo features:
  - Base team Elo, modest home-field advantage, recent-form adjustment over the last `10` games, capped at `16` Elo points, via `pipeline/config.py:121-133` and `pipeline/run.py:245-284`.
  - No rest penalties are applied (`back_to_back_penalty=0`, `fatigue_penalty=0`).
- Results-feature model:
  - Walk-forward season/recent win% and scoring-margin features from historical game results in `pipeline/models.py`.
- Pitcher matchup model:
  - Walk-forward starter RA9, recent RA9, K-BB per inning, recent team margin in starts, recent innings workload, days-rest differential, and start-count differential from ESPN summary data cached by `pipeline/fetch_mlb.py`.
- Bullpen matchup model:
  - Walk-forward bullpen RA9, K-BB per inning, recent innings workload, and team margin support from aggregated non-starter pitching lines in `pipeline/fetch_mlb.py` and `pipeline/models.py`.
- Handedness matchup model:
  - Walk-forward team run and margin splits versus left-handed and right-handed starters using cached starter throwing-hand metadata from ESPN core athlete records in `pipeline/fetch_mlb.py` and `pipeline/models.py`.
- Run-environment model:
  - Walk-forward team runs scored, runs allowed, recent scoring form, recent prevention form, recent margin, and home park-factor context using `MLB_PARK_FACTORS` in `pipeline/config.py` and `pipeline/models.py`.
- Totals model:
  - Walk-forward regression of expected game runs using team scoring form, starter run prevention, bullpen run prevention, and park context, then live over/under pricing with weather adjustment in `pipeline/models.py` and `pipeline/run.py`.
- Fetched but unused inputs:
  - Bullpen leverage roles and catcher/framing context are still not modeled.

#### MMA

Configured models live in `pipeline/config.py:138-160`.

- Elo features:
  - Fighter Elo only, with no home advantage in `pipeline/config.py:147-158` and `pipeline/models.py:458-469`.
  - Recent-form adjustment over the last `4` fights, capped at `20` Elo points, via `pipeline/run.py:245-284`.
  - MMA historical outcomes are derived from ESPN fight winners as binary `1/0` results in `pipeline/fetch_mma.py:63-94`.
- Results-feature model:
  - Walk-forward recent/season win and margin features from prior fight outcomes in `pipeline/models.py`.
- Fetched but unused inputs:
  - No fighter-level stats such as age, reach, stance, significant strike rate, takedown rate, or layoff length are fetched or used.

### Ensemble mechanics

- Active model lists are sport-specific in `pipeline/config.py:62-160`.
  - NBA uses six side models: `elo`, `efficiency`, `four_factors`, `results_features`, `recent_boxscore`, `nba_matchup`.
  - NBA now also has a totals lane driven by `NbaTotalsModel` plus live totals-market pricing.
  - NCAAM uses five models: `elo`, `efficiency`, `four_factors`, `results_features`, `recent_boxscore`.
  - MLB uses six side models: `elo`, `results_features`, `pitcher_features`, `bullpen_features`, `run_environment`, `handedness_features`.
  - MLB also now has a separate totals lane driven by `MlbTotalsModel` plus live totals-market pricing.
  - MMA uses two models: `elo`, `results_features`.
- Historical rolling accuracy for each model is read from `model_accuracy.json` and transformed into softmax weights in `pipeline/backtest.py:72-87` and `pipeline/run.py:775-781`.
- The softmax temperature is now configurable per sport:
  - `3.0` for NBA and NCAAM
  - `1.5` for MLB and MMA
  This lives in `pipeline/config.py:68`, `pipeline/config.py:93`, `pipeline/config.py:118`, and `pipeline/config.py:144`.
- Final blended probabilities are a weighted average in `pipeline/ensemble.py:55-69`.
- After blending, isotonic calibrators are fit from resolved historical ensemble outputs and blended back into the live probability distribution in `pipeline/ensemble.py:72-136`.

### Confidence score, odds handling, and pick gating

- `compute_edges()` in `pipeline/ensemble.py:210-276` now does five things for each outcome:
  - converts decimal odds to raw implied probability
  - removes vig by normalizing active-outcome implied probabilities
  - shrinks the model toward the fair no-vig market with `calibrate_probability()`
  - computes probability edge and expected value
  - computes full Kelly and sport-specific fractional Kelly
- The output edge fields are:
  - `market_implied_prob`: raw `1 / decimal_odds`
  - `implied_prob`: fair no-vig probability after removing hold
  - `hold`: total market hold across active outcomes
  - `edge`: `calibrated_model_prob - implied_prob`
  - `expected_value`: expected profit per 1 unit staked
- Confidence is still score-based rather than a true calibrated confidence interval. `pipeline/ensemble.py:139-183` combines:
  - model agreement via standard deviation
  - model win probability
  - edge size
  - penalties for large market divergence and sub-45% underdogs
- MLB totals use the same vig removal / EV / Kelly logic through `compute_totals_edges()` in `pipeline/ensemble.py`, but they operate on `over` and `under` rather than `home` and `away`.
- Pick filters in `pipeline/run.py:445-605` now gate on EV in addition to probability edge:
  - `slop_locks`: `edge >= 0.03`, `model_prob > 0.45`, `expected_value >= min_expected_value`, maximum 3 picks, additional picks require `confidence_score >= 65`
  - `longslop`: `american_odds >= +500`, `confidence_score >= 65`, `edge >= 0`, `expected_value >= min_expected_value`
  - `slimegrinder`: bounded odds window, positive edge, positive EV threshold, then ranked by model probability

### Calibration, validation, and tracking

- Resolved predictions are evaluated with:
  - correctness and predicted winner in `pipeline/backtest.py:11-57`
  - Brier score in `pipeline/backtest.py:60-69`
  - rolling hit-rate logs for model weighting in `pipeline/backtest.py:114-169`
- Historical summaries are available via:
  - `summarize_prediction_history()` in `pipeline/backtest.py:172-201`
  - `summarize_pick_history()` in `pipeline/backtest.py:204-227`
  - CLI report builder in `pipeline/backtest.py:230-285`
- Ongoing post-result logging now persists resolved predictions and picks to `data/tracking/results_log.csv` through `pipeline/run.py:87-183`.
- The stack now has a true date-ordered replay layer over settled tracked outputs in `pipeline/backtest.py`, including daily cumulative prediction accuracy, pick ROI, and calibration bins. It still does not retrain the underlying sport models from raw inputs for every historical date, but it is no longer limited to static aggregate summaries.

## Changes Implemented

### Quick Wins

#### 1. Added sport-specific recent-form Elo adjustments

- What changed:
  - Added `recent_form_window` and `recent_form_max_adjustment` to each sport config in `pipeline/config.py:62-160`.
  - Added `_recent_form_adjustment()` in `pipeline/run.py:245-284`.
  - Applied recent-form adjustments in the Elo prediction path in `pipeline/run.py:839-862`.
- Why it matters:
  - The original Elo path treated a team on a strong short-term run the same as one on a stale season-long baseline.
  - This adds lightweight recency without rewriting the model stack.
- Current settings:
  - NBA: 6 games / 35 Elo points
  - NCAAM: 8 games / 25 Elo points
  - MLB: 10 games / 16 Elo points
  - MMA: 4 fights / 20 Elo points

#### 2. Replaced hardcoded NBA-only rest logic with sport-configurable rest/fatigue logic

- What changed:
  - Added `back_to_back_penalty`, `fatigue_window_days`, `fatigue_threshold_games`, `fatigue_penalty`, `rest_bonus_days`, and `rest_bonus_points` to `pipeline/config.py:62-160`.
  - Added `_rest_adjustment()` in `pipeline/run.py:286-319`.
  - Wired those adjustments into Elo predictions for every sport in `pipeline/run.py:839-862`.
- Why it matters:
  - Rest is a real signal in NBA and NCAAM, and it was previously either hardcoded or absent.
  - Moving these values into config makes the logic tunable per sport instead of magic-number-driven.

### Core Model Improvements

#### 3. Added no-vig market probabilities and hold tracking

- What changed:
  - Added `no_vig_probabilities()` in `pipeline/ensemble.py:22-33`.
  - `compute_edges()` now stores both raw market implied probability and fair no-vig probability, plus `hold`, in `pipeline/ensemble.py:231-274`.
- Why it matters:
  - Comparing model probabilities to vig-inflated implied probabilities understates edge and distorts ranking.
  - The pipeline now evaluates against a fairer approximation of the market.

#### 4. Added proper expected-value calculation and EV-aware pick filters

- What changed:
  - Added `expected_value()` in `pipeline/ensemble.py:36-41`.
  - `compute_edges()` now stores `expected_value` for each outcome in `pipeline/ensemble.py:243-269`.
  - Added `min_expected_value` to each sport in `pipeline/config.py:76`, `pipeline/config.py:101`, `pipeline/config.py:125`, `pipeline/config.py:151`.
  - `slop_locks`, `longslop`, `slimegrinder`, and refresh-time pick rebuilding all now gate on EV in `pipeline/run.py:445-605` and `pipeline/refresh_picks.py:44-144`.
- Why it matters:
  - A raw probability gap is not the same as expected betting return.
  - The selection logic now prefers spots where the payout structure actually supports the model's edge.

#### 5. Added full Kelly and fractional Kelly sizing outputs

- What changed:
  - Added `kelly_fraction()` in `pipeline/ensemble.py:44-52`.
  - `compute_edges()` now emits `kelly_fraction` and `fractional_kelly` in `pipeline/ensemble.py:245-269`.
  - Added per-sport `kelly_fraction` config in `pipeline/config.py:77`, `pipeline/config.py:102`, `pipeline/config.py:126`, and `pipeline/config.py:152`.
  - Those values are written into match records, picks, and refreshed outputs in `pipeline/run.py:923-956` and `pipeline/refresh_picks.py:66-140`.
- Why it matters:
  - Position sizing is now tied to modeled edge instead of being purely implicit.
  - The frontend JSON schema remains compatible while exposing better bankroll guidance.

#### 6. Added historical isotonic probability calibration

- What changed:
  - Added `fit_probability_calibrators()` and `apply_probability_calibration()` in `pipeline/ensemble.py:72-136`.
  - Added per-sport calibration thresholds in `pipeline/config.py:69-70`, `pipeline/config.py:94-95`, `pipeline/config.py:119-120`, and `pipeline/config.py:145-146`.
  - The live run now fits calibrators from resolved `history.json` and applies them after blending in `pipeline/run.py:783-797` and `pipeline/run.py:891-897`.
- Why it matters:
  - Raw ensemble probabilities rank outcomes, but they are not guaranteed to be well calibrated.
  - This creates a feedback loop from realized outcomes back into probability quality.

#### 7. Added sport-specific ensemble weighting temperature

- What changed:
  - `compute_model_weights()` now accepts a `temperature` argument in `pipeline/backtest.py:72-87`.
  - `run_sport_pipeline()` passes sport-level temperatures from config in `pipeline/run.py:775-781`.
- Why it matters:
  - The multi-model basketball sports benefit from sharper weighting toward recent better-performing models.
  - Single-model sports do not need the same weighting behavior.

### Advanced Improvements Implemented

#### 8. Added persistent resolved-results tracking

- What changed:
  - Added tracking path constants in `pipeline/config.py:5-6`.
  - Added results-log helpers in `pipeline/run.py:87-183`.
  - Resolved predictions and picks now append to `data/tracking/results_log.csv` during normal pipeline runs in `pipeline/run.py:981-1057`.
- Why it matters:
  - This creates a clean evaluation dataset for future calibration, CLV, and model-drift analysis.
  - It also removes dependence on only semi-structured JSON history files.

#### 9. Added historical reporting CLI with per-sport accuracy, log loss, Brier, and ROI

- What changed:
  - Added `compute_brier_score()`, `summarize_prediction_history()`, `summarize_pick_history()`, and `build_backtest_report()` in `pipeline/backtest.py:60-285`.
  - The module now runs as a CLI and prints a JSON report from saved histories.
- Why it matters:
  - The project now has an executable reporting layer instead of only raw stored history.
  - This is not yet a full walk-forward replay harness, but it materially improves visibility into model quality by sport.

#### 10. Added walk-forward results-feature models across all four sports

- What changed:
  - Added `ResultsFeatureModel` and `results_features_predict()` in `pipeline/models.py`.
  - Enabled it in `pipeline/config.py` for NBA, NCAAM, MLB, and MMA.
  - Wired it into the ensemble path in `pipeline/run.py`.
- Why it matters:
  - NBA/NCAAM/MLB/MMA now all get a second opinion built from recent and season-long results form instead of leaning purely on Elo or season-average team stats.
  - MMA in particular is no longer Elo-only.

#### 11. Added a recency-weighted basketball box-score matchup model

- What changed:
  - Added `RecentBoxScoreModel` and `recent_boxscore_predict()` in `pipeline/models.py`.
  - Enabled it for NBA and NCAAM in `pipeline/config.py`.
  - Wired it into the ensemble in `pipeline/run.py`.
- Why it matters:
  - The existing efficiency and four-factors layers are mostly season-level snapshots.
  - This new model gives NBA and NCAAM a walk-forward recent-form layer derived from net rating, shooting quality, turnover control, rebounding, free-throw pressure, and pace.

#### 12. Added pitcher-aware MLB modeling backed by ESPN summary data

- What changed:
  - Added starter extraction helpers to `pipeline/fetch_mlb.py` so completed MLB games cache starter names and basic pitching lines.
  - Added `PitcherMatchupModel` and `pitcher_matchup_predict()` in `pipeline/models.py`.
  - Enabled the model in MLB config and wired it into `pipeline/run.py`.
- Why it matters:
  - Starting pitchers are one of the strongest MLB moneyline inputs available in the current free-data stack.
  - This turns previously decorative probable-pitcher fields into an actual prediction signal.

#### 13. Added bullpen-aware MLB modeling from ESPN summary data

- What changed:
  - Extended `pipeline/fetch_mlb.py` to aggregate non-starter pitching lines into per-team bullpen innings, runs, earned runs, walks, and strikeouts.
  - Added `BullpenMatchupModel` and `bullpen_matchup_predict()` in `pipeline/models.py`.
  - Enabled the model in MLB config and wired it into `pipeline/run.py`.
- Why it matters:
  - MLB moneylines are not decided by starters alone, especially when starters work shorter outings.
  - This gives the ensemble an explicit view of bullpen quality and recent bullpen workload instead of treating every post-starter inning as noise.

#### 14. Added MLB run-environment modeling with park context

- What changed:
  - Added `MLB_PARK_FACTORS` to `pipeline/config.py`.
  - Added `RunEnvironmentModel` and `run_environment_predict()` in `pipeline/models.py`.
  - Enabled the model in MLB config and wired it into `pipeline/run.py`.
- Why it matters:
  - Baseball outcomes are more sensitive to scoring environment than the other sports in the stack.
  - This adds a baseball-specific offense/prevention layer and makes the home venue part of the forecast instead of pure decoration.

#### 15. Added starter rest and workload context to the MLB pitcher model

- What changed:
  - Extended `PitcherMatchupModel` in `pipeline/models.py` to track recent innings workload and days since last start.
  - Updated `pipeline/run.py` to pass fixture dates into starter predictions so rest is evaluated in the live path.
- Why it matters:
  - A pitcher on normal rest and a pitcher on a short or irregular turnaround should not be treated identically.
  - This improves the existing starter model without requiring another data source.

#### 16. Added live MLB weather context

- What changed:
  - Added `MLB_BALLPARKS` and `OPEN_METEO_BASE` in `pipeline/config.py`.
  - Added weather fetching/caching helpers in `pipeline/fetch_mlb.py`.
  - Added a modest live weather adjustment in `pipeline/run.py` that uses the run-environment model as directionality rather than pretending weather alone predicts a winner.
- Why it matters:
  - Outdoor baseball behaves differently on warm, windy nights than it does in cold, suppressive conditions.
  - This gives the live MLB path real environment awareness without inventing a fake historical-weather training set.

#### 17. Added handedness-aware MLB matchup modeling

- What changed:
  - Added ESPN core-player handedness fetch/caching in `pipeline/fetch_mlb.py`.
  - Added `HandednessMatchupModel` and `handedness_matchup_predict()` in `pipeline/models.py`.
  - Enabled the model in MLB config and wired it into `pipeline/run.py`.
- Why it matters:
  - Team offenses do not perform identically into lefties and righties.
  - This adds a real baseball-specific split layer using data already accessible from the current free stack.

#### 18. Added MLB over/under market support

- What changed:
  - Extended `pipeline/fetch_data.py` to fetch and normalize totals markets with a consensus totals line.
  - Added `MlbTotalsModel` and `mlb_totals_predict()` in `pipeline/models.py`.
  - Added `compute_totals_edges()` in `pipeline/ensemble.py`.
  - Wired a parallel `totals_matches` / `totals_locks` lane into `pipeline/run.py`.
  - Surfaced totals locks in `index.html` and `pipeline/notify_discord.py`.
- Why it matters:
  - MLB now supports a second real betting market besides moneyline sides.
  - The totals lane reuses the current run-environment, pitcher, bullpen, park, and weather work instead of requiring a separate product.

#### 19. Added current-roster MLB lineup strength and platoon adjustments

- What changed:
  - Added ESPN roster fetching/caching plus active-hitter handedness summaries in `pipeline/fetch_mlb.py`.
  - Added config-driven lineup adjustment caps in `pipeline/config.py`.
  - Added roster-based side and totals adjustments in `pipeline/run.py` that use active hitter availability, injury count, and handedness mix versus the opposing probable starter.
- Why it matters:
  - The previous MLB live path knew the starter and the weather, but it still treated the batting side too generically at lock time.
  - This adds a real pregame roster-health and platoon-composition signal without pretending ESPN gives a perfectly confirmed batting order.

#### 20. Added an NBA matchup-context model

- What changed:
  - Added `NbaMatchupModel` and `nba_matchup_predict()` in `pipeline/models.py`.
  - Enabled it in `pipeline/config.py` and wired it into the NBA ensemble in `pipeline/run.py`.
- Why it matters:
  - The prior NBA stack had season and recent-form views, but it still lacked a dedicated layer for home/road split strength, rest differential, and pace/style interaction.
  - This gives NBA a more game-specific matchup read instead of relying only on generic team-strength snapshots.

#### 21. Added live NBA availability adjustment

- What changed:
  - Added ESPN roster-based availability fetching/caching in `pipeline/fetch_nba.py`.
  - Added a live availability adjustment in `pipeline/run.py` using injury burden plus current team-leader absence signals.
  - Added NBA config control for the adjustment cap in `pipeline/config.py`.
- Why it matters:
  - A team missing its current high-usage scorer or playmaker should not be treated like a full-strength roster.
  - This improves the live NBA path without requiring a fragile player-projection or beat-reporter feed.

#### 22. Fixed MMA slate windowing so recent UFC cards survive the daily run

- What changed:
  - Expanded the MMA schedule fetch window in `pipeline/fetch_mma.py` to include the previous ET day, not just “today forward.”
- Why it matters:
  - UFC cards were falling out of the live slate too early, producing `fixtures_fetched > 0` but `fixtures_in_window = 0`.
  - This is an operational fix that allows real cards to reach the model instead of disappearing before pricing.

#### 23. Added NBA over/under market support

- What changed:
  - Added `NbaTotalsModel` and `nba_totals_predict()` in `pipeline/models.py`.
  - Enabled NBA totals fetches in `pipeline/run.py` by requesting totals markets from The Odds API.
  - Wired NBA into the existing `totals_matches` / `totals_locks` pipeline lane in `pipeline/run.py`.
- Why it matters:
  - NBA now supports a second real betting market besides moneyline sides.
  - The totals lane reuses the live box-score and pace data the project already fetches instead of depending on a new source.

#### 24. Deepened the NBA availability signal with weighted leader absences

- What changed:
  - Extended `pipeline/fetch_nba.py` so roster availability is cached as reusable player-status rows, then scored with weighted leader categories from the ESPN scoreboard.
  - Updated the live availability adjustment in `pipeline/run.py` to penalize missing scoring and playmaking leaders more heavily than minor-category leaders.
- Why it matters:
  - Not every absence has the same impact.
  - This makes the live NBA pregame adjustment more sensitive to who is missing, not just how many players are listed.

#### 25. Deepened the NHL matchup model with summary-level possession and special-teams context

- What changed:
  - Extended `pipeline/fetch_nhl.py` so final historical NHL games are enriched from ESPN `summary` data with shots, faceoff percentage, power-play rate, takeaways, giveaways, and penalty minutes.
  - Expanded `NhlMatchupModel` in `pipeline/models.py` to use those signals alongside goal differential, shooting, save percentage, and rest.
  - Added coverage in `tests/test_fetch_nhl.py`, `tests/test_models.py`, and `tests/test_run.py`.
- Why it matters:
  - The first NHL pass mostly saw goals, shots, and save percentage, which is too thin for a real hockey model.
  - This gives NHL a more credible read on puck-control and special-teams strength without changing the public output contract.

#### 26. Added goalie-aware NHL modeling with probable starters

- What changed:
  - Extended `pipeline/fetch_nhl.py` to parse actual lead-goalie stats from historical ESPN summaries and probable starting goalies from the live ESPN scoreboard.
  - Expanded `NhlMatchupModel` in `pipeline/models.py` to use recent probable-goalie save percentage, goals allowed, and goalie rest as explicit features.
  - Wired probable goalie names through `pipeline/run.py` and added coverage in `tests/test_fetch_nhl.py`, `tests/test_models.py`, and `tests/test_run.py`.
- Why it matters:
  - Hockey moneylines are unusually sensitive to the crease.
  - This gives the NHL model a direct goaltending signal instead of forcing it to infer everything from team-level save percentage.

#### 27. Added a generated reporting dashboard for the site

- What changed:
  - Added `build_dashboard_data()` and recent-window helpers to `pipeline/backtest.py`.
  - The full pipeline now writes `data/dashboard.json` in `pipeline/run.py`, and the refresh path updates it in `pipeline/refresh_picks.py`.
  - The site now has a dedicated `DASHBOARD` view in `index.html` with aggregate record, recent windows, sport splits, leaders, and threshold guidance.
- Why it matters:
  - The tracking stack was already collecting useful information, but it was buried in JSON and CSV files.
  - This turns the reporting layer into an operational surface you can actually use to diagnose quality, volume, and live slate coverage.

#### 28. Made the dashboard operational with ranked recommended actions

- What changed:
  - Added `_build_recommended_actions()` in `pipeline/backtest.py`.
  - The dashboard payload now includes ranked operating actions derived from live odds coverage gaps, lane performance, recent-vs-baseline ROI drift, and CLV tension.
  - The site renders those recommendations in `index.html`, and Discord posts them as a `CONTROL PANEL` embed in `pipeline/notify_discord.py`.
- Why it matters:
  - The dashboard now tells you what to do, not just what happened.
  - This reduces the number of “why is today’s card thin?” debugging passes needed during live operation.

#### 29. Upgraded the late-day refresh into a live-input refresh path

- What changed:
  - `pipeline/refresh_picks.py` now refetches live schedule metadata for NBA, MLB, and NHL instead of refreshing odds only.
  - The refresh path reapplies the same live adjustments used in the full run:
    - NBA availability side and totals adjustments
    - MLB weather and lineup side and totals adjustments
    - NHL goalie-status adjustment
  - It now preserves `base_model_probs` / `base_expected_total`, rebuilds diagnostics, updates `dashboard.json`, and refreshes saved market snapshots.
- Why it matters:
  - Late-day runs now react to more than line movement.
  - This closes a real gap between the morning full pipeline and the afternoon refresh/deploy path.

#### 30. Added a live NHL goalie-confirmation adjustment and deeper special-teams context

- What changed:
  - Added `_apply_nhl_goalie_status_adjustment()` in `pipeline/run.py` and applied it in both the full pipeline and `pipeline/refresh_picks.py`.
  - Expanded `NhlMatchupModel` in `pipeline/models.py` with recent shot suppression, blocked-shot differential, and penalty-kill differential on top of the earlier goalie-aware feature set.
- Why it matters:
  - NHL sides are sensitive not just to who the probable goalie is, but also to how certain that goalie information is and how the teams are defending recently.
  - This gives the live NHL lane more hockey-specific context without changing the public data contract.

#### 31. Improved NBA late-news handling near tipoff

- What changed:
  - Extended `pipeline/fetch_nba.py` so live availability profiles now distinguish confirmed injuries from late uncertain statuses such as questionable, day-to-day, and doubtful.
  - Added near-tipoff urgency scaling in `pipeline/run.py` so questionable star absences matter more as game time approaches.
  - Wired the same logic into `pipeline/refresh_picks.py` so the afternoon refresh reacts more sharply to unresolved live news.
- Why it matters:
  - NBA edges can swing late in the day on a single star-status change.
  - This makes the refresh path more sensitive to exactly the kind of pregame uncertainty that was previously too blunt.

#### 32. Made MLB lineup quality aware of missing top bats

- What changed:
  - Extended `pipeline/fetch_mlb.py` so live lineup profiles can weight missing hitters using ESPN scoreboard leader categories such as OPS, home runs, and RBI.
  - Expanded the lineup index in `pipeline/run.py` so missing middle-of-the-order production drags side and totals adjustments more than generic bench injuries.
- Why it matters:
  - A lineup missing its best hitters is materially different from a lineup missing generic depth pieces.
  - This moves MLB closer to real same-day batting quality without requiring a separate paid lineup feed.

#### 33. Tightened NHL live odds coverage and diagnostics

- What changed:
  - Extended `pipeline/fetch_nhl.py` normalization with alias and accent handling for common live-market naming variants.
  - Expanded pipeline diagnostics in `pipeline/run.py` to carry `coverage_gap_examples`, then surfaced those gaps on the site and in Discord through `index.html` and `pipeline/notify_discord.py`.
- Why it matters:
  - When NHL is thin, it is now easier to tell whether the model disliked the slate or whether the live odds feed simply failed to match a few fixtures.
  - Better name normalization also improves the chance that live NHL markets get picked up in the first place.

#### 34. Added a true date-ordered replay report over settled predictions and picks

- What changed:
  - Added `build_walkforward_report()` plus daily/cumulative replay helpers and calibration bins in `pipeline/backtest.py`.
  - The dashboard payload now includes a `walkforward` section with aggregate replay summaries, calibration buckets, and recent daily replay rows.
  - Added focused coverage in `tests/test_backtest.py`.
- Why it matters:
  - The reporting layer is no longer just a static aggregate over saved histories.
  - You can now inspect how the stack has behaved day by day, how probabilities have calibrated, and whether pick quality is drifting over time.

#### 35. Added confirmed MLB batting-order context and live bullpen-fatigue adjustments

- What changed:
  - `pipeline/fetch_mlb.py` now fetches same-day ESPN summary payloads during schedule generation and extracts confirmed batting-order rosters with handedness mix and top-order depth.
  - `pipeline/run.py` now uses those confirmed-lineup fields inside the MLB lineup adjustment, and it computes a same-day bullpen fatigue tax from recent bullpen innings before applying extra side and totals adjustments.
  - `pipeline/refresh_picks.py` now reapplies those bullpen-fatigue adjustments during live refreshes, and the new behavior is covered in `tests/test_fetch_mlb.py` and `tests/test_run.py`.
- Why it matters:
  - MLB sides and totals now react to actual same-day lineups instead of only generic roster health.
  - Bullpen quality is no longer treated as purely static season form; recent bullpen strain now nudges the projection too.

#### 36. Added event-specific NBA and NHL injury refresh signals

- What changed:
  - `pipeline/fetch_nba.py` now pulls each event summary during schedule refresh and merges event-specific injury burdens into team availability profiles.
  - `pipeline/fetch_nhl.py` now pulls event summaries to build same-day skater injury profiles alongside probable-goalie status.
  - `pipeline/run.py` and `pipeline/refresh_picks.py` now apply those event-specific NBA/NHL injury adjustments during both the full run and the late refresh path.
  - Added coverage in `tests/test_fetch_nba.py`, `tests/test_fetch_nhl.py`, and `tests/test_run.py`.
- Why it matters:
  - Late-news handling is now based on game-specific injury state, not just generic roster listings.
  - That gives the model a better chance to react to day-of absences and uncertainties before lock.

#### 37. Added a raw-data walk-forward replay harness

- What changed:
  - Added `build_raw_walkforward_report()` and `run_raw_walkforward_for_sport()` in `pipeline/backtest.py`.
  - The new replay path fits sport-specific models on prior historical games only, then predicts each later day in order using rolling model-accuracy weights.
  - Added CLI support via `python -m pipeline.backtest --raw-walkforward ...` plus focused coverage in `tests/test_backtest.py`.
- Why it matters:
  - The project now has an actual historical rebuild path over raw match inputs instead of only summaries over saved outputs and settled pick logs.
  - That makes threshold and model tuning much less guess-driven, even though it still does not replay every live odds and injury snapshot perfectly.

## Future Improvements

### 1. Deepen the raw replay toward full live-state parity

`pipeline/backtest.py` now has a raw-data walk-forward harness, but it still does not replay every live market and late-news input exactly as they existed historically. The next step is a deeper replay script that:

- reconstruct the feature state as of each historical date
- reconstruct the historical odds state per market, not just the game results state
- run the live model logic against that state
- compare generated picks to actual outcomes
- emit ROI, Brier, log loss, and calibration curves by sport

Primary touchpoints: `pipeline/backtest.py`, `pipeline/run.py`, all `pipeline/fetch_*.py` modules.

### 2. Add truly confirmed batting-order quality and bullpen-role context to MLB

The MLB stack now covers starters, bullpens, park context, handedness, live weather, confirmed batting-order shape, and lineup-aware totals, but it still lacks:

- bullpen leverage role quality beyond simple aggregate workload
- totals-specific bullpen availability and umpire context

### 3. Add deeper late-day confirmed news closer to lock

The refresh path now refetches core live metadata in `pipeline/refresh_picks.py`, but it still does not fully rebuild each sport from scratch. The next step is to push closer to confirmed pregame information, especially:

- NBA late scratches and minute-allocation expectations closer to tip
- NHL confirmed skater availability and special-teams usage closer to puck drop

### 4. Fix training leakage in efficiency and four-factors models

`AdjustedEfficiency` and `FourFactorsModel` both train on season-to-date weighted aggregates built from the full available box-score set in `pipeline/models.py:510-605` and `pipeline/models.py:674-781`. That means earlier training examples can be informed by later games in the same season. A better version would freeze features as of each game date during training.

### 5. Add richer MMA features

MMA remains Elo-only. The next meaningful lift would require fighter-level stats such as age, reach, striking efficiency, takedown success, takedown defense, finish rates, and layoff length.

### 6. Improve odds-market consistency

`pipeline/fetch_data.py:38-68` still takes the best price independently per outcome across books. That is useful for shopping, but it can create a synthetic market when home and away prices come from different bookmakers. A future version should optionally store book-consistent markets alongside best-price markets.

## Discovered Issues

### 1. Cold local full runs are still slower than the live refresh path

The late-day refresh and the pushed GitHub workflows are now the practical operating path. A cold local full run can still spend a long time rebuilding ESPN caches before it reaches useful output, especially for sports with heavier historical fetches. That is not a correctness bug, but it does make local ad hoc verification slower than it should be.

### 2. Single-model sports still get structurally high agreement scores

`compute_confidence_score()` in `pipeline/ensemble.py:139-183` uses cross-model dispersion as part of confidence. MMA still has only two relatively similar models, so its agreement component can look better calibrated than it really is. That should eventually be capped or reformulated for sparse-model sports.

### 3. MMA remains comparatively under-modeled

The MLB stack is substantially better than before, but MMA is still constrained by sparse features. The infrastructure is better; the sport-specific signal depth is still thin where fighter-level data is absent.
