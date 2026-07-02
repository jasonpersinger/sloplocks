# SLOP LOCKS

Static PWA for model-driven sports betting picks, pick history, and pipeline health reporting.

Live site: [jasonpersinger.me/sloplocks](https://jasonpersinger.me/sloplocks/) (GitHub Pages)

SLOP LOCKS is intentionally simple at runtime: there is no backend app server and no client-side build step. A Python pipeline writes committed JSON/CSV files into `data/`, and the single-file frontend reads those static files directly.

## Current Scope

Active sports:

| Sport | Status | Main moneyline models | Notes |
| --- | --- | --- | --- |
| NBA | Active | Elo, Results Features, Recent Boxscore, NBA Matchup | Totals are modeled, but publication can be health-gated. |
| WNBA | Active | Elo, Results Features, Recent Boxscore | Newer surface with smaller settled sample. |
| NHL | Active | Elo, Results Features, NHL Matchup | Moneyline only. |
| MLB | Active | Elo, Results Features, Bullpen Features, Run Environment, Handedness | Totals enabled; MLB Stats API probable-pitcher fallback is used when ESPN has `TBD`. |
| NCAAM | Season-disabled | Historical code retained | Skipped by live pipeline while disabled. |

Removed from the live product:

- MMA
- EPL/MLS era references from older docs and branches

The canonical branch is `master`. Deploys and data automation are wired to `master`.

## System Overview

The daily pipeline does this:

1. Fetches schedules, results, and sport context.
2. Fetches moneyline and supported totals markets from The Odds API.
3. Builds sport-specific historical features.
4. Fits and scores model families.
5. Blends probabilities, applies calibration and market-respect logic, and computes edge/EV/Kelly.
6. Selects publishable picks with lane-specific gates.
7. Applies publication guards based on settled evidence, ROI, CLV, and sample size.
8. Grades settled picks and updates ledgers.
9. Writes static data for the frontend.
10. Commits generated `data/` output from GitHub Actions.

The frontend reads only static files:

- `data/manifest.json`
- `data/dashboard.json`
- `data/{sport}/predictions.json`
- `data/{sport}/pick_history.json`

Because there is no API layer, data-shape changes must update both the writer and the frontend reader.

## Repository Layout

```text
sloplocks/
|-- index.html                  # Full static frontend, CSS, and client-side JS
|-- manifest.json               # Root PWA manifest
|-- sw.js                       # Service worker
|-- data/                       # Committed generated data and tracking ledgers
|-- pipeline/
|   |-- config.py               # Sport registry, thresholds, API bases, data paths
|   |-- run.py                  # Main daily pipeline orchestrator
|   |-- refresh_picks.py        # Fast odds refresh without full retraining
|   |-- fetch_data.py           # The Odds API client
|   |-- fetch_nba.py            # ESPN NBA data fetcher
|   |-- fetch_wnba.py           # ESPN WNBA data fetcher
|   |-- fetch_nhl.py            # ESPN NHL data fetcher
|   |-- fetch_mlb.py            # ESPN MLB fetcher plus MLB Stats API fallback
|   |-- fetch_ncaam.py          # Retained NCAAM fetcher for historical/off-season use
|   |-- models.py               # Model implementations
|   |-- ensemble.py             # Calibration, probability blending, edge, Kelly, tiers
|   |-- backtest.py             # Reporting, replay, dashboard payloads, lane health
|   |-- reset_public_record.py  # Archive-first public-record maintenance
|   |-- notify_discord.py       # Discord message formatting and sending
|   |-- qualitative_analysis.py # Optional OpenAI game-context layer
|   `-- build_draft_tab.py      # Display-only NFL Draft special payload builder
|-- tests/                      # Pytest suite
`-- .github/workflows/
    |-- daily.yml               # Full pipeline, commit data, Discord notify
    |-- refresh-picks.yml       # Manual odds refresh, commit data, Discord notify
```

## Data Products

Shared generated files:

| Path | Purpose |
| --- | --- |
| `data/manifest.json` | Frontend sport status, current run status, and diagnostics summary. |
| `data/dashboard.json` | BOARD tab payload with aggregate record, replay results, lane health, and recommendations. |
| `data/tracking/results_log.csv` | Mutable live settled-results log. |
| `data/tracking/results_audit_log.csv` | Append-only settled-results audit ledger. |
| `data/tracking/odds_history.csv` | Market snapshots used for CLV tracking. |
| `data/tracking/pick_decisions.csv` | Append-oriented decision-time pick ledger. |
| `data/tracking/snapshots/YYYY-MM-DD/{sport}/*.json` | Immutable run snapshots for replay and audit. |

Per-sport files:

| Path | Purpose |
| --- | --- |
| `data/{sport}/predictions.json` | Current slate, modeled matches, live picks, diagnostics, publication guard, model weights. |
| `data/{sport}/history.json` | Saved modeled match history. |
| `data/{sport}/pick_history.json` | Published picks plus settled outcomes and CLV fields. |
| `data/{sport}/model_accuracy.json` | Rolling model scoring history. |
| `data/{sport}/espn_cache.json` | Sport fetch cache. This can change during live runs. |

Tracking files are not disposable cache. Treat them as product data unless a task explicitly calls for a migration or reset.

## Data Sources

Keyed:

- The Odds API for moneyline and supported totals odds.
- Optional balldontlie data paths for NBA-related enrichment.
- Optional OpenAI qualitative context when enabled.
- Discord webhook for notifications.

Free/keyless:

- ESPN site/core APIs for schedules, results, rosters, team context, and sport-specific metadata.
- MLB Stats API for probable-pitcher fallback when ESPN lists a starter as `TBD`. MLB pitcher assignments carry `*_pitcher_source` and `*_pitcher_last_checked`; Stats API probable-pitcher cache entries are TTL-validated and refreshed near first pitch.
- Open-Meteo for MLB weather context.

## Pick Selection

SLOP LOCKS are moneyline picks selected from modeled match edges. The current selector uses explicit lanes:

| Lane | Purpose |
| --- | --- |
| `core` | Standard edge, probability, and EV gate. |
| `value_dog` | Positive-EV underdogs that intentionally sit below the core win-probability floor. |
| `near_favorite` | Modestly priced favorites and short dogs with a smaller edge requirement. |

Each sport config can define:

- edge floor
- model probability floor
- confidence floor
- minimum expected value
- American odds band
- max picks per lane
- global max picks

The selected lane is preserved on published picks as `selection_lane`. SLOP LOCK publication also rejects candidates below the configured confidence floor or with a computed `NO_PLAY` tier. Older historical picks without `selection_lane` should be treated as `core` in reports.

Totals picks are selected separately where enabled. MLB totals are live; NBA totals are modeled but can be suppressed by config and lane-health gates.

## Publication Guards

Selection and publication are deliberately separate.

The model can find a candidate, but the publication guard can still suppress it when settled evidence is too thin or unhealthy. Guard inputs include:

- evaluated pick count
- recent ROI
- CLV
- calibration health
- market family
- sport-specific thresholds

Sports on a live hold can publish a small capped number of picks when configured health thresholds are met. This keeps the system from being all-or-nothing while still avoiding uncontrolled volume expansion.

Important integrity rules:

- Do not display or publish stale picks after the scheduled start time.
- Do not publish `NO_PLAY` candidates as SLOP LOCKS.
- Do not force low-confidence fallback picks into official output.
- Do not combine moneyline CLV and totals CLV into one naive unit.
- Prefer immutable/audit ledgers for reporting when available.
- Preserve append-only behavior for decision and audit ledgers.

## Diagnostics

`predictions.json` includes current-slate diagnostics that explain low-pick days. Useful fields include:

- `fixtures_in_window`
- `fixtures_with_odds`
- `matches_modeled`
- `matches_with_positive_ev`
- `lock_eligible_matches`
- `lock_eligible_outcomes`
- `slop_locks_posted`
- `gate_failures`
- `candidate_lanes`
- `publication_guard`
- MLB `pitcher_warnings` on match/pick records when starters remain `TBD` inside the configured pregame warning window.

Typical low-volume causes:

- small playoff slate
- no odds coverage
- candidates have edge but fail probability floor
- candidates qualify only in research lanes
- live publication guard suppresses the sport due to sample size, CLV, or recent health
- WNBA has insufficient settled evidence early in the season

## Frontend

The frontend is a static, single-file app in `index.html`.

Design direction:

- old-terminal / CRT styling
- green-on-black shell
- monospace typography
- scanline overlay
- dense pick and dashboard cards

Keep this visual direction unless a task explicitly asks for a redesign.

The site has no build step. Open `index.html` directly for basic static review, or serve the repo root with a simple local server if browser caching or service-worker behavior matters.

When changing pipeline output:

1. Search `index.html` for the field name.
2. Update the data writer.
3. Update the frontend reader.
4. Add or update tests around the output contract.

## Setup

```bash
git clone https://github.com/jasonpersinger/sloplocks.git
cd sloplocks
python -m venv venv
source venv/bin/activate
pip install -r pipeline/requirements.txt
```

Copy `.env.template` to `.env` and fill in the keys you need.

## Environment Variables

Loaded from `.env` when present.

| Key | Used for | Required |
| --- | --- | --- |
| `ODDS_API_KEY` | The Odds API odds ingestion. | Required for live odds. |
| `BALLDONTLIE_API_KEY` | Optional deeper NBA data path. | Optional. |
| `OPENAI_API_KEY` | Game-level qualitative context via `gpt-4o-mini`. | Optional. |
| `ENABLE_QUALITATIVE` | Enables qualitative adjustment paths. | Optional. |
| `DISCORD_WEBHOOK_URL` | Discord notifications from workflows. | Optional. |
| `GEMINI_API_KEY` | Legacy/draft-special workflow compatibility. | Usually unused. |
| `ANTHROPIC_API_KEY` | Legacy/configured key. | Usually unused. |

## Commands

Run the full active-sport pipeline:

```bash
python -m pipeline.run
```

Run one sport:

```bash
python -m pipeline.run --sport nba
python -m pipeline.run --sport mlb
```

Write output somewhere other than `data/`:

```bash
python -m pipeline.run --sport mlb --output-dir /tmp/sloplocks-data/mlb
```

Refresh odds and recompute picks without full retraining:

```bash
python -m pipeline.refresh_picks
python -m pipeline.refresh_picks nba nhl
```

Run reporting:

```bash
python -m pipeline.backtest
python -m pipeline.backtest --walkforward
python -m pipeline.backtest --raw-walkforward
python -m pipeline.backtest --snapshot-replay
python -m pipeline.backtest --decision-replay
```

Limit reporting to specific sports:

```bash
python -m pipeline.backtest mlb
python -m pipeline.backtest nba nhl --walkforward
```

Run tests:

```bash
pytest tests/ -v
```

Fast verification after pipeline/report changes:

```bash
python -m compileall pipeline tests
pytest -q tests/test_run.py tests/test_backtest.py tests/test_refresh_picks.py tests/test_fetch_mlb.py tests/test_notify_discord.py
```

Full verification:

```bash
python -m compileall pipeline tests
pytest -q
```

## Automation

GitHub Actions:

| Workflow | Trigger | Behavior |
| --- | --- | --- |
| `.github/workflows/daily.yml` | Daily at 12:00 UTC and manual dispatch | Runs full pipeline, commits updated `data/`, posts Discord notification if configured. |
| `.github/workflows/refresh-picks.yml` | Manual dispatch | Runs fast odds refresh, commits updated `data/`, posts Discord notification if configured. |

Daily and refresh workflows commit as `sloplocks-bot`.

## Deployment

Production deploys from `master`.

Normal code path:

1. Branch from `master`.
2. Make code/docs changes.
3. Run relevant verification.
4. Open a PR into `master`.
5. Merge to `master`.
6. GitHub Pages publishes the static site automatically on push to `master`.

Generated `data/` changes are expected from automation. Do not mix unrelated generated-data churn into code commits unless the task specifically requires it.

## Development Guidance

Before model or pick-selection changes, inspect:

- `pipeline/config.py`
- `pipeline/run.py`
- `pipeline/ensemble.py`
- `pipeline/backtest.py`
- `pipeline/models.py`

Before frontend output changes, inspect:

- the writer in `pipeline/run.py` or `pipeline/backtest.py`
- the reader in `index.html`
- representative payloads under `data/{sport}/`

Before tracking changes, inspect:

- `data/tracking/results_log.csv`
- `data/tracking/results_audit_log.csv`
- `data/tracking/odds_history.csv`
- `data/tracking/pick_decisions.csv`
- `data/tracking/snapshots/`

Prefer additive migrations and backfills over destructive rewrites.

## Troubleshooting

Low or zero picks:

1. Check `data/{sport}/predictions.json`.
2. Read `diagnostics.fixtures_with_odds` vs `fixtures_in_window`.
3. Read `diagnostics.gate_failures`.
4. Read `diagnostics.candidate_lanes`.
5. Read `publication_guard.status` and `publication_guard.reason`.
6. Confirm whether the sport has enough settled evidence.

Unexpected frontend display:

1. Confirm `data/manifest.json` and `data/dashboard.json` were regenerated.
2. Confirm the sport's `predictions.json` has the field the frontend expects.
3. Search `index.html` for that field.
4. Clear service-worker/browser cache if local behavior differs from committed data.

Unexpected large diff:

- `data/` is committed and normal pipeline runs can legitimately create large generated diffs.
- `data/*/espn_cache.json` can change during live runs.
- Snapshot folders are immutable run artifacts and should not be casually rewritten.

## License

MIT
