# SLOP LOCKS

Static PWA for model-driven sports betting picks, dashboard reporting, and pick-history tracking.

**Live:** [sloplocks.lol](https://sloplocks.lol)

SLOP LOCKS has no backend app server. A Python pipeline writes committed JSON/CSV data under `data/`, and the single-file frontend (`index.html`) reads those static files directly.

## Supported Sports

| Sport | Status | Active moneyline models |
| --- | --- | --- |
| NBA | Active | Elo, Results Features, Recent Boxscore, NBA Matchup |
| WNBA | Active | Elo, Results Features, Recent Boxscore |
| NHL | Active | Elo, Results Features, NHL Matchup |
| MLB | Active | Elo, Results Features, Bullpen Features, Run Environment, Handedness |
| NCAAM | Season-disabled | Historical code/data retained; live picks disabled |

Current config disables NBA `four_factors` and MLB `pitcher_features` at runtime. MLB totals are enabled; NBA totals are modeled but publication depends on config and lane-health gates.

## How It Works

1. GitHub Actions runs the pipeline on a schedule or by manual dispatch.
2. `pipeline.run` fetches ESPN schedules/results/context, fetches odds from The Odds API, fits sport-specific models, blends probabilities, computes edge/EV/Kelly values, validates publishable picks, grades settled picks, and writes JSON/CSV output.
3. `index.html` reads static data from `data/manifest.json`, `data/dashboard.json`, and each sport's `predictions.json` / `pick_history.json`.
4. Netlify deploys the static site from `master`.

The pipeline also writes immutable run snapshots and append-oriented ledgers so reporting can replay what was actually available at decision time.

## Data Products

Shared:

- `data/manifest.json` - frontend sport status manifest
- `data/dashboard.json` - BOARD tab payload with aggregate record, replay, lane health, and recommendations
- `data/tracking/results_log.csv` - mutable live settled-results log
- `data/tracking/results_audit_log.csv` - append-only settled-results audit ledger
- `data/tracking/odds_history.csv` - market snapshots for CLV
- `data/tracking/pick_decisions.csv` - decision-time pick ledger
- `data/tracking/snapshots/YYYY-MM-DD/{sport}/*.json` - immutable run snapshots

Per sport:

- `data/{sport}/predictions.json` - current slate, picks, diagnostics, guards, model weights
- `data/{sport}/history.json` - saved modeled match history
- `data/{sport}/pick_history.json` - published picks plus settled outcomes and CLV fields
- `data/{sport}/model_accuracy.json` - rolling model scoring history
- `data/{sport}/espn_cache.json` - ESPN cache, expected to change during live runs

There is also a display-only NFL Draft special path (`pipeline/build_draft_tab.py`, `data/nfl-draft*.json`). It is separate from the live sport pipeline and excluded from site totals.

## Pick Controls

Publication is gated by settled evidence, not just current model output. A sport or lane can be suppressed when sample size, recent ROI, CLV, or calibration health is too weak.

Important current rules:

- Moneyline SLOP LOCKS support explicit selection lanes:
  - `core` - standard edge/probability/EV gate
  - `value_dog` - positive-EV underdogs with lower win-probability floor
  - `near_favorite` - modestly priced favorites and short dogs with smaller edge floor
- A sport on a live-publication hold can publish a capped pick count when recent health clears the configured hold threshold.
- Current-slate diagnostics include gate-failure counts and lane-candidate counts so low-volume days can be explained from `predictions.json`.
- `pick_decisions.csv` is the authoritative decision-time ledger.
- Backtests and dashboard reporting prefer immutable/audit ledgers where available.
- CLV is separated by market family. Moneyline CLV and totals-line CLV are not merged into a single unit.
- Low-confidence fallback picks should not be forced into official surfaces.
- MMA is intentionally removed from the live product and pipeline.

Free/keyless data currently used includes ESPN schedule/team/context endpoints, Open-Meteo for MLB weather, and MLB Stats API as a probable-pitcher fallback when ESPN leaves starters as `TBD`.

## Setup

```bash
git clone https://github.com/jasonpersinger/sloplocks.git
cd sloplocks
python -m venv venv
source venv/bin/activate
pip install -r pipeline/requirements.txt
```

Copy `.env.template` to `.env` and fill in the keys you need.

## Commands

Run the full active-sport pipeline:

```bash
python -m pipeline.run
```

Run one sport:

```bash
python -m pipeline.run --sport nba
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

Run tests:

```bash
pytest tests/ -v
```

Fast verification after pipeline/report changes:

```bash
python -m compileall pipeline tests
pytest -q tests/test_run.py tests/test_backtest.py tests/test_reset_public_record.py tests/test_notify_discord.py
```

## Environment Variables

Loaded from `.env` when present.

| Key | Used for | Required |
| --- | --- | --- |
| `ODDS_API_KEY` | The Odds API odds ingestion | Yes for live odds |
| `BALLDONTLIE_API_KEY` | Optional deeper NBA data path | Optional |
| `OPENAI_API_KEY` | Game-level qualitative context (`gpt-4o-mini`) | Optional |
| `GEMINI_API_KEY` | Draft-special qualitative commentary | Optional |
| `ENABLE_QUALITATIVE` | Enables qualitative adjustment paths | Optional |
| `DISCORD_WEBHOOK_URL` | Discord notifications from workflows | Optional |
| `NETLIFY_AUTH_TOKEN` / `NETLIFY_SITE_ID` | Netlify deployment workflow | Required for deploy workflow |
| `ANTHROPIC_API_KEY` | Legacy/configured key | Usually unused |

## Project Structure

```text
sloplocks/
|-- index.html                  # Static frontend, embedded CSS/JS
|-- manifest.json               # PWA manifest
|-- sw.js                       # Service worker
|-- netlify.toml                # Static Netlify config
|-- data/                       # Committed generated output and tracking ledgers
|-- pipeline/
|   |-- config.py               # Sport registry, thresholds, paths, keys
|   |-- run.py                  # Main pipeline orchestrator
|   |-- refresh_picks.py        # Fast odds refresh without retraining
|   |-- fetch_data.py           # The Odds API client
|   |-- fetch_nba.py            # ESPN NBA client
|   |-- fetch_wnba.py           # ESPN WNBA client
|   |-- fetch_nhl.py            # ESPN NHL client
|   |-- fetch_mlb.py            # ESPN MLB client plus MLB Stats API pitcher fallback
|   |-- fetch_ncaam.py          # Retained historical/off-season NCAAM client
|   |-- models.py               # Model implementations
|   |-- ensemble.py             # Blending, calibration, edge math, confidence
|   |-- backtest.py             # Reports, replay, dashboard payload
|   |-- reset_public_record.py  # Archive-first public-record maintenance
|   |-- notify_discord.py       # Discord webhook formatting/sending
|   `-- qualitative_analysis.py # Optional game-level OpenAI context layer
|-- tests/                      # Pytest suite
`-- .github/workflows/
    |-- daily.yml               # Full pipeline, commit data, Discord notify
    |-- refresh-picks.yml       # Manual fast refresh, commit data, Discord notify
    `-- deploy-site.yml         # Netlify deploy on master
```

## Frontend

The frontend is intentionally old-terminal / CRT styled: green-on-black, monospace, scanlines, and dense cards. It has no build step. The sport tabs and BOARD tab are rendered by client-side JavaScript in `index.html`.

When changing output JSON shape, update both the pipeline writer and the frontend reader. Search `index.html` for a field before renaming or removing it.

## Deployment

Push to `master` triggers the Netlify deploy workflow. The daily workflow runs at 12:00 UTC, commits updated `data/`, and posts Discord output when `DISCORD_WEBHOOK_URL` is configured. The refresh workflow is manual and intended for quick odds updates.

## License

MIT
