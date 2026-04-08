# SLOP LOCKS: LLM Working Guide

This file is the machine-facing overview for this repository.

Read this first if you are asked to work in this folder.

`README.md` and `CLAUDE.md` are partly stale. They describe an older EPL-focused version of the project. Use this file plus the live code as the current source of truth.

## Purpose

SLOP LOCKS is a static PWA that publishes model-driven sports betting picks.

There is no backend app server. The Python pipeline writes JSON into `data/`, and the single-file frontend reads those generated files directly.

Current active sports:
- `nba`
- `nhl`
- `ncaam`
- `mlb`

MMA was intentionally removed from the live product and pipeline.

## Stack

- Pipeline: Python 3.11
- Core libs: `pandas`, `numpy`, `scipy`, `requests`, `scikit-learn`, `beautifulsoup4`
- Frontend: static `index.html` with embedded CSS and JS
- PWA shell: root `manifest.json` and `sw.js`
- Hosting: Netlify, deployed from `master`
- Automation: GitHub Actions

## High-Level Architecture

1. `pipeline.run` fetches sport data, fits models, blends probabilities, computes edges, selects picks, grades settled results, and writes output JSON.
2. The frontend reads:
   - `data/manifest.json`
   - `data/dashboard.json`
   - `data/{sport}/predictions.json`
   - `data/{sport}/pick_history.json`
3. The site is deployed as static files only.

Important consequence:
- Most changes are either pipeline/data changes or single-file frontend changes.
- There is no API layer to patch around mistakes.

## Repo Layout

- [index.html](/home/jason/sloplocks/index.html)
  - Entire UI shell and client-side rendering logic.
- [manifest.json](/home/jason/sloplocks/manifest.json)
  - Root PWA manifest for the site.
- [sw.js](/home/jason/sloplocks/sw.js)
  - Service worker and static asset caching.
- [pipeline/config.py](/home/jason/sloplocks/pipeline/config.py)
  - Central configuration and the `SPORTS` registry.
- [pipeline/run.py](/home/jason/sloplocks/pipeline/run.py)
  - Main orchestrator. Most critical file in the repo.
- [pipeline/refresh_picks.py](/home/jason/sloplocks/pipeline/refresh_picks.py)
  - Fast odds refresh without full retraining.
- [pipeline/fetch_data.py](/home/jason/sloplocks/pipeline/fetch_data.py)
  - Odds ingestion from The Odds API.
- [pipeline/fetch_nba.py](/home/jason/sloplocks/pipeline/fetch_nba.py)
- [pipeline/fetch_nhl.py](/home/jason/sloplocks/pipeline/fetch_nhl.py)
- [pipeline/fetch_ncaam.py](/home/jason/sloplocks/pipeline/fetch_ncaam.py)
- [pipeline/fetch_mlb.py](/home/jason/sloplocks/pipeline/fetch_mlb.py)
  - Sport-specific ESPN and context ingestion.
- [pipeline/models.py](/home/jason/sloplocks/pipeline/models.py)
  - Model implementations.
- [pipeline/ensemble.py](/home/jason/sloplocks/pipeline/ensemble.py)
  - Blending, calibration, edge math, Kelly, confidence.
- [pipeline/backtest.py](/home/jason/sloplocks/pipeline/backtest.py)
  - Reporting, replay, ROI, CLV, walk-forward summaries.
- [pipeline/reset_public_record.py](/home/jason/sloplocks/pipeline/reset_public_record.py)
  - Archive-first public-record maintenance tool.
- [pipeline/notify_discord.py](/home/jason/sloplocks/pipeline/notify_discord.py)
  - Discord post formatting and sending.
- [data/](/home/jason/sloplocks/data)
  - Generated live output and tracking ledgers. This is committed.
- [tests/](/home/jason/sloplocks/tests)
  - Pytest suite.

## Current Data Products

Per sport:
- `predictions.json`
  - current slate, live picks, diagnostics, publication guard, model weights
- `history.json`
  - saved modeled match history
- `pick_history.json`
  - published pick history plus settled results and CLV fields
- `model_accuracy.json`
  - rolling model scoring history

Shared tracking:
- `data/tracking/results_log.csv`
  - mutable live results log
- `data/tracking/results_audit_log.csv`
  - append-only settled results ledger
- `data/tracking/odds_history.csv`
  - tracked market snapshots
- `data/tracking/pick_decisions.csv`
  - append-only decision-time pick ledger
- `data/tracking/snapshots/YYYY-MM-DD/{sport}/*.json`
  - immutable run snapshots

These tracking files matter. Do not treat them like disposable cache.

## Important Current Behavior

The project has had a recent integrity pass. Relevant current rules:

- Publication is gated by settled evidence.
  - Some sports can be suppressed from posting official picks live if the sample is too thin.
- MMA is removed.
- Backtests and reporting now prefer immutable ledgers where available.
- Decision replay exists.
  - `pick_decisions.csv` is the authoritative decision-time ledger.
- CLV is separated by market family.
  - Moneyline and totals CLV should not be merged into one naive unit.
- Older historical data is mixed quality.
  - The repo now backfills some missing market/CLV context, but legacy rows are still less complete than fresh rows.

## Current Frontend State

The frontend is intentionally old-terminal / CRT styled:
- green-on-black shell
- monospace typography
- scanline overlay
- data-dense cards

Preserve the current visual direction unless the task explicitly asks for a redesign.

## Current Model / Pipeline Notes

This is not a toy site anymore. The repo has already fixed several serious problems:
- synthetic cross-book no-vig benchmarking
- leaking four-factors training logic
- forced low-confidence “slop lock” picks
- mixed-unit CLV reporting
- mutable public record defaults
- MMA product surface

Do not casually reintroduce those issues.

If you touch model or backtest logic, inspect:
- [pipeline/run.py](/home/jason/sloplocks/pipeline/run.py)
- [pipeline/ensemble.py](/home/jason/sloplocks/pipeline/ensemble.py)
- [pipeline/backtest.py](/home/jason/sloplocks/pipeline/backtest.py)
- [pipeline/models.py](/home/jason/sloplocks/pipeline/models.py)

## Automation

GitHub workflows:
- [daily.yml](/home/jason/sloplocks/.github/workflows/daily.yml)
  - full pipeline run, commit updated `data/`, notify Discord
- [refresh-picks.yml](/home/jason/sloplocks/.github/workflows/refresh-picks.yml)
  - fast odds refresh, commit updated `data/`, notify Discord
- [deploy-site.yml](/home/jason/sloplocks/.github/workflows/deploy-site.yml)
  - deploy static site to Netlify on push to `master`

Netlify deploy config is in [netlify.toml](/home/jason/sloplocks/netlify.toml).

## Commands

Setup:

```bash
python -m venv venv
source venv/bin/activate
pip install -r pipeline/requirements.txt
```

Run full pipeline:

```bash
python -m pipeline.run
```

Run one sport:

```bash
python -m pipeline.run --sport nba
```

Refresh odds without retraining:

```bash
python -m pipeline.refresh_picks
```

Run reporting:

```bash
python -m pipeline.backtest
python -m pipeline.backtest --walkforward
python -m pipeline.backtest --snapshot-replay
python -m pipeline.backtest --decision-replay
```

Run tests:

```bash
pytest tests/ -v
```

Fast verification commonly used after pipeline/report changes:

```bash
python -m compileall pipeline tests
pytest -q tests/test_run.py tests/test_backtest.py tests/test_reset_public_record.py tests/test_notify_discord.py
```

## Environment Variables

Loaded from `.env` if present via `dotenv`.

Main keys:
- `ODDS_API_KEY`
- `BALLDONTLIE_API_KEY`
- `ANTHROPIC_API_KEY`
- `ENABLE_QUALITATIVE`

Some workflows also use:
- `GEMINI_API_KEY`
- `DISCORD_WEBHOOK_URL`
- `NETLIFY_AUTH_TOKEN`
- `NETLIFY_SITE_ID`

## Practical Editing Guidance

When changing frontend:
- most logic is inside `index.html`
- keep dynamic placeholders intact
- remember the root `manifest.json` and `sw.js` are separate from generated `data/manifest.json`

When changing pipeline output shape:
- inspect both pipeline writers and frontend readers
- search `index.html` for the field name before changing or removing it

When changing pick or reporting logic:
- inspect `pick_history.json`, `results_log.csv`, `results_audit_log.csv`, `pick_decisions.csv`, and `odds_history.csv`
- avoid mutating historical truth unless the task explicitly calls for a migration

When changing tracking logic:
- preserve append-only behavior where intended
- prefer additive migrations and backfills over destructive rewrites

## Known Truths / Gotchas

- `README.md` still references EPL and older models. It is not current.
- `CLAUDE.md` is also partially stale.
- `master` is the live deployment branch.
- `data/` is committed and expected to change as part of normal pipeline work.
- `data/*/espn_cache.json` can change during live runs.
- A successful code change may legitimately produce large generated-data diffs.
- Zero odds coverage on a given run is possible and does not automatically mean the models failed.

## If You Need To Orient Quickly

Start in this order:

1. [pipeline/config.py](/home/jason/sloplocks/pipeline/config.py)
2. [pipeline/run.py](/home/jason/sloplocks/pipeline/run.py)
3. [pipeline/backtest.py](/home/jason/sloplocks/pipeline/backtest.py)
4. [index.html](/home/jason/sloplocks/index.html)
5. one live sport payload in `data/{sport}/predictions.json`

That is enough to understand most tasks in this repository.
