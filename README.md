# SLOP LOCKS

Static PWA for model-driven sports betting picks, pick history, and pipeline health reporting.

Live: [jasonpersinger.me/sloplocks](https://jasonpersinger.me/sloplocks/) (GitHub Pages, deployed from `master`)

There is no backend and no build step. A Python pipeline writes committed JSON/CSV into `data/`, and the single-file frontend reads those static files directly. Because there is no API layer, **any data-shape change must update both the writer and the frontend reader.**

## Sports

| Sport | Models (moneyline) | Totals | Notes |
| --- | --- | --- | --- |
| NBA | Elo, Results Features, Recent Boxscore, NBA Matchup | modeled, publication gated | |
| WNBA | Elo, Results Features, Recent Boxscore | — | Small settled sample. |
| NHL | Elo, Results Features, NHL Matchup | — | |
| MLB | Elo, Results Features, Bullpen, Run Environment, Handedness | live | MLB Stats API probable-pitcher fallback when ESPN says `TBD`. |
| NFL | Elo, Results Features | live | Publishes a pick on **every** game (see Full Slate). |
| NCAAF | Elo, Results Features | live | FBS only; non-FBS opponents collapse to one synthetic rating entity. |
| NCAAM | — | — | Season-disabled: code and history retained, skipped by live runs. |

A sport is **active** purely by being a key in `SPORTS` in `pipeline/config.py`. `SEASON_DISABLED_SPORTS` is the parallel dict for retained-but-inactive sports. Everything downstream — the orchestrator, manifest, CLI choices, Discord — iterates those two.

### Football specifics

Football is weekly and scores on a different scale, so several shared assumptions are config-gated (defaults preserve every other sport exactly):

- `elo_margin_divisor` / `elo_margin_cap` — margins are measured in touchdowns, so a 24-point win doesn't apply a 4.4× K multiplier.
- `elo_season_carryover` + `history_seasons` — Elo trains on 3 seasons and regresses toward the mean at each season boundary. Without it a 17-game NFL season sits flat at 1500 until roughly week 5. NFL keeps 0.67 (draft and free agency force parity); NCAAF keeps 0.72 (program strength persists harder).
- `results_feature_rest_cap_days` — rest was clamped at 7 days, which erased bye weeks entirely; football uses 14.
- `short_rest_days` / `short_rest_penalty` — the Thursday-game case. `rest_bonus_max_days` stops a season-opening gap being scored as a bye.

## Pipeline

Daily: fetch schedules/results/context → fetch odds → build features → fit and score models → blend, calibrate, apply market respect → compute edge/EV/Kelly → select picks by lane → apply publication guards → grade settled picks → write `data/` → commit from Actions.

**Selection and publication are separate.** The model can find a candidate that the publication guard still suppresses when settled evidence is thin or unhealthy (evaluated pick count, recent ROI, CLV, calibration health, per-sport thresholds). A newly activated sport therefore runs in `research` mode, recording **shadow picks** (`published: false`) as guard evidence until it earns publication. `_is_live_public_output` only enforces the guard against the real `data/` dir, so custom `--output-dir` runs let you inspect candidates offline.

Selection lanes: `core` (standard edge/probability/EV gate), `value_dog` (positive-EV underdogs below the core probability floor), `near_favorite` (modest edge requirement on short prices). Each sport config sets edge floor, probability floor, confidence floor, minimum EV, odds band, and per-lane and global max picks. The chosen lane persists as `selection_lane`; treat older picks lacking it as `core`.

Integrity rules: never publish stale picks past start time, `NO_PLAY` tiers, or forced low-confidence fallbacks; never merge moneyline and totals CLV into one unit; prefer immutable/audit ledgers for reporting; keep decision and audit ledgers append-only.

## Full Slate (NFL)

With `publish_full_slate`, the sport tab renders a **FULL SLATE** table under the locks: one row per game with pick, odds, model probability, confidence, edge, and the AI rationale. Rows that ascended to a Slop Lock are badged rather than hidden.

This board renders independently of the publication guard and is labelled model output — locks remain the gated public record. Note that a predicted *winner* is often not a *bet*: a heavy favourite can be the correct pick and still carry negative edge, which the analysis labels `avoid`.

## AI analysis

Game and totals analysis run on **Gemini 2.5 Flash** (`GEMINI_API_KEY`) via `pipeline/qualitative_analysis.py`. The draft tab uses the same provider in `draft_qualitative_analysis.py`.

The analyst receives the model's own reasoning — pick, model probability, market implied probability, edge, per-component model breakdown, Elo ratings — alongside scraped situational context, and returns:

- `home_impact` / `away_impact` / `individual_factors` — numeric scores that feed a capped probability adjustment.
- `pick_rationale` — the explanatory blurb the site renders.
- `model_vs_market`, `key_risk`, `confidence_label`.

Situational context is optional: college football has no public injury feed, so when context is absent the analyst is explicitly told not to speculate about injuries, weather, or news. The layer is off unless `ENABLE_QUALITATIVE=true`, and degrades to neutral defaults without a key.

## Repository layout

```text
sloplocks/
|-- index.html                     # Entire frontend: markup, CSS, client JS
|-- sw.js, manifest.json           # Service worker + PWA manifest
|-- data/                          # Committed generated data and tracking ledgers
|-- pipeline/
|   |-- config.py                  # Sport registry, thresholds, API bases, paths
|   |-- run.py                     # Daily orchestrator
|   |-- refresh_picks.py           # Fast odds refresh without retraining
|   |-- models.py                  # Elo, results/matchup models, totals models
|   |-- ensemble.py                # Calibration, blending, edge, Kelly, tiers
|   |-- backtest.py                # Reporting, replay, dashboard, lane health
|   |-- fetch_data.py              # The Odds API client
|   |-- fetch_{nba,wnba,nhl,mlb,nfl,ncaaf,ncaam}.py
|   |-- qualitative_analysis.py    # Gemini game + totals analysis
|   |-- draft_qualitative_analysis.py, build_draft_tab.py
|   |-- context_scraper.py         # Situational context for the AI layer
|   |-- notify_discord.py
|   `-- reset_public_record.py     # Archive-first public-record maintenance
|-- tests/
`-- .github/workflows/{daily,refresh-picks}.yml
```

## Data products

| Path | Purpose |
| --- | --- |
| `data/manifest.json` | Sport status, run status, diagnostics summary. |
| `data/dashboard.json` | BOARD tab: aggregate record, replay, lane health. |
| `data/{sport}/predictions.json` | Slate, modeled matches, picks, diagnostics, guard, weights. |
| `data/{sport}/pick_history.json` | Picks plus settled outcomes and CLV; `published` flags shadows. |
| `data/{sport}/history.json` | Saved modeled match history. |
| `data/{sport}/model_accuracy.json` | Rolling model scoring history. |
| `data/{sport}/espn_cache.json` | Fetch cache; can change during live runs. |
| `data/tracking/results_log.csv` | Mutable settled-results log. |
| `data/tracking/results_audit_log.csv` | Append-only settled-results audit ledger. |
| `data/tracking/odds_history.csv` | Market snapshots for CLV. |
| `data/tracking/pick_decisions.csv` | Decision-time pick ledger. |
| `data/tracking/snapshots/YYYY-MM-DD/{sport}/*.json` | Immutable run snapshots. |

Tracking files are product data, not disposable cache. Prefer additive migrations over destructive rewrites.

## Data sources

**Keyed:** The Odds API (moneyline + totals), Gemini (qualitative analysis), balldontlie (optional NBA enrichment), Discord webhook.

**Keyless:** ESPN site/core APIs (schedules, results, rosters, team and conference metadata), MLB Stats API (probable-pitcher fallback), Open-Meteo (MLB weather).

Note that ESPN's site `/teams` endpoint ignores `?groups` and carries no conference field, so NCAAF derives FBS membership and conferences from the **core API group tree** (group 80). That means realignment is tracked automatically.

## Setup

```bash
git clone https://github.com/jasonpersinger/sloplocks.git
cd sloplocks
python -m venv venv && source venv/bin/activate
pip install -r pipeline/requirements.txt
cp .env.template .env      # then fill in keys
```

| Key | Used for | Required |
| --- | --- | --- |
| `ODDS_API_KEY` | Odds ingestion. | Yes, for live odds. |
| `GEMINI_API_KEY` | Game and totals AI analysis. | Optional; analysis is neutral without it. |
| `ENABLE_QUALITATIVE` | Master switch for the AI layer. | Optional (`true` in CI). |
| `BALLDONTLIE_API_KEY` | Deeper NBA data path. | Optional. |
| `DISCORD_WEBHOOK_URL` | Workflow notifications. | Optional. |

The test suite strips these keys and disables the AI layer via an autouse fixture in `tests/conftest.py`. `pipeline/config.py` calls `load_dotenv(override=True)`, so without that guard a populated `.env` would make the suite issue real billed API calls.

## Commands

```bash
python -m pipeline.run                       # all active sports
python -m pipeline.run --sport nfl           # one sport
python -m pipeline.run --sport mlb --output-dir /tmp/out/mlb   # research run

python -m pipeline.refresh_picks             # odds refresh, no retraining
python -m pipeline.refresh_picks nba nhl

python -m pipeline.backtest                  # reporting
python -m pipeline.backtest --walkforward    # also: --raw-walkforward,
                                             # --snapshot-replay, --decision-replay
python -m pipeline.backtest nba nhl --walkforward

pytest -q                                    # full suite
python -m compileall pipeline tests          # syntax check
```

There is no lint or type tooling configured; pytest is the only gate.

## Automation

| Workflow | Trigger | Behavior |
| --- | --- | --- |
| `daily.yml` | 12:00 UTC daily + manual | Full pipeline, commits `data/`, Discord notify. |
| `refresh-picks.yml` | Manual | Odds refresh, commits `data/`, Discord notify. |

Both commit as `sloplocks-bot` directly to `master`, so **pull with `--rebase` before pushing.** Generated `data/` churn is expected; avoid mixing it into unrelated code commits.

## Development

Branch from `master`, verify, PR, merge — Pages publishes on push.

Before touching models or selection: `config.py`, `run.py`, `ensemble.py`, `models.py`, `backtest.py`. Note that `run.py` and `backtest.py` each construct models independently, so **model config must be wired in both** or the backtest silently diverges from live.

Before changing frontend output: the writer in `run.py`/`backtest.py`, the reader in `index.html`, and a representative payload under `data/{sport}/`. Bump `CACHE_NAME` in `sw.js` when static assets change.

Frontend design direction — old-terminal/CRT, green-on-black, monospace, scanline overlay, dense cards. Keep it unless a task explicitly asks for a redesign.

## Troubleshooting

**Low or zero picks** — read `data/{sport}/predictions.json`:

1. `diagnostics.fixtures_with_odds` vs `fixtures_in_window` — odds coverage.
2. `diagnostics.gate_failures` — which floor rejected candidates.
3. `diagnostics.candidate_lanes` — all candidates in `value_dog` and none in `near_favorite` usually means the model has little signal and is fading every favourite.
4. `publication_guard.status` / `.reason` — `research` means the sport has not earned publication yet; picks are being recorded as shadows.
5. `historical_matches` — a near-constant probability across the slate means the model has no data (early season).

**Unexpected frontend display** — confirm `manifest.json`/`dashboard.json` regenerated, confirm the field exists in `predictions.json`, search `index.html` for it, clear service-worker cache.

**Large diffs** — `data/` is committed; `espn_cache.json` changes on live runs; snapshot folders are immutable artifacts.

## License

MIT
