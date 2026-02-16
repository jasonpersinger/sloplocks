# SLOP LOCKS — Project Brief

## What is this?
Multi-sport prediction engine with ensemble models and automated betting edge detection. Live at https://sloplocks.lol.

## Supported Sports
- **EPL** (Premier League) — 3-way outcomes (home/draw/away), Dixon-Coles + xG + Elo ensemble
- **NBA** — 2-way outcomes (home/away), Elo-only model with 100-point home court advantage

## Tech Stack
- Pipeline: Python 3.11+ (scipy, pandas, numpy, requests, beautifulsoup4)
- Frontend: Single-file HTML/CSS/JS (index.html)
- Automation: GitHub Actions (daily cron at 6am UTC)
- Hosting: Netlify (auto-deploy from GitHub on push to master)
- Domain: sloplocks.lol

## Architecture
- GitHub Action runs daily at 6am UTC
- Python pipeline fetches data, runs models, writes JSON to `data/{sport}/`
- Static frontend reads per-sport `predictions.json` via sport toggle, no backend
- `data/manifest.json` lists available sports and their status

## Sport Config System
`pipeline/config.py` contains a `SPORTS` dict with per-sport configuration:
- `outcomes` — `["home", "draw", "away"]` (EPL) or `["home", "away"]` (NBA)
- `models` — which models to run per sport
- `elo_k_factor`, `elo_home_advantage` — sport-specific Elo tuning
- `odds_sport` — The Odds API sport key
- `data_dir` — output directory for this sport

## The Ensemble

### EPL (3 models, softmax-weighted by rolling accuracy)
1. **Dixon-Coles** — Attack/defense parameters from actual goals, time-decay, low-score correction (rho)
2. **xG-adjusted Dixon-Coles** — Same model using expected goals from Understat
3. **Elo ratings** — Dynamic power ratings with 65-point home advantage, goal-diff multiplier

### NBA (Elo only)
1. **Elo ratings** — 2-way logistic model with 100-point home court advantage, K=20

## Data Sources
- football-data.org (EPL results, fixtures) — API key in `FOOTBALL_DATA_API_KEY`
- Understat (EPL xG) — scraped, no key needed
- The Odds API (bookmaker odds, all sports) — API key in `ODDS_API_KEY`
- balldontlie.io (NBA results, schedule) — API key in `BALLDONTLIE_API_KEY`

## Commands
- Run pipeline: `python -m pipeline.run`
- Run single sport: `python -c "from pipeline.run import run_sport_pipeline; run_sport_pipeline('nba')"`
- Run tests: `pytest tests/ -v`
- Install deps: `pip install -r pipeline/requirements.txt`

## File Structure
```
sloplocks/
├── index.html              ← Frontend (single file, ~2200 lines)
├── manifest.json           ← PWA manifest
├── sw.js                   ← Service worker
├── data/                   ← Generated daily by pipeline
│   ├── manifest.json       ← Sport status/availability
│   ├── epl/
│   │   ├── predictions.json
│   │   ├── history.json
│   │   └── model_accuracy.json
│   └── nba/
│       ├── predictions.json
│       ├── history.json
│       └── model_accuracy.json
├── pipeline/
│   ├── config.py           ← Central config, API keys, SPORTS dict
│   ├── fetch_data.py       ← football-data.org + Odds API clients
│   ├── fetch_nba.py        ← balldontlie.io NBA client
│   ├── fetch_xg.py         ← Understat xG scraper
│   ├── models.py           ← Dixon-Coles + Elo (2-way and 3-way)
│   ├── ensemble.py         ← Blending + edge detection (generic)
│   ├── backtest.py         ← Accuracy tracking + ROI
│   └── run.py              ← Pipeline orchestrator (multi-sport)
├── tests/
│   ├── conftest.py         ← Shared fixtures (EPL sample matches)
│   ├── test_fetch_data.py
│   ├── test_fetch_nba.py
│   ├── test_fetch_xg.py
│   ├── test_models.py
│   ├── test_ensemble.py
│   ├── test_backtest.py
│   └── test_run.py
├── .github/workflows/
│   └── daily.yml           ← Daily prediction cron
└── icons/
    ├── icon-192.png
    └── icon-512.png
```

## Color Scheme
- Background: #0a0a0a
- Value/edge highlight: #39FF14 (slime green)
- Text: #EAEAEA
- Secondary: #888888
- Negative edge: #FF2D95

## Model Parameters (config.py)
- TIME_DECAY_RATE = 0.005
- FORM_WINDOW = 6
- ELO_K_FACTOR = 20 (both sports)
- ELO_HOME_ADVANTAGE = 65 (EPL), 100 (NBA)
- VALUE_EDGE_THRESHOLD = 0.05 (5% edge = value bet)
- MAX_GOALS = 6 (scoreline matrix size, EPL only)
- ENSEMBLE_ACCURACY_WINDOW = 10 (rolling window for model weights)

## Deployment
Push to master auto-deploys to Netlify. Pipeline runs daily via GitHub Action, commits updated `data/{sport}/predictions.json`. Bump `CACHE_NAME` in sw.js when updating static assets.

## Adding a New Sport
1. Add sport config to `SPORTS` dict in `config.py`
2. Create a `fetch_{sport}.py` with `fetch_{sport}_games()` and `fetch_{sport}_schedule()`
3. Add sport-specific branch in `run_sport_pipeline()` in `run.py`
4. Add sport pill to the frontend toggle in `index.html`
5. Add `{SPORT}_API_KEY` to GitHub Actions secrets if needed
