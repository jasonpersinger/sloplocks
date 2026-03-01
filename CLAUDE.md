# SLOP LOCKS — Project Brief

## What is this?
Multi-sport prediction engine with ensemble models and automated betting edge detection. Live at https://sloplocks.lol.

## Supported Sports
- **NBA** — 2-way outcomes (home/away), Elo + AdjustedEfficiency + FourFactors ensemble with B2B rest adjustment

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
- `outcomes` — `["home", "away"]` (all current sports)
- `models` — which models to run per sport
- `elo_k_factor`, `elo_home_advantage` — sport-specific Elo tuning
- `odds_sport` — The Odds API sport key
- `data_dir` — output directory for this sport

## The Ensemble

### NBA / NCAAM (3 models, softmax-weighted by rolling accuracy)
1. **Elo ratings** — Dynamic power ratings with 65-point home advantage, K=20, B2B rest penalty (NBA)
2. **AdjustedEfficiency** — Offensive/defensive efficiency ratings (points per 100 possessions) adjusted for schedule strength
3. **FourFactors** — Dean Oliver's four factors model (eFG%, TOV%, ORB%, FT rate)

## Data Sources
- The Odds API (bookmaker odds, all sports) — API key in `ODDS_API_KEY`
- ESPN API (NBA/NCAAM results, schedule) — no key needed

## Commands
- Run pipeline: `python -m pipeline.run`
- Run single sport: `python -c "from pipeline.run import run_sport_pipeline; run_sport_pipeline('nba')"`
- Run tests: `pytest tests/ -v`
- Install deps: `pip install -r pipeline/requirements.txt`

## File Structure
```
sloplocks/
├── index.html              ← Frontend (single file)
├── manifest.json           ← PWA manifest
├── sw.js                   ← Service worker
├── data/                   ← Generated daily by pipeline
│   ├── manifest.json       ← Sport status/availability
│   ├── nba/
│   │   ├── predictions.json
│   │   ├── history.json
│   │   └── model_accuracy.json
│   └── ncaam/
│       ├── predictions.json
│       ├── history.json
│       └── model_accuracy.json
├── pipeline/
│   ├── config.py           ← Central config, API keys, SPORTS dict
│   ├── fetch_data.py       ← Odds API client
│   ├── fetch_nba.py        ← ESPN NBA client (games, schedule, box scores)
│   ├── fetch_ncaam.py      ← ESPN NCAAM client
│   ├── models.py           ← Elo + Efficiency + FourFactors models
│   ├── ensemble.py         ← Blending + edge detection (generic)
│   ├── backtest.py         ← Accuracy tracking + ROI
│   └── run.py              ← Pipeline orchestrator (multi-sport)
├── tests/
│   ├── conftest.py         ← Shared fixtures
│   ├── test_fetch_data.py
│   ├── test_fetch_nba.py
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
- ELO_K_FACTOR = 20 (both sports)
- ELO_HOME_ADVANTAGE = 65 (both sports)
- NBA_B2B_PENALTY = 30 (Elo points deducted for back-to-back games)
- VALUE_EDGE_THRESHOLD = 0.05 (5% edge = value bet)
- ENSEMBLE_ACCURACY_WINDOW = 10 (rolling window for model weights)

## Deployment
Push to master auto-deploys to Netlify. Pipeline runs daily via GitHub Action, commits updated `data/{sport}/predictions.json`. Bump `CACHE_NAME` in sw.js when updating static assets.

## Adding a New Sport
1. Add sport config to `SPORTS` dict in `config.py`
2. Create a `fetch_{sport}.py` with `fetch_{sport}_games()` and `fetch_{sport}_schedule()`
3. Add sport-specific branch in `run_sport_pipeline()` in `run.py`
4. Add sport pill to the frontend toggle in `index.html`
5. Add `{SPORT}_API_KEY` to GitHub Actions secrets if needed
