# SLOP LOCKS — Project Brief

## What is this?
EPL match prediction engine with a 3-model ensemble and automated betting edge detection. Live at https://sloplocks.lol.

## Tech Stack
- Pipeline: Python 3.11+ (scipy, pandas, numpy, requests, beautifulsoup4)
- Frontend: Single-file HTML/CSS/JS (index.html)
- Automation: GitHub Actions (daily cron at 6am UTC)
- Hosting: Netlify (auto-deploy from GitHub on push to master)
- Domain: sloplocks.lol

## Architecture
- GitHub Action runs daily at 6am UTC
- Python pipeline fetches data, runs models, writes JSON to `data/`
- Static frontend reads `data/predictions.json`, no backend

## The Ensemble
Three models blended by rolling accuracy (softmax-weighted):
1. **Dixon-Coles** — Attack/defense parameters from actual goals, time-decay, low-score correction (rho)
2. **xG-adjusted Dixon-Coles** — Same model using expected goals from Understat
3. **Elo ratings** — Dynamic power ratings with home advantage, goal-diff multiplier

## Data Sources
- football-data.org (results, fixtures) — API key in `FOOTBALL_DATA_API_KEY`
- Understat (xG) — scraped, no key needed
- The Odds API (bookmaker odds) — API key in `ODDS_API_KEY`

## Commands
- Run pipeline: `python -m pipeline.run`
- Run tests: `pytest tests/ -v`
- Install deps: `pip install -r pipeline/requirements.txt`

## File Structure
```
sloplocks/
├── index.html              ← Frontend (single file, 1600+ lines)
├── manifest.json           ← PWA manifest
├── sw.js                   ← Service worker
├── data/                   ← Generated daily by pipeline
│   └── predictions.json
├── pipeline/
│   ├── config.py           ← Central config, API keys, model params
│   ├── fetch_data.py       ← football-data.org + Odds API clients
│   ├── fetch_xg.py         ← Understat xG scraper
│   ├── models.py           ← Dixon-Coles + Elo implementations
│   ├── ensemble.py         ← Blending + edge detection
│   ├── backtest.py         ← Accuracy tracking + ROI
│   └── run.py              ← Pipeline orchestrator
├── tests/
│   ├── conftest.py         ← Shared fixtures (10 sample matches)
│   ├── test_fetch_data.py
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
- Value/edge highlight: #00ff88 (neon green)
- Text: #ffffff
- Secondary: #888888
- Negative edge: #ff4444

## Model Parameters (config.py)
- TIME_DECAY_RATE = 0.005
- FORM_WINDOW = 6
- ELO_K_FACTOR = 20
- ELO_HOME_ADVANTAGE = 65
- VALUE_EDGE_THRESHOLD = 0.05 (5% edge = value bet)
- MAX_GOALS = 6 (scoreline matrix size)
- ENSEMBLE_ACCURACY_WINDOW = 10 (rolling window for model weights)

## Deployment
Push to master auto-deploys to Netlify. Pipeline runs daily via GitHub Action, commits updated `data/predictions.json`. Bump `CACHE_NAME` in sw.js when updating static assets.
