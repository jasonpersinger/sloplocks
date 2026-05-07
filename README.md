# SLOP LOCKS

Multi-sport prediction engine with ensemble models and automated edge detection.

**Live:** [sloplocks.lol](https://sloplocks.lol)

---

## Supported Sports

| Sport | Status | Models |
|-------|--------|--------|
| NBA | Active | Elo, Results Features, Recent Boxscore, NBA Matchup |
| NHL | Active | Elo, Results Features, NHL Matchup |
| MLB | Active | Elo, Results Features, Pitcher Features, Bullpen Features, Run Environment, Handedness |
| NCAAM | Off-season | Historical data retained; live picks disabled |

---

## How It Works

A GitHub Action runs daily at 12pm UTC. The pipeline fetches fresh schedules, results, and bookmaker odds, fits ensemble models, and writes `predictions.json` for each sport. Netlify auto-deploys the static frontend on every push to `master`.

### Ensemble Architecture

Each sport runs 3–6 sport-specific models. Model weights are determined by rolling accuracy over a configurable window (softmax-scaled), so recent performance shifts the blend automatically.

**NBA models:**
- **Elo** — Dynamic power ratings with 65-point home advantage, K=20, back-to-back rest penalty
- **Results Features** — Logistic regression on recent game outcomes and margin features
- **Recent Boxscore** — Efficiency metrics from the last N box scores
- **NBA Matchup** — Head-to-head style matchup features

**NHL models:**
- **Elo** — Dynamic ratings with 28-point home advantage, K=18
- **Results Features** — Recent outcome and goal-differential features
- **NHL Matchup** — Goalie status and line matchup features

**MLB models:**
- **Elo** — Low-K ratings for 162-game season, 24-point home advantage
- **Results Features** — Recent W/L, run differential trends
- **Pitcher Features** — Starting pitcher ERA, K/9, recent workload
- **Bullpen Features** — Bullpen ERA, recent usage, fatigue
- **Run Environment** — Park factors, weather adjustments, handedness splits
- **Handedness Features** — Batter vs. pitcher handedness matchup

### Probability Calibration

Raw model probabilities are passed through two calibration layers before edge calculation:

1. **Isotonic regression** — Fitted on resolved historical predictions to correct systematic over/under-confidence
2. **Market-respect blend** — Weighted average with the bookmaker's no-vig implied probability (30% market weight). Extreme model divergence (>20% from the market) triggers heavy shrinkage.

### Pick Tiers

Every candidate pick is classified into one of four tiers based on confidence, win probability, and edge — in that order. Probability is the primary gate; a large edge cannot promote a low-probability pick.

| Tier | Confidence | Win Prob | Edge | Description |
|------|-----------|---------|------|-------------|
| **STRONG** | ≥ 62 | ≥ 57% | ≥ 2% | High confidence + clear value |
| **LEAN** | ≥ 54 | ≥ 53% | ≥ 1% | Solid prediction, modest edge |
| **WATCHLIST** | ≥ 48 | ≥ 52% | any | Interesting angle, thin confirmation |
| **NO PLAY** | — | < 52% | — | Insufficient evidence |

### Confidence Score

Each pick's 0–100 confidence score is weighted:

- **45%** — Win probability (model's predicted likelihood of the pick hitting)
- **30%** — Model agreement (low variance across ensemble models)
- **15%** — Edge (model probability vs. bookmaker implied probability)
- **10%** — Expected value

### Pick Types

- **Slop Locks** — Main moneyline picks, sorted by start time
- **Totals Locks** — Over/under picks for supported sports

---

## Setup

```bash
git clone https://github.com/jasonpersinger/sloplocks.git
cd sloplocks
python -m venv venv
source venv/bin/activate
pip install -r pipeline/requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys (see API Keys below), then:

```bash
# Run full pipeline (all active sports)
python -m pipeline.run

# Run a single sport
python -c "from pipeline.run import run_sport_pipeline; run_sport_pipeline('nba')"

# Run tests
pytest tests/ -v
```

---

## API Keys

| Key | Source | Required |
|-----|--------|----------|
| `ODDS_API_KEY` | [the-odds-api.com](https://the-odds-api.com/) | Yes |
| `BALLDONTLIE_API_KEY` | [balldontlie.io](https://www.balldontlie.io/) | Yes (NBA) |
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/) | Optional (qualitative analysis, gpt-4o-mini) |

Set these as GitHub Secrets for the daily Action. ESPN schedule/results data requires no key.

---

## Project Structure

```
sloplocks/
├── index.html                  ← Frontend (single file, no build step)
├── data/                       ← Generated daily by pipeline
│   ├── manifest.json
│   ├── nba/
│   │   ├── predictions.json
│   │   ├── history.json
│   │   └── model_accuracy.json
│   ├── nhl/
│   └── mlb/
├── pipeline/
│   ├── config.py               ← Central config, sport definitions, thresholds
│   ├── run.py                  ← Pipeline orchestrator
│   ├── ensemble.py             ← Blending, calibration, edge detection, pick tiers
│   ├── models.py               ← All model implementations
│   ├── backtest.py             ← Accuracy tracking, ROI, lane health guards
│   ├── fetch_nba.py            ← ESPN NBA client
│   ├── fetch_nhl.py            ← ESPN NHL client
│   ├── fetch_mlb.py            ← ESPN MLB client
│   ├── fetch_data.py           ← The Odds API client
│   ├── notify_discord.py       ← Discord webhook notifications
│   └── qualitative_analysis.py ← LLM-based context layer (optional)
├── tests/
└── .github/workflows/
    └── daily.yml               ← Runs at 12pm UTC, commits data/, deploys via Netlify
```

---

## Deployment

Push to `master` → Netlify auto-deploys the frontend. The daily GitHub Action runs at 12pm UTC (~8am ET), commits updated `data/` JSON, and triggers a fresh deploy. No server required — the frontend reads static JSON files directly.

To add a new sport:

1. Add a sport config entry to `SPORTS` in `pipeline/config.py`
2. Create `pipeline/fetch_{sport}.py` with schedule and results fetchers
3. Add sport-specific models to `pipeline/models.py`
4. Wire up the sport branch in `pipeline/run.py`
5. Add the sport pill to the frontend toggle in `index.html`
6. Add any new API keys to GitHub Secrets and `daily.yml`

---

## License

MIT
