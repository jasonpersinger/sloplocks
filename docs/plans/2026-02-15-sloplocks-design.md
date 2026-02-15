# SLOP LOCKS — Design Document

**Goal:** A daily-updating EPL match prediction engine with a 3-model ensemble and automated betting edge detection, deployed as a free static site at sloplocks.lol.

**Architecture:** Hybrid — GitHub Action runs a Python pipeline daily, generates JSON predictions, commits to repo, Netlify auto-deploys the static frontend.

**Tech Stack:** Python (scipy, pandas, numpy, requests, beautifulsoup4), HTML/CSS/JS, GitHub Actions, Netlify

---

## Architecture

```
GitHub Action (daily, 6am UTC)
  → Python pipeline:
    1. Fetch results from football-data.org
    2. Scrape xG from Understat
    3. Pull live odds from The Odds API
    4. Run 3-model ensemble
    5. Compute edges vs bookmaker lines
    6. Write predictions.json + history.json + model_accuracy.json
    7. Commit & push → triggers Netlify deploy

Static Frontend (sloplocks.lol)
  → Single HTML file reads JSON files
  → No backend, no signup, no API calls at runtime
```

### Repo Structure

```
sloplocks/
├── index.html              ← The entire frontend
├── data/
│   ├── predictions.json    ← Generated daily by Action
│   ├── history.json        ← Rolling log of past predictions vs results
│   └── model_accuracy.json ← Backtesting stats
├── pipeline/
│   ├── fetch_data.py       ← API + scraping
│   ├── models.py           ← Dixon-Coles, xG, Elo
│   ├── ensemble.py         ← Blending + edge detection
│   ├── backtest.py         ← Accuracy tracking
│   └── run.py              ← Orchestrator (Action calls this)
├── .github/
│   └── workflows/
│       └── daily.yml       ← Cron: 0 6 * * *
├── manifest.json           ← PWA manifest
├── sw.js                   ← Service worker
├── CLAUDE.md
└── README.md
```

---

## The Ensemble Model

Three models, blended by recent accuracy:

### 1. Dixon-Coles (Goal-Based)

The academic gold standard for soccer prediction. A modified Poisson model that:

- Estimates per-team attack and defense strength parameters (20 teams = 80 params: home attack, home defense, away attack, away defense)
- Applies time-decay weighting — recent matches count ~2x vs early season
- Corrects for low-score correlation (0-0, 1-0, 0-1, 1-1 are more frequent than independent Poisson predicts)
- Outputs a full scoreline probability matrix (0-0 through 5-5), which collapses to home/draw/away probabilities

### 2. xG-Adjusted Dixon-Coles

Same mathematical framework as Dixon-Coles, but instead of using actual goals scored/conceded, uses expected goals (xG) from Understat. This separates skill from luck:

- A team that scores 3 goals from 0.4 xG is lucky, not good
- A team that creates 3.2 xG but scores 0 is unlucky, not bad
- xG-based ratings are a sharper, less noisy signal for underlying quality

### 3. Elo Ratings

Dynamic power ratings that update after every match:

- Start-of-season ratings based on previous season finish
- Update magnitude based on result vs expectation (upset = bigger shift) and goal margin
- Home advantage baked into the rating calculation
- Converts rating difference to win/draw/loss probabilities via logistic function

### Ensemble Blending

- Each model produces home/draw/away probabilities
- Models weighted by rolling 10-match prediction accuracy
- Final probability = weighted average of all three models
- If one model is hot, it gets more influence

### Additional Factors

- **Home/away splits:** Separate attack/defense parameters for home vs away matches
- **Form momentum:** Last 6 matches weighted 2x in Dixon-Coles time-decay
- **Fixture congestion:** Penalty applied to teams with <4 days rest between matches
- **Head-to-head venue history:** Historical results at the specific ground factor into the prediction

---

## Edge Finding

For each upcoming match and each outcome (home/draw/away):

1. **Model probability** — from the ensemble (e.g., 42% Arsenal win)
2. **Bookmaker implied probability** — converted from best available odds across books (e.g., +150 American → 40% implied)
3. **Edge** = model probability − implied probability (e.g., +2% edge)
4. **Value threshold** — edges >5% flagged as value bets
5. **Closing Line Value (CLV)** — track whether our flagged edges at prediction time beat the closing line. Consistent CLV is the strongest indicator of a real edge.

---

## Data Sources

| Source | Data | Cost | Access |
|--------|------|------|--------|
| football-data.org | Match results, standings, fixtures, basic odds | Free | REST API, 10 req/min |
| Understat | xG per match, per team, per player | Free | Scrape (BeautifulSoup) |
| The Odds API | Live bookmaker odds across 10+ books | Free (500 req/mo) | REST API |

---

## Frontend

Single HTML file, dark theme, irreverent tone.

### Layout

1. **Header** — SLOP LOCKS branding
2. **Upcoming Matches** — Cards for next gameweek:
   - Team names, date/kickoff time
   - Model probabilities: Home W% / Draw% / Away W%
   - Best bookmaker odds (American format: +150, -200, etc.)
   - Edge % per outcome, highlighted green when >5%
   - Confidence indicators for strong edges
3. **Model Accuracy** — Season stats:
   - Value bet hit rate
   - ROI %
   - Units profit/loss
4. **CLV Tracker** — Chart showing closing line value over time
5. **Recent Predictions** — Last few gameweeks with results marked

### Visual Style

- Black background, neon green for value bets, white text
- Compact match cards
- Terminal/sportsbook aesthetic — not corporate, not boring
- Monospace for numbers/odds
- Responsive for mobile

---

## Deployment

- **Hosting:** Netlify (free tier), auto-deploy from GitHub push
- **Domain:** sloplocks.lol → Netlify
- **Schedule:** GitHub Action cron `0 6 * * *` (6am UTC daily)
- **PWA:** manifest.json + sw.js for offline support / installable on phone

---

## API Keys Required

- **football-data.org:** Free API key (register at football-data.org)
- **The Odds API:** Free API key (register at the-odds-api.com)
- Both stored as GitHub Actions secrets (`FOOTBALL_DATA_API_KEY`, `ODDS_API_KEY`)
