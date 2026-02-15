# SLOP LOCKS

EPL match predictions powered by a Dixon-Coles + xG + Elo ensemble.

**Live:** [sloplocks.lol](https://sloplocks.lol)

## How It Works

A GitHub Action runs daily at 6am UTC, pulling EPL results, xG data, and bookmaker odds. Three prediction models are blended into an ensemble, and the output is compared against bookmaker lines to surface value bets.

### The Models

- **Dixon-Coles** — Modified Poisson model with team attack/defense strengths, time-decay weighting, and low-score correction
- **xG Dixon-Coles** — Same architecture but fitted on expected goals (Understat) instead of actual goals
- **Elo** — Dynamic power ratings updated after each match with goal-difference multiplier

Model weights are determined by rolling accuracy over the last 10 predictions (softmax-scaled).

### Edge Detection

For each upcoming match, the ensemble probability is compared against bookmaker implied probability. An edge of 5%+ is flagged as a value bet.

## Setup

```bash
# Clone and install
git clone https://github.com/yourusername/sloplocks.git
cd sloplocks
python -m venv venv
source venv/bin/activate
pip install -r pipeline/requirements.txt

# Set API keys
export FOOTBALL_DATA_API_KEY="your-key"
export ODDS_API_KEY="your-key"

# Run pipeline
python -m pipeline.run

# Run tests
pytest tests/ -v
```

## API Keys

- **football-data.org** — Free tier (10 requests/minute). [Get a key](https://www.football-data.org/client/register)
- **The Odds API** — Free tier (500 requests/month). [Get a key](https://the-odds-api.com/)

Set these as GitHub Secrets (`FOOTBALL_DATA_API_KEY`, `ODDS_API_KEY`) for the daily Action.

## Data Sources

- [football-data.org](https://www.football-data.org/) — Match results and fixtures
- [Understat](https://understat.com/) — Expected goals (xG)
- [The Odds API](https://the-odds-api.com/) — Bookmaker odds

## License

MIT
