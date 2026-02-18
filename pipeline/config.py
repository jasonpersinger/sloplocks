"""Central configuration for the SLOP LOCKS pipeline."""

import os

# API Keys (from environment / GitHub Secrets)
FOOTBALL_DATA_API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "")
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
BALLDONTLIE_API_KEY = os.environ.get("BALLDONTLIE_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# football-data.org
FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"
EPL_COMPETITION_ID = "PL"

# Understat
UNDERSTAT_BASE = "https://understat.com"
CURRENT_SEASON = "2025"  # Understat uses start year of season

# The Odds API
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_SPORT = "soccer_epl"
ODDS_REGIONS = "us"
ODDS_MARKETS = "h2h"

# balldontlie.io
BALLDONTLIE_BASE = "https://api.balldontlie.io/v1"

# ESPN (no API key needed)
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
NBA_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
NBA_B2B_PENALTY = 30  # Elo points subtracted for back-to-back game

# Model parameters
TIME_DECAY_RATE = 0.005
FORM_WINDOW = 6
FORM_WEIGHT_MULTIPLIER = 2.0
CONGESTION_THRESHOLD_DAYS = 4
CONGESTION_PENALTY = 0.05
ELO_K_FACTOR = 20
ELO_HOME_ADVANTAGE = 65
VALUE_EDGE_THRESHOLD = 0.05
ENSEMBLE_ACCURACY_WINDOW = 10
MAX_GOALS = 6

# Output paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "predictions.json")
HISTORY_PATH = os.path.join(DATA_DIR, "history.json")
ACCURACY_PATH = os.path.join(DATA_DIR, "model_accuracy.json")

# Per-sport configuration
SPORTS = {
    "epl": {
        "name": "EPL",
        "display_name": "Premier League",
        "odds_sport": "soccer_epl",
        "outcomes": ["home", "draw", "away"],
        "models": ["dixon_coles", "xg", "elo"],
        "elo_k_factor": 20,
        "elo_home_advantage": 65,
        "data_dir": os.path.join(DATA_DIR, "epl"),
    },
    "nba": {
        "name": "NBA",
        "display_name": "NBA",
        "odds_sport": "basketball_nba",
        "outcomes": ["home", "away"],
        "models": ["elo", "efficiency", "four_factors"],
        "elo_k_factor": 20,
        "elo_home_advantage": 65,
        "efficiency_home_bonus": 3.5,
        "data_dir": os.path.join(DATA_DIR, "nba"),
    },
    "ncaam": {
        "name": "NCAAM",
        "display_name": "NCAAM",
        "odds_sport": "basketball_ncaab",
        "outcomes": ["home", "away"],
        "models": ["elo", "efficiency", "four_factors"],
        "elo_k_factor": 32,
        "elo_home_advantage": 125,
        "efficiency_home_bonus": 3.5,
        "data_dir": os.path.join(DATA_DIR, "ncaam"),
    },
}
