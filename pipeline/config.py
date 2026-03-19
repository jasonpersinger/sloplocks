"""Central configuration for the SLOP LOCKS pipeline."""

import os

# API Keys (from environment / GitHub Secrets)
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
BALLDONTLIE_API_KEY = os.environ.get("BALLDONTLIE_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# The Odds API
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_REGIONS = "us"
ODDS_MARKETS = "h2h"

# balldontlie.io
BALLDONTLIE_BASE = "https://api.balldontlie.io/v1"

# ESPN (no API key needed)
NCAAM_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
NBA_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
MLB_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"
MMA_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc"

# Model parameters
TIME_DECAY_RATE = 0.005
FORM_WINDOW = 6
FORM_WEIGHT_MULTIPLIER = 2.0
CONGESTION_THRESHOLD_DAYS = 4
CONGESTION_PENALTY = 0.05
ELO_K_FACTOR = 20
ELO_HOME_ADVANTAGE = 65
NBA_B2B_PENALTY = 30  # Elo points subtracted for back-to-back game
VALUE_EDGE_THRESHOLD = 0.05
SLOP_LOCK_MIN_ODDS = -150          # American odds lower bound for Slop Locks
SLOP_LOCK_MAX_ODDS = 195           # American odds upper bound for Slop Locks
SLOP_LOCK_FALLBACK_MIN_ODDS = -350 # Hard floor for fallback picks outside the preferred window

# Slimegrinder: Conservative bankroll building
SLIMEGRINDER_MIN_ODDS = -250
SLIMEGRINDER_MAX_ODDS = 165

ENSEMBLE_ACCURACY_WINDOW = 10
MAX_GOALS = 6

# Output paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "predictions.json")
HISTORY_PATH = os.path.join(DATA_DIR, "history.json")
ACCURACY_PATH = os.path.join(DATA_DIR, "model_accuracy.json")

# Per-sport configuration
SPORTS = {
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
    "mlb": {
        "name": "MLB",
        "display_name": "MLB",
        "odds_sport": "baseball_mlb",
        "outcomes": ["home", "away"],
        "models": ["elo"],
        "elo_k_factor": 4, # Lower K for 162 game season
        "elo_home_advantage": 24, # Standard MLB home edge
        "data_dir": os.path.join(DATA_DIR, "mlb"),
    },
    "mma": {
        "name": "MMA",
        "display_name": "MMA",
        "odds_sport": "mma_mixed_martial_arts", # Standard Odds API key
        "outcomes": ["home", "away"], # Home = Favorite/First, Away = Underdog/Second
        "models": ["elo"],
        "elo_k_factor": 32,
        "elo_home_advantage": 0, # No home advantage in MMA generally
        "data_dir": os.path.join(DATA_DIR, "mma"),
    },
}
