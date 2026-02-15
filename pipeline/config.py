"""Central configuration for the SLOP LOCKS pipeline."""

import os

# API Keys (from environment / GitHub Secrets)
FOOTBALL_DATA_API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "")
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")

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
