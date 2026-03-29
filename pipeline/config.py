"""Central configuration for the SLOP LOCKS pipeline."""

import os

TRACKING_DIRNAME = "tracking"
RESULTS_LOG_FILENAME = "results_log.csv"

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
MLB_CORE_API_BASE = "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb"
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"

# MLB park factors are coarse run-environment multipliers keyed by ESPN short
# team names. 1.00 is neutral; above 1.00 boosts offense slightly.
MLB_PARK_FACTORS = {
    "Angels": 1.01,
    "Astros": 0.98,
    "Athletics": 1.03,
    "Blue Jays": 1.02,
    "Braves": 1.01,
    "Brewers": 1.00,
    "Cardinals": 0.96,
    "Cubs": 1.04,
    "D-backs": 1.05,
    "Dodgers": 0.97,
    "Giants": 0.93,
    "Guardians": 0.99,
    "Mariners": 0.95,
    "Marlins": 0.94,
    "Mets": 0.97,
    "Nationals": 1.01,
    "Orioles": 1.02,
    "Padres": 0.98,
    "Phillies": 1.03,
    "Pirates": 0.98,
    "Rangers": 1.04,
    "Rays": 0.97,
    "Red Sox": 1.05,
    "Reds": 1.08,
    "Rockies": 1.12,
    "Royals": 1.00,
    "Tigers": 0.96,
    "Twins": 1.01,
    "White Sox": 1.03,
    "Yankees": 1.02,
}

MLB_BALLPARKS = {
    "Angels": {"latitude": 33.8003, "longitude": -117.8827, "weather_exposed": True},
    "Astros": {"latitude": 29.7573, "longitude": -95.3555, "weather_exposed": False},
    "Athletics": {"latitude": 38.5804, "longitude": -121.5139, "weather_exposed": True},
    "Blue Jays": {"latitude": 43.6414, "longitude": -79.3894, "weather_exposed": False},
    "Braves": {"latitude": 33.8907, "longitude": -84.4677, "weather_exposed": True},
    "Brewers": {"latitude": 43.0280, "longitude": -87.9712, "weather_exposed": False},
    "Cardinals": {"latitude": 38.6226, "longitude": -90.1928, "weather_exposed": True},
    "Cubs": {"latitude": 41.9484, "longitude": -87.6553, "weather_exposed": True},
    "D-backs": {"latitude": 33.4453, "longitude": -112.0667, "weather_exposed": False},
    "Dodgers": {"latitude": 34.0739, "longitude": -118.2400, "weather_exposed": True},
    "Giants": {"latitude": 37.7786, "longitude": -122.3893, "weather_exposed": True},
    "Guardians": {"latitude": 41.4962, "longitude": -81.6852, "weather_exposed": True},
    "Mariners": {"latitude": 47.5914, "longitude": -122.3326, "weather_exposed": False},
    "Marlins": {"latitude": 25.7781, "longitude": -80.2197, "weather_exposed": False},
    "Mets": {"latitude": 40.7571, "longitude": -73.8458, "weather_exposed": True},
    "Nationals": {"latitude": 38.8730, "longitude": -77.0074, "weather_exposed": True},
    "Orioles": {"latitude": 39.2838, "longitude": -76.6217, "weather_exposed": True},
    "Padres": {"latitude": 32.7073, "longitude": -117.1573, "weather_exposed": True},
    "Phillies": {"latitude": 39.9061, "longitude": -75.1665, "weather_exposed": True},
    "Pirates": {"latitude": 40.4469, "longitude": -80.0057, "weather_exposed": True},
    "Rangers": {"latitude": 32.7513, "longitude": -97.0825, "weather_exposed": False},
    "Rays": {"latitude": 27.7682, "longitude": -82.6534, "weather_exposed": False},
    "Red Sox": {"latitude": 42.3467, "longitude": -71.0972, "weather_exposed": True},
    "Reds": {"latitude": 39.0979, "longitude": -84.5083, "weather_exposed": True},
    "Rockies": {"latitude": 39.7562, "longitude": -104.9942, "weather_exposed": True},
    "Royals": {"latitude": 39.0517, "longitude": -94.4803, "weather_exposed": True},
    "Tigers": {"latitude": 42.3390, "longitude": -83.0485, "weather_exposed": True},
    "Twins": {"latitude": 44.9817, "longitude": -93.2776, "weather_exposed": True},
    "White Sox": {"latitude": 41.8300, "longitude": -87.6338, "weather_exposed": True},
    "Yankees": {"latitude": 40.8296, "longitude": -73.9262, "weather_exposed": True},
}

# Model parameters
TIME_DECAY_RATE = 0.005
FORM_WINDOW = 6
FORM_WEIGHT_MULTIPLIER = 2.0
CONGESTION_THRESHOLD_DAYS = 4
CONGESTION_PENALTY = 0.05
ELO_K_FACTOR = 20
ELO_HOME_ADVANTAGE = 65
NBA_B2B_PENALTY = 30  # Elo points subtracted for back-to-back game
NBA_3IN4_PENALTY = 15  # Elo points subtracted for 3 games in 4 nights
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

# Phase 4: Calibration & Market Respect
MARKET_RESPECT_FACTOR = 0.4  # 0.4 means 40% weight to market, 60% to model
MAX_ALLOWED_DIVERGENCE = 0.20 # If model is >20% from books, it's flagged as unrealistic
RTM_WINDOW = 20              # Games required before reducing shrinkage

# Per-sport configuration
SPORTS = {
    "nba": {
        "name": "NBA",
        "display_name": "NBA",
        "odds_sport": "basketball_nba",
        "outcomes": ["home", "away"],
        "models": ["elo", "efficiency", "four_factors", "results_features", "recent_boxscore"],
        "accuracy_softmax_temperature": 3.0,
        "probability_calibration_min_samples": 30,
        "probability_calibration_blend": 0.5,
        "elo_k_factor": 20,
        "elo_home_advantage": 65,
        "efficiency_home_bonus": 3.5,
        "results_feature_window": 8,
        "results_feature_min_games": 30,
        "recent_boxscore_window": 8,
        "recent_boxscore_min_games": 30,
        "recent_form_window": 6,
        "recent_form_max_adjustment": 35,
        "min_expected_value": 0.0,
        "kelly_fraction": 0.25,
        "slop_lock_edge_threshold": 0.02,
        "slop_lock_probability_floor": 0.50,
        "slop_lock_confidence_threshold": 52,
        "slop_lock_confidence_dropoff": 8,
        "slop_lock_max_picks": 5,
        "longslop_confidence_threshold": 55,
        "slimegrinder_confidence_threshold": 55,
        "back_to_back_penalty": NBA_B2B_PENALTY,
        "fatigue_window_days": 4,
        "fatigue_threshold_games": 3,
        "fatigue_penalty": NBA_3IN4_PENALTY,
        "rest_bonus_days": 3,
        "rest_bonus_points": 8,
        "accuracy_window": 40,
        "data_dir": os.path.join(DATA_DIR, "nba"),
    },
    "ncaam": {
        "name": "NCAAM",
        "display_name": "NCAAM",
        "odds_sport": "basketball_ncaab",
        "outcomes": ["home", "away"],
        "models": ["elo", "efficiency", "four_factors", "results_features", "recent_boxscore"],
        "accuracy_softmax_temperature": 3.0,
        "probability_calibration_min_samples": 30,
        "probability_calibration_blend": 0.5,
        "elo_k_factor": 32,
        "elo_home_advantage": 125,
        "efficiency_home_bonus": 3.5,
        "results_feature_window": 10,
        "results_feature_min_games": 40,
        "recent_boxscore_window": 10,
        "recent_boxscore_min_games": 40,
        "recent_form_window": 8,
        "recent_form_max_adjustment": 25,
        "min_expected_value": 0.0,
        "kelly_fraction": 0.25,
        "slop_lock_edge_threshold": 0.02,
        "slop_lock_probability_floor": 0.50,
        "slop_lock_confidence_threshold": 52,
        "slop_lock_confidence_dropoff": 8,
        "slop_lock_max_picks": 5,
        "longslop_confidence_threshold": 55,
        "slimegrinder_confidence_threshold": 55,
        "back_to_back_penalty": 12,
        "fatigue_window_days": 4,
        "fatigue_threshold_games": 3,
        "fatigue_penalty": 6,
        "rest_bonus_days": 4,
        "rest_bonus_points": 5,
        "accuracy_window": 40,
        "data_dir": os.path.join(DATA_DIR, "ncaam"),
    },
    "mlb": {
        "name": "MLB",
        "display_name": "MLB",
        "odds_sport": "baseball_mlb",
        "outcomes": ["home", "away"],
        "models": ["elo", "results_features", "pitcher_features", "bullpen_features", "run_environment", "handedness_features"],
        "accuracy_softmax_temperature": 1.5,
        "probability_calibration_min_samples": 20,
        "probability_calibration_blend": 0.35,
        "elo_k_factor": 4, # Lower K for 162 game season
        "elo_home_advantage": 24, # Standard MLB home edge
        "pitcher_feature_window": 8,
        "pitcher_feature_min_games": 20,
        "bullpen_feature_window": 12,
        "bullpen_recent_usage_window": 5,
        "bullpen_feature_min_games": 20,
        "handedness_feature_window": 18,
        "handedness_feature_min_games": 20,
        "run_environment_window": 12,
        "run_environment_min_games": 20,
        "weather_adjustment_max_delta": 0.02,
        "lineup_adjustment_max_delta": 0.015,
        "lineup_total_adjustment_max_delta": 0.35,
        "totals_feature_window": 12,
        "totals_feature_min_games": 20,
        "totals_min_expected_value": 0.0,
        "totals_edge_threshold": 0.02,
        "totals_probability_floor": 0.53,
        "totals_confidence_threshold": 54,
        "totals_max_picks": 3,
        "totals_default_stddev": 3.1,
        "results_feature_window": 12,
        "results_feature_min_games": 50,
        "recent_form_window": 10,
        "recent_form_max_adjustment": 16,
        "min_expected_value": 0.0,
        "kelly_fraction": 0.2,
        "slop_lock_edge_threshold": 0.015,
        "slop_lock_probability_floor": 0.52,
        "slop_lock_confidence_threshold": 48,
        "slop_lock_confidence_dropoff": 6,
        "slop_lock_max_picks": 4,
        "longslop_confidence_threshold": 50,
        "slimegrinder_confidence_threshold": 50,
        "back_to_back_penalty": 0,
        "fatigue_window_days": 0,
        "fatigue_threshold_games": 0,
        "fatigue_penalty": 0,
        "rest_bonus_days": 0,
        "rest_bonus_points": 0,
        "accuracy_window": 80,
        "season_start_month": 2, # Feb 20 — captures spring training as Elo warmup
        "season_start_day": 20,
        "data_dir": os.path.join(DATA_DIR, "mlb"),
    },
    "mma": {
        "name": "MMA",
        "display_name": "MMA",
        "odds_sport": "mma_mixed_martial_arts", # Standard Odds API key
        "outcomes": ["home", "away"], # Home = Favorite/First, Away = Underdog/Second
        "models": ["elo", "results_features"],
        "accuracy_softmax_temperature": 1.5,
        "probability_calibration_min_samples": 12,
        "probability_calibration_blend": 0.35,
        "elo_k_factor": 32,
        "elo_home_advantage": 0, # No home advantage in MMA generally
        "results_feature_window": 5,
        "results_feature_min_games": 12,
        "recent_form_window": 4,
        "recent_form_max_adjustment": 20,
        "min_expected_value": 0.0,
        "kelly_fraction": 0.15,
        "slop_lock_edge_threshold": 0.02,
        "slop_lock_probability_floor": 0.50,
        "slop_lock_confidence_threshold": 50,
        "slop_lock_confidence_dropoff": 8,
        "slop_lock_max_picks": 4,
        "longslop_confidence_threshold": 52,
        "slimegrinder_confidence_threshold": 52,
        "back_to_back_penalty": 0,
        "fatigue_window_days": 0,
        "fatigue_threshold_games": 0,
        "fatigue_penalty": 0,
        "rest_bonus_days": 0,
        "rest_bonus_points": 0,
        "accuracy_window": 20,
        "data_dir": os.path.join(DATA_DIR, "mma"),
    },
}
