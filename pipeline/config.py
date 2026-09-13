"""Central configuration for the SLOP LOCKS pipeline."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv(override=True)

TRACKING_DIRNAME = "tracking"
RESULTS_LOG_FILENAME = "results_log.csv"
RESULTS_AUDIT_LOG_FILENAME = "results_audit_log.csv"
ODDS_HISTORY_FILENAME = "odds_history.csv"
PICK_DECISION_LOG_FILENAME = "pick_decisions.csv"
PUBLIC_RECORD_ARCHIVE_DIRNAME = "public_record_archives"

# API Keys (from environment / GitHub Secrets)
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
BALLDONTLIE_API_KEY = os.environ.get("BALLDONTLIE_API_KEY", "")

# The Odds API
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_REGIONS = "us"
ODDS_MARKETS = "h2h"

# balldontlie.io
BALLDONTLIE_BASE = "https://api.balldontlie.io/v1"

# ESPN (no API key needed)
NCAAM_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
NBA_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
WNBA_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba"
MLB_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"
NHL_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl"
NFL_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
NCAAF_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"
NCAAF_CORE_API_BASE = "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football"
MLB_CORE_API_BASE = "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb"
MLB_STATS_API_BASE = "https://statsapi.mlb.com/api/v1"
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
SLOP_LOCK_MIN_ODDS = -220          # American odds lower bound for Slop Locks
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
MARKET_RESPECT_FACTOR = 0.3  # 0.3 means 30% weight to market, 70% to model
MAX_ALLOWED_DIVERGENCE = 0.20 # If model is >20% from books, it's flagged as unrealistic
RTM_WINDOW = 20              # Games required before reducing shrinkage

# Phase 5: Qualitative Gemini Integration
ENABLE_QUALITATIVE = os.environ.get("ENABLE_QUALITATIVE", "false").lower() == "true"
QUALITATIVE_DEFAULT_WEIGHT = 0.005 # Default probability delta per impact point

# Sports that retain historical data/code but are intentionally not active in
# live pipeline runs or public notification surfaces.
SEASON_DISABLED_SPORTS = {
    "ncaam": {
        "name": "NCAAM",
        "display_name": "NCAAM",
        "status": "season_disabled",
        "reason": "NCAAM is out of season; historical data remains available, but live picks are disabled.",
        "data_dir": os.path.join(DATA_DIR, "ncaam"),
    },
}

# Per-sport active live configuration
SPORTS = {
    "nba": {
        "name": "NBA",
        "display_name": "NBA",
        "odds_sport": "basketball_nba",
        "totals_enabled": True,
        "outcomes": ["home", "away"],
        "models": ["elo", "four_factors", "results_features", "recent_boxscore", "nba_matchup"],
        "disabled_models": ["four_factors"],
        "accuracy_softmax_temperature": 3.0,
        "probability_calibration_min_samples": 30,
        "probability_calibration_blend": 0.5,
        "probability_calibration_window_days": 180,
        "probability_calibration_holdout_days": 7,
        "elo_k_factor": 20,
        "elo_home_advantage": 65,
        "efficiency_home_bonus": 3.5,
        "results_feature_window": 8,
        "results_feature_min_games": 30,
        "recent_boxscore_window": 8,
        "recent_boxscore_min_games": 30,
        "nba_matchup_window": 8,
        "nba_matchup_min_games": 30,
        "availability_adjustment_max_delta": 0.02,
        "availability_uncertainty_weight": 0.35,
        "availability_leader_uncertainty_weight": 0.35,
        "availability_tipoff_partial_hours": 12,
        "availability_tipoff_full_hours": 2,
        "availability_total_adjustment_max_points": 2.2,
        "qualitative_total_adjustment_max_points": 2.5,
        "totals_feature_window": 8,
        "totals_feature_min_games": 30,
        "totals_min_expected_value": 0.0,
        "totals_edge_threshold": 0.02,
        "totals_probability_floor": 0.53,
        "totals_confidence_threshold": 54,
        "totals_max_picks": 0,
        "totals_default_stddev": 13.5,
        "recent_form_window": 6,
        "recent_form_max_adjustment": 35,
        "min_expected_value": 0.0,
        "kelly_fraction": 0.25,
        "slop_lock_edge_threshold": 0.02,
        "slop_lock_probability_floor": 0.50,
        "slop_lock_confidence_threshold": 52,
        "slop_lock_confidence_dropoff": 8,
        "slop_lock_max_picks": 5,
        "moneyline_hold_max_picks": 1,
        "moneyline_hold_min_health_score": 0.7,
        "totals_hold_max_picks": 1,
        "totals_hold_min_health_score": 0.75,
        "slop_lock_lanes": {
            "value_dog": {
                "enabled": True,
                "edge_floor": 0.04,
                "probability_floor": 0.35,
                "min_expected_value": 0.05,
                "american_odds_min": 120,
                "american_odds_max": 500,
                "max_picks": 2,
            },
            "near_favorite": {
                "enabled": True,
                "edge_floor": 0.01,
                "probability_floor": 0.52,
                "min_expected_value": 0.0,
                "american_odds_min": -180,
                "american_odds_max": 120,
                "max_picks": 3,
            },
        },
        "publication_min_evaluated_picks": 20,
        "publication_min_evaluated_totals_picks": 10,
        "moneyline_health_recent_window": 8,
        "moneyline_health_min_recent_evaluated": 5,
        "moneyline_health_min_recent_roi": 0.0,
        "moneyline_health_max_overconfidence_gap": 0.12,
        "moneyline_clv_guard_window": 5,
        "moneyline_clv_guard_min_tracked": 3,
        "moneyline_clv_guard_min_avg": -0.005,
        "totals_health_recent_window": 8,
        "totals_health_min_recent_evaluated": 5,
        "totals_health_min_recent_roi": 0.0,
        "totals_health_max_overconfidence_gap": 0.1,
        "totals_clv_guard_window": 5,
        "totals_clv_guard_min_tracked": 5,
        "totals_clv_guard_min_avg": 0.0,
        "enable_longslop": False,
        "enable_slimegrinder": False,
        "longslop_confidence_threshold": 55,
        "slimegrinder_confidence_threshold": 55,
        "back_to_back_penalty": NBA_B2B_PENALTY,
        "fatigue_window_days": 4,
        "fatigue_threshold_games": 3,
        "fatigue_penalty": NBA_3IN4_PENALTY,
        "rest_bonus_days": 3,
        "rest_bonus_points": 8,
        "accuracy_window": 40,
        "qualitative_weight": 0.5,
        "enable_qualitative": True,
        "data_dir": os.path.join(DATA_DIR, "nba"),
    },
    "nhl": {
        "name": "NHL",
        "display_name": "NHL",
        "odds_sport": "icehockey_nhl",
        "outcomes": ["home", "away"],
        "models": ["elo", "results_features", "nhl_matchup"],
        "disabled_models": [],
        "accuracy_softmax_temperature": 2.5,
        "probability_calibration_min_samples": 20,
        "probability_calibration_blend": 0.4,
        "probability_calibration_window_days": 180,
        "probability_calibration_holdout_days": 7,
        "elo_k_factor": 18,
        "elo_home_advantage": 28,
        "results_feature_window": 10,
        "results_feature_min_games": 40,
        "nhl_matchup_window": 10,
        "nhl_matchup_min_games": 40,
        "recent_form_window": 8,
        "recent_form_max_adjustment": 24,
        "injury_adjustment_max_delta": 0.01,
        "goalie_status_adjustment_max_delta": 0.012,
        "min_expected_value": 0.0,
        "kelly_fraction": 0.2,
        "slop_lock_edge_threshold": 0.015,
        "slop_lock_probability_floor": 0.5,
        "slop_lock_confidence_threshold": 50,
        "slop_lock_confidence_dropoff": 7,
        "slop_lock_max_picks": 5,
        "moneyline_hold_max_picks": 1,
        "moneyline_hold_min_health_score": 0.7,
        "slop_lock_lanes": {
            "value_dog": {
                "enabled": True,
                "edge_floor": 0.04,
                "probability_floor": 0.35,
                "min_expected_value": 0.05,
                "american_odds_min": 120,
                "american_odds_max": 500,
                "max_picks": 2,
            },
            "near_favorite": {
                "enabled": True,
                "edge_floor": 0.01,
                "probability_floor": 0.52,
                "min_expected_value": 0.0,
                "american_odds_min": -180,
                "american_odds_max": 120,
                "max_picks": 3,
            },
        },
        "publication_min_evaluated_picks": 20,
        "moneyline_health_recent_window": 8,
        "moneyline_health_min_recent_evaluated": 5,
        "moneyline_health_min_recent_roi": -0.15,
        "moneyline_health_max_overconfidence_gap": 0.15,
        "moneyline_clv_guard_window": 5,
        "moneyline_clv_guard_min_tracked": 3,
        "moneyline_clv_guard_min_avg": -0.025,
        "enable_longslop": False,
        "enable_slimegrinder": False,
        "longslop_confidence_threshold": 54,
        "slimegrinder_confidence_threshold": 52,
        "back_to_back_penalty": 14,
        "fatigue_window_days": 4,
        "fatigue_threshold_games": 3,
        "fatigue_penalty": 8,
        "rest_bonus_days": 3,
        "rest_bonus_points": 6,
        "accuracy_window": 40,
        "qualitative_weight": 0.4,
        "enable_qualitative": True,
        "data_dir": os.path.join(DATA_DIR, "nhl"),
    },
    "wnba": {
        "name": "WNBA",
        "display_name": "WNBA",
        "odds_sport": "basketball_wnba",
        "outcomes": ["home", "away"],
        "models": ["elo", "results_features", "recent_boxscore"],
        "disabled_models": [],
        "accuracy_softmax_temperature": 2.5,
        "probability_calibration_min_samples": 20,
        "probability_calibration_blend": 0.4,
        "probability_calibration_window_days": 150,
        "probability_calibration_holdout_days": 7,
        "elo_k_factor": 20,
        "elo_home_advantage": 50,
        "results_feature_window": 8,
        "results_feature_min_games": 20,
        "recent_boxscore_window": 8,
        "recent_boxscore_min_games": 20,
        "availability_adjustment_max_delta": 0.02,
        "availability_uncertainty_weight": 0.35,
        "availability_leader_uncertainty_weight": 0.35,
        "availability_tipoff_partial_hours": 12,
        "availability_tipoff_full_hours": 2,
        "totals_max_picks": 0,
        "publication_min_evaluated_totals_picks": 0,
        "recent_form_window": 6,
        "recent_form_max_adjustment": 30,
        "min_expected_value": 0.0,
        "kelly_fraction": 0.2,
        "slop_lock_edge_threshold": 0.02,
        "slop_lock_probability_floor": 0.52,
        "slop_lock_confidence_threshold": 52,
        "slop_lock_confidence_dropoff": 7,
        "slop_lock_max_picks": 3,
        "moneyline_hold_max_picks": 1,
        "moneyline_hold_min_health_score": 0.7,
        "slop_lock_lanes": {
            "value_dog": {
                "enabled": True,
                "edge_floor": 0.04,
                "probability_floor": 0.35,
                "min_expected_value": 0.05,
                "american_odds_min": 120,
                "american_odds_max": 500,
                "max_picks": 1,
            },
            "near_favorite": {
                "enabled": True,
                "edge_floor": 0.01,
                "probability_floor": 0.52,
                "min_expected_value": 0.0,
                "american_odds_min": -180,
                "american_odds_max": 120,
                "max_picks": 2,
            },
        },
        "publication_min_evaluated_picks": 15,
        "moneyline_health_recent_window": 8,
        "moneyline_health_min_recent_evaluated": 4,
        "moneyline_health_min_recent_roi": -0.10,
        "moneyline_health_max_overconfidence_gap": 0.15,
        "moneyline_clv_guard_window": 5,
        "moneyline_clv_guard_min_tracked": 3,
        "moneyline_clv_guard_min_avg": -0.005,
        "enable_longslop": False,
        "enable_slimegrinder": False,
        "longslop_confidence_threshold": 54,
        "slimegrinder_confidence_threshold": 52,
        "back_to_back_penalty": 20,
        "fatigue_window_days": 3,
        "fatigue_threshold_games": 3,
        "fatigue_penalty": 10,
        "rest_bonus_days": 3,
        "rest_bonus_points": 6,
        "accuracy_window": 30,
        "qualitative_weight": 0.4,
        "enable_qualitative": True,
        "data_dir": os.path.join(DATA_DIR, "wnba"),
    },
    "mlb": {
        "name": "MLB",
        "display_name": "MLB",
        "odds_sport": "baseball_mlb",
        "totals_enabled": True,
        "outcomes": ["home", "away"],
        "models": ["elo", "results_features", "pitcher_features", "bullpen_features", "run_environment", "handedness_features"],
        "disabled_models": ["pitcher_features"],
        "accuracy_softmax_temperature": 1.5,
        "probability_calibration_min_samples": 20,
        "probability_calibration_blend": 0.35,
        "probability_calibration_window_days": 180,
        "probability_calibration_holdout_days": 7,
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
        "pitcher_cache_ttl_minutes": 120,
        "pitcher_force_refresh_hours": 4,
        "pitcher_tbd_warning_hours": 2,
        "bullpen_fatigue_window_days": 3,
        "bullpen_fatigue_usage_baseline": 6.5,
        "bullpen_last_game_usage_baseline": 3.5,
        "bullpen_availability_adjustment_max_delta": 0.012,
        "bullpen_total_adjustment_max_delta": 0.3,
        "qualitative_total_adjustment_max_points": 0.5,
        "totals_feature_window": 12,
        "totals_feature_min_games": 20,
        "totals_min_expected_value": 0.0,
        "totals_edge_threshold": 0.02,
        "totals_probability_floor": 0.53,
        "totals_confidence_threshold": 54,
        "totals_max_picks": 3,
        "totals_default_stddev": 3.1,
        # Regression runs hot+flat vs market lines; recenter each slate's
        # projections on the market mean so only relative signal creates edges.
        "totals_recenter_to_market": True,
        "totals_recenter_min_slate": 5,
        "results_feature_window": 12,
        "results_feature_min_games": 50,
        "recent_form_window": 10,
        "recent_form_max_adjustment": 16,
        "min_expected_value": 0.0,
        "kelly_fraction": 0.2,
        "slop_lock_edge_threshold": 0.025,
        "slop_lock_probability_floor": 0.55,
        "slop_lock_confidence_threshold": 56,
        "slop_lock_confidence_dropoff": 4,
        "slop_lock_max_picks": 5,
        "moneyline_hold_max_picks": 1,
        "moneyline_hold_min_health_score": 0.7,
        "totals_hold_max_picks": 2,
        "totals_hold_min_health_score": 0.75,
        "slop_lock_lanes": {
            "value_dog": {
                "enabled": True,
                "edge_floor": 0.04,
                "probability_floor": 0.35,
                "min_expected_value": 0.05,
                "american_odds_min": 120,
                "american_odds_max": 500,
                "max_picks": 2,
            },
            "near_favorite": {
                "enabled": True,
                "edge_floor": 0.01,
                "probability_floor": 0.52,
                "min_expected_value": 0.0,
                "american_odds_min": -180,
                "american_odds_max": 120,
                "max_picks": 3,
            },
        },
        "publication_min_evaluated_picks": 20,
        "publication_min_evaluated_totals_picks": 10,
        "moneyline_health_recent_window": 8,
        "moneyline_health_min_recent_evaluated": 5,
        "moneyline_health_min_recent_roi": -0.25,
        "moneyline_health_max_overconfidence_gap": 0.25,
        "moneyline_clv_guard_window": 8,
        "moneyline_clv_guard_min_tracked": 5,
        "moneyline_clv_guard_min_avg": -0.02,
        "totals_health_recent_window": 8,
        "totals_health_min_recent_evaluated": 5,
        "totals_health_min_recent_roi": -0.10,
        "totals_health_max_overconfidence_gap": 0.1,
        "totals_clv_guard_window": 8,
        "totals_clv_guard_min_tracked": 5,
        "totals_clv_guard_min_avg": -0.02,
        "enable_longslop": False,
        "enable_slimegrinder": False,
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
        "qualitative_weight": 0.3,
        "enable_qualitative": True,
        "data_dir": os.path.join(DATA_DIR, "mlb"),
    },
    "nfl": {
        "name": "NFL",
        "display_name": "NFL",
        "odds_sport": "americanfootball_nfl",
        # Publish a moneyline pick for every game, not just Slop Locks.
        "publish_full_slate": True,
        "outcomes": ["home", "away"],
        "models": ["elo", "results_features"],
        "disabled_models": [],
        "accuracy_softmax_temperature": 2.5,
        "probability_calibration_min_samples": 20,
        "probability_calibration_blend": 0.4,
        "probability_calibration_window_days": 240,
        "probability_calibration_holdout_days": 7,
        # 17-game season: K stays moderate, and margins are divided into
        # touchdown-sized units before the Elo goal-diff multiplier so a
        # 3-score win does not swing ratings like a 3-goal hockey win.
        "elo_k_factor": 20,
        "elo_home_advantage": 48,
        "elo_margin_divisor": 7.0,
        # Three seasons of history give Elo a real week-1 prior; without it
        # every team sits at 1500 until roughly week 5 of a 17-game season.
        "history_seasons": 3,
        # Keep two thirds of last season's rating, regress one third to the
        # mean. The draft and free agency are built for parity, so the NFL
        # reverts harder year over year than college does.
        "elo_season_carryover": 0.67,
        # Football is weekly: a 5-game window is ~1/3 of the season.
        "results_feature_window": 5,
        "results_feature_min_games": 30,
        # Bye weeks produce 13-14 days of rest, so the rest feature must not
        # be clamped at the 7-day default used by the daily-cadence sports.
        "results_feature_rest_cap_days": 14,
        "recent_form_window": 5,
        "recent_form_max_adjustment": 30,
        "min_expected_value": 0.0,
        "kelly_fraction": 0.2,
        "slop_lock_edge_threshold": 0.03,
        "slop_lock_probability_floor": 0.52,
        "slop_lock_confidence_threshold": 53,
        "slop_lock_confidence_dropoff": 7,
        "slop_lock_max_picks": 3,
        "moneyline_hold_max_picks": 1,
        "moneyline_hold_min_health_score": 0.7,
        # Totals run off FootballTotalsModel (final scores only).
        "totals_enabled": True,
        "totals_feature_window": 5,
        "totals_feature_min_games": 60,
        "totals_default_stddev": 10.5,
        "totals_projection_min": 20.0,
        "totals_projection_max": 80.0,
        "totals_min_expected_value": 0.0,
        "totals_edge_threshold": 0.03,
        "totals_probability_floor": 0.54,
        "totals_confidence_threshold": 55,
        "totals_max_picks": 2,
        "totals_hold_max_picks": 1,
        "totals_hold_min_health_score": 0.75,
        "qualitative_total_adjustment_max_points": 2.0,
        "publication_min_evaluated_totals_picks": 10,
        "totals_health_recent_window": 8,
        "totals_health_min_recent_evaluated": 5,
        "totals_health_min_recent_roi": -0.15,
        "totals_health_max_overconfidence_gap": 0.15,
        "totals_clv_guard_window": 5,
        "totals_clv_guard_min_tracked": 3,
        "totals_clv_guard_min_avg": -0.025,
        "slop_lock_lanes": {
            "value_dog": {
                "enabled": True,
                "edge_floor": 0.04,
                "probability_floor": 0.35,
                "min_expected_value": 0.05,
                "american_odds_min": 120,
                "american_odds_max": 500,
                "max_picks": 1,
            },
            "near_favorite": {
                "enabled": True,
                "edge_floor": 0.015,
                "probability_floor": 0.52,
                "min_expected_value": 0.0,
                "american_odds_min": -180,
                "american_odds_max": 120,
                "max_picks": 2,
            },
        },
        "publication_min_evaluated_picks": 20,
        "moneyline_health_recent_window": 8,
        "moneyline_health_min_recent_evaluated": 5,
        "moneyline_health_min_recent_roi": -0.15,
        "moneyline_health_max_overconfidence_gap": 0.15,
        "moneyline_clv_guard_window": 5,
        "moneyline_clv_guard_min_tracked": 3,
        "moneyline_clv_guard_min_avg": -0.025,
        "enable_longslop": False,
        "enable_slimegrinder": False,
        "longslop_confidence_threshold": 55,
        "slimegrinder_confidence_threshold": 55,
        # Football never plays back-to-back days. Its fatigue case is the
        # short week (Thursday game on 4 days rest); its rest case is the bye.
        "back_to_back_penalty": 0,
        "fatigue_window_days": 0,
        "fatigue_threshold_games": 0,
        "fatigue_penalty": 0,
        "short_rest_days": 5,
        "short_rest_penalty": 15,
        "rest_bonus_days": 10,
        "rest_bonus_max_days": 21,
        "rest_bonus_points": 12,
        "accuracy_window": 40,
        "season_start_month": 8,
        "season_start_day": 1,
        "qualitative_weight": 0.4,
        "enable_qualitative": True,
        "data_dir": os.path.join(DATA_DIR, "nfl"),
    },
    "ncaaf": {
        "name": "NCAAF",
        "display_name": "NCAAF",
        "odds_sport": "americanfootball_ncaaf",
        "outcomes": ["home", "away"],
        "models": ["elo", "results_features"],
        "disabled_models": [],
        "accuracy_softmax_temperature": 2.5,
        "probability_calibration_min_samples": 40,
        "probability_calibration_blend": 0.4,
        "probability_calibration_window_days": 240,
        "probability_calibration_holdout_days": 7,
        # ~134 FBS teams with a far wider talent spread than the NFL, on a
        # 12-game regular season: a higher K is needed for ratings to separate
        # within one year, and the home edge is larger than the NFL's.
        "elo_k_factor": 26,
        "elo_home_advantage": 65,
        "elo_margin_divisor": 7.0,
        # Blowouts are routine in non-conference play, so cap the margin that
        # can feed the Elo multiplier to stop cupcake wins inflating ratings.
        "elo_margin_cap": 28,
        "history_seasons": 3,
        # College keeps more of its rating than the NFL: there is no draft or
        # salary cap, so program strength persists strongly year over year.
        "elo_season_carryover": 0.72,
        "results_feature_window": 5,
        "results_feature_min_games": 120,
        "results_feature_rest_cap_days": 14,
        "recent_form_window": 5,
        "recent_form_max_adjustment": 40,
        "min_expected_value": 0.0,
        "kelly_fraction": 0.15,
        # Wider spread of team strength means more lopsided prices; require a
        # larger edge than the NFL before a pick is eligible.
        "slop_lock_edge_threshold": 0.04,
        "slop_lock_probability_floor": 0.55,
        "slop_lock_confidence_threshold": 55,
        "slop_lock_confidence_dropoff": 7,
        "slop_lock_max_picks": 4,
        "moneyline_hold_max_picks": 1,
        "moneyline_hold_min_health_score": 0.7,
        # College scoring is higher and far more variable than the NFL, so the
        # projection band is wider and the residual sigma larger.
        "totals_enabled": True,
        "totals_feature_window": 5,
        "totals_feature_min_games": 200,
        "totals_default_stddev": 13.5,
        "totals_projection_min": 20.0,
        "totals_projection_max": 100.0,
        "totals_min_expected_value": 0.0,
        "totals_edge_threshold": 0.04,
        "totals_probability_floor": 0.56,
        "totals_confidence_threshold": 57,
        "totals_max_picks": 3,
        "totals_hold_max_picks": 1,
        "totals_hold_min_health_score": 0.75,
        "qualitative_total_adjustment_max_points": 2.5,
        "publication_min_evaluated_totals_picks": 12,
        "totals_health_recent_window": 10,
        "totals_health_min_recent_evaluated": 6,
        "totals_health_min_recent_roi": -0.15,
        "totals_health_max_overconfidence_gap": 0.15,
        "totals_clv_guard_window": 8,
        "totals_clv_guard_min_tracked": 5,
        "totals_clv_guard_min_avg": -0.025,
        "slop_lock_lanes": {
            "value_dog": {
                "enabled": True,
                "edge_floor": 0.05,
                "probability_floor": 0.35,
                "min_expected_value": 0.05,
                "american_odds_min": 120,
                "american_odds_max": 500,
                "max_picks": 2,
            },
            "near_favorite": {
                "enabled": True,
                "edge_floor": 0.02,
                "probability_floor": 0.55,
                "min_expected_value": 0.0,
                "american_odds_min": -180,
                "american_odds_max": 120,
                "max_picks": 2,
            },
        },
        "publication_min_evaluated_picks": 25,
        "moneyline_health_recent_window": 10,
        "moneyline_health_min_recent_evaluated": 6,
        "moneyline_health_min_recent_roi": -0.15,
        "moneyline_health_max_overconfidence_gap": 0.15,
        "moneyline_clv_guard_window": 8,
        "moneyline_clv_guard_min_tracked": 5,
        "moneyline_clv_guard_min_avg": -0.025,
        "enable_longslop": False,
        "enable_slimegrinder": False,
        "longslop_confidence_threshold": 57,
        "slimegrinder_confidence_threshold": 57,
        "back_to_back_penalty": 0,
        "fatigue_window_days": 0,
        "fatigue_threshold_games": 0,
        "fatigue_penalty": 0,
        # College bye weeks are common and scattered; short weeks are rare.
        "short_rest_days": 5,
        "short_rest_penalty": 10,
        "rest_bonus_days": 10,
        "rest_bonus_max_days": 21,
        "rest_bonus_points": 10,
        "accuracy_window": 80,
        "season_start_month": 8,
        "season_start_day": 1,
        # FBS only. FCS opponents are folded into a single synthetic team so
        # their results still inform FBS ratings without polluting the pool.
        "espn_groups": [80],
        "qualitative_weight": 0.35,
        "enable_qualitative": True,
        "data_dir": os.path.join(DATA_DIR, "ncaaf"),
    },
}
