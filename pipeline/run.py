"""Pipeline orchestrator for SLOP LOCKS.

Ties together data fetching, model fitting, ensemble blending, backtesting,
and JSON output into a single ``run_pipeline()`` entry point.
"""

import json
import os
import csv
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from scipy.stats import norm

from pipeline.config import (
    ANTHROPIC_API_KEY,
    DATA_DIR,
    NBA_B2B_PENALTY,
    NBA_3IN4_PENALTY,
    TRACKING_DIRNAME,
    RESULTS_LOG_FILENAME,
    ODDS_HISTORY_FILENAME,
    SLOP_LOCK_MIN_ODDS,
    SLOP_LOCK_MAX_ODDS,
    SLOP_LOCK_FALLBACK_MIN_ODDS,
    SLIMEGRINDER_MIN_ODDS,
    SLIMEGRINDER_MAX_ODDS,
    SPORTS,
)
from pipeline.fetch_data import fetch_odds
from pipeline.fetch_nba import fetch_nba_games, fetch_nba_schedule, normalize_nba_team_name, fetch_nba_espn_games, fetch_nba_espn_schedule
from pipeline.fetch_nhl import fetch_nhl_games, fetch_nhl_schedule, normalize_nhl_team_name
from pipeline.fetch_ncaam import fetch_ncaam_games, fetch_ncaam_schedule, normalize_ncaam_team_name
from pipeline.fetch_mlb import fetch_mlb_games, fetch_mlb_schedule, normalize_mlb_team_name
from pipeline.fetch_mma import fetch_mma_games, fetch_mma_schedule, normalize_mma_name
from pipeline.models import (
    AdjustedEfficiency,
    BullpenMatchupModel,
    EloRatings,
    FourFactorsModel,
    HandednessMatchupModel,
    MlbTotalsModel,
    NbaMatchupModel,
    NbaTotalsModel,
    NhlMatchupModel,
    PitcherMatchupModel,
    RecentBoxScoreModel,
    bullpen_matchup_predict,
    efficiency_predict,
    elo_predict,
    four_factors_predict,
    handedness_matchup_predict,
    mlb_totals_predict,
    nba_matchup_predict,
    nba_totals_predict,
    nhl_matchup_predict,
    ResultsFeatureModel,
    pitcher_matchup_predict,
    recent_boxscore_predict,
    RunEnvironmentModel,
    run_environment_predict,
    results_features_predict,
)
from pipeline.ensemble import (
    blend_predictions,
    compute_edges,
    compute_totals_edges,
    decimal_to_american,
    compute_confidence_stars,
    no_vig_probabilities,
)
from pipeline.ensemble import fit_probability_calibrators, apply_probability_calibration
from pipeline.backtest import (
    compute_model_weights,
    compute_roi,
    evaluate_prediction,
    get_rolling_accuracy,
    update_accuracy_log,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _load_json(path):
    """Load a JSON file, returning an empty dict if the file doesn't exist."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_json(path, data):
    """Write *data* to *path* as pretty-printed JSON, creating dirs as needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)


def _lookup_match_odds(odds_lookup: dict, sport_key: str, home_team: str, away_team: str) -> dict | None:
    """Return odds for a fixture, with MMA-specific reversed-order fallback."""
    match_odds = odds_lookup.get((home_team, away_team))
    if match_odds is not None:
        return match_odds
    if sport_key == "mma":
        return odds_lookup.get((away_team, home_team))
    return None


def _resolve_start_time(fixture: dict, match_odds: dict | None) -> str | None:
    """Return the best available start time for a fixture."""
    start_time = fixture.get("start_time")
    if start_time:
        return start_time
    if match_odds:
        return match_odds.get("commence_time")
    return None


_RESULTS_LOG_FIELDS = [
    "logged_at",
    "sport",
    "entry_type",
    "market_type",
    "home_team",
    "away_team",
    "match_date",
    "pick",
    "actual",
    "won",
    "push",
    "model_prob",
    "home_prob",
    "away_prob",
    "draw_prob",
    "implied_prob",
    "market_implied_prob",
    "edge",
    "expected_value",
    "american_odds",
    "decimal_odds",
    "total_line",
    "confidence_score",
    "kelly_fraction",
    "fractional_kelly",
    "closing_american_odds",
    "closing_decimal_odds",
    "closing_implied_prob",
    "closing_market_implied_prob",
    "closing_total_line",
    "closing_line_value",
]

_ODDS_HISTORY_FIELDS = [
    "logged_at",
    "sport",
    "market_type",
    "home_team",
    "away_team",
    "match_date",
    "start_time",
    "outcome",
    "total_line",
    "decimal_odds",
    "american_odds",
    "implied_prob",
    "market_implied_prob",
]


def _results_log_path(base_dir: str) -> str:
    """Return the CSV path for the persistent resolved-results log."""
    return os.path.join(base_dir, TRACKING_DIRNAME, RESULTS_LOG_FILENAME)


def _odds_history_path(base_dir: str) -> str:
    """Return the CSV path for tracked odds snapshots."""
    return os.path.join(base_dir, TRACKING_DIRNAME, ODDS_HISTORY_FILENAME)


def _results_log_key(row: dict) -> tuple:
    """Stable dedupe key for results-log rows."""
    return (
        row["sport"],
        row["entry_type"],
        row["home_team"],
        row["away_team"],
        row["match_date"],
        row["pick"],
    )


def _build_results_log_row(sport_key: str, entry_type: str, record: dict, match_date: str, actual: str) -> dict:
    """Normalize an evaluated prediction or pick into one CSV results-log row."""
    model_probs = record.get("model_probs", {})
    pick = record.get("pick", "")
    market_type = record.get("market_type", "moneyline")
    push = actual == "push" or bool(record.get("push"))
    return {
        "logged_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sport": sport_key,
        "entry_type": entry_type,
        "market_type": market_type,
        "home_team": record["home_team"],
        "away_team": record["away_team"],
        "match_date": match_date,
        "pick": pick,
        "actual": actual,
        "won": str((pick == actual) and not push).lower(),
        "push": str(push).lower(),
        "model_prob": record.get("model_prob"),
        "home_prob": model_probs.get("home"),
        "away_prob": model_probs.get("away"),
        "draw_prob": model_probs.get("draw"),
        "implied_prob": record.get("implied_prob"),
        "market_implied_prob": record.get("market_implied_prob"),
        "edge": record.get("edge"),
        "expected_value": record.get("expected_value"),
        "american_odds": record.get("american_odds"),
        "decimal_odds": record.get("decimal_odds"),
        "total_line": record.get("total_line"),
        "confidence_score": record.get("confidence_score"),
        "kelly_fraction": record.get("kelly_fraction"),
        "fractional_kelly": record.get("fractional_kelly"),
        "closing_american_odds": record.get("closing_american_odds"),
        "closing_decimal_odds": record.get("closing_decimal_odds"),
        "closing_implied_prob": record.get("closing_implied_prob"),
        "closing_market_implied_prob": record.get("closing_market_implied_prob"),
        "closing_total_line": record.get("closing_total_line"),
        "closing_line_value": record.get("closing_line_value"),
    }


def _append_results_log(path: str, rows: list[dict]) -> None:
    """Append deduped rows to the persistent results log."""
    if not rows:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing_keys = set()
    if os.path.exists(path):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_keys.add(_results_log_key(row))

    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_RESULTS_LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            key = _results_log_key(row)
            if key in existing_keys:
                continue
            writer.writerow({field: row.get(field, "") for field in _RESULTS_LOG_FIELDS})
            existing_keys.add(key)


def _odds_snapshot_key(row: dict) -> tuple:
    """Stable dedupe key for one odds snapshot state."""
    return (
        row["sport"],
        row["market_type"],
        row["home_team"],
        row["away_team"],
        row["match_date"],
        row["outcome"],
        str(row.get("total_line", "")),
        str(row.get("decimal_odds", "")),
    )


def _append_odds_snapshot_log(path: str, rows: list[dict]) -> None:
    """Append deduped odds snapshots for later CLV analysis."""
    if not rows:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing_keys = set()
    if os.path.exists(path):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_keys.add(_odds_snapshot_key(row))

    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_ODDS_HISTORY_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            key = _odds_snapshot_key(row)
            if key in existing_keys:
                continue
            writer.writerow({field: row.get(field, "") for field in _ODDS_HISTORY_FIELDS})
            existing_keys.add(key)


def _safe_float(value):
    """Convert CSV-ish scalar values to float when possible."""
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value):
    """Convert CSV-ish scalar values to int when possible."""
    if value in (None, "", "None"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _build_odds_snapshot_rows(sport_key: str, odds_list: list[dict]) -> list[dict]:
    """Normalize current odds into per-outcome snapshot rows."""
    logged_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []

    for odds in odds_list:
        match_date = str(odds.get("commence_time", ""))[:10]

        moneyline_decimals = {
            outcome: odds.get(f"{outcome}_odds", 0.0)
            for outcome in ("home", "away", "draw")
            if odds.get(f"{outcome}_odds", 0.0) and odds.get(f"{outcome}_odds", 0.0) > 0
        }
        _, fair_probs, raw_probs = None, None, None
        if moneyline_decimals:
            raw_probs, fair_probs, _ = no_vig_probabilities(moneyline_decimals)
            for outcome, decimal_odds in moneyline_decimals.items():
                rows.append({
                    "logged_at": logged_at,
                    "sport": sport_key,
                    "market_type": "moneyline",
                    "home_team": odds["home_team"],
                    "away_team": odds["away_team"],
                    "match_date": match_date,
                    "start_time": odds.get("commence_time"),
                    "outcome": outcome,
                    "total_line": "",
                    "decimal_odds": round(float(decimal_odds), 4),
                    "american_odds": decimal_to_american(float(decimal_odds)),
                    "implied_prob": round(float(fair_probs.get(outcome, 0.0)), 4),
                    "market_implied_prob": round(float(raw_probs.get(outcome, 0.0)), 4),
                })

        totals_decimals = {
            outcome: odds.get(f"{outcome}_odds", 0.0)
            for outcome in ("over", "under")
            if odds.get(f"{outcome}_odds", 0.0) and odds.get(f"{outcome}_odds", 0.0) > 0
        }
        if totals_decimals and odds.get("total_line") is not None:
            raw_probs, fair_probs, _ = no_vig_probabilities(totals_decimals)
            for outcome, decimal_odds in totals_decimals.items():
                rows.append({
                    "logged_at": logged_at,
                    "sport": sport_key,
                    "market_type": "total",
                    "home_team": odds["home_team"],
                    "away_team": odds["away_team"],
                    "match_date": match_date,
                    "start_time": odds.get("commence_time"),
                    "outcome": outcome,
                    "total_line": round(float(odds["total_line"]), 3),
                    "decimal_odds": round(float(decimal_odds), 4),
                    "american_odds": decimal_to_american(float(decimal_odds)),
                    "implied_prob": round(float(fair_probs.get(outcome, 0.0)), 4),
                    "market_implied_prob": round(float(raw_probs.get(outcome, 0.0)), 4),
                })

    return rows


def _load_latest_odds_snapshots(path: str, sport_key: str) -> dict[tuple, dict]:
    """Load the latest stored odds snapshot per market/outcome."""
    if not os.path.exists(path):
        return {}

    latest = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("sport") != sport_key:
                continue
            key = (
                row.get("market_type", "moneyline"),
                row.get("home_team"),
                row.get("away_team"),
                row.get("match_date"),
                row.get("outcome"),
            )
            current = latest.get(key)
            if current is None or row.get("logged_at", "") > current.get("logged_at", ""):
                latest[key] = row
    return latest


def _apply_latest_market_snapshots(picks: list[dict], snapshot_lookup: dict[tuple, dict]) -> None:
    """Annotate picks with the latest tracked odds snapshot for CLV analysis."""
    for pick in picks:
        market_type = pick.get("market_type", "moneyline")
        key = (
            market_type,
            pick.get("home_team"),
            pick.get("away_team"),
            str(pick.get("match_date") or pick.get("date") or "")[:10],
            pick.get("pick"),
        )
        snapshot = snapshot_lookup.get(key)
        if not snapshot:
            continue

        pick["closing_decimal_odds"] = _safe_float(snapshot.get("decimal_odds"))
        pick["closing_american_odds"] = _safe_int(snapshot.get("american_odds"))
        pick["closing_implied_prob"] = _safe_float(snapshot.get("implied_prob"))
        pick["closing_market_implied_prob"] = _safe_float(snapshot.get("market_implied_prob"))

        if market_type == "total":
            closing_total_line = _safe_float(snapshot.get("total_line"))
            pick["closing_total_line"] = closing_total_line
            opening_total_line = _safe_float(pick.get("total_line"))
            if closing_total_line is not None and opening_total_line is not None:
                if pick.get("pick") == "over":
                    pick["closing_line_value"] = round(closing_total_line - opening_total_line, 3)
                elif pick.get("pick") == "under":
                    pick["closing_line_value"] = round(opening_total_line - closing_total_line, 3)
        else:
            opening_prob = pick.get("market_implied_prob")
            if opening_prob is None:
                opening_prob = pick.get("implied_prob")
            opening_prob = _safe_float(opening_prob)
            closing_prob = _safe_float(snapshot.get("implied_prob"))
            if opening_prob is not None and closing_prob is not None:
                pick["closing_line_value"] = round(closing_prob - opening_prob, 4)



def _days_since_last_game(team: str, before_date: str, matches: pd.DataFrame) -> int | None:
    """Return days since team's most recent game strictly before before_date.

    Parameters
    ----------
    team : str
        Normalised team name.
    before_date : str
        ISO date string (YYYY-MM-DD) of the upcoming fixture.
    matches : pd.DataFrame
        Historical game results with a ``date`` column.

    Returns
    -------
    int or None
        Days since last game, or None if the team has no recorded games.
    """
    cutoff = pd.to_datetime(before_date[:10])
    team_mask = (matches["home_team"] == team) | (matches["away_team"] == team)
    team_games = matches[team_mask].copy()
    team_games["_dt"] = pd.to_datetime(team_games["date"])
    past_games = team_games[team_games["_dt"] < cutoff]

    if past_games.empty:
        return None

    last_game = past_games["_dt"].max()
    return (cutoff - last_game).days


def _games_in_window(team: str, before_date: str, matches: pd.DataFrame, days: int = 4) -> int:
    """Count how many games a team played in a rolling window before a date.

    Parameters
    ----------
    team : str
        Normalised team name.
    before_date : str
        ISO date string of the upcoming fixture.
    matches : pd.DataFrame
        Historical game results with a ``date`` column.
    days : int
        Window size in days (e.g. 4 for 3-in-4-nights detection).

    Returns
    -------
    int
        Number of games played in the window.
    """
    cutoff = pd.to_datetime(before_date[:10])
    window_start = cutoff - timedelta(days=days)
    team_mask = (matches["home_team"] == team) | (matches["away_team"] == team)
    team_games = matches[team_mask].copy()
    team_games["_dt"] = pd.to_datetime(team_games["date"])
    in_window = team_games[(team_games["_dt"] >= window_start) & (team_games["_dt"] < cutoff)]
    return len(in_window)


def _recent_form_adjustment(
    team: str,
    before_date: str,
    matches: pd.DataFrame,
    window: int = 6,
    max_adjustment: float = 0.0,
) -> float:
    """Return an Elo-point adjustment from a team's recent results.

    The adjustment is centered on a 0.500 results score:
    wins = 1.0, draws = 0.5, losses = 0.0.
    It scales down automatically when fewer than ``window`` prior games exist.
    """
    if matches is None or matches.empty or window <= 0 or max_adjustment <= 0:
        return 0.0

    cutoff = pd.to_datetime(before_date[:10])
    team_mask = (matches["home_team"] == team) | (matches["away_team"] == team)
    team_games = matches[team_mask].copy()
    if team_games.empty:
        return 0.0

    team_games["_dt"] = pd.to_datetime(team_games["date"])
    past_games = team_games[team_games["_dt"] < cutoff].sort_values("_dt", ascending=False).head(window)
    if past_games.empty:
        return 0.0

    scores = []
    for _, row in past_games.iterrows():
        if row["home_goals"] == row["away_goals"]:
            scores.append(0.5)
        elif row["home_team"] == team:
            scores.append(1.0 if row["home_goals"] > row["away_goals"] else 0.0)
        else:
            scores.append(1.0 if row["away_goals"] > row["home_goals"] else 0.0)

    form_score = sum(scores) / len(scores)
    sample_scale = min(1.0, len(scores) / float(window))
    return ((form_score - 0.5) * 2.0) * max_adjustment * sample_scale


def _rest_adjustment(team: str, before_date: str, matches: pd.DataFrame, sport: dict) -> float:
    """Return a sport-specific Elo-point rest/fatigue adjustment."""
    if matches is None or matches.empty:
        return 0.0

    adjustment = 0.0
    days_since = _days_since_last_game(team, before_date, matches)

    back_to_back_penalty = sport.get("back_to_back_penalty", 0.0)
    if days_since == 1 and back_to_back_penalty:
        adjustment -= back_to_back_penalty

    rest_bonus_days = sport.get("rest_bonus_days", 0)
    rest_bonus_points = sport.get("rest_bonus_points", 0.0)
    if (
        rest_bonus_days
        and rest_bonus_points
        and days_since is not None
        and days_since >= rest_bonus_days
    ):
        adjustment += rest_bonus_points

    fatigue_window_days = sport.get("fatigue_window_days", 0)
    fatigue_threshold_games = sport.get("fatigue_threshold_games", 0)
    fatigue_penalty = sport.get("fatigue_penalty", 0.0)
    if (
        fatigue_window_days
        and fatigue_threshold_games
        and fatigue_penalty
        and _games_in_window(team, before_date, matches, days=fatigue_window_days) >= fatigue_threshold_games
    ):
        adjustment -= fatigue_penalty

    return adjustment


def _normalize_two_way_probs(probs: dict[str, float]) -> dict[str, float]:
    """Renormalize two-way probabilities after a heuristic adjustment."""
    total = max(1e-9, probs.get("home", 0.0) + probs.get("away", 0.0))
    return {
        "home": probs.get("home", 0.0) / total,
        "away": probs.get("away", 0.0) / total,
    }


def _apply_mlb_weather_adjustment(
    blended: dict[str, float],
    run_environment_probs: dict[str, float] | None,
    weather: dict | None,
    max_delta: float = 0.02,
) -> dict[str, float]:
    """Apply a modest MLB weather adjustment using offense-environment context.

    Warm, windy outdoor conditions amplify run-environment edges. Cold or wet
    conditions compress them slightly. The adjustment is intentionally small so
    weather informs the projection without dominating pitcher and bullpen data.
    """
    if not run_environment_probs or not weather or not weather.get("weather_exposed"):
        return blended

    temperature_f = weather.get("temperature_f")
    wind_mph = weather.get("wind_mph")
    precip_probability = weather.get("precipitation_probability")
    if temperature_f is None or wind_mph is None:
        return blended

    warm_term = max(-1.0, min(1.0, (float(temperature_f) - 68.0) / 18.0))
    wind_term = max(-1.0, min(1.0, (float(wind_mph) - 10.0) / 15.0))
    precip_term = 0.0
    if precip_probability is not None:
        precip_term = max(0.0, min(1.0, (float(precip_probability) - 20.0) / 50.0))

    environment_index = max(-1.0, min(1.0, (0.55 * warm_term) + (0.35 * wind_term) - (0.45 * precip_term)))
    if abs(environment_index) < 0.05:
        return blended

    run_bias = float(run_environment_probs.get("home", 0.5)) - 0.5
    delta = max(-max_delta, min(max_delta, environment_index * run_bias * 0.8))
    if abs(delta) < 0.001:
        return blended

    adjusted = {
        "home": min(0.99, max(0.01, blended.get("home", 0.5) + delta)),
        "away": min(0.99, max(0.01, blended.get("away", 0.5) - delta)),
    }
    return _normalize_two_way_probs(adjusted)


def _apply_mlb_weather_total_adjustment(
    expected_total: float,
    weather: dict | None,
    max_runs_delta: float = 0.8,
) -> float:
    """Shift an MLB total modestly for outdoor weather conditions."""
    if not weather or not weather.get("weather_exposed"):
        return expected_total

    temperature_f = weather.get("temperature_f")
    wind_mph = weather.get("wind_mph")
    precip_probability = weather.get("precipitation_probability")
    if temperature_f is None or wind_mph is None:
        return expected_total

    warm_term = max(-1.0, min(1.0, (float(temperature_f) - 68.0) / 18.0))
    wind_term = max(-1.0, min(1.0, (float(wind_mph) - 10.0) / 15.0))
    precip_term = 0.0
    if precip_probability is not None:
        precip_term = max(0.0, min(1.0, (float(precip_probability) - 20.0) / 50.0))

    environment_index = max(-1.0, min(1.0, (0.55 * warm_term) + (0.35 * wind_term) - (0.45 * precip_term)))
    if abs(environment_index) < 0.05:
        return expected_total

    delta = max(-max_runs_delta, min(max_runs_delta, environment_index * max_runs_delta))
    return max(4.5, min(16.0, expected_total + delta))


def _compute_mlb_lineup_index(lineup_profile: dict | None, opposing_pitcher_hand: str | None) -> float:
    """Score a current MLB roster profile for the handedness of today's matchup."""
    if not lineup_profile:
        return 0.0

    lefty_share = float(lineup_profile.get("lefty_share", 0.0) or 0.0)
    righty_share = float(lineup_profile.get("righty_share", 0.0) or 0.0)
    switch_share = float(lineup_profile.get("switch_share", 0.0) or 0.0)
    active_hitters = float(lineup_profile.get("active_hitters", 0.0) or 0.0)
    available_hitters = float(lineup_profile.get("available_hitters", 0.0) or 0.0)
    injured_hitters = float(lineup_profile.get("injured_hitters", 0.0) or 0.0)

    pitcher_hand = (opposing_pitcher_hand or "R").upper()
    if pitcher_hand == "L":
        platoon_term = righty_share + (0.6 * switch_share) - 0.5
    else:
        platoon_term = lefty_share + (0.6 * switch_share) - 0.5

    depth_term = max(-1.0, min(1.0, (available_hitters - 11.0) / 3.0))
    availability_rate = 0.0
    if active_hitters > 0:
        availability_rate = max(-1.0, min(1.0, (available_hitters / active_hitters) - 1.0))
    injury_term = max(-1.0, min(1.0, injured_hitters / 4.0))

    return (0.65 * platoon_term) + (0.2 * depth_term) + (0.15 * availability_rate) - (0.2 * injury_term)


def _apply_mlb_lineup_adjustment(
    blended: dict[str, float],
    home_lineup_profile: dict | None,
    away_lineup_profile: dict | None,
    home_pitcher_hand: str | None,
    away_pitcher_hand: str | None,
    max_delta: float = 0.015,
) -> dict[str, float]:
    """Apply a small MLB side adjustment from current roster platoon/health context."""
    home_index = _compute_mlb_lineup_index(home_lineup_profile, away_pitcher_hand)
    away_index = _compute_mlb_lineup_index(away_lineup_profile, home_pitcher_hand)
    differential = home_index - away_index
    if abs(differential) < 0.05:
        return blended

    delta = max(-max_delta, min(max_delta, differential * max_delta * 0.85))
    if abs(delta) < 0.001:
        return blended

    adjusted = {
        "home": min(0.99, max(0.01, blended.get("home", 0.5) + delta)),
        "away": min(0.99, max(0.01, blended.get("away", 0.5) - delta)),
    }
    return _normalize_two_way_probs(adjusted)


def _apply_mlb_lineup_total_adjustment(
    expected_total: float,
    home_lineup_profile: dict | None,
    away_lineup_profile: dict | None,
    home_pitcher_hand: str | None,
    away_pitcher_hand: str | None,
    max_runs_delta: float = 0.35,
) -> float:
    """Shift an MLB total modestly for current platoon-friendly or depleted rosters."""
    home_index = _compute_mlb_lineup_index(home_lineup_profile, away_pitcher_hand)
    away_index = _compute_mlb_lineup_index(away_lineup_profile, home_pitcher_hand)
    combined = home_index + away_index
    if abs(combined) < 0.05:
        return expected_total

    delta = max(-max_runs_delta, min(max_runs_delta, combined * max_runs_delta * 0.7))
    return max(4.5, min(16.0, expected_total + delta))


def _compute_nba_availability_index(profile: dict | None) -> float:
    """Score a live NBA availability profile."""
    if not profile:
        return 0.0
    active_players = float(profile.get("active_players", 0.0) or 0.0)
    available_core_players = float(profile.get("available_core_players", 0.0) or 0.0)
    injury_burden = float(profile.get("injury_burden", 0.0) or 0.0)
    key_absence_score = float(profile.get("key_absence_score", 0.0) or 0.0)
    leader_absence_burden = float(profile.get("leader_absence_burden", 0.0) or 0.0)

    depth_term = max(-1.0, min(1.0, (available_core_players - 9.0) / 4.0))
    active_term = 0.0
    if active_players > 0:
        active_term = max(-1.0, min(1.0, (available_core_players / active_players) - 0.75))

    return (
        (0.35 * depth_term)
        + (0.25 * active_term)
        - (0.35 * injury_burden)
        - (0.45 * key_absence_score)
        - (0.7 * leader_absence_burden)
    )


def _apply_nba_availability_adjustment(
    blended: dict[str, float],
    home_profile: dict | None,
    away_profile: dict | None,
    max_delta: float = 0.02,
) -> dict[str, float]:
    """Apply a small live NBA availability adjustment from roster status."""
    differential = _compute_nba_availability_index(home_profile) - _compute_nba_availability_index(away_profile)
    if abs(differential) < 0.05:
        return blended

    delta = max(-max_delta, min(max_delta, differential * max_delta * 0.6))
    if abs(delta) < 0.001:
        return blended

    adjusted = {
        "home": min(0.99, max(0.01, blended.get("home", 0.5) + delta)),
        "away": min(0.99, max(0.01, blended.get("away", 0.5) - delta)),
    }
    return _normalize_two_way_probs(adjusted)


def _apply_nba_availability_total_adjustment(
    expected_total: float,
    home_profile: dict | None,
    away_profile: dict | None,
    max_points_delta: float = 2.2,
) -> float:
    """Adjust an NBA total modestly for current roster depletion."""
    if not home_profile and not away_profile:
        return expected_total

    def _offense_drag(profile: dict | None) -> float:
        if not profile:
            return 0.0
        leader_absence = float(profile.get("leader_absence_burden", 0.0) or 0.0)
        injury_burden = float(profile.get("injury_burden", 0.0) or 0.0)
        active_players = float(profile.get("active_players", 0.0) or 0.0)
        available_core = float(profile.get("available_core_players", 0.0) or 0.0)
        depth_gap = max(0.0, 10.0 - available_core)
        active_gap = 0.0
        if active_players > 0:
            active_gap = max(0.0, 0.75 - (available_core / active_players))
        return (0.65 * leader_absence) + (0.25 * injury_burden) + (0.2 * depth_gap) + (0.35 * active_gap)

    combined_drag = _offense_drag(home_profile) + _offense_drag(away_profile)
    if combined_drag <= 0.05:
        return expected_total

    delta = max(-max_points_delta, min(0.0, -combined_drag * max_points_delta * 0.45))
    return max(180.0, min(270.0, expected_total + delta))


# ---------------------------------------------------------------------------
# Blurb generation (via Claude API)
# ---------------------------------------------------------------------------

def _generate_blurbs(picks, pick_type="lock"):
    """Generate short analysis blurbs for picks using Claude.

    Parameters
    ----------
    picks : list[dict] or dict or None
        SLOP LOCK list or single LONGSLOP dict.
    pick_type : str
        "lock" or "longslop" — controls the prompt tone.

    Returns the picks with a "blurb" field added to each. Fails silently
    (blurb = "") if the API key is missing or the call fails.
    """
    if not ANTHROPIC_API_KEY:
        if isinstance(picks, list):
            for p in picks:
                p["blurb"] = ""
        elif picks:
            picks["blurb"] = ""
        return picks

    if picks is None:
        return None

    items = picks if isinstance(picks, list) else [picks]
    if not items:
        return picks

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    except Exception:
        for p in items:
            p["blurb"] = ""
        return picks

    for p in items:
        try:
            # Build individual model breakdown
            models_str = ""
            ind = p.get("individual_models", {})
            pick_outcome = p["pick"]
            for model_name, probs in ind.items():
                prob = probs.get(pick_outcome, 0)
                models_str += f"  {model_name}: {prob:.1%}\n"

            if pick_type == "lock":
                prompt = (
                    f"You are the analytics voice for SLOP LOCKS, a sports betting predictions site. "
                    f"Write exactly 1-2 sentences explaining why the model is confident in this pick. Be direct, "
                    f"confident, concise. No hedging. Reference the model probability and why this outcome is likely.\n\n"
                    f"Match: {p['home_team']} vs {p['away_team']}\n"
                    f"Pick: {pick_outcome}\n"
                    f"Model probability: {p['model_prob']:.1%}\n"
                    f"Books implied: {p['implied_prob']:.1%}\n"
                    f"Edge: {p['edge']:.1%}\n"
                    f"Odds: {p['american_odds']:+d}\n"
                    f"Individual models:\n{models_str}"
                )
            else:
                prompt = (
                    f"You are the analytics voice for SLOP LOCKS, a sports betting predictions site. "
                    f"Write exactly 1-2 sentences explaining why this longshot may hit. Be bold, "
                    f"intriguing, concise. This is a +500 or longer pick our model believes in.\n\n"
                    f"Match: {p['home_team']} vs {p['away_team']}\n"
                    f"Pick: {pick_outcome}\n"
                    f"Model probability: {p['model_prob']:.1%}\n"
                    f"Books implied: {p['implied_prob']:.1%}\n"
                    f"Edge: {p['edge']:.1%}\n"
                    f"Odds: {p['american_odds']:+d}\n"
                    f"Individual models:\n{models_str}"
                )

            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )
            p["blurb"] = resp.content[0].text.strip()
        except Exception:
            p["blurb"] = ""

    # Strip individual_models from final output (only needed for blurb gen)
    for p in items:
        p.pop("individual_models", None)

    return picks


def _exclude_opponent_conflicts(locks: list[dict]) -> list[dict]:
    """Remove picks where a picked team is also the losing side of another pick.

    If we pick Team A to beat Team B, but Team B is also picked in a different
    game, we drop the lower-edge pick. Both Brentford and Brighton appearing as
    locks when they play each other looks contradictory to users.
    """
    def picked(lock):
        return lock["home_team"] if lock["pick"] == "home" else lock["away_team"]

    def unpicked(lock):
        return lock["away_team"] if lock["pick"] == "home" else lock["home_team"]

    # Map: unpicked team name -> the lock where they are the losing side
    unpicked_map = {unpicked(l): l for l in locks}

    to_remove = set()
    for lock in locks:
        pt = picked(lock)
        if pt in unpicked_map and unpicked_map[pt] is not lock:
            # pt is picked here but is the loser in another lock — conflict
            other = unpicked_map[pt]
            if lock["edge"] <= other["edge"]:
                to_remove.add(id(lock))
            else:
                to_remove.add(id(other))

    return [l for l in locks if id(l) not in to_remove]


def _compute_slop_locks(
    prediction_records,
    outcomes,
    min_expected_value: float = 0.0,
    edge_floor: float = 0.03,
    probability_floor: float = 0.45,
    additional_confidence_floor: float = 65.0,
    confidence_dropoff: float = 0.0,
    max_picks: int = 3,
):
    """Extract SLOP LOCKS: High-confidence picks meeting Phase 3 criteria.

    The top eligible candidate is always included. Additional picks can qualify
    either by clearing an absolute confidence floor or by staying close enough
    to the slate's top score. This avoids emptying the card on slates where the
    market compresses all scores into a narrow band.
    """
    candidates = []
    for rec in prediction_records:
        if rec.get("completed"):
            continue
        edges = rec.get("edges", {})
        
        for outcome in outcomes:
            e = edges.get(outcome)
            if not e:
                continue
            
            conf = e.get("confidence_score", 0)
            edge = e.get("edge", 0)
            prob = e.get("model_prob", 0)
            ev = e.get("expected_value", 0.0)
            
            # Basic eligibility for any pick
            if edge >= edge_floor and prob >= probability_floor and ev >= min_expected_value:
                candidates.append({
                    "home_team": rec["home_team"],
                    "away_team": rec["away_team"],
                    "date": rec["date"],
                    "start_time": rec.get("start_time"),
                    "matchday": rec.get("matchday"),
                    "pick": outcome,
                    "model_prob": round(prob, 4),
                    "implied_prob": round(e["implied_prob"], 4),
                    "market_implied_prob": round(e.get("market_implied_prob", e["implied_prob"]), 4),
                    "edge": round(edge, 4),
                    "expected_value": round(ev, 4),
                    "american_odds": e["american_odds"],
                    "decimal_odds": e["decimal_odds"],
                    "kelly_fraction": round(e.get("kelly_fraction", 0.0), 4),
                    "fractional_kelly": round(e.get("fractional_kelly", 0.0), 4),
                    "confidence_score": conf,
                    "individual_models": rec.get("individual_models", {}),
                })

    # Sort all eligible candidates by confidence score
    candidates.sort(key=lambda x: (x["confidence_score"], x["edge"]), reverse=True)
    
    if not candidates:
        return []

    # 1. The Pick of the Day (Always the #1 eligible candidate)
    selected = [candidates[0]]
    top_confidence = candidates[0]["confidence_score"]
    
    # 2. Additional Locks
    for c in candidates[1:]:
        if len(selected) >= max_picks:
            break
        if (
            c["confidence_score"] >= additional_confidence_floor
            or c["confidence_score"] >= (top_confidence - confidence_dropoff)
        ):
            selected.append(c)

    return _exclude_opponent_conflicts(selected)



def _compute_longslop(
    prediction_records,
    outcomes,
    min_expected_value: float = 0.0,
    confidence_floor: float = 65.0,
):
    """Extract LONGSLOP: High-upside longshots (+500 or better) that pass Phase 3.

    Must meet the same 65+ confidence threshold as standard locks.
    """
    longslop_candidates = []
    for rec in prediction_records:
        if rec.get("completed"):
            continue
        edges = rec.get("edges", {})
        
        for outcome in outcomes:
            e = edges.get(outcome)
            if not e:
                continue
            
            american = e.get("american_odds", 0)
            if american < 500:
                continue
            
            # Phase 3 Thresholds
            conf = e.get("confidence_score", 0)
            edge = e.get("edge", 0)
            prob = e.get("model_prob", 0)
            ev = e.get("expected_value", 0.0)
            
            # For longslop, we require high confidence and positive edge
            if conf >= confidence_floor and edge >= 0 and ev >= min_expected_value:
                longslop_candidates.append({
                    "home_team": rec["home_team"],
                    "away_team": rec["away_team"],
                    "date": rec["date"],
                    "start_time": rec.get("start_time"),
                    "matchday": rec.get("matchday"),
                    "pick": outcome,
                    "model_prob": round(prob, 4),
                    "implied_prob": round(e["implied_prob"], 4),
                    "market_implied_prob": round(e.get("market_implied_prob", e["implied_prob"]), 4),
                    "edge": round(edge, 4),
                    "expected_value": round(ev, 4),
                    "american_odds": american,
                    "decimal_odds": e["decimal_odds"],
                    "kelly_fraction": round(e.get("kelly_fraction", 0.0), 4),
                    "fractional_kelly": round(e.get("fractional_kelly", 0.0), 4),
                    "confidence_score": conf,
                    "individual_models": rec.get("individual_models", {}),
                })

    longslop_candidates.sort(key=lambda x: (x["confidence_score"], x["edge"]), reverse=True)
    return longslop_candidates[0] if longslop_candidates else None


def _compute_totals_locks(
    total_records,
    min_expected_value: float = 0.0,
    edge_floor: float = 0.02,
    probability_floor: float = 0.53,
    confidence_floor: float = 54.0,
    max_picks: int = 3,
):
    """Extract curated MLB totals picks from over/under records."""
    candidates = []
    for rec in total_records:
        if rec.get("completed"):
            continue
        edges = rec.get("edges") or {}
        for outcome in ("over", "under"):
            edge_data = edges.get(outcome)
            if not edge_data:
                continue
            if (
                edge_data.get("edge", 0.0) >= edge_floor
                and edge_data.get("model_prob", 0.0) >= probability_floor
                and edge_data.get("expected_value", 0.0) >= min_expected_value
                and edge_data.get("confidence_score", 0.0) >= confidence_floor
            ):
                candidates.append({
                    "market_type": "total",
                    "home_team": rec["home_team"],
                    "away_team": rec["away_team"],
                    "date": rec["date"],
                    "start_time": rec.get("start_time"),
                    "pick": outcome,
                    "total_line": rec.get("total_line"),
                    "expected_total": rec.get("expected_total"),
                    "weather": rec.get("weather"),
                    "model_prob": round(edge_data["model_prob"], 4),
                    "implied_prob": round(edge_data["implied_prob"], 4),
                    "market_implied_prob": round(edge_data.get("market_implied_prob", edge_data["implied_prob"]), 4),
                    "edge": round(edge_data["edge"], 4),
                    "expected_value": round(edge_data["expected_value"], 4),
                    "american_odds": edge_data["american_odds"],
                    "decimal_odds": edge_data["decimal_odds"],
                    "kelly_fraction": round(edge_data.get("kelly_fraction", 0.0), 4),
                    "fractional_kelly": round(edge_data.get("fractional_kelly", 0.0), 4),
                    "confidence_score": edge_data.get("confidence_score", 0.0),
                })

    candidates.sort(
        key=lambda item: (
            item.get("confidence_score", 0.0),
            item.get("expected_value", 0.0),
            item.get("edge", 0.0),
        ),
        reverse=True,
    )
    return candidates[:max_picks]


def _compute_slimegrinder(
    prediction_records,
    outcomes,
    min_expected_value: float = 0.0,
    confidence_floor: float = 65.0,
):
    """Extract SLIMEGRINDER: High-confidence, likely winners (odds -250 to +165).
    
    Must meet the same 65+ confidence threshold as Locks.
    """
    candidates = []
    for rec in prediction_records:
        if rec.get("completed"):
            continue
        edges = rec.get("edges", {})

        for outcome in outcomes:
            e = edges.get(outcome)
            if not e:
                continue

            american = e.get("american_odds", 0)
            if not (SLIMEGRINDER_MIN_ODDS <= american <= SLIMEGRINDER_MAX_ODDS):
                continue

            conf = e.get("confidence_score", 0)
            edge = e.get("edge", 0)
            prob = e.get("model_prob", 0)
            ev = e.get("expected_value", 0.0)

            # Strict Phase 3 filter
            if conf >= confidence_floor and edge > 0 and ev >= min_expected_value:
                candidates.append({
                    "home_team": rec["home_team"],
                    "away_team": rec["away_team"],
                    "date": rec["date"],
                    "start_time": rec.get("start_time"),
                    "matchday": rec.get("matchday"),
                    "pick": outcome,
                    "model_prob": round(prob, 4),
                    "implied_prob": round(e["implied_prob"], 4),
                    "edge": round(edge, 4),
                    "expected_value": round(ev, 4),
                    "american_odds": american,
                    "decimal_odds": e["decimal_odds"],
                    "kelly_fraction": round(e.get("kelly_fraction", 0.0), 4),
                    "fractional_kelly": round(e.get("fractional_kelly", 0.0), 4),
                    "confidence_score": conf,
                })

    # Sort by raw model probability (descending) to find the likeliest winners
    candidates.sort(key=lambda x: x["model_prob"], reverse=True)
    return _exclude_opponent_conflicts(candidates)[:3]


def _compute_pick_stats(picks):
    """Compute aggregate stats from evaluated picks.

    Returns a dict with stats broken out by pick type.
    """
    stats = {}
    for pick_type in ("slop_lock", "total_lock", "longslop", "all"):
        subset = [p for p in picks if pick_type == "all" or p["type"] == pick_type]
        evaluated = [p for p in subset if p.get("evaluated")]
        wins = [p for p in evaluated if p.get("won")]
        pushes = [p for p in evaluated if p.get("push")]
        losses = [p for p in evaluated if not p.get("won") and not p.get("push")]

        # ROI: flat $100 bet per pick
        bets = []
        for p in evaluated:
            bets.append({
                "stake": 100.0,
                "odds": p.get("decimal_odds", 0),
                "won": p.get("won", False),
                "push": p.get("push", False),
            })
        roi = compute_roi(bets)

        stats[pick_type] = {
            "total": len(subset),
            "evaluated": len(evaluated),
            "wins": len(wins),
            "losses": len(losses),
            "pushes": len(pushes),
            "pending": len(subset) - len(evaluated),
            "hit_rate": round(len(wins) / max(len(evaluated), 1), 3) if evaluated else None,
            "roi": round(roi, 3) if evaluated else None,
        }

    return stats


def _match_key(record: dict) -> tuple[str, str, str]:
    """Return a stable key for one matchup record."""
    return (
        record["home_team"],
        record["away_team"],
        str(record.get("date", ""))[:10],
    )


def _build_pipeline_diagnostics(
    matches: pd.DataFrame,
    fixtures_fetched: list[dict],
    fixtures_in_window: list[dict],
    odds_list: list[dict],
    odds_lookup: dict,
    prediction_records: list[dict],
    outcomes: list[str],
    sport_key: str,
    sport: dict,
    slop_locks: list[dict],
    longslop: dict | None,
    slimegrinder: list[dict],
) -> dict:
    """Summarize why a slate did or did not produce picks."""
    min_expected_value = sport.get("min_expected_value", 0.0)
    edge_floor = sport.get("slop_lock_edge_threshold", 0.03)
    probability_floor = sport.get("slop_lock_probability_floor", 0.45)

    fixtures_with_odds = 0
    for fixture in fixtures_in_window:
        if _lookup_match_odds(odds_lookup, sport_key, fixture["home_team"], fixture["away_team"]):
            fixtures_with_odds += 1

    matches_with_market = 0
    matches_with_positive_ev = set()
    lock_eligible_matches = set()
    lock_eligible_outcomes = 0

    for record in prediction_records:
        edges = record.get("edges") or {}
        if edges:
            matches_with_market += 1

        has_positive_ev = False
        has_lock_eligible_outcome = False
        for outcome in outcomes:
            edge_data = edges.get(outcome)
            if not edge_data:
                continue
            edge_value = edge_data.get("edge", 0.0)
            expected_value = edge_data.get("expected_value", 0.0)
            model_prob = edge_data.get("model_prob", 0.0)

            if edge_value > 0 and expected_value >= min_expected_value:
                has_positive_ev = True
            if (
                edge_value >= edge_floor
                and model_prob >= probability_floor
                and expected_value >= min_expected_value
            ):
                has_lock_eligible_outcome = True
                lock_eligible_outcomes += 1

        key = _match_key(record)
        if has_positive_ev:
            matches_with_positive_ev.add(key)
        if has_lock_eligible_outcome:
            lock_eligible_matches.add(key)

    completed_fixtures = sum(1 for fixture in fixtures_in_window if fixture.get("completed"))
    diagnostics = {
        "historical_matches": int(len(matches) if matches is not None else 0),
        "fixtures_fetched": len(fixtures_fetched),
        "fixtures_in_window": len(fixtures_in_window),
        "fixtures_completed": completed_fixtures,
        "fixtures_pending": len(fixtures_in_window) - completed_fixtures,
        "odds_events_fetched": len(odds_list),
        "fixtures_with_odds": fixtures_with_odds,
        "fixtures_without_odds": len(fixtures_in_window) - fixtures_with_odds,
        "matches_modeled": len(prediction_records),
        "matches_with_market": matches_with_market,
        "matches_with_positive_ev": len(matches_with_positive_ev),
        "lock_eligible_matches": len(lock_eligible_matches),
        "lock_eligible_outcomes": lock_eligible_outcomes,
        "slop_locks_posted": len(slop_locks),
        "longslop_posted": 1 if longslop else 0,
        "slimegrinders_posted": len(slimegrinder),
    }
    diagnostics["summary"] = (
        f"modeled={diagnostics['matches_modeled']} | "
        f"odds={diagnostics['fixtures_with_odds']}/{diagnostics['fixtures_in_window']} | "
        f"+ev={diagnostics['matches_with_positive_ev']} | "
        f"eligible={diagnostics['lock_eligible_matches']} | "
        f"locks={diagnostics['slop_locks_posted']}"
    )
    return diagnostics


def _print_pipeline_diagnostics(sport_key: str, diagnostics: dict) -> None:
    """Emit a concise per-sport diagnostics line for CLI / Actions logs."""
    print(f"[{sport_key}] {diagnostics.get('summary', '')}")


# ---------------------------------------------------------------------------
# Per-sport pipeline
# ---------------------------------------------------------------------------

def run_sport_pipeline(sport_key, output_dir=None):
    """Run prediction pipeline for a single sport.

    Parameters
    ----------
    sport_key : str
        Key into ``SPORTS`` config dict (e.g. "nba", "ncaam").
    output_dir : str or None
        Override the sport's data directory.
    """
    sport = SPORTS[sport_key]
    sport_dir = output_dir or sport["data_dir"]
    outcomes = sport["outcomes"]

    predictions_path = os.path.join(sport_dir, "predictions.json")
    history_path = os.path.join(sport_dir, "history.json")
    accuracy_path = os.path.join(sport_dir, "model_accuracy.json")
    results_log_path = _results_log_path(os.path.dirname(sport_dir))
    odds_history_path = _odds_history_path(os.path.dirname(sport_dir))

    # ------------------------------------------------------------------
    # 1. Fetch data (sport-specific)
    # ------------------------------------------------------------------
    box_scores_df = None

    if sport_key == "nba":
        games_df, box_scores_df = fetch_nba_espn_games(
            cache_path=os.path.join(sport_dir, "espn_cache.json")
        )
        fixtures = fetch_nba_espn_schedule(
            cache_path=os.path.join(sport_dir, "espn_cache.json")
        )
        matches = games_df
    elif sport_key == "nhl":
        games_df, box_scores_df = fetch_nhl_games(
            cache_path=os.path.join(sport_dir, "espn_cache.json")
        )
        fixtures = fetch_nhl_schedule(
            cache_path=os.path.join(sport_dir, "espn_cache.json")
        )
        matches = games_df
    elif sport_key == "ncaam":
        games_df, box_scores_df = fetch_ncaam_games(
            cache_path=os.path.join(sport_dir, "espn_cache.json")
        )
        fixtures = fetch_ncaam_schedule()
        # Fallback: if scoreboard is empty, check cache for any games matching the allowed window
        if not fixtures and games_df is not None:
            today_utc = datetime.now(timezone.utc).date()
            allowed = {
                (today_utc - timedelta(days=1)).strftime("%Y-%m-%d"),
                today_utc.strftime("%Y-%m-%d"),
                (today_utc + timedelta(days=1)).strftime("%Y-%m-%d")
            }
            upcoming = games_df[games_df["date"].isin(allowed)]
            for _, row in upcoming.iterrows():
                fixtures.append({
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "date": row["date"],
                    "completed": row.get("completed", False)
                })
        matches = games_df
    elif sport_key == "mlb":
        games_df, box_scores_df = fetch_mlb_games(
            cache_path=os.path.join(sport_dir, "espn_cache.json")
        )
        fixtures = fetch_mlb_schedule(
            cache_path=os.path.join(sport_dir, "espn_cache.json")
        )
        matches = games_df
    elif sport_key == "mma":
        games_df, box_scores_df = fetch_mma_games(
            cache_path=os.path.join(sport_dir, "espn_cache.json")
        )
        fixtures = fetch_mma_schedule()
        matches = games_df
    else:
        raise ValueError(f"Unknown sport: {sport_key}")

    # Odds are generic — just pass the right sport key
    odds_list = []
    try:
        odds_list = fetch_odds(
            sport_key=sport["odds_sport"],
            include_totals=(sport_key in {"mlb", "nba"}),
        )
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 2. Fit models
    # ------------------------------------------------------------------
    # Elo ratings (with sport-specific parameters)
    elo = None
    if "elo" in sport["models"]:
        all_teams = []
        if matches is not None and not matches.empty:
            all_teams = sorted(
                set(matches["home_team"].unique()) | set(matches["away_team"].unique())
            )
        
        # Add teams from current fixtures to all_teams so Elo has them
        fixture_teams = set()
        for f in fixtures:
            fixture_teams.add(f["home_team"])
            fixture_teams.add(f["away_team"])
        
        all_teams = sorted(set(all_teams) | fixture_teams)

        elo = EloRatings(
            all_teams,
            k_factor=sport["elo_k_factor"],
            home_advantage=sport["elo_home_advantage"],
        )
        if matches is not None and not matches.empty:
            elo.process_season(matches)

    # Adjusted Efficiency model (NCAAM)
    efficiency_model = None
    if "efficiency" in sport["models"] and box_scores_df is not None:
        efficiency_model = AdjustedEfficiency(box_scores_df, matches)

    # Four Factors model (NCAAM)
    four_factors_model = None
    if "four_factors" in sport["models"] and box_scores_df is not None:
        four_factors_model = FourFactorsModel(box_scores_df, matches)

    # Results-feature logistic model (uses only historical game outcomes)
    results_feature_model = None
    if "results_features" in sport["models"] and matches is not None and not matches.empty:
        results_feature_model = ResultsFeatureModel(
            matches,
            feature_window=sport.get("results_feature_window", 8),
            min_games=sport.get("results_feature_min_games", 30),
        )

    recent_boxscore_model = None
    if "recent_boxscore" in sport["models"] and box_scores_df is not None and matches is not None:
        recent_boxscore_model = RecentBoxScoreModel(
            box_scores_df,
            matches,
            feature_window=sport.get("recent_boxscore_window", 8),
            min_games=sport.get("recent_boxscore_min_games", 30),
        )

    nba_matchup_model = None
    if sport_key == "nba" and "nba_matchup" in sport["models"] and box_scores_df is not None and matches is not None:
        nba_matchup_model = NbaMatchupModel(
            box_scores_df,
            matches,
            feature_window=sport.get("nba_matchup_window", 8),
            min_games=sport.get("nba_matchup_min_games", 30),
        )

    nhl_matchup_model = None
    if sport_key == "nhl" and "nhl_matchup" in sport["models"] and matches is not None and not matches.empty:
        nhl_matchup_model = NhlMatchupModel(
            matches,
            feature_window=sport.get("nhl_matchup_window", 10),
            min_games=sport.get("nhl_matchup_min_games", 40),
        )

    pitcher_feature_model = None
    if "pitcher_features" in sport["models"] and matches is not None and not matches.empty:
        pitcher_feature_model = PitcherMatchupModel(
            matches,
            feature_window=sport.get("pitcher_feature_window", 8),
            min_games=sport.get("pitcher_feature_min_games", 20),
        )

    bullpen_feature_model = None
    if "bullpen_features" in sport["models"] and matches is not None and not matches.empty:
        bullpen_feature_model = BullpenMatchupModel(
            matches,
            feature_window=sport.get("bullpen_feature_window", 12),
            recent_usage_window=sport.get("bullpen_recent_usage_window", 5),
            min_games=sport.get("bullpen_feature_min_games", 20),
        )

    run_environment_model = None
    if "run_environment" in sport["models"] and matches is not None and not matches.empty:
        run_environment_model = RunEnvironmentModel(
            matches,
            feature_window=sport.get("run_environment_window", 12),
            min_games=sport.get("run_environment_min_games", 20),
        )

    handedness_feature_model = None
    if "handedness_features" in sport["models"] and matches is not None and not matches.empty:
        handedness_feature_model = HandednessMatchupModel(
            matches,
            feature_window=sport.get("handedness_feature_window", 18),
            min_games=sport.get("handedness_feature_min_games", 20),
        )

    totals_model = None
    if sport_key == "mlb" and matches is not None and not matches.empty:
        totals_model = MlbTotalsModel(
            matches,
            feature_window=sport.get("totals_feature_window", 12),
            min_games=sport.get("totals_feature_min_games", 20),
            default_stddev=sport.get("totals_default_stddev", 3.1),
        )
    elif sport_key == "nba" and box_scores_df is not None and matches is not None and not matches.empty:
        totals_model = NbaTotalsModel(
            box_scores_df,
            matches,
            feature_window=sport.get("totals_feature_window", 8),
            min_games=sport.get("totals_feature_min_games", 30),
            default_stddev=sport.get("totals_default_stddev", 13.5),
        )

    # ------------------------------------------------------------------
    # 3. Load accuracy log and compute model weights
    # ------------------------------------------------------------------
    accuracy_log = _load_json(accuracy_path)
    if not isinstance(accuracy_log, dict):
        accuracy_log = {}

    # Build list of active models for this run
    model_names = []
    if elo is not None:
        model_names.append("elo")
    if efficiency_model is not None:
        model_names.append("efficiency")
    if four_factors_model is not None:
        model_names.append("four_factors")
    if results_feature_model is not None:
        model_names.append("results_features")
    if recent_boxscore_model is not None:
        model_names.append("recent_boxscore")
    if nba_matchup_model is not None:
        model_names.append("nba_matchup")
    if nhl_matchup_model is not None:
        model_names.append("nhl_matchup")
    if pitcher_feature_model is not None:
        model_names.append("pitcher_features")
    if bullpen_feature_model is not None:
        model_names.append("bullpen_features")
    if run_environment_model is not None:
        model_names.append("run_environment")
    if handedness_feature_model is not None:
        model_names.append("handedness_features")

    accuracy_window = sport.get("accuracy_window", None)
    accuracies = [get_rolling_accuracy(accuracy_log, name, window=accuracy_window) for name in model_names]
    weights = compute_model_weights(
        accuracies,
        temperature=sport.get("accuracy_softmax_temperature", 2.0),
    )
    model_weight_dict = dict(zip(model_names, weights))

    history = _load_json(history_path)
    if not isinstance(history, dict):
        history = {}
    past_predictions = history.get("predictions", [])
    probability_calibrators = fit_probability_calibrators(
        past_predictions,
        outcomes,
        min_samples=sport.get("probability_calibration_min_samples", 20),
    )

    # ------------------------------------------------------------------
    # 4. Normalize odds team names and build lookup
    # ------------------------------------------------------------------
    if sport_key == "nba":
        normalizer = normalize_nba_team_name
    elif sport_key == "ncaam":
        normalizer = normalize_ncaam_team_name
    elif sport_key == "mlb":
        normalizer = normalize_mlb_team_name
    elif sport_key == "mma":
        normalizer = normalize_mma_name
    elif sport_key == "nhl":
        normalizer = normalize_nhl_team_name
    else:
        normalizer = lambda x: x
    for o in odds_list:
        o["home_team"] = normalizer(o["home_team"])
        o["away_team"] = normalizer(o["away_team"])

    odds_lookup = {}
    for o in odds_list:
        key = (o["home_team"], o["away_team"])
        odds_lookup[key] = o
    _append_odds_snapshot_log(odds_history_path, _build_odds_snapshot_rows(sport_key, odds_list))
    latest_snapshot_lookup = _load_latest_odds_snapshots(odds_history_path, sport_key)

    # ------------------------------------------------------------------
    # 5. Predict each fixture
    # Generate predictions for games scheduled today or tomorrow (UTC).
    # This ensures evening ET games (which shift to the next UTC day) are included.
    # ------------------------------------------------------------------
    today_utc = datetime.now(timezone.utc).date()
    allowed_dates = {
        (today_utc - timedelta(days=1)).strftime("%Y-%m-%d"),
        today_utc.strftime("%Y-%m-%d"),
        (today_utc + timedelta(days=1)).strftime("%Y-%m-%d")
    }
    fixtures_fetched = list(fixtures)
    fixtures = [f for f in fixtures if str(f.get("date", ""))[:10] in allowed_dates]

    prediction_records = []
    totals_prediction_records = []

    for fix in fixtures:
        home = fix["home_team"]
        away = fix["away_team"]
        is_neutral = fix.get("neutral", False)

        individual_preds = []
        blend_weights = []
        individual_models = {}

        # Elo (all sports)
        if elo is not None:
            home_rest_adj = _rest_adjustment(home, fix["date"], matches, sport)
            away_rest_adj = _rest_adjustment(away, fix["date"], matches, sport)
            recent_form_window = sport.get("recent_form_window", 0)
            recent_form_max_adjustment = sport.get("recent_form_max_adjustment", 0.0)
            home_form_adj = _recent_form_adjustment(
                home, fix["date"], matches,
                window=recent_form_window,
                max_adjustment=recent_form_max_adjustment,
            )
            away_form_adj = _recent_form_adjustment(
                away, fix["date"], matches,
                window=recent_form_window,
                max_adjustment=recent_form_max_adjustment,
            )
            
            # Disable home advantage for neutral sites
            current_home_adv = 0.0 if is_neutral else elo.home_advantage
            
            elo_probs = elo_predict(elo, home, away, outcomes=outcomes,
                                    home_rest_adj=home_rest_adj + home_form_adj,
                                    away_rest_adj=away_rest_adj + away_form_adj,
                                    home_advantage_override=current_home_adv)
            individual_preds.append(elo_probs)
            blend_weights.append(model_weight_dict["elo"])
            individual_models["elo"] = elo_probs

        # Adjusted Efficiency (NCAAM)
        if efficiency_model is not None and home in efficiency_model.off_efficiency and away in efficiency_model.off_efficiency:
            # Disable home bonus for neutral sites
            current_home_bonus = 0.0 if is_neutral else sport.get("efficiency_home_bonus", 3.5)
            
            eff_probs = efficiency_predict(
                efficiency_model, home, away,
                home_bonus=current_home_bonus,
            )
            individual_preds.append(eff_probs)
            blend_weights.append(model_weight_dict["efficiency"])
            individual_models["efficiency"] = eff_probs

        # Four Factors (NCAAM)
        if four_factors_model is not None and four_factors_model.model is not None:
            if home in four_factors_model.team_stats and away in four_factors_model.team_stats:
                ff_probs = four_factors_predict(four_factors_model, home, away)
                individual_preds.append(ff_probs)
                blend_weights.append(model_weight_dict["four_factors"])
                individual_models["four_factors"] = ff_probs

        if results_feature_model is not None:
            rf_probs = results_features_predict(
                results_feature_model,
                home,
                away,
                neutral_site=is_neutral,
                game_date=fix["date"],
            )
            individual_preds.append(rf_probs)
            blend_weights.append(model_weight_dict["results_features"])
            individual_models["results_features"] = rf_probs

        if recent_boxscore_model is not None:
            recent_boxscore_probs = recent_boxscore_predict(
                recent_boxscore_model,
                home,
                away,
            )
            individual_preds.append(recent_boxscore_probs)
            blend_weights.append(model_weight_dict["recent_boxscore"])
            individual_models["recent_boxscore"] = recent_boxscore_probs

        if nba_matchup_model is not None:
            nba_matchup_probs = nba_matchup_predict(
                nba_matchup_model,
                home,
                away,
                game_date=fix.get("date"),
            )
            individual_preds.append(nba_matchup_probs)
            blend_weights.append(model_weight_dict["nba_matchup"])
            individual_models["nba_matchup"] = nba_matchup_probs

        if nhl_matchup_model is not None:
            nhl_matchup_probs = nhl_matchup_predict(
                nhl_matchup_model,
                home,
                away,
                game_date=fix.get("date"),
                home_goalie=fix.get("home_goalie"),
                away_goalie=fix.get("away_goalie"),
            )
            individual_preds.append(nhl_matchup_probs)
            blend_weights.append(model_weight_dict["nhl_matchup"])
            individual_models["nhl_matchup"] = nhl_matchup_probs

        if pitcher_feature_model is not None:
            pitcher_probs = pitcher_matchup_predict(
                pitcher_feature_model,
                fix.get("home_pitcher"),
                fix.get("away_pitcher"),
                game_date=fix.get("date"),
            )
            individual_preds.append(pitcher_probs)
            blend_weights.append(model_weight_dict["pitcher_features"])
            individual_models["pitcher_features"] = pitcher_probs

        if bullpen_feature_model is not None:
            bullpen_probs = bullpen_matchup_predict(
                bullpen_feature_model,
                home,
                away,
            )
            individual_preds.append(bullpen_probs)
            blend_weights.append(model_weight_dict["bullpen_features"])
            individual_models["bullpen_features"] = bullpen_probs

        if run_environment_model is not None:
            run_environment_probs = run_environment_predict(
                run_environment_model,
                home,
                away,
            )
            individual_preds.append(run_environment_probs)
            blend_weights.append(model_weight_dict["run_environment"])
            individual_models["run_environment"] = run_environment_probs
        else:
            run_environment_probs = None

        if handedness_feature_model is not None:
            handedness_probs = handedness_matchup_predict(
                handedness_feature_model,
                home,
                away,
                home_pitcher_hand=fix.get("home_pitcher_hand"),
                away_pitcher_hand=fix.get("away_pitcher_hand"),
            )
            individual_preds.append(handedness_probs)
            blend_weights.append(model_weight_dict["handedness_features"])
            individual_models["handedness_features"] = handedness_probs

        if not individual_preds:
            continue

        # Blend
        blended = blend_predictions(individual_preds, blend_weights)
        if sport_key == "mlb":
            blended = _apply_mlb_weather_adjustment(
                blended,
                run_environment_probs,
                fix.get("weather"),
                max_delta=sport.get("weather_adjustment_max_delta", 0.02),
            )
            blended = _apply_mlb_lineup_adjustment(
                blended,
                fix.get("home_lineup_profile"),
                fix.get("away_lineup_profile"),
                fix.get("home_pitcher_hand"),
                fix.get("away_pitcher_hand"),
                max_delta=sport.get("lineup_adjustment_max_delta", 0.015),
            )
        if sport_key == "nba":
            blended = _apply_nba_availability_adjustment(
                blended,
                fix.get("home_availability_profile"),
                fix.get("away_availability_profile"),
                max_delta=sport.get("availability_adjustment_max_delta", 0.02),
            )
        blended = apply_probability_calibration(
            blended,
            probability_calibrators,
            blend=sport.get("probability_calibration_blend", 0.5),
        )

        # Edges and best odds
        match_odds = _lookup_match_odds(odds_lookup, sport_key, home, away)
        edges = {}
        best_odds = {}
        if match_odds:
            # Pass individual_preds to enable Agreement Score
            edges = compute_edges(
                blended,
                match_odds,
                individual_probs=individual_preds,
                fractional_kelly=sport.get("kelly_fraction", 0.25),
            )
            best_odds = {}
            for out in outcomes:
                odds_key = f"{out}_odds"
                dec = match_odds.get(odds_key, 0)
                if dec > 0:
                    best_odds[out] = decimal_to_american(dec)

        # Determine the best pick and its confidence rating
        pick = max(blended.keys(), key=lambda k: blended[k])
        model_prob = blended[pick]
        edge_data = edges.get(pick, {})
        edge = edge_data.get("edge", 0.0)
        expected_value_value = edge_data.get("expected_value", 0.0)
        kelly_fraction_value = edge_data.get("kelly_fraction", 0.0)
        fractional_kelly_value = edge_data.get("fractional_kelly", 0.0)
        confidence_score = edge_data.get("confidence_score", 0)
        
        # Legacy stars for frontend compatibility
        stars = compute_confidence_stars(model_prob, edge)

        record = {
            "home_team": home,
            "away_team": away,
            "date": fix["date"],
            "start_time": _resolve_start_time(fix, match_odds),
            "matchday": fix.get("matchday"),
            "completed": fix.get("completed", False),
            "neutral": is_neutral,
            "home_availability_profile": fix.get("home_availability_profile"),
            "away_availability_profile": fix.get("away_availability_profile"),
            "home_goalie": fix.get("home_goalie"),
            "away_goalie": fix.get("away_goalie"),
            "home_goalie_status": fix.get("home_goalie_status"),
            "away_goalie_status": fix.get("away_goalie_status"),
            "home_pitcher": fix.get("home_pitcher"),
            "home_pitcher_hand": fix.get("home_pitcher_hand"),
            "away_pitcher": fix.get("away_pitcher"),
            "away_pitcher_hand": fix.get("away_pitcher_hand"),
            "home_lineup_profile": fix.get("home_lineup_profile"),
            "away_lineup_profile": fix.get("away_lineup_profile"),
            "weather": fix.get("weather"),
            "pick": pick,
            "model_prob": round(model_prob, 4),
            "edge": round(edge, 4),
            "expected_value": round(expected_value_value, 4),
            "kelly_fraction": round(kelly_fraction_value, 4),
            "fractional_kelly": round(fractional_kelly_value, 4),
            "confidence_score": confidence_score,
            "confidence_stars": stars,
            "american_odds": best_odds.get(pick),
            "model_probs": {k: round(v, 4) for k, v in blended.items()},
            "individual_models": {
                name: {k: round(v, 4) for k, v in probs.items()}
                for name, probs in individual_models.items()
            },
            "edges": edges,
            "best_odds": best_odds,
        }
        prediction_records.append(record)

        if totals_model is not None and match_odds and match_odds.get("total_line") is not None:
            if sport_key == "mlb":
                total_projection = mlb_totals_predict(
                    totals_model,
                    fix,
                    total_line=float(match_odds["total_line"]),
                )
                total_projection["expected_total"] = _apply_mlb_weather_total_adjustment(
                    total_projection["expected_total"],
                    fix.get("weather"),
                )
                total_projection["expected_total"] = _apply_mlb_lineup_total_adjustment(
                    total_projection["expected_total"],
                    fix.get("home_lineup_profile"),
                    fix.get("away_lineup_profile"),
                    fix.get("home_pitcher_hand"),
                    fix.get("away_pitcher_hand"),
                    max_runs_delta=sport.get("lineup_total_adjustment_max_delta", 0.35),
                )
            else:
                total_projection = nba_totals_predict(
                    totals_model,
                    fix,
                    total_line=float(match_odds["total_line"]),
                )
                total_projection["expected_total"] = _apply_nba_availability_total_adjustment(
                    total_projection["expected_total"],
                    fix.get("home_availability_profile"),
                    fix.get("away_availability_profile"),
                    max_points_delta=sport.get("availability_total_adjustment_max_points", 2.2),
                )
            sigma = max(1.5, float(total_projection.get("stddev", sport.get("totals_default_stddev", 3.1))))
            over_prob = float(
                1.0 - norm.cdf(
                    float(match_odds["total_line"]),
                    loc=total_projection["expected_total"],
                    scale=sigma,
                )
            )
            over_prob = max(0.01, min(0.99, over_prob))
            total_model_probs = {"over": over_prob, "under": 1.0 - over_prob}
            total_edges = compute_totals_edges(
                total_model_probs,
                match_odds,
                individual_probs=[total_model_probs],
                fractional_kelly=sport.get("kelly_fraction", 0.25),
            )
            total_pick = max(total_model_probs, key=total_model_probs.get)
            totals_prediction_records.append({
                "market_type": "total",
                "home_team": home,
                "away_team": away,
                "date": fix["date"],
                "start_time": _resolve_start_time(fix, match_odds),
                "completed": fix.get("completed", False),
                "home_pitcher": fix.get("home_pitcher"),
                "away_pitcher": fix.get("away_pitcher"),
                "home_lineup_profile": fix.get("home_lineup_profile"),
                "away_lineup_profile": fix.get("away_lineup_profile"),
                "weather": fix.get("weather"),
                "total_line": float(match_odds["total_line"]),
                "expected_total": round(total_projection["expected_total"], 3),
                "total_stddev": round(sigma, 3),
                "pick": total_pick,
                "model_prob": round(total_model_probs[total_pick], 4),
                "confidence_score": total_edges.get(total_pick, {}).get("confidence_score", 0.0),
                "american_odds": total_edges.get(total_pick, {}).get("american_odds"),
                "model_probs": {k: round(v, 4) for k, v in total_model_probs.items()},
                "individual_models": {"totals_model": {k: round(v, 4) for k, v in total_model_probs.items()}},
                "edges": total_edges,
            })

    # ------------------------------------------------------------------
    # 5b. SLOP LOCKS + LONGSLOP
    # ------------------------------------------------------------------
    min_expected_value = sport.get("min_expected_value", 0.0)
    slop_locks = _compute_slop_locks(
        prediction_records,
        outcomes,
        min_expected_value=min_expected_value,
        edge_floor=sport.get("slop_lock_edge_threshold", 0.03),
        probability_floor=sport.get("slop_lock_probability_floor", 0.45),
        additional_confidence_floor=sport.get("slop_lock_confidence_threshold", 65.0),
        confidence_dropoff=sport.get("slop_lock_confidence_dropoff", 0.0),
        max_picks=sport.get("slop_lock_max_picks", 3),
    )
    longslop = _compute_longslop(
        prediction_records,
        outcomes,
        min_expected_value=min_expected_value,
        confidence_floor=sport.get("longslop_confidence_threshold", 65.0),
    )
    slimegrinder = _compute_slimegrinder(
        prediction_records,
        outcomes,
        min_expected_value=min_expected_value,
        confidence_floor=sport.get("slimegrinder_confidence_threshold", 65.0),
    )
    totals_locks = _compute_totals_locks(
        totals_prediction_records,
        min_expected_value=sport.get("totals_min_expected_value", min_expected_value),
        edge_floor=sport.get("totals_edge_threshold", 0.02),
        probability_floor=sport.get("totals_probability_floor", 0.53),
        confidence_floor=sport.get("totals_confidence_threshold", 54.0),
        max_picks=sport.get("totals_max_picks", 3),
    ) if totals_prediction_records else []

    # Generate analysis blurbs via Claude
    slop_locks = _generate_blurbs(slop_locks, pick_type="lock")
    longslop = _generate_blurbs(longslop, pick_type="longslop")

    result_lookup = {}
    if matches is not None and not matches.empty:
        for _, row in matches.iterrows():
            date_str = str(row["date"])[:10]
            score = (int(row["home_goals"]), int(row["away_goals"]))
            result_lookup[(row["home_team"], row["away_team"], date_str)] = score
            # Also index under date+1 to handle UTC date shift for evening ET games
            next_date = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            result_lookup.setdefault((row["home_team"], row["away_team"], next_date), score)

    updated_past = []
    resolved_results_rows = []
    for pred in past_predictions:
        if pred.get("evaluated"):
            updated_past.append(pred)
            continue

        date_str = str(pred.get("date", ""))[:10]
        key = (pred["home_team"], pred["away_team"], date_str)
        result = result_lookup.get(key)

        if result is not None:
            hg, ag = result
            pred["evaluated"] = True
            pred["home_goals"] = hg
            pred["away_goals"] = ag

            for model_name in pred.get("individual_models", {}):
                model_probs = pred["individual_models"][model_name]
                eval_result = evaluate_prediction(model_probs, hg, ag)
                update_accuracy_log(accuracy_log, model_name, eval_result, window=accuracy_window)

            if "model_probs" in pred:
                eval_result = evaluate_prediction(pred["model_probs"], hg, ag)
                update_accuracy_log(accuracy_log, "ensemble", eval_result, window=accuracy_window)

            resolved_results_rows.append(
                _build_results_log_row(
                    sport_key,
                    "prediction",
                    pred,
                    date_str,
                    eval_result["actual"],
                )
            )

        updated_past.append(pred)

    # ------------------------------------------------------------------
    # 6b. Track and evaluate picks
    # ------------------------------------------------------------------
    pick_history_path = os.path.join(sport_dir, "pick_history.json")
    pick_history = _load_json(pick_history_path)
    if not isinstance(pick_history, dict):
        pick_history = {}
    past_picks = pick_history.get("picks", [])
    _apply_latest_market_snapshots(past_picks, latest_snapshot_lookup)

    # Evaluate unevaluated past picks against results
    for pick in past_picks:
        pick.setdefault("market_type", "moneyline")
        if pick.get("evaluated"):
            continue
        match_date = str(pick.get("match_date", ""))[:10]
        key = (pick["home_team"], pick["away_team"], match_date)
        result = result_lookup.get(key)
        if result is not None:
            hg, ag = result
            if pick.get("market_type") == "total":
                total_line = float(pick.get("total_line", 0.0))
                total_score = hg + ag
                if total_score > total_line:
                    actual = "over"
                elif total_score < total_line:
                    actual = "under"
                else:
                    actual = "push"
            else:
                if hg > ag:
                    actual = "home"
                elif hg == ag:
                    actual = "draw"
                else:
                    actual = "away"
            pick["evaluated"] = True
            pick["actual"] = actual
            pick["push"] = actual == "push"
            pick["won"] = pick["pick"] == actual and not pick["push"]
            pick["home_goals"] = hg
            pick["away_goals"] = ag
            resolved_results_rows.append(
                _build_results_log_row(
                    sport_key,
                    pick.get("type", "pick"),
                    pick,
                    match_date,
                    actual,
                )
            )

    # Append today's new picks (deduplicate by game identity, not pick_date)
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    existing_keys = set()
    for p in past_picks:
        existing_keys.add((p["type"], p["home_team"], p["away_team"],
                           p.get("match_date"), p["pick"]))

    for lock in slop_locks:
        pk = ("slop_lock", lock["home_team"], lock["away_team"],
              str(lock["date"])[:10], lock["pick"])
        if pk not in existing_keys:
            past_picks.append({
                "pick_date": today_str,
                "type": "slop_lock",
                "market_type": "moneyline",
                "home_team": lock["home_team"],
                "away_team": lock["away_team"],
                "match_date": str(lock["date"])[:10],
                "start_time": lock.get("start_time"),
                "pick": lock["pick"],
                "model_prob": lock["model_prob"],
                "implied_prob": lock["implied_prob"],
                "market_implied_prob": lock.get("market_implied_prob"),
                "edge": lock["edge"],
                "expected_value": lock.get("expected_value", 0.0),
                "american_odds": lock["american_odds"],
                "decimal_odds": lock["decimal_odds"],
                "confidence_score": lock.get("confidence_score"),
                "kelly_fraction": lock.get("kelly_fraction"),
                "fractional_kelly": lock.get("fractional_kelly"),
                "evaluated": False,
            })

    if longslop:
        pk = ("longslop", longslop["home_team"], longslop["away_team"],
              str(longslop["date"])[:10], longslop["pick"])
        if pk not in existing_keys:
            past_picks.append({
                "pick_date": today_str,
                "type": "longslop",
                "market_type": "moneyline",
                "home_team": longslop["home_team"],
                "away_team": longslop["away_team"],
                "match_date": str(longslop["date"])[:10],
                "start_time": longslop.get("start_time"),
                "pick": longslop["pick"],
                "model_prob": longslop["model_prob"],
                "implied_prob": longslop["implied_prob"],
                "market_implied_prob": longslop.get("market_implied_prob"),
                "edge": longslop["edge"],
                "expected_value": longslop.get("expected_value", 0.0),
                "american_odds": longslop["american_odds"],
                "decimal_odds": longslop["decimal_odds"],
                "confidence_score": longslop.get("confidence_score"),
                "kelly_fraction": longslop.get("kelly_fraction"),
                "fractional_kelly": longslop.get("fractional_kelly"),
                "evaluated": False,
            })

    for total_lock in totals_locks:
        pk = ("total_lock", total_lock["home_team"], total_lock["away_team"],
              str(total_lock["date"])[:10], total_lock["pick"])
        if pk not in existing_keys:
            past_picks.append({
                "pick_date": today_str,
                "type": "total_lock",
                "market_type": "total",
                "home_team": total_lock["home_team"],
                "away_team": total_lock["away_team"],
                "match_date": str(total_lock["date"])[:10],
                "start_time": total_lock.get("start_time"),
                "pick": total_lock["pick"],
                "total_line": total_lock.get("total_line"),
                "expected_total": total_lock.get("expected_total"),
                "model_prob": total_lock["model_prob"],
                "implied_prob": total_lock["implied_prob"],
                "market_implied_prob": total_lock.get("market_implied_prob"),
                "edge": total_lock["edge"],
                "expected_value": total_lock.get("expected_value", 0.0),
                "american_odds": total_lock["american_odds"],
                "decimal_odds": total_lock["decimal_odds"],
                "confidence_score": total_lock.get("confidence_score"),
                "kelly_fraction": total_lock.get("kelly_fraction"),
                "fractional_kelly": total_lock.get("fractional_kelly"),
                "evaluated": False,
            })

    _apply_latest_market_snapshots(past_picks, latest_snapshot_lookup)

    pick_history = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "picks": past_picks,
    }
    _save_json(pick_history_path, pick_history)
    _append_results_log(results_log_path, resolved_results_rows)

    # Compute pick stats for output
    pick_stats = _compute_pick_stats(past_picks)

    # ------------------------------------------------------------------
    # 7. Compute season stats
    # ------------------------------------------------------------------
    total_matches = len(matches)
    home_wins = int((matches["home_goals"] > matches["away_goals"]).sum())
    away_wins = int((matches["home_goals"] < matches["away_goals"]).sum())

    season_stats = {
        "total_matches": total_matches,
        "home_wins": home_wins,
        "away_wins": away_wins,
        "home_win_pct": round(home_wins / max(total_matches, 1), 3),
        "away_win_pct": round(away_wins / max(total_matches, 1), 3),
    }

    if "draw" in outcomes:
        draws = int((matches["home_goals"] == matches["away_goals"]).sum())
        season_stats["draws"] = draws
        season_stats["draw_pct"] = round(draws / max(total_matches, 1), 3)

    # ------------------------------------------------------------------
    # 8. Write output files
    # ------------------------------------------------------------------
    predictions_output = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sport": sport_key,
        "sport_name": sport["display_name"],
        "outcomes": outcomes,
        "slop_locks": slop_locks,
        "totals_locks": totals_locks,
        "slimegrinder": slimegrinder,
        "longslop": longslop,
        "matches": prediction_records,
        "totals_matches": totals_prediction_records,
        "season_stats": season_stats,
        "model_weights": {k: round(v, 4) for k, v in model_weight_dict.items()},
        "pick_stats": pick_stats,
    }
    predictions_output["diagnostics"] = _build_pipeline_diagnostics(
        matches=matches,
        fixtures_fetched=fixtures_fetched,
        fixtures_in_window=fixtures,
        odds_list=odds_list,
        odds_lookup=odds_lookup,
        prediction_records=prediction_records,
        outcomes=outcomes,
        sport_key=sport_key,
        sport=sport,
        slop_locks=slop_locks,
        longslop=longslop,
        slimegrinder=slimegrinder,
    )
    _save_json(predictions_path, predictions_output)

    # Deduplicate: only add prediction_records not already in history
    existing_game_keys = set()
    for p in updated_past:
        gk = (p["home_team"], p["away_team"], str(p.get("date", ""))[:10])
        existing_game_keys.add(gk)

    new_predictions = [
        rec for rec in prediction_records
        if (rec["home_team"], rec["away_team"], str(rec.get("date", ""))[:10])
        not in existing_game_keys
    ]

    history_output = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "predictions": updated_past + new_predictions,
    }
    _save_json(history_path, history_output)

    _save_json(accuracy_path, accuracy_log)

    _print_pipeline_diagnostics(sport_key, predictions_output["diagnostics"])

    return predictions_output


# ---------------------------------------------------------------------------
# Main pipeline — runs all sports
# ---------------------------------------------------------------------------

def run_pipeline(output_dir=None):
    """Run the SLOP LOCKS pipeline for all configured sports.

    Parameters
    ----------
    output_dir : str or None
        Base directory for output. Each sport writes to a subdirectory.
        Defaults to ``DATA_DIR`` from config.
    """
    base_dir = output_dir or DATA_DIR
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    manifest = {
        "updated_at": now,
        "sports": {},
    }

    for sport_key in SPORTS:
        sport_dir = os.path.join(base_dir, sport_key) if output_dir else None
        try:
            sport_output = run_sport_pipeline(sport_key, output_dir=sport_dir)
            manifest["sports"][sport_key] = {
                "name": SPORTS[sport_key]["display_name"],
                "status": "ok",
                "updated_at": now,
                "diagnostics": sport_output.get("diagnostics", {}),
            }
        except Exception as exc:
            manifest["sports"][sport_key] = {
                "name": SPORTS[sport_key]["display_name"],
                "status": "error",
                "error": str(exc),
                "updated_at": now,
            }

    _save_json(os.path.join(base_dir, "manifest.json"), manifest)

    return manifest


def _main(argv=None):
    """CLI entry point for the pipeline module."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run the SLOP LOCKS pipeline.")
    parser.add_argument("--sport", choices=sorted(SPORTS.keys()))
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)

    if args.sport:
        run_sport_pipeline(args.sport, output_dir=args.output_dir)
        return 0

    manifest = run_pipeline(output_dir=args.output_dir)
    errors = [s["error"] for s in manifest["sports"].values() if s["status"] == "error"]
    if errors:
        for err in errors:
            print(f"Error in pipeline: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main())
