"""Backtesting and accuracy-tracking utilities for SLOP LOCKS."""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from pipeline.config import ENSEMBLE_ACCURACY_WINDOW, SPORTS
from pipeline.ensemble import blend_predictions
from pipeline.fetch_mlb import fetch_mlb_games
from pipeline.fetch_mma import fetch_mma_games
from pipeline.fetch_nba import fetch_nba_espn_games
from pipeline.fetch_ncaam import fetch_ncaam_games
from pipeline.fetch_nhl import fetch_nhl_games
from pipeline.models import (
    AdjustedEfficiency,
    BullpenMatchupModel,
    EloRatings,
    FourFactorsModel,
    HandednessMatchupModel,
    NbaMatchupModel,
    NbaTotalsModel,
    NhlMatchupModel,
    PitcherMatchupModel,
    RecentBoxScoreModel,
    ResultsFeatureModel,
    RunEnvironmentModel,
    bullpen_matchup_predict,
    efficiency_predict,
    elo_predict,
    four_factors_predict,
    handedness_matchup_predict,
    nba_matchup_predict,
    nhl_matchup_predict,
    pitcher_matchup_predict,
    recent_boxscore_predict,
    results_features_predict,
    run_environment_predict,
)


def _safe_float(value):
    """Convert stored scalar values into floats when possible."""
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value):
    """Parse stored truthy/falsey strings into bools."""
    if isinstance(value, bool):
        return value
    if value in (None, "", "None"):
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _american_to_decimal(american):
    """Convert American odds to decimal odds."""
    american = _safe_float(american)
    if american is None or american == 0:
        return None
    if american > 0:
        return round(1.0 + (american / 100.0), 4)
    return round(1.0 + (100.0 / abs(american)), 4)


def _load_results_rows(data_dir: str, sports: list[str] | None = None) -> list[dict]:
    """Load tracked settled result rows from the shared results log."""
    path = os.path.join(data_dir, "tracking", "results_log.csv")
    if not os.path.exists(path):
        return []

    selected_sports = set(sports or SPORTS.keys())
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("sport") not in selected_sports:
                continue
            match_date = str(row.get("match_date") or "")[:10]
            try:
                datetime.strptime(match_date, "%Y-%m-%d")
            except ValueError:
                continue
            home_team = row.get("home_team")
            away_team = row.get("away_team")
            if home_team in {"moneyline", "total", "prediction", "slop_lock", "total_lock"}:
                continue
            if away_team in {"moneyline", "total", "prediction", "slop_lock", "total_lock"}:
                continue
            rows.append({
                "logged_at": row.get("logged_at"),
                "sport": row.get("sport"),
                "entry_type": row.get("entry_type"),
                "home_team": home_team,
                "away_team": away_team,
                "match_date": match_date,
                "pick": row.get("pick"),
                "actual": row.get("actual"),
                "won": _safe_bool(row.get("won")),
                "model_prob": _safe_float(row.get("model_prob")),
                "home_prob": _safe_float(row.get("home_prob")),
                "away_prob": _safe_float(row.get("away_prob")),
                "draw_prob": _safe_float(row.get("draw_prob")),
                "implied_prob": _safe_float(row.get("implied_prob")),
                "market_implied_prob": _safe_float(row.get("market_implied_prob")),
                "edge": _safe_float(row.get("edge")),
                "expected_value": _safe_float(row.get("expected_value")),
                "american_odds": _safe_float(row.get("american_odds")),
                "decimal_odds": _safe_float(row.get("decimal_odds")),
                "confidence_score": _safe_float(row.get("confidence_score")),
                "kelly_fraction": _safe_float(row.get("kelly_fraction")),
                "fractional_kelly": _safe_float(row.get("fractional_kelly")),
            })
    return rows


def _prediction_probs_from_row(row: dict) -> dict[str, float]:
    """Build a probability dict from one tracked prediction row."""
    probs = {}
    for key, outcome in (("home_prob", "home"), ("away_prob", "away"), ("draw_prob", "draw")):
        value = row.get(key)
        if value is None:
            continue
        probs[outcome] = float(value)
    return probs


def _score_prediction_row(row: dict) -> dict | None:
    """Evaluate one settled tracked prediction row."""
    probs = _prediction_probs_from_row(row)
    actual = row.get("actual")
    if not probs or actual not in {"home", "away", "draw"}:
        return None
    actual_prob = max(probs.get(actual, 1e-15), 1e-15)
    predicted = max(probs, key=probs.get)
    brier = sum((probs.get(outcome, 0.0) - (1.0 if outcome == actual else 0.0)) ** 2 for outcome in probs)
    return {
        "predicted": predicted,
        "actual": actual,
        "correct": predicted == actual,
        "actual_prob": actual_prob,
        "log_loss": -math.log(actual_prob),
        "brier": brier,
        "model_prob": row.get("model_prob") or probs.get(predicted),
    }


def _summarize_prediction_rows(rows: list[dict]) -> dict:
    """Summarize settled tracked prediction rows."""
    scored = [item for item in (_score_prediction_row(row) for row in rows) if item is not None]
    if not scored:
        return {
            "evaluated": 0,
            "accuracy": None,
            "avg_log_loss": None,
            "avg_brier": None,
        }
    evaluated = len(scored)
    correct = sum(1 for item in scored if item["correct"])
    return {
        "evaluated": evaluated,
        "accuracy": round(correct / evaluated, 4),
        "avg_log_loss": round(sum(item["log_loss"] for item in scored) / evaluated, 4),
        "avg_brier": round(sum(item["brier"] for item in scored) / evaluated, 4),
    }


def _summarize_pick_rows(rows: list[dict]) -> dict:
    """Summarize settled tracked pick rows from the results log."""
    if not rows:
        return {
            "evaluated": 0,
            "wins": 0,
            "losses": 0,
            "record": "0-0",
            "hit_rate": None,
            "roi": None,
            "avg_expected_value": None,
            "avg_confidence": None,
        }

    bets = []
    wins = 0
    losses = 0
    pushes = 0
    expected_values = []
    confidence_values = []
    for row in rows:
        won = row.get("won")
        if won is True:
            wins += 1
        elif won is False:
            losses += 1
        else:
            pushes += 1

        decimal_odds = row.get("decimal_odds")
        if decimal_odds is None:
            decimal_odds = _american_to_decimal(row.get("american_odds"))
        if decimal_odds and decimal_odds > 1.0 and won is not None:
            bets.append({
                "stake": 100.0,
                "odds": decimal_odds,
                "won": bool(won),
                "push": False,
            })
        if row.get("expected_value") is not None:
            expected_values.append(float(row["expected_value"]))
        if row.get("confidence_score") is not None:
            confidence_values.append(float(row["confidence_score"]))

    record = f"{wins}-{losses}" + (f"-{pushes}" if pushes else "")
    evaluated = wins + losses + pushes
    return {
        "evaluated": evaluated,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "record": record,
        "hit_rate": round(wins / evaluated, 4) if evaluated else None,
        "roi": round(compute_roi(bets), 4) if bets else None,
        "avg_expected_value": round(sum(expected_values) / len(expected_values), 4) if expected_values else None,
        "avg_confidence": round(sum(confidence_values) / len(confidence_values), 4) if confidence_values else None,
    }


def _calibration_bin_summary(rows: list[dict], bucket_size: float = 0.1) -> list[dict]:
    """Summarize how predicted probabilities have calibrated against outcomes."""
    buckets = defaultdict(list)
    for row in rows:
        scored = _score_prediction_row(row)
        if scored is None:
            continue
        model_prob = row.get("model_prob")
        if model_prob is None:
            model_prob = scored.get("model_prob")
        if model_prob is None:
            continue
        bucket = min(int(float(model_prob) / bucket_size), int((1.0 / bucket_size) - 1))
        buckets[bucket].append({
            "model_prob": float(model_prob),
            "won": 1.0 if scored["correct"] else 0.0,
        })

    summary = []
    for bucket in sorted(buckets):
        bucket_rows = buckets[bucket]
        low = round(bucket * bucket_size, 2)
        high = round(low + bucket_size, 2)
        sample = len(bucket_rows)
        avg_prob = sum(item["model_prob"] for item in bucket_rows) / sample
        win_rate = sum(item["won"] for item in bucket_rows) / sample
        summary.append({
            "bucket": f"{low:.2f}-{high:.2f}",
            "sample": sample,
            "avg_model_prob": round(avg_prob, 4),
            "actual_win_rate": round(win_rate, 4),
            "calibration_gap": round(avg_prob - win_rate, 4),
        })
    return summary


def _summarize_scored_predictions(rows: list[dict]) -> dict:
    """Summarize a list of already-scored prediction rows."""
    if not rows:
        return {
            "evaluated": 0,
            "accuracy": None,
            "avg_log_loss": None,
            "avg_brier": None,
        }
    evaluated = len(rows)
    correct = sum(1 for row in rows if row.get("correct"))
    return {
        "evaluated": evaluated,
        "accuracy": round(correct / evaluated, 4),
        "avg_log_loss": round(sum(float(row.get("log_loss", 0.0)) for row in rows) / evaluated, 4),
        "avg_brier": round(sum(float(row.get("brier", 0.0)) for row in rows) / evaluated, 4),
    }


def _load_raw_walkforward_inputs(sport_key: str, data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load historical raw inputs for one sport using the local ESPN caches."""
    cache_path = os.path.join(data_dir, sport_key, "espn_cache.json")
    if sport_key == "nba":
        return fetch_nba_espn_games(cache_path=cache_path)
    if sport_key == "ncaam":
        return fetch_ncaam_games(cache_path=cache_path)
    if sport_key == "mlb":
        return fetch_mlb_games(cache_path=cache_path)
    if sport_key == "nhl":
        return fetch_nhl_games(cache_path=cache_path)
    if sport_key == "mma":
        return fetch_mma_games(cache_path=cache_path)
    raise ValueError(f"Unsupported sport: {sport_key}")


def _build_walkforward_models(
    sport_key: str,
    sport: dict,
    train_matches: pd.DataFrame,
    train_box_scores: pd.DataFrame | None,
    model_names: list[str],
) -> dict:
    """Fit the active model set for one walk-forward training slice."""
    models = {}
    if train_matches is None or train_matches.empty:
        return models

    teams = sorted(set(train_matches["home_team"].unique()) | set(train_matches["away_team"].unique()))
    if "elo" in model_names and teams:
        elo = EloRatings(
            teams,
            k_factor=sport["elo_k_factor"],
            home_advantage=sport["elo_home_advantage"],
        )
        elo.process_season(train_matches)
        models["elo"] = elo

    if "results_features" in model_names:
        models["results_features"] = ResultsFeatureModel(
            train_matches,
            feature_window=sport.get("results_feature_window", 8),
            min_games=sport.get("results_feature_min_games", 30),
        )

    if sport_key in {"nba", "ncaam"} and train_box_scores is not None and not train_box_scores.empty:
        if "efficiency" in model_names:
            models["efficiency"] = AdjustedEfficiency(train_box_scores, train_matches)
        if "four_factors" in model_names:
            models["four_factors"] = FourFactorsModel(train_box_scores, train_matches)
        if "recent_boxscore" in model_names:
            models["recent_boxscore"] = RecentBoxScoreModel(
                train_box_scores,
                train_matches,
                feature_window=sport.get("recent_boxscore_window", 8),
                min_games=sport.get("recent_boxscore_min_games", 30),
            )
    if sport_key == "nba" and train_box_scores is not None and not train_box_scores.empty and "nba_matchup" in model_names:
        models["nba_matchup"] = NbaMatchupModel(
            train_box_scores,
            train_matches,
            feature_window=sport.get("nba_matchup_window", 8),
            min_games=sport.get("nba_matchup_min_games", 30),
        )
    if sport_key == "nhl" and "nhl_matchup" in model_names:
        models["nhl_matchup"] = NhlMatchupModel(
            train_matches,
            feature_window=sport.get("nhl_matchup_window", 10),
            min_games=sport.get("nhl_matchup_min_games", 40),
        )
    if sport_key == "mlb":
        if "pitcher_features" in model_names:
            models["pitcher_features"] = PitcherMatchupModel(
                train_matches,
                feature_window=sport.get("pitcher_feature_window", 8),
                min_games=sport.get("pitcher_feature_min_games", 20),
            )
        if "bullpen_features" in model_names:
            models["bullpen_features"] = BullpenMatchupModel(
                train_matches,
                feature_window=sport.get("bullpen_feature_window", 12),
                recent_usage_window=sport.get("bullpen_recent_usage_window", 5),
                min_games=sport.get("bullpen_feature_min_games", 20),
            )
        if "run_environment" in model_names:
            models["run_environment"] = RunEnvironmentModel(
                train_matches,
                feature_window=sport.get("run_environment_window", 12),
                min_games=sport.get("run_environment_min_games", 20),
            )
        if "handedness_features" in model_names:
            models["handedness_features"] = HandednessMatchupModel(
                train_matches,
                feature_window=sport.get("handedness_feature_window", 18),
                min_games=sport.get("handedness_feature_min_games", 20),
            )
    return models


def _predict_walkforward_fixture(
    sport_key: str,
    sport: dict,
    fixture: dict,
    models: dict,
    accuracy_log: dict[str, list[dict]],
    model_names: list[str],
) -> tuple[dict | None, dict[str, dict[str, float]]]:
    """Generate a blended walk-forward prediction for one historical fixture."""
    home = fixture["home_team"]
    away = fixture["away_team"]
    game_date = str(fixture.get("date", ""))[:10]
    is_neutral = bool(fixture.get("neutral", False))

    predictions = []
    weights = []
    individual = {}

    active_model_names = [name for name in model_names if name in models]
    if not active_model_names:
        return None, {}

    accuracies = [get_rolling_accuracy(accuracy_log, name, window=sport.get("accuracy_window")) for name in active_model_names]
    weight_values = compute_model_weights(accuracies, temperature=sport.get("accuracy_softmax_temperature", 2.0))
    weight_map = dict(zip(active_model_names, weight_values))

    if "elo" in models:
        probs = elo_predict(models["elo"], home, away, outcomes=sport["outcomes"])
        predictions.append(probs)
        weights.append(weight_map["elo"])
        individual["elo"] = probs
    if "results_features" in models:
        probs = results_features_predict(models["results_features"], home, away, neutral_site=is_neutral, game_date=game_date)
        predictions.append(probs)
        weights.append(weight_map["results_features"])
        individual["results_features"] = probs
    if "efficiency" in models and home in models["efficiency"].off_efficiency and away in models["efficiency"].off_efficiency:
        probs = efficiency_predict(models["efficiency"], home, away, home_bonus=0.0 if is_neutral else sport.get("efficiency_home_bonus", 3.5))
        predictions.append(probs)
        weights.append(weight_map["efficiency"])
        individual["efficiency"] = probs
    if "four_factors" in models and models["four_factors"].model is not None:
        if home in models["four_factors"].team_stats and away in models["four_factors"].team_stats:
            probs = four_factors_predict(models["four_factors"], home, away)
            predictions.append(probs)
            weights.append(weight_map["four_factors"])
            individual["four_factors"] = probs
    if "recent_boxscore" in models:
        probs = recent_boxscore_predict(models["recent_boxscore"], home, away)
        predictions.append(probs)
        weights.append(weight_map["recent_boxscore"])
        individual["recent_boxscore"] = probs
    if "nba_matchup" in models:
        probs = nba_matchup_predict(models["nba_matchup"], home, away, game_date=game_date)
        predictions.append(probs)
        weights.append(weight_map["nba_matchup"])
        individual["nba_matchup"] = probs
    if "nhl_matchup" in models:
        probs = nhl_matchup_predict(
            models["nhl_matchup"],
            home,
            away,
            game_date=game_date,
            home_goalie=fixture.get("home_goalie"),
            away_goalie=fixture.get("away_goalie"),
        )
        predictions.append(probs)
        weights.append(weight_map["nhl_matchup"])
        individual["nhl_matchup"] = probs
    if "pitcher_features" in models:
        probs = pitcher_matchup_predict(models["pitcher_features"], fixture.get("home_pitcher"), fixture.get("away_pitcher"), game_date=game_date)
        predictions.append(probs)
        weights.append(weight_map["pitcher_features"])
        individual["pitcher_features"] = probs
    if "bullpen_features" in models:
        probs = bullpen_matchup_predict(models["bullpen_features"], home, away)
        predictions.append(probs)
        weights.append(weight_map["bullpen_features"])
        individual["bullpen_features"] = probs
    if "run_environment" in models:
        probs = run_environment_predict(models["run_environment"], home, away)
        predictions.append(probs)
        weights.append(weight_map["run_environment"])
        individual["run_environment"] = probs
    if "handedness_features" in models:
        probs = handedness_matchup_predict(
            models["handedness_features"],
            home,
            away,
            home_pitcher_hand=fixture.get("home_pitcher_hand"),
            away_pitcher_hand=fixture.get("away_pitcher_hand"),
        )
        predictions.append(probs)
        weights.append(weight_map["handedness_features"])
        individual["handedness_features"] = probs

    if not predictions:
        return None, {}
    return blend_predictions(predictions, weights), individual


def run_raw_walkforward_for_sport(
    sport_key: str,
    matches: pd.DataFrame,
    box_scores_df: pd.DataFrame | None = None,
    max_days: int | None = None,
    model_names: list[str] | None = None,
    min_training_games: int = 20,
) -> dict:
    """Run a raw-data walk-forward replay for one sport using historical inputs."""
    sport = SPORTS[sport_key]
    if matches is None or matches.empty:
        return {
            "sport": sport_key,
            "dates_evaluated": 0,
            "predictions": _summarize_scored_predictions([]),
            "models": {},
            "daily": [],
        }

    games = matches.copy().sort_values("date").reset_index(drop=True)
    if box_scores_df is not None and not box_scores_df.empty:
        box_scores_df = box_scores_df.copy().sort_values("date").reset_index(drop=True)

    dates = sorted(str(value)[:10] for value in games["date"].dropna().unique())
    if max_days is not None and max_days > 0:
        dates = dates[-max_days:]

    active_model_names = model_names or list(sport.get("models", []))
    accuracy_log: dict[str, list[dict]] = {}
    all_model_results: dict[str, list[dict]] = defaultdict(list)
    scored_rows = []
    daily = []

    for current_date in dates:
        train_matches = games[games["date"].astype(str).str[:10] < current_date].copy()
        eval_matches = games[games["date"].astype(str).str[:10] == current_date].copy()
        if len(train_matches) < min_training_games or eval_matches.empty:
            continue

        train_box = None
        if box_scores_df is not None and not box_scores_df.empty:
            train_box = box_scores_df[box_scores_df["date"].astype(str).str[:10] < current_date].copy()

        models = _build_walkforward_models(sport_key, sport, train_matches, train_box, active_model_names)
        day_rows = []
        for _, row in eval_matches.iterrows():
            fixture = row.to_dict()
            blended, individual = _predict_walkforward_fixture(sport_key, sport, fixture, models, accuracy_log, active_model_names)
            if blended is None:
                continue

            scored = evaluate_prediction(blended, int(row["home_goals"]), int(row["away_goals"]))
            scored["brier"] = compute_brier_score(blended, int(row["home_goals"]), int(row["away_goals"]))
            scored["date"] = current_date
            scored["home_team"] = row["home_team"]
            scored["away_team"] = row["away_team"]
            scored["model_prob"] = round(float(blended.get(scored["predicted"], 0.0)), 4)
            day_rows.append(scored)
            scored_rows.append(scored)

            for name, probs in individual.items():
                result = evaluate_prediction(probs, int(row["home_goals"]), int(row["away_goals"]))
                result["brier"] = compute_brier_score(probs, int(row["home_goals"]), int(row["away_goals"]))
                update_accuracy_log(accuracy_log, name, result, window=sport.get("accuracy_window"))
                all_model_results[name].append(result)

        if day_rows:
            cumulative = _summarize_scored_predictions(scored_rows)
            daily.append({
                "date": current_date,
                "games": len(day_rows),
                "predictions": _summarize_scored_predictions(day_rows),
                "cumulative_predictions": cumulative,
            })

    model_summaries = {
        name: _summarize_scored_predictions(results)
        for name, results in sorted(all_model_results.items())
        if results
    }
    return {
        "sport": sport_key,
        "name": sport.get("display_name", sport_key.upper()),
        "dates_evaluated": len(daily),
        "predictions": _summarize_scored_predictions(scored_rows),
        "models": model_summaries,
        "daily": daily,
        "_scored_rows": scored_rows,
    }


def build_raw_walkforward_report(
    data_dir: str = "data",
    sports: list[str] | None = None,
    max_days: int | None = None,
    min_training_games: int = 20,
) -> dict:
    """Build a raw-data walk-forward replay report across sports."""
    selected_sports = sports or list(SPORTS.keys())
    sport_reports = {}
    aggregate_rows = []

    for sport_key in selected_sports:
        matches, box_scores_df = _load_raw_walkforward_inputs(sport_key, data_dir=data_dir)
        report = run_raw_walkforward_for_sport(
            sport_key,
            matches,
            box_scores_df=box_scores_df,
            max_days=max_days,
            min_training_games=min_training_games,
        )
        sport_reports[sport_key] = report
        aggregate_rows.extend(report.get("_scored_rows", []))

    for report in sport_reports.values():
        report.pop("_scored_rows", None)

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sports": sport_reports,
        "aggregate": _summarize_scored_predictions(aggregate_rows),
        "max_days": max_days,
        "min_training_games": min_training_games,
    }


def build_walkforward_report(data_dir: str = "data", sports: list[str] | None = None, as_of: str | None = None) -> dict:
    """Build a date-ordered replay report from the shared settled results log."""
    rows = _load_results_rows(data_dir, sports=sports)
    if as_of:
        anchor = datetime.fromisoformat(str(as_of).replace("Z", "+00:00")).date()
        rows = [row for row in rows if row.get("match_date") and datetime.strptime(row["match_date"][:10], "%Y-%m-%d").date() <= anchor]

    rows.sort(key=lambda row: (row.get("match_date") or "", row.get("logged_at") or "", row.get("sport") or "", row.get("entry_type") or ""))
    selected_sports = sports or list(SPORTS.keys())

    prediction_rows = [row for row in rows if row.get("entry_type") == "prediction"]
    pick_rows = [row for row in rows if row.get("entry_type") != "prediction"]

    per_day = []
    by_day = defaultdict(list)
    for row in rows:
        if row.get("match_date"):
            by_day[row["match_date"][:10]].append(row)

    cumulative_predictions = []
    cumulative_picks = []
    for match_date in sorted(by_day):
        day_rows = by_day[match_date]
        day_predictions = [row for row in day_rows if row.get("entry_type") == "prediction"]
        day_picks = [row for row in day_rows if row.get("entry_type") != "prediction"]
        cumulative_predictions.extend(day_predictions)
        cumulative_picks.extend(day_picks)
        day_summary = {
            "date": match_date,
            "predictions": _summarize_prediction_rows(day_predictions),
            "picks": _summarize_pick_rows(day_picks),
            "cumulative_predictions": _summarize_prediction_rows(cumulative_predictions),
            "cumulative_picks": _summarize_pick_rows(cumulative_picks),
        }
        per_day.append(day_summary)

    sport_rows = {}
    for sport_key in selected_sports:
        sport_group = [row for row in rows if row.get("sport") == sport_key]
        sport_predictions = [row for row in sport_group if row.get("entry_type") == "prediction"]
        sport_picks = [row for row in sport_group if row.get("entry_type") != "prediction"]
        if not sport_group:
            continue
        sport_rows[sport_key] = {
            "predictions": _summarize_prediction_rows(sport_predictions),
            "picks": _summarize_pick_rows(sport_picks),
            "calibration": _calibration_bin_summary(sport_predictions),
        }

    aggregate_predictions = _summarize_prediction_rows(prediction_rows)
    aggregate_picks = _summarize_pick_rows(pick_rows)
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sports": sport_rows,
        "aggregate": {
            "predictions": aggregate_predictions,
            "picks": aggregate_picks,
            "calibration": _calibration_bin_summary(prediction_rows),
        },
        "daily": per_day,
    }


def evaluate_prediction(probs, home_goals, away_goals):
    """Score a single prediction against the actual result.

    Parameters
    ----------
    probs : dict
        Predicted probabilities with keys ``home``, ``draw``, ``away``.
    home_goals : int
        Goals scored by the home team.
    away_goals : int
        Goals scored by the away team.

    Returns
    -------
    dict
        ``predicted`` – outcome with the highest predicted probability,
        ``actual`` – outcome derived from the scoreline,
        ``correct`` – whether the prediction was right,
        ``log_loss`` – negative log of the probability assigned to the
        actual outcome (lower is better),
        ``actual_prob`` – the probability the model assigned to the
        actual outcome.
    """
    # Determine which outcome was predicted (highest probability).
    predicted = max(probs, key=probs.get)

    # Determine the actual outcome from the scoreline.
    if home_goals > away_goals:
        actual = "home"
    elif home_goals == away_goals:
        actual = "draw"
    else:
        actual = "away"

    # Guard: if actual outcome isn't in probs (e.g. "draw" for a 2-outcome sport),
    # treat it as a miss with a near-zero probability.
    actual_prob = probs.get(actual, 1e-15)
    # Clamp to avoid log(0).
    clamped = max(actual_prob, 1e-15)

    return {
        "predicted": predicted,
        "actual": actual,
        "correct": predicted == actual,
        "log_loss": -math.log(clamped),
        "actual_prob": actual_prob,
    }


def compute_brier_score(probs, home_goals, away_goals):
    """Compute a one-vs-all Brier score for the available outcomes."""
    if home_goals > away_goals:
        actual = "home"
    elif home_goals == away_goals:
        actual = "draw"
    else:
        actual = "away"

    return sum((probs.get(outcome, 0.0) - (1.0 if outcome == actual else 0.0)) ** 2 for outcome in probs)


def compute_model_weights(accuracies, temperature: float = 2.0):
    """Convert per-model accuracies into ensemble weights via softmax scaling.

    Parameters
    ----------
    accuracies : list[float]
        Accuracy values (0–1) for each model.

    Returns
    -------
    list[float]
        Weights that sum to 1.  Higher-accuracy models receive more weight.
    """
    scaled = [math.exp(acc * temperature) for acc in accuracies]
    total = sum(scaled)
    return [s / total for s in scaled]


def compute_roi(bets):
    """Compute return-on-investment for a sequence of bets.

    Parameters
    ----------
    bets : list[dict]
        Each dict must have ``stake`` (float), ``odds`` (decimal float),
        and ``won`` (bool). Optional ``push`` entries return stake with no
        profit or loss.

    Returns
    -------
    float
        ROI as a fraction of total staked (profit / total_staked).
        Returns 0.0 when the list is empty.
    """
    if not bets:
        return 0.0

    total_staked = sum(b["stake"] for b in bets)
    total_return = 0.0
    for bet in bets:
        if bet.get("push"):
            total_return += bet["stake"]
        elif bet["won"]:
            total_return += bet["stake"] * bet["odds"]
    profit = total_return - total_staked
    return profit / total_staked


def update_accuracy_log(accuracy_log, model_name, prediction_result, window=None):
    """Append a prediction result to the rolling accuracy log.

    The log is kept to the last *window* entries per model so that only
    recent performance drives the ensemble weights.

    Parameters
    ----------
    accuracy_log : dict[str, list]
        Mutable mapping from model name to list of result dicts.
    model_name : str
        Identifier for the model.
    prediction_result : dict
        A result dict (as returned by :func:`evaluate_prediction`).
    window : int or None
        Rolling window size. Falls back to ``ENSEMBLE_ACCURACY_WINDOW``.
    """
    if window is None:
        window = ENSEMBLE_ACCURACY_WINDOW

    if model_name not in accuracy_log:
        accuracy_log[model_name] = []

    accuracy_log[model_name].append(prediction_result)

    # Trim to the configured window size.
    if len(accuracy_log[model_name]) > window:
        accuracy_log[model_name] = accuracy_log[model_name][-window:]


def get_rolling_accuracy(accuracy_log, model_name, window=None):
    """Return the fraction of correct predictions in the log for a model.

    Parameters
    ----------
    accuracy_log : dict[str, list]
        The accuracy log (see :func:`update_accuracy_log`).
    model_name : str
        Identifier for the model.
    window : int or None
        Only consider the last *window* entries. Uses all entries if None.

    Returns
    -------
    float
        Fraction correct, or 0.5 (uninformative prior) when no data
        exists for the model.
    """
    entries = accuracy_log.get(model_name, [])
    if window is not None:
        entries = entries[-window:]
    if not entries:
        return 0.5

    correct_count = sum(1 for e in entries if e["correct"])
    return correct_count / len(entries)


def summarize_prediction_history(predictions):
    """Summarize evaluated historical predictions."""
    evaluated = [p for p in predictions if p.get("evaluated") and p.get("model_probs")]
    if not evaluated:
        return {
            "evaluated": 0,
            "accuracy": None,
            "avg_log_loss": None,
            "avg_brier": None,
        }

    scored = [
        evaluate_prediction(p["model_probs"], p["home_goals"], p["away_goals"])
        for p in evaluated
    ]
    briers = [
        compute_brier_score(p["model_probs"], p["home_goals"], p["away_goals"])
        for p in evaluated
    ]

    accuracy = sum(1 for item in scored if item["correct"]) / len(scored)
    avg_log_loss = sum(item["log_loss"] for item in scored) / len(scored)
    avg_brier = sum(briers) / len(briers)

    return {
        "evaluated": len(evaluated),
        "accuracy": round(accuracy, 4),
        "avg_log_loss": round(avg_log_loss, 4),
        "avg_brier": round(avg_brier, 4),
    }


def summarize_pick_history(picks):
    """Summarize evaluated picks and their ROI."""
    total_picks = len(picks)
    evaluated = [p for p in picks if p.get("evaluated")]
    pending = total_picks - len(evaluated)
    confidence_values = [float(p["confidence_score"]) for p in picks if p.get("confidence_score") is not None]
    expected_values = [float(p["expected_value"]) for p in picks if p.get("expected_value") is not None]
    if not evaluated:
        return {
            "total_picks": total_picks,
            "evaluated": 0,
            "pending": pending,
            "hit_rate": None,
            "roi": None,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "avg_confidence": round(sum(confidence_values) / len(confidence_values), 4) if confidence_values else None,
            "avg_expected_value": round(sum(expected_values) / len(expected_values), 4) if expected_values else None,
        }

    bets = [
        {
            "stake": 100.0,
            "odds": p.get("decimal_odds", 0.0),
            "won": p.get("won", False),
            "push": p.get("push", False),
        }
        for p in evaluated
    ]
    wins = sum(1 for p in evaluated if p.get("won"))
    pushes = sum(1 for p in evaluated if p.get("push"))
    losses = sum(1 for p in evaluated if not p.get("won") and not p.get("push"))
    return {
        "total_picks": total_picks,
        "evaluated": len(evaluated),
        "pending": pending,
        "hit_rate": round(wins / len(evaluated), 4),
        "roi": round(compute_roi(bets), 4),
        "push_rate": round(pushes / len(evaluated), 4),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "avg_confidence": round(sum(confidence_values) / len(confidence_values), 4) if confidence_values else None,
        "avg_expected_value": round(sum(expected_values) / len(expected_values), 4) if expected_values else None,
    }


def summarize_closing_line_value(picks):
    """Summarize CLV-style movement from saved pick records."""
    tracked = [p for p in picks if p.get("closing_line_value") is not None]
    if not tracked:
        return {
            "tracked": 0,
            "avg_clv": None,
            "positive_rate": None,
            "non_negative_rate": None,
        }

    values = [float(p["closing_line_value"]) for p in tracked]
    positives = sum(1 for value in values if value > 0)
    non_negative = sum(1 for value in values if value >= 0)
    return {
        "tracked": len(tracked),
        "avg_clv": round(sum(values) / len(values), 4),
        "positive_rate": round(positives / len(values), 4),
        "non_negative_rate": round(non_negative / len(values), 4),
    }


def _pick_lane_key(pick: dict, lane: str) -> str:
    """Return the grouping key for one pick lane."""
    if lane == "type":
        return str(pick.get("type") or "unknown")
    if lane == "market_type":
        return str(pick.get("market_type") or "moneyline")
    raise ValueError(f"Unsupported lane: {lane}")


def summarize_pick_breakdowns(picks):
    """Return per-type and per-market summaries."""
    breakdowns = {}
    for lane in ("type", "market_type"):
        groups = defaultdict(list)
        for pick in picks:
            groups[_pick_lane_key(pick, lane)].append(pick)
        breakdowns[lane] = {
            key: {
                **summarize_pick_history(group),
                "clv": summarize_closing_line_value(group),
            }
            for key, group in sorted(groups.items())
        }
    return breakdowns


def _parse_pick_date(pick: dict) -> date | None:
    """Return the best available calendar date for a stored pick."""
    for key in ("pick_date", "match_date", "logged_at"):
        value = pick.get(key)
        if not value:
            continue
        text = str(value).strip()
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        except ValueError:
            pass
        try:
            return datetime.strptime(text[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
    return None


def summarize_pick_window(picks, days: int, as_of: str | None = None):
    """Summarize picks inside a trailing window ending on *as_of*."""
    if as_of:
        anchor = datetime.fromisoformat(str(as_of).replace("Z", "+00:00")).date()
    else:
        anchor = datetime.now(timezone.utc).date()
    cutoff = anchor - timedelta(days=max(days - 1, 0))
    window_picks = []
    for pick in picks:
        pick_day = _parse_pick_date(pick)
        if pick_day is None or pick_day < cutoff or pick_day > anchor:
            continue
        window_picks.append(pick)
    return {
        **summarize_pick_history(window_picks),
        "clv": summarize_closing_line_value(window_picks),
        "window_days": days,
    }


def _record_label(summary: dict) -> str:
    """Format a concise W-L-P label from a pick summary."""
    wins = int(summary.get("wins") or 0)
    losses = int(summary.get("losses") or 0)
    pushes = int(summary.get("pushes") or 0)
    label = f"{wins}-{losses}"
    if pushes:
        label += f"-{pushes}"
    return label


def _sport_dashboard_summary(
    sport_key: str,
    sport_report: dict,
    manifest_sport: dict | None = None,
    current_output: dict | None = None,
) -> dict:
    """Condense one sport into dashboard-friendly fields."""
    picks = sport_report.get("picks", {})
    clv = picks.get("clv", {})
    diagnostics = (manifest_sport or {}).get("diagnostics", {})
    locks = int(diagnostics.get("slop_locks_posted") or 0)
    totals = len((current_output or {}).get("totals_locks", []) or [])

    return {
        "sport": sport_key,
        "name": SPORTS.get(sport_key, {}).get("display_name", sport_key.upper()),
        "status": (manifest_sport or {}).get("status", "ok"),
        "current": {
            "modeled": int(diagnostics.get("matches_modeled") or 0),
            "odds_coverage": (
                f"{int(diagnostics.get('fixtures_with_odds') or 0)}/"
                f"{int(diagnostics.get('fixtures_in_window') or 0)}"
            ),
            "positive_ev": int(diagnostics.get("matches_with_positive_ev") or 0),
            "eligible": int(diagnostics.get("lock_eligible_matches") or 0),
            "locks": locks,
            "totals": totals,
            "summary": diagnostics.get("summary"),
        },
        "performance": {
            "record": _record_label(picks),
            "evaluated": int(picks.get("evaluated") or 0),
            "pending": int(picks.get("pending") or 0),
            "hit_rate": picks.get("hit_rate"),
            "roi": picks.get("roi"),
            "avg_confidence": picks.get("avg_confidence"),
            "avg_expected_value": picks.get("avg_expected_value"),
            "avg_clv": clv.get("avg_clv"),
        },
    }


def _lane_leaders(report: dict) -> dict:
    """Return simple leaderboard entries from the aggregate report."""
    leaders = {}

    sports = []
    for sport_key, sport_report in report.get("sports", {}).items():
        picks = sport_report.get("picks", {})
        if int(picks.get("evaluated") or 0) > 0:
            sports.append({
                "sport": sport_key,
                "name": SPORTS.get(sport_key, {}).get("display_name", sport_key.upper()),
                "roi": picks.get("roi"),
                "hit_rate": picks.get("hit_rate"),
                "avg_clv": (picks.get("clv") or {}).get("avg_clv"),
                "evaluated": int(picks.get("evaluated") or 0),
            })

    roi_candidates = [item for item in sports if item.get("roi") is not None]
    if roi_candidates:
        leaders["best_roi_sport"] = max(roi_candidates, key=lambda item: item["roi"])

    clv_candidates = [item for item in sports if item.get("avg_clv") is not None]
    if clv_candidates:
        leaders["best_clv_sport"] = max(clv_candidates, key=lambda item: item["avg_clv"])

    lane_candidates = []
    for lane, groups in (report.get("aggregate", {}).get("picks", {}).get("breakdowns", {}).items()):
        for key, summary in groups.items():
            if int(summary.get("evaluated") or 0) <= 0 or summary.get("roi") is None:
                continue
            lane_candidates.append({
                "lane": lane,
                "key": key,
                "roi": summary.get("roi"),
                "hit_rate": summary.get("hit_rate"),
                "evaluated": int(summary.get("evaluated") or 0),
            })
    if lane_candidates:
        leaders["best_lane"] = max(lane_candidates, key=lambda item: item["roi"])

    return leaders


def _build_dashboard_insights(report: dict, manifest: dict, windows: dict, leaders: dict) -> list[str]:
    """Generate concise human-readable dashboard insights."""
    insights = []
    aggregate = report.get("aggregate", {}).get("picks", {})
    market_breakdowns = aggregate.get("breakdowns", {}).get("market_type", {})
    moneyline = market_breakdowns.get("moneyline", {})
    totals = market_breakdowns.get("total", {})

    if totals.get("evaluated") and moneyline.get("evaluated"):
        if (totals.get("roi") or -999) > (moneyline.get("roi") or -999):
            insights.append("Totals are currently outperforming moneylines on both hit rate and ROI.")
        else:
            insights.append("Moneylines are still the steadier lane; totals need more settled sample before loosening.")

    recent7 = windows.get("7d", {})
    recent30 = windows.get("30d", {})
    if recent7.get("evaluated") and recent30.get("evaluated"):
        r7 = recent7.get("roi")
        r30 = recent30.get("roi")
        if r7 is not None and r30 is not None:
            if r7 > r30:
                insights.append("Recent 7-day ROI is running ahead of the 30-day baseline.")
            elif r7 < r30:
                insights.append("The last 7 days are trailing the 30-day baseline; stay selective.")

    if leaders.get("best_roi_sport"):
        best = leaders["best_roi_sport"]
        insights.append(
            f"{best['name']} is the top settled sport right now by ROI "
            f"({best['roi']:+.1%} across {best['evaluated']} graded picks)."
        )

    sports_manifest = manifest.get("sports", {})
    coverage_gaps = []
    for sport_key, sport_meta in sports_manifest.items():
        diagnostics = sport_meta.get("diagnostics", {})
        in_window = int(diagnostics.get("fixtures_in_window") or 0)
        with_odds = int(diagnostics.get("fixtures_with_odds") or 0)
        if in_window > with_odds:
            coverage_gaps.append(f"{SPORTS.get(sport_key, {}).get('display_name', sport_key.upper())} {with_odds}/{in_window}")
    if coverage_gaps:
        insights.append("Odds coverage is still the main live bottleneck for: " + ", ".join(coverage_gaps) + ".")

    if not insights:
        insights.append("Tracking is live and the dashboard is ready; let the settled sample grow before retuning thresholds.")
    return insights


def _build_recommended_actions(report: dict, manifest: dict, windows: dict, leaders: dict) -> list[dict]:
    """Build ranked, user-facing operating recommendations."""
    actions = []
    aggregate = report.get("aggregate", {}).get("picks", {})
    aggregate_clv = aggregate.get("clv", {})
    sports_manifest = manifest.get("sports", {})

    coverage_gaps = []
    for sport_key, sport_meta in sports_manifest.items():
        diagnostics = sport_meta.get("diagnostics", {})
        in_window = int(diagnostics.get("fixtures_in_window") or 0)
        with_odds = int(diagnostics.get("fixtures_with_odds") or 0)
        if in_window > with_odds:
            coverage_gaps.append((sport_key, with_odds, in_window))
    if coverage_gaps:
        coverage_gaps.sort(key=lambda item: (item[2] - item[1], item[2]), reverse=True)
        labels = ", ".join(
            f"{SPORTS.get(s, {}).get('display_name', s.upper())} {got}/{total}"
            for s, got, total in coverage_gaps[:3]
        )
        actions.append({
            "priority": "high",
            "title": "Fix Live Coverage Gaps",
            "detail": f"Late-day value is still being lost to missing odds or schedule coverage in {labels}.",
        })

    best_lane = leaders.get("best_lane")
    if best_lane and best_lane.get("evaluated", 0) >= 5:
        lane_name = f"{best_lane['lane']}:{best_lane['key']}"
        actions.append({
            "priority": "medium",
            "title": "Lean Into The Strongest Lane",
            "detail": f"{lane_name} is currently the best settled lane at {best_lane['roi']:+.1%} ROI.",
        })

    recent7 = windows.get("7d", {})
    recent30 = windows.get("30d", {})
    if recent7.get("evaluated", 0) >= 5 and recent30.get("evaluated", 0) >= 15:
        r7 = recent7.get("roi")
        r30 = recent30.get("roi")
        if r7 is not None and r30 is not None:
            if r7 < 0 <= r30:
                actions.append({
                    "priority": "medium",
                    "title": "Stay Selective This Week",
                    "detail": "The 7-day window is trailing the broader 30-day baseline. Do not loosen thresholds.",
                })
            elif r7 > r30 and r7 > 0:
                actions.append({
                    "priority": "low",
                    "title": "Current Form Is Strong",
                    "detail": "Recent results are ahead of the 30-day baseline. Hold thresholds steady and let the sample build.",
                })

    if aggregate.get("evaluated", 0) >= 20:
        roi = aggregate.get("roi")
        avg_clv = aggregate_clv.get("avg_clv")
        if roi is not None and avg_clv is not None:
            if roi > 0 and avg_clv < 0:
                actions.append({
                    "priority": "medium",
                    "title": "Watch CLV Before Expanding",
                    "detail": "Realized ROI is positive, but closing-line value is lagging. Avoid adding volume until the market read improves.",
                })
            elif roi < 0 and avg_clv > 0:
                actions.append({
                    "priority": "medium",
                    "title": "Hold Nerve On Thresholds",
                    "detail": "CLV is positive even though realized ROI is down. The model may be right before the results catch up.",
                })

    if not actions:
        actions.append({
            "priority": "low",
            "title": "Hold Current Gates",
            "detail": "The current sample does not justify a threshold change yet. Keep collecting settled picks.",
        })

    priority_order = {"high": 0, "medium": 1, "low": 2}
    actions.sort(key=lambda item: (priority_order.get(item["priority"], 9), item["title"]))
    return actions[:4]


def build_dashboard_data(data_dir: str = "data", sports: list[str] | None = None, as_of: str | None = None) -> dict:
    """Build a site-friendly reporting dashboard payload."""
    selected_sports = sports or list(SPORTS.keys())
    report = build_backtest_report(data_dir=data_dir, sports=selected_sports)
    walkforward = build_walkforward_report(data_dir=data_dir, sports=selected_sports, as_of=as_of)

    manifest_path = os.path.join(data_dir, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f) or {}

    all_picks = []
    for sport_key in selected_sports:
        pick_history_path = os.path.join(data_dir, sport_key, "pick_history.json")
        if os.path.exists(pick_history_path):
            with open(pick_history_path) as f:
                all_picks.extend((json.load(f) or {}).get("picks", []))

    windows = {
        "7d": summarize_pick_window(all_picks, 7, as_of=as_of),
        "30d": summarize_pick_window(all_picks, 30, as_of=as_of),
    }
    leaders = _lane_leaders(report)

    sports_summary = []
    for sport_key in selected_sports:
        current_output = {}
        predictions_path = os.path.join(data_dir, sport_key, "predictions.json")
        if os.path.exists(predictions_path):
            with open(predictions_path) as f:
                current_output = json.load(f) or {}
        sports_summary.append(
            _sport_dashboard_summary(
                sport_key,
                report.get("sports", {}).get(sport_key, {}),
                (manifest.get("sports") or {}).get(sport_key, {}),
                current_output,
            )
        )

    aggregate_picks = report.get("aggregate", {}).get("picks", {})
    aggregate_record = {
        "record": _record_label(aggregate_picks),
        "evaluated": int(aggregate_picks.get("evaluated") or 0),
        "pending": int(aggregate_picks.get("pending") or 0),
        "hit_rate": aggregate_picks.get("hit_rate"),
        "roi": aggregate_picks.get("roi"),
        "avg_confidence": aggregate_picks.get("avg_confidence"),
        "avg_expected_value": aggregate_picks.get("avg_expected_value"),
        "avg_clv": (aggregate_picks.get("clv") or {}).get("avg_clv"),
    }

    slate = {
        "modeled": sum(int(((manifest.get("sports") or {}).get(s, {}).get("diagnostics", {}) or {}).get("matches_modeled") or 0) for s in selected_sports),
        "with_odds": sum(int(((manifest.get("sports") or {}).get(s, {}).get("diagnostics", {}) or {}).get("fixtures_with_odds") or 0) for s in selected_sports),
        "fixtures": sum(int(((manifest.get("sports") or {}).get(s, {}).get("diagnostics", {}) or {}).get("fixtures_in_window") or 0) for s in selected_sports),
        "positive_ev": sum(int(((manifest.get("sports") or {}).get(s, {}).get("diagnostics", {}) or {}).get("matches_with_positive_ev") or 0) for s in selected_sports),
        "eligible": sum(int(((manifest.get("sports") or {}).get(s, {}).get("diagnostics", {}) or {}).get("lock_eligible_matches") or 0) for s in selected_sports),
        "locks": sum(int(((manifest.get("sports") or {}).get(s, {}).get("diagnostics", {}) or {}).get("slop_locks_posted") or 0) for s in selected_sports),
    }

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_updated_at": manifest.get("updated_at"),
        "aggregate": {
            "record": aggregate_record,
            "predictions": report.get("aggregate", {}).get("predictions", {}),
            "breakdowns": aggregate_picks.get("breakdowns", {}),
            "threshold_guidance": report.get("aggregate", {}).get("threshold_guidance", []),
            "slate": slate,
        },
        "windows": windows,
        "sports": sports_summary,
        "leaders": leaders,
        "walkforward": {
            "predictions": walkforward.get("aggregate", {}).get("predictions", {}),
            "picks": walkforward.get("aggregate", {}).get("picks", {}),
            "calibration": walkforward.get("aggregate", {}).get("calibration", []),
            "recent_days": walkforward.get("daily", [])[-7:],
        },
        "recommended_actions": _build_recommended_actions(report, manifest, windows, leaders),
        "insights": _build_dashboard_insights(report, manifest, windows, leaders),
    }


def build_threshold_guidance(pick_summary: dict) -> list[str]:
    """Generate simple evidence-based threshold guidance from report metrics."""
    guidance = []
    evaluated = int(pick_summary.get("evaluated") or 0)
    tracked = int(((pick_summary.get("clv") or {}).get("tracked")) or 0)
    roi = pick_summary.get("roi")
    avg_clv = (pick_summary.get("clv") or {}).get("avg_clv")

    if evaluated < 20:
        guidance.append("Insufficient settled pick volume to retune thresholds confidently; hold current gates for now.")
        return guidance

    if tracked < 15:
        guidance.append("CLV sample is still thin; use ROI and hit rate cautiously before loosening any lane.")

    if roi is not None and avg_clv is not None:
        if roi > 0.05 and avg_clv > 0:
            guidance.append("Current thresholds look healthy; do not loosen until a larger tracked sample confirms the edge.")
        elif roi < 0 and avg_clv < 0:
            guidance.append("Results and CLV are both negative; tighten thresholds or reduce low-confidence volume.")
        elif roi > 0 and avg_clv < 0:
            guidance.append("Results are positive but CLV is weak; keep thresholds steady and watch for regression.")
        elif roi < 0 and avg_clv > 0:
            guidance.append("CLV is positive but realized ROI is lagging; avoid reactive threshold cuts until the sample matures.")

    return guidance


def build_backtest_report(data_dir: str = "data", sports: list[str] | None = None) -> dict:
    """Build a historical performance report from saved data files."""
    selected_sports = sports or list(SPORTS.keys())
    report = {
        "data_dir": data_dir,
        "sports": {},
        "aggregate": {
            "predictions": {"evaluated": 0, "accuracy": None, "avg_log_loss": None, "avg_brier": None},
            "picks": {"evaluated": 0, "hit_rate": None, "roi": None},
        },
    }

    all_predictions = []
    all_picks = []

    for sport in selected_sports:
        sport_dir = os.path.join(data_dir, sport)
        history_path = os.path.join(sport_dir, "history.json")
        pick_history_path = os.path.join(sport_dir, "pick_history.json")

        predictions = []
        picks = []
        if os.path.exists(history_path):
            with open(history_path) as f:
                predictions = (json.load(f) or {}).get("predictions", [])
        if os.path.exists(pick_history_path):
            with open(pick_history_path) as f:
                picks = (json.load(f) or {}).get("picks", [])

        pred_summary = summarize_prediction_history(predictions)
        pick_summary = summarize_pick_history(picks)
        report["sports"][sport] = {
            "predictions": pred_summary,
            "picks": {
                **pick_summary,
                "clv": summarize_closing_line_value(picks),
                "breakdowns": summarize_pick_breakdowns(picks),
            },
        }
        report["sports"][sport]["threshold_guidance"] = build_threshold_guidance(report["sports"][sport]["picks"])

        all_predictions.extend(predictions)
        all_picks.extend(picks)

    report["aggregate"]["predictions"] = summarize_prediction_history(all_predictions)
    report["aggregate"]["picks"] = {
        **summarize_pick_history(all_picks),
        "clv": summarize_closing_line_value(all_picks),
        "breakdowns": summarize_pick_breakdowns(all_picks),
    }
    report["aggregate"]["threshold_guidance"] = build_threshold_guidance(report["aggregate"]["picks"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize historical model and pick performance.")
    parser.add_argument("sports", nargs="*", help="Optional sport keys to limit the report.")
    parser.add_argument("--data-dir", default="data", help="Directory containing per-sport pipeline outputs.")
    parser.add_argument("--walkforward", action="store_true", help="Emit the walk-forward replay report instead of the aggregate summary.")
    parser.add_argument("--raw-walkforward", action="store_true", help="Replay models day by day from raw historical inputs.")
    parser.add_argument("--max-days", type=int, default=None, help="Optional cap on replay days for raw walk-forward runs.")
    parser.add_argument("--min-training-games", type=int, default=20, help="Minimum prior games required before evaluating a replay day.")
    args = parser.parse_args()

    if args.raw_walkforward:
        report = build_raw_walkforward_report(
            data_dir=args.data_dir,
            sports=args.sports or None,
            max_days=args.max_days,
            min_training_games=args.min_training_games,
        )
    elif args.walkforward:
        report = build_walkforward_report(data_dir=args.data_dir, sports=args.sports or None)
    else:
        report = build_backtest_report(data_dir=args.data_dir, sports=args.sports or None)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
