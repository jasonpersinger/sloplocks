"""Pipeline orchestrator for SLOP LOCKS.

Ties together data fetching, model fitting, ensemble blending, backtesting,
and JSON output into a single ``run_pipeline()`` entry point.
"""

import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from pipeline.config import (
    CONGESTION_THRESHOLD_DAYS,
    DATA_DIR,
    PREDICTIONS_PATH,
    HISTORY_PATH,
    ACCURACY_PATH,
)
from pipeline.fetch_data import fetch_epl_matches, fetch_epl_fixtures, fetch_odds
from pipeline.fetch_xg import fetch_understat_xg
from pipeline.models import (
    EloRatings,
    dixon_coles_predict,
    elo_predict,
    fit_dixon_coles,
    scoreline_to_probabilities,
)
from pipeline.ensemble import blend_predictions, compute_edges, decimal_to_american
from pipeline.backtest import (
    compute_model_weights,
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)


def _check_congestion(team, fixtures, matches, threshold_days=CONGESTION_THRESHOLD_DAYS):
    """Return True if *team* has a fixture within *threshold_days* of another.

    Looks at both upcoming fixtures and recent match dates to decide whether
    the team is in a congested schedule period.
    """
    team_dates = []

    # Gather dates from completed matches
    for _, row in matches.iterrows():
        if row["home_team"] == team or row["away_team"] == team:
            dt = pd.to_datetime(row["date"])
            if dt.tzinfo is not None:
                dt = dt.tz_localize(None)
            team_dates.append(dt)

    # Gather dates from upcoming fixtures
    for fix in fixtures:
        if fix["home_team"] == team or fix["away_team"] == team:
            dt = pd.to_datetime(fix["date"])
            if dt.tzinfo is not None:
                dt = dt.tz_localize(None)
            team_dates.append(dt)

    if len(team_dates) < 2:
        return False

    team_dates.sort()
    for i in range(1, len(team_dates)):
        gap = (team_dates[i] - team_dates[i - 1]).days
        if 0 < gap <= threshold_days:
            return True

    return False


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(output_dir=None):
    """Run the full SLOP LOCKS prediction pipeline.

    Parameters
    ----------
    output_dir : str or None
        Directory to write output JSON files. Defaults to ``DATA_DIR``
        from config.
    """
    if output_dir is None:
        output_dir = DATA_DIR

    predictions_path = os.path.join(output_dir, "predictions.json")
    history_path = os.path.join(output_dir, "history.json")
    accuracy_path = os.path.join(output_dir, "model_accuracy.json")

    # ------------------------------------------------------------------
    # 1. Fetch data
    # ------------------------------------------------------------------
    matches = fetch_epl_matches()
    fixtures = fetch_epl_fixtures()

    # xG and odds are optional -- degrade gracefully
    xg_data = None
    try:
        xg_data = fetch_understat_xg()
    except Exception:
        pass

    odds_list = []
    try:
        odds_list = fetch_odds()
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 2. Fit models
    # ------------------------------------------------------------------

    # Dixon-Coles on actual goals
    dc_params = fit_dixon_coles(matches)

    # Dixon-Coles on xG (if enough data)
    xg_params = None
    if xg_data is not None and len(xg_data) >= 20:
        xg_params = fit_dixon_coles(
            xg_data,
            goals_col_home="home_xg",
            goals_col_away="away_xg",
        )

    # Elo ratings
    all_teams = sorted(
        set(matches["home_team"].unique()) | set(matches["away_team"].unique())
    )
    elo = EloRatings(all_teams)
    elo.process_season(matches)

    # ------------------------------------------------------------------
    # 3. Load accuracy log and compute model weights
    # ------------------------------------------------------------------
    accuracy_log = _load_json(accuracy_path)
    if not isinstance(accuracy_log, dict):
        accuracy_log = {}

    model_names = ["dixon_coles", "elo"]
    if xg_params is not None:
        model_names.insert(1, "xg")

    accuracies = [get_rolling_accuracy(accuracy_log, name) for name in model_names]
    weights = compute_model_weights(accuracies)

    model_weight_dict = dict(zip(model_names, weights))

    # ------------------------------------------------------------------
    # 4. Build odds lookup
    # ------------------------------------------------------------------
    odds_lookup = {}
    for o in odds_list:
        key = (o["home_team"], o["away_team"])
        odds_lookup[key] = o

    # ------------------------------------------------------------------
    # 5. Predict each fixture
    # ------------------------------------------------------------------
    prediction_records = []

    for fix in fixtures:
        home = fix["home_team"]
        away = fix["away_team"]

        # Skip if either team is unknown to the models
        if home not in dc_params["attack"] or away not in dc_params["attack"]:
            continue

        # Congestion flags
        cong_home = _check_congestion(home, fixtures, matches)
        cong_away = _check_congestion(away, fixtures, matches)

        # Dixon-Coles prediction
        dc_matrix = dixon_coles_predict(home, away, dc_params,
                                        congestion_home=cong_home,
                                        congestion_away=cong_away)
        dc_probs = scoreline_to_probabilities(dc_matrix)

        # Collect individual model predictions and weights for blending
        individual_preds = [dc_probs]
        blend_weights = [model_weight_dict["dixon_coles"]]
        individual_models = {"dixon_coles": dc_probs}

        # xG prediction
        if xg_params is not None and home in xg_params["attack"] and away in xg_params["attack"]:
            xg_matrix = dixon_coles_predict(home, away, xg_params,
                                            congestion_home=cong_home,
                                            congestion_away=cong_away)
            xg_probs = scoreline_to_probabilities(xg_matrix)
            individual_preds.append(xg_probs)
            blend_weights.append(model_weight_dict["xg"])
            individual_models["xg"] = xg_probs

        # Elo prediction
        elo_probs = elo_predict(elo, home, away)
        individual_preds.append(elo_probs)
        blend_weights.append(model_weight_dict["elo"])
        individual_models["elo"] = elo_probs

        # Blend
        blended = blend_predictions(individual_preds, blend_weights)

        # Edges and best odds (if odds available)
        match_odds = odds_lookup.get((home, away))
        edges = {}
        best_odds = {}
        if match_odds:
            edges = compute_edges(blended, match_odds)
            best_odds = {
                "home": decimal_to_american(match_odds["home_odds"]),
                "draw": decimal_to_american(match_odds["draw_odds"]),
                "away": decimal_to_american(match_odds["away_odds"]),
            }

        record = {
            "home_team": home,
            "away_team": away,
            "date": fix["date"],
            "matchday": fix.get("matchday"),
            "model_probs": {k: round(v, 4) for k, v in blended.items()},
            "individual_models": {
                name: {k: round(v, 4) for k, v in probs.items()}
                for name, probs in individual_models.items()
            },
            "edges": edges,
            "best_odds": best_odds,
        }
        prediction_records.append(record)

    # ------------------------------------------------------------------
    # 6. Evaluate past predictions against new results
    # ------------------------------------------------------------------
    history = _load_json(history_path)
    if not isinstance(history, dict):
        history = {}
    past_predictions = history.get("predictions", [])

    # Build result lookup from matches
    result_lookup = {}
    for _, row in matches.iterrows():
        key = (row["home_team"], row["away_team"], str(row["date"])[:10])
        result_lookup[key] = (int(row["home_goals"]), int(row["away_goals"]))

    updated_past = []
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

            # Evaluate each model and update accuracy log
            for model_name in pred.get("individual_models", {}):
                model_probs = pred["individual_models"][model_name]
                eval_result = evaluate_prediction(model_probs, hg, ag)
                update_accuracy_log(accuracy_log, model_name, eval_result)

            # Evaluate blended prediction
            if "model_probs" in pred:
                eval_result = evaluate_prediction(pred["model_probs"], hg, ag)
                update_accuracy_log(accuracy_log, "ensemble", eval_result)

        updated_past.append(pred)

    # ------------------------------------------------------------------
    # 7. Compute season stats
    # ------------------------------------------------------------------
    total_matches = len(matches)
    home_wins = int((matches["home_goals"] > matches["away_goals"]).sum())
    draws = int((matches["home_goals"] == matches["away_goals"]).sum())
    away_wins = int((matches["home_goals"] < matches["away_goals"]).sum())

    season_stats = {
        "total_matches": total_matches,
        "home_wins": home_wins,
        "draws": draws,
        "away_wins": away_wins,
        "home_win_pct": round(home_wins / max(total_matches, 1), 3),
        "draw_pct": round(draws / max(total_matches, 1), 3),
        "away_win_pct": round(away_wins / max(total_matches, 1), 3),
    }

    # ------------------------------------------------------------------
    # 8. Write output files
    # ------------------------------------------------------------------
    predictions_output = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "matches": prediction_records,
        "season_stats": season_stats,
        "model_weights": {k: round(v, 4) for k, v in model_weight_dict.items()},
    }
    _save_json(predictions_path, predictions_output)

    # History: current predictions become the new past predictions
    history_output = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "predictions": updated_past + prediction_records,
    }
    _save_json(history_path, history_output)

    # Accuracy log
    _save_json(accuracy_path, accuracy_log)

    return predictions_output
