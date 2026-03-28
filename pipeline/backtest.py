"""Backtesting and accuracy-tracking utilities for SLOP LOCKS."""

import argparse
import json
import math
import os

from pipeline.config import ENSEMBLE_ACCURACY_WINDOW, SPORTS


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
        and ``won`` (bool).

    Returns
    -------
    float
        ROI as a fraction of total staked (profit / total_staked).
        Returns 0.0 when the list is empty.
    """
    if not bets:
        return 0.0

    total_staked = sum(b["stake"] for b in bets)
    total_return = sum(b["stake"] * b["odds"] if b["won"] else 0.0 for b in bets)
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
    evaluated = [p for p in picks if p.get("evaluated")]
    if not evaluated:
        return {
            "evaluated": 0,
            "hit_rate": None,
            "roi": None,
        }

    bets = [
        {
            "stake": 100.0,
            "odds": p.get("decimal_odds", 0.0),
            "won": p.get("won", False),
        }
        for p in evaluated
    ]
    wins = sum(1 for p in evaluated if p.get("won"))
    return {
        "evaluated": len(evaluated),
        "hit_rate": round(wins / len(evaluated), 4),
        "roi": round(compute_roi(bets), 4),
    }


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
            "picks": pick_summary,
        }

        all_predictions.extend(predictions)
        all_picks.extend(picks)

    report["aggregate"]["predictions"] = summarize_prediction_history(all_predictions)
    report["aggregate"]["picks"] = summarize_pick_history(all_picks)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize historical model and pick performance.")
    parser.add_argument("sports", nargs="*", help="Optional sport keys to limit the report.")
    parser.add_argument("--data-dir", default="data", help="Directory containing per-sport pipeline outputs.")
    args = parser.parse_args()

    report = build_backtest_report(data_dir=args.data_dir, sports=args.sports or None)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
