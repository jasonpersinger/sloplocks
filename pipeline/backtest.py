"""Backtesting and accuracy-tracking utilities for SLOP LOCKS."""

import argparse
import json
import math
import os
from collections import defaultdict

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
    args = parser.parse_args()

    report = build_backtest_report(data_dir=args.data_dir, sports=args.sports or None)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
