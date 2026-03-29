"""Backtesting and accuracy-tracking utilities for SLOP LOCKS."""

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

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
    args = parser.parse_args()

    report = build_backtest_report(data_dir=args.data_dir, sports=args.sports or None)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
