from typing import Optional, Union
"""Backtesting and accuracy-tracking utilities for SLOP LOCKS."""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from pipeline.config import (
    ENSEMBLE_ACCURACY_WINDOW,
    ODDS_HISTORY_FILENAME,
    PICK_DECISION_LOG_FILENAME,
    RESULTS_AUDIT_LOG_FILENAME,
    RESULTS_LOG_FILENAME,
    TRACKING_DIRNAME,
    SPORTS,
)
from pipeline.ensemble import blend_predictions
from pipeline.fetch_mlb import fetch_mlb_games
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


def _safe_json(value):
    """Parse JSON strings when possible."""
    if value in (None, "", "None"):
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _american_to_decimal(american):
    """Convert American odds to decimal odds."""
    american = _safe_float(american)
    if american is None or american == 0:
        return None
    if american > 0:
        return round(1.0 + (american / 100.0), 4)
    return round(1.0 + (100.0 / abs(american)), 4)


def _load_results_rows(data_dir: str, sports:Optional[ list[str] ] = None) -> list[dict]:
    """Load tracked settled result rows from the shared results log."""
    audit_path = os.path.join(data_dir, TRACKING_DIRNAME, RESULTS_AUDIT_LOG_FILENAME)
    path = audit_path if os.path.exists(audit_path) else os.path.join(data_dir, TRACKING_DIRNAME, RESULTS_LOG_FILENAME)
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
            entry_type = row.get("entry_type")
            market_type = row.get("market_type")
            if not market_type:
                if entry_type == "total_lock" or str(row.get("pick") or "").lower() in {"over", "under"}:
                    market_type = "total"
                else:
                    market_type = "moneyline"
            rows.append({
                "logged_at": row.get("logged_at"),
                "sport": row.get("sport"),
                "entry_type": entry_type,
                "market_type": market_type,
                "home_team": home_team,
                "away_team": away_team,
                "match_date": match_date,
                "pick": row.get("pick"),
                "actual": row.get("actual"),
                "won": _safe_bool(row.get("won")),
                "push": _safe_bool(row.get("push")),
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
                "total_line": _safe_float(row.get("total_line")),
                "confidence_score": _safe_float(row.get("confidence_score")),
                "kelly_fraction": _safe_float(row.get("kelly_fraction")),
                "fractional_kelly": _safe_float(row.get("fractional_kelly")),
                "closing_line_value": _safe_float(row.get("closing_line_value")),
                "closing_line_value_unit": row.get("closing_line_value_unit"),
            })
    return rows


def _load_pick_decision_rows(data_dir: str, sports:Optional[ list[str] ] = None) -> list[dict]:
    """Load immutable pick-decision rows from the shared ledger."""
    path = os.path.join(data_dir, TRACKING_DIRNAME, PICK_DECISION_LOG_FILENAME)
    if not os.path.exists(path):
        return []

    selected_sports = set(sports or SPORTS.keys())
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("sport") not in selected_sports:
                continue
            rows.append({
                "logged_at": row.get("logged_at"),
                "run_id": row.get("run_id"),
                "run_type": row.get("run_type"),
                "snapshot_timestamp": row.get("snapshot_timestamp"),
                "snapshot_path": row.get("snapshot_path"),
                "sport": row.get("sport"),
                "pick_type": row.get("pick_type"),
                "market_type": row.get("market_type") or "moneyline",
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "match_date": str(row.get("match_date") or "")[:10],
                "start_time": row.get("start_time"),
                "pick": row.get("pick"),
                "total_line": _safe_float(row.get("total_line")),
                "expected_total": _safe_float(row.get("expected_total")),
                "total_stddev": _safe_float(row.get("total_stddev")),
                "model_prob": _safe_float(row.get("model_prob")),
                "implied_prob": _safe_float(row.get("implied_prob")),
                "market_implied_prob": _safe_float(row.get("market_implied_prob")),
                "edge": _safe_float(row.get("edge")),
                "expected_value": _safe_float(row.get("expected_value")),
                "american_odds": _safe_float(row.get("american_odds")),
                "decimal_odds": _safe_float(row.get("decimal_odds")),
                "confidence_score": _safe_float(row.get("confidence_score")),
                "kelly_fraction": _safe_float(row.get("kelly_fraction")),
                "fractional_kelly": _safe_float(row.get("fractional_kelly")),
                "market_source": row.get("market_source"),
                "market_books": _safe_float(row.get("market_books")),
                "hold": _safe_float(row.get("hold")),
                "publication_guard_status": row.get("publication_guard_status"),
                "publication_guard_reason": row.get("publication_guard_reason"),
                "publication_guard_enforced": _safe_bool(row.get("publication_guard_enforced")),
                "publication_guard_evaluated_picks": _safe_float(row.get("publication_guard_evaluated_picks")),
                "publication_guard_evaluated_totals_picks": _safe_float(row.get("publication_guard_evaluated_totals_picks")),
                "calibration_sample_size": _safe_float(row.get("calibration_sample_size")),
                "selection_min_expected_value": _safe_float(row.get("selection_min_expected_value")),
                "selection_edge_floor": _safe_float(row.get("selection_edge_floor")),
                "selection_probability_floor": _safe_float(row.get("selection_probability_floor")),
                "selection_confidence_floor": _safe_float(row.get("selection_confidence_floor")),
                "selection_confidence_dropoff": _safe_float(row.get("selection_confidence_dropoff")),
                "selection_max_picks": _safe_float(row.get("selection_max_picks")),
                "market_snapshot": _safe_json(row.get("market_snapshot_json")),
                "model_probs": _safe_json(row.get("model_probs_json")),
                "individual_models": _safe_json(row.get("individual_models_json")),
                "decision_context": _safe_json(row.get("decision_context_json")),
                "gate_context": _safe_json(row.get("gate_context_json")),
            })
    return rows


def _load_latest_odds_snapshots(data_dir: str, sports:Optional[ list[str] ] = None) -> dict[tuple, dict]:
    """Load the latest tracked odds snapshot per market/outcome."""
    path = os.path.join(data_dir, TRACKING_DIRNAME, ODDS_HISTORY_FILENAME)
    if not os.path.exists(path):
        return {}

    selected_sports = set(sports or SPORTS.keys())
    latest = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("sport") not in selected_sports:
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


def _apply_closing_snapshot_fallback(row: dict, snapshot: Optional[dict]) -> dict:
    """Hydrate missing closing-line fields from tracked odds snapshots."""
    if not snapshot:
        return row

    result = dict(row)
    result.setdefault("closing_decimal_odds", _safe_float(snapshot.get("decimal_odds")))
    result.setdefault("closing_american_odds", _safe_float(snapshot.get("american_odds")))
    result.setdefault("closing_implied_prob", _safe_float(snapshot.get("implied_prob")))
    result.setdefault("closing_market_implied_prob", _safe_float(snapshot.get("market_implied_prob")))
    result.setdefault("closing_market_source", snapshot.get("market_source"))
    result.setdefault("closing_market_books", _safe_float(snapshot.get("market_books")))
    result.setdefault("closing_hold", _safe_float(snapshot.get("hold")))
    result.setdefault("closing_market_snapshot", _safe_json(snapshot.get("market_snapshot_json")))

    if result.get("closing_line_value") is not None:
        return result

    market_type = str(result.get("market_type") or "moneyline")
    if market_type == "total":
        opening_total = _safe_float(result.get("total_line"))
        closing_total = _safe_float(snapshot.get("total_line"))
        if opening_total is None or closing_total is None:
            return result
        result["closing_total_line"] = closing_total
        if result.get("pick") == "over":
            result["closing_line_value"] = round(closing_total - opening_total, 3)
        elif result.get("pick") == "under":
            result["closing_line_value"] = round(opening_total - closing_total, 3)
        if result.get("closing_line_value") is not None:
            result["closing_line_value_unit"] = "total_points"
        return result

    opening_prob = _safe_float(result.get("market_implied_prob"))
    if opening_prob is None:
        opening_prob = _safe_float(result.get("implied_prob"))
    closing_prob = _safe_float(snapshot.get("implied_prob"))
    if opening_prob is not None and closing_prob is not None:
        result["closing_line_value"] = round(closing_prob - opening_prob, 4)
        result["closing_line_value_unit"] = "implied_probability_points"
    return result


def _prediction_probs_from_row(row: dict) -> dict[str, float]:
    """Build a probability dict from one tracked prediction row."""
    probs = {}
    for key, outcome in (("home_prob", "home"), ("away_prob", "away"), ("draw_prob", "draw")):
        value = row.get(key)
        if value is None:
            continue
        probs[outcome] = float(value)
    return probs


def _score_prediction_row(row: dict) -> Optional[dict]:
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
        push = bool(row.get("push"))
        if push:
            pushes += 1
        elif won is True:
            wins += 1
        elif won is False:
            losses += 1
        else:
            continue

        decimal_odds = row.get("decimal_odds")
        if decimal_odds is None:
            decimal_odds = _american_to_decimal(row.get("american_odds"))
        if decimal_odds and decimal_odds > 1.0 and won is not None:
            bets.append({
                "stake": 100.0,
                "odds": decimal_odds,
                "won": bool(won),
                "push": push,
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


def _snapshot_pick_signature(record: dict) -> Optional[tuple]:
    """Build a stable comparison key for one saved pick."""
    if not isinstance(record, dict):
        return None
    market_type = str(record.get("market_type") or "moneyline")
    match_date = str(record.get("date") or record.get("match_date") or "")[:10]
    total_line = record.get("total_line")
    if total_line is not None:
        total_line = round(float(total_line), 3)
    return (
        market_type,
        record.get("home_team"),
        record.get("away_team"),
        match_date,
        record.get("pick"),
        total_line if market_type == "total" else None,
    )


def _snapshot_results_lookup(rows: list[dict]) -> dict[tuple, dict]:
    """Index settled results rows by the same pick signature used in snapshots."""
    lookup = {}
    for row in rows:
        signature = _snapshot_pick_signature({
            "market_type": row.get("market_type"),
            "home_team": row.get("home_team"),
            "away_team": row.get("away_team"),
            "match_date": row.get("match_date"),
            "pick": row.get("pick"),
            "total_line": row.get("total_line"),
        })
        if signature is None or signature in lookup:
            continue
        lookup[signature] = row
    return lookup


def _lookup_snapshot_result(result_lookup: dict[tuple, dict], record: dict) -> Optional[dict]:
    """Return the settled result row for one snapshot pick, with legacy fallback."""
    signature = _snapshot_pick_signature(record)
    settled = result_lookup.get(signature)
    if settled is not None:
        return settled
    if signature and signature[0] == "total":
        # Fallback: older result logs did not always persist total_line, so use a
        # line-agnostic lookup for historical totals snapshots when needed.
        return result_lookup.get((signature[0], signature[1], signature[2], signature[3], signature[4], None))
    return None


def _snapshot_pick_to_result_row(record: dict, settled: Optional[dict], snapshot_lookup: dict[tuple,Optional[ dict] ] = None) -> Optional[dict]:
    """Convert one archived snapshot pick into the summary row shape."""
    if settled is None:
        return None
    decimal_odds = record.get("decimal_odds")
    if decimal_odds is None:
        decimal_odds = _american_to_decimal(record.get("american_odds"))
    row = {
        "evaluated": True,
        "market_type": record.get("market_type", "moneyline"),
        "home_team": record.get("home_team"),
        "away_team": record.get("away_team"),
        "match_date": str(record.get("date") or record.get("match_date") or "")[:10],
        "pick": record.get("pick"),
        "won": settled.get("won"),
        "push": settled.get("push"),
        "decimal_odds": decimal_odds,
        "american_odds": record.get("american_odds"),
        "expected_value": record.get("expected_value"),
        "confidence_score": record.get("confidence_score"),
        "total_line": record.get("total_line"),
        "implied_prob": record.get("implied_prob"),
        "market_implied_prob": record.get("market_implied_prob"),
        "closing_line_value": settled.get("closing_line_value"),
        "closing_line_value_unit": settled.get("closing_line_value_unit"),
    }
    if snapshot_lookup is None:
        return row
    snapshot = snapshot_lookup.get((
        row.get("market_type", "moneyline"),
        row.get("home_team"),
        row.get("away_team"),
        row.get("match_date"),
        row.get("pick"),
    ))
    return _apply_closing_snapshot_fallback(row, snapshot)


def _decision_pick_to_result_row(record: dict, settled: Optional[dict], snapshot_lookup: dict[tuple,Optional[ dict] ] = None) -> Optional[dict]:
    """Convert one decision-ledger pick into the summary row shape."""
    if settled is None:
        return None
    decimal_odds = record.get("decimal_odds")
    if decimal_odds is None:
        decimal_odds = _american_to_decimal(record.get("american_odds"))
    row = {
        "type": record.get("pick_type"),
        "evaluated": True,
        "market_type": record.get("market_type", "moneyline"),
        "home_team": record.get("home_team"),
        "away_team": record.get("away_team"),
        "match_date": str(record.get("match_date") or "")[:10],
        "pick": record.get("pick"),
        "won": settled.get("won"),
        "push": settled.get("push"),
        "actual": settled.get("actual"),
        "decimal_odds": decimal_odds,
        "american_odds": record.get("american_odds"),
        "expected_value": record.get("expected_value"),
        "confidence_score": record.get("confidence_score"),
        "closing_line_value": settled.get("closing_line_value"),
        "closing_line_value_unit": settled.get("closing_line_value_unit"),
        "total_line": record.get("total_line"),
        "implied_prob": record.get("implied_prob"),
        "market_implied_prob": record.get("market_implied_prob"),
    }
    if snapshot_lookup is None:
        return row
    snapshot = snapshot_lookup.get((
        row.get("market_type", "moneyline"),
        row.get("home_team"),
        row.get("away_team"),
        row.get("match_date"),
        row.get("pick"),
    ))
    return _apply_closing_snapshot_fallback(row, snapshot)


def _snapshot_exclude_opponent_conflicts(locks: list[dict]) -> list[dict]:
    """Mirror the production conflict filter for moneyline locks."""
    def picked(lock):
        return lock["home_team"] if lock["pick"] == "home" else lock["away_team"]

    def unpicked(lock):
        return lock["away_team"] if lock["pick"] == "home" else lock["home_team"]

    unpicked_map = {unpicked(lock): lock for lock in locks}
    to_remove = set()
    for lock in locks:
        picked_team = picked(lock)
        other = unpicked_map.get(picked_team)
        if other is None or other is lock:
            continue
        if float(lock.get("edge", 0.0)) <= float(other.get("edge", 0.0)):
            to_remove.add(id(lock))
        else:
            to_remove.add(id(other))
    return [lock for lock in locks if id(lock) not in to_remove]


def _snapshot_compute_slop_locks(records: list[dict], outcomes: list[str], config: dict) -> list[dict]:
    """Recompute SLOP LOCKS from one saved snapshot."""
    candidates = []
    for rec in records:
        if rec.get("completed"):
            continue
        for outcome in outcomes:
            edge_data = (rec.get("edges") or {}).get(outcome)
            if not edge_data:
                continue
            confidence = float(edge_data.get("confidence_score", 0.0) or 0.0)
            edge = float(edge_data.get("edge", 0.0) or 0.0)
            prob = float(edge_data.get("model_prob", 0.0) or 0.0)
            ev = float(edge_data.get("expected_value", 0.0) or 0.0)
            if (
                edge >= float(config.get("edge_floor", 0.03))
                and prob >= float(config.get("probability_floor", 0.45))
                and ev >= float(config.get("min_expected_value", 0.0))
                and confidence >= float(config.get("additional_confidence_floor", 65.0))
            ):
                candidates.append({
                    "market_type": "moneyline",
                    "home_team": rec["home_team"],
                    "away_team": rec["away_team"],
                    "date": str(rec.get("date") or rec.get("match_date") or "")[:10],
                    "pick": outcome,
                    "edge": edge,
                    "confidence_score": confidence,
                })

    candidates.sort(key=lambda item: (item["confidence_score"], item["edge"]), reverse=True)
    if not candidates:
        return []

    selected = [candidates[0]]
    top_confidence = candidates[0]["confidence_score"]
    for candidate in candidates[1:]:
        if len(selected) >= int(config.get("max_picks", 3)):
            break
        if (
            candidate["confidence_score"] >= float(config.get("additional_confidence_floor", 65.0))
            and candidate["confidence_score"] >= (top_confidence - float(config.get("confidence_dropoff", 0.0)))
        ):
            selected.append(candidate)
    return _snapshot_exclude_opponent_conflicts(selected)


def _snapshot_compute_longslop(records: list[dict], outcomes: list[str], config: dict) -> Optional[dict]:
    """Recompute LONGSLOP from one saved snapshot."""
    candidates = []
    for rec in records:
        if rec.get("completed"):
            continue
        for outcome in outcomes:
            edge_data = (rec.get("edges") or {}).get(outcome)
            if not edge_data:
                continue
            american = edge_data.get("american_odds")
            if american is None or float(american) < 500:
                continue
            confidence = float(edge_data.get("confidence_score", 0.0) or 0.0)
            edge = float(edge_data.get("edge", 0.0) or 0.0)
            ev = float(edge_data.get("expected_value", 0.0) or 0.0)
            if confidence >= float(config.get("confidence_floor", 65.0)) and edge >= 0.0 and ev >= float(config.get("min_expected_value", 0.0)):
                candidates.append({
                    "market_type": "moneyline",
                    "home_team": rec["home_team"],
                    "away_team": rec["away_team"],
                    "date": str(rec.get("date") or rec.get("match_date") or "")[:10],
                    "pick": outcome,
                    "edge": edge,
                    "confidence_score": confidence,
                })
    candidates.sort(key=lambda item: (item["confidence_score"], item["edge"]), reverse=True)
    return candidates[0] if candidates else None


def _snapshot_compute_totals_locks(records: list[dict], config: dict) -> list[dict]:
    """Recompute totals locks from one saved snapshot."""
    candidates = []
    for rec in records:
        if rec.get("completed"):
            continue
        for outcome in ("over", "under"):
            edge_data = (rec.get("edges") or {}).get(outcome)
            if not edge_data:
                continue
            edge = float(edge_data.get("edge", 0.0) or 0.0)
            prob = float(edge_data.get("model_prob", 0.0) or 0.0)
            ev = float(edge_data.get("expected_value", 0.0) or 0.0)
            confidence = float(edge_data.get("confidence_score", 0.0) or 0.0)
            if (
                edge >= float(config.get("edge_floor", 0.02))
                and prob >= float(config.get("probability_floor", 0.53))
                and ev >= float(config.get("min_expected_value", 0.0))
                and confidence >= float(config.get("confidence_floor", 54.0))
            ):
                candidates.append({
                    "market_type": "total",
                    "home_team": rec["home_team"],
                    "away_team": rec["away_team"],
                    "date": str(rec.get("date") or rec.get("match_date") or "")[:10],
                    "pick": outcome,
                    "total_line": rec.get("total_line"),
                    "edge": edge,
                    "confidence_score": confidence,
                    "expected_value": ev,
                })
    candidates.sort(
        key=lambda item: (
            item.get("confidence_score", 0.0),
            item.get("expected_value", 0.0),
            item.get("edge", 0.0),
        ),
        reverse=True,
    )
    return candidates[: int(config.get("max_picks", 3))]


def _iter_snapshot_payloads(data_dir: str = "data", sports:Optional[ list[str] ] = None) -> list[dict]:
    """Load saved live-state snapshots from disk."""
    root = os.path.join(data_dir, "tracking", "snapshots")
    if not os.path.exists(root):
        return []
    selected_sports = set(sports or SPORTS.keys())
    payloads = []
    for current_root, _, files in os.walk(root):
        for filename in files:
            if not filename.endswith(".json"):
                continue
            path = os.path.join(current_root, filename)
            try:
                with open(path) as f:
                    payload = json.load(f) or {}
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("sport") not in selected_sports:
                continue
            payload["_path"] = path
            payloads.append(payload)
    payloads.sort(key=lambda item: (item.get("snapshot_timestamp") or "", item.get("sport") or "", item.get("run_id") or ""))
    return payloads


def _replay_snapshot_payload(snapshot: dict) -> dict:
    """Replay official pick selection from one saved snapshot bundle."""
    selection_config = snapshot.get("selection_config") or {}
    outcomes = selection_config.get("outcomes") or snapshot.get("outcomes") or []
    records = (snapshot.get("records") or {}).get("matches") or []
    totals_records = (snapshot.get("records") or {}).get("totals_matches") or []
    outputs = snapshot.get("outputs") or {}

    replayed = []
    replayed.extend(_snapshot_compute_slop_locks(records, outcomes, selection_config.get("slop_locks") or {}))
    longslop = _snapshot_compute_longslop(records, outcomes, selection_config.get("longslop") or {})
    if longslop:
        replayed.append(longslop)
    replayed.extend(_snapshot_compute_totals_locks(totals_records, selection_config.get("totals_locks") or {}))

    expected = []
    expected.extend(outputs.get("slop_locks") or [])
    if isinstance(outputs.get("longslop"), dict):
        expected.append(outputs["longslop"])
    expected.extend(outputs.get("totals_locks") or [])

    expected_signatures = sorted(filter(None, (_snapshot_pick_signature(item) for item in expected)))
    replayed_signatures = sorted(filter(None, (_snapshot_pick_signature(item) for item in replayed)))
    expected_set = set(expected_signatures)
    replayed_set = set(replayed_signatures)

    return {
        "run_id": snapshot.get("run_id"),
        "run_type": snapshot.get("run_type"),
        "sport": snapshot.get("sport"),
        "snapshot_timestamp": snapshot.get("snapshot_timestamp"),
        "snapshot_path": snapshot.get("_path"),
        "expected_picks": expected,
        "replayed_picks": replayed,
        "expected_count": len(expected_signatures),
        "replayed_count": len(replayed_signatures),
        "matched_count": len(expected_set & replayed_set),
        "exact_match": expected_signatures == replayed_signatures,
        "missing_in_replay": [list(item) for item in sorted(expected_set - replayed_set)],
        "unexpected_in_replay": [list(item) for item in sorted(replayed_set - expected_set)],
    }


def build_snapshot_replay_report(data_dir: str = "data", sports:Optional[ list[str] ] = None) -> dict:
    """Replay selection from saved snapshots instead of live fetchers."""
    snapshots = _iter_snapshot_payloads(data_dir=data_dir, sports=sports)
    result_lookup = _snapshot_results_lookup(_load_results_rows(data_dir=data_dir, sports=sports))
    odds_snapshot_lookup = _load_latest_odds_snapshots(data_dir=data_dir, sports=sports)
    sport_reports = defaultdict(lambda: {
        "snapshots": 0,
        "exact_matches": 0,
        "exact_match_rate": None,
        "mismatches": 0,
        "recent_mismatches": [],
        "expected_picks": _summarize_pick_rows([]),
        "replayed_picks": _summarize_pick_rows([]),
    })
    entries = []
    aggregate_expected_rows = []
    aggregate_replayed_rows = []
    for snapshot in snapshots:
        replay = _replay_snapshot_payload(snapshot)
        expected_rows = [
            row
            for row in (
                _snapshot_pick_to_result_row(
                    item,
                    _lookup_snapshot_result(result_lookup, item),
                    odds_snapshot_lookup,
                )
                for item in replay.get("expected_picks", [])
            )
            if row is not None
        ]
        replayed_rows = [
            row
            for row in (
                _snapshot_pick_to_result_row(
                    item,
                    _lookup_snapshot_result(result_lookup, item),
                    odds_snapshot_lookup,
                )
                for item in replay.get("replayed_picks", [])
            )
            if row is not None
        ]
        replay["expected_pick_performance"] = _summarize_pick_rows(expected_rows)
        replay["replayed_pick_performance"] = _summarize_pick_rows(replayed_rows)
        entries.append(replay)
        sport_report = sport_reports[replay["sport"]]
        sport_report["snapshots"] += 1
        aggregate_expected_rows.extend(expected_rows)
        aggregate_replayed_rows.extend(replayed_rows)
        if replay["exact_match"]:
            sport_report["exact_matches"] += 1
        else:
            sport_report["mismatches"] += 1
            if len(sport_report["recent_mismatches"]) < 5:
                sport_report["recent_mismatches"].append(replay)
        sport_expected_rows = sport_report.get("_expected_rows", [])
        sport_expected_rows.extend(expected_rows)
        sport_report["_expected_rows"] = sport_expected_rows
        sport_replayed_rows = sport_report.get("_replayed_rows", [])
        sport_replayed_rows.extend(replayed_rows)
        sport_report["_replayed_rows"] = sport_replayed_rows

    for sport_report in sport_reports.values():
        if sport_report["snapshots"] > 0:
            sport_report["exact_match_rate"] = round(sport_report["exact_matches"] / sport_report["snapshots"], 4)
        sport_report["expected_picks"] = _summarize_pick_rows(sport_report.pop("_expected_rows", []))
        sport_report["replayed_picks"] = _summarize_pick_rows(sport_report.pop("_replayed_rows", []))

    aggregate = {
        "snapshots": len(entries),
        "exact_matches": sum(1 for entry in entries if entry["exact_match"]),
        "exact_match_rate": round(sum(1 for entry in entries if entry["exact_match"]) / len(entries), 4) if entries else None,
        "mismatches": sum(1 for entry in entries if not entry["exact_match"]),
        "expected_picks": _summarize_pick_rows(aggregate_expected_rows),
        "replayed_picks": _summarize_pick_rows(aggregate_replayed_rows),
    }
    recent_mismatches = [entry for entry in reversed(entries) if not entry["exact_match"]][:5]
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "aggregate": aggregate,
        "sports": dict(sport_reports),
        "recent_mismatches": recent_mismatches,
    }


def build_pick_decision_replay_report(data_dir: str = "data", sports:Optional[ list[str] ] = None) -> dict:
    """Grade picks from the immutable decision ledger against settled results."""
    decisions = _load_pick_decision_rows(data_dir=data_dir, sports=sports)
    result_lookup = _snapshot_results_lookup(_load_results_rows(data_dir=data_dir, sports=sports))
    odds_snapshot_lookup = _load_latest_odds_snapshots(data_dir=data_dir, sports=sports)
    sport_reports = defaultdict(lambda: {
        "logged_picks": 0,
        "settled_picks": {
            "evaluated": 0,
            "wins": 0,
            "losses": 0,
            "record": "0-0",
            "hit_rate": None,
            "roi": None,
            "avg_expected_value": None,
            "avg_confidence": None,
            "clv": {"tracked": 0, "avg_clv": None, "positive_rate": None, "non_negative_rate": None},
            "breakdowns": {},
        },
        "unsettled_logged_picks": 0,
        "recent_unsettled": [],
    })
    aggregate_rows = []

    for decision in decisions:
        sport_report = sport_reports[decision["sport"]]
        sport_report["logged_picks"] += 1
        settled = _lookup_snapshot_result(result_lookup, decision)
        result_row = _decision_pick_to_result_row(decision, settled, odds_snapshot_lookup)
        if result_row is None:
            sport_report["unsettled_logged_picks"] += 1
            if len(sport_report["recent_unsettled"]) < 5:
                sport_report["recent_unsettled"].append({
                    "pick_type": decision.get("pick_type"),
                    "market_type": decision.get("market_type"),
                    "home_team": decision.get("home_team"),
                    "away_team": decision.get("away_team"),
                    "match_date": decision.get("match_date"),
                    "pick": decision.get("pick"),
                    "snapshot_timestamp": decision.get("snapshot_timestamp"),
                })
            continue
        sport_rows = sport_report.get("_rows", [])
        sport_rows.append(result_row)
        sport_report["_rows"] = sport_rows
        aggregate_rows.append(result_row)

    for sport_report in sport_reports.values():
        rows = sport_report.pop("_rows", [])
        sport_report["settled_picks"] = {
            **_summarize_pick_rows(rows),
            "clv": summarize_closing_line_value(rows),
            "breakdowns": summarize_pick_breakdowns(rows),
        }

    aggregate = {
        "logged_picks": len(decisions),
        "settled_picks": {
            **_summarize_pick_rows(aggregate_rows),
            "clv": summarize_closing_line_value(aggregate_rows),
            "breakdowns": summarize_pick_breakdowns(aggregate_rows),
        },
        "unsettled_logged_picks": max(0, len(decisions) - len(aggregate_rows)),
    }
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "aggregate": aggregate,
        "sports": dict(sport_reports),
    }


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


def _load_raw_walkforward_inputs(sport_key: str, data_dir: str = "data") -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
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
    raise ValueError(f"Unsupported sport: {sport_key}")


def _build_walkforward_models(
    sport_key: str,
    sport: dict,
    train_matches: pd.DataFrame,
    train_box_scores: Optional[pd.DataFrame],
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
) -> tuple[Optional[dict], dict[str, dict[str, float]]]:
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

    weight_values = compute_model_weights(
        accuracy_log,
        model_names=active_model_names,
        temperature=sport.get("accuracy_softmax_temperature", 2.0),
        window=sport.get("accuracy_window"),
    )
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
    box_scores_df: Optional[pd.DataFrame] = None,
    max_days: Optional[int] = None,
    model_names:Optional[ list[str] ] = None,
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
    sports:Optional[ list[str] ] = None,
    max_days: Optional[int] = None,
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


def build_walkforward_report(data_dir: str = "data", sports:Optional[ list[str] ] = None, as_of: Optional[str] = None) -> dict:
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


def compute_model_weights(
    accuracy_log_or_accuracies,
    model_names:Optional[ list[str] ] = None,
    temperature: float = 2.0,
    window: Optional[int] = None,
    prior_strength: float = 12.0,
    log_loss_blend: float = 0.35,
):
    """Convert model histories into ensemble weights via shrunk softmax scaling.

    When a full accuracy log is available, weights are based on:
    - rolling correctness
    - rolling log-loss quality
    - sample-size shrinkage back toward a neutral prior

    The legacy list-of-accuracies call shape still works as a fallback.
    """
    if model_names is None:
        accuracies = list(accuracy_log_or_accuracies or [])
        if not accuracies:
            return []
        scaled = [math.exp(acc * temperature) for acc in accuracies]
        total = sum(scaled)
        return [s / total for s in scaled]

    raw_scores = []
    for model_name in model_names:
        entries = list((accuracy_log_or_accuracies or {}).get(model_name, []))
        if window is not None:
            entries = entries[-window:]
        sample_size = len(entries)
        if not entries:
            raw_scores.append(0.5)
            continue

        correct_count = sum(1 for entry in entries if entry.get("correct"))
        shrunk_accuracy = (correct_count + (0.5 * prior_strength)) / (sample_size + prior_strength)

        log_losses = [
            float(entry.get("log_loss"))
            for entry in entries
            if entry.get("log_loss") is not None
        ]
        # Fallback: if log loss is missing, treat the model as neutral instead of
        # rewarding it off a tiny pure-accuracy sample.
        log_loss_quality = math.exp(-sum(log_losses) / len(log_losses)) if log_losses else 0.5
        blended_quality = ((1.0 - log_loss_blend) * shrunk_accuracy) + (log_loss_blend * log_loss_quality)

        sample_scale = sample_size / (sample_size + prior_strength)
        stabilized_score = (sample_scale * blended_quality) + ((1.0 - sample_scale) * 0.5)
        raw_scores.append(stabilized_score)

    scaled = [math.exp(score * temperature) for score in raw_scores]
    total = sum(scaled)
    return [s / total for s in scaled]


def build_model_health_snapshot(
    accuracy_log: dict,
    model_names: list[str],
    temperature: float = 2.0,
    window: Optional[int] = None,
    min_samples: int = 20,
    disable_sample_min: int = 30,
    disable_log_loss_margin: float = 0.08,
    disable_accuracy_floor: float = 0.5,
    disable_accuracy_margin: float = 0.08,
) -> dict:
    """Summarize per-model health from the rolling accuracy log."""
    accuracy_log = accuracy_log or {}
    weights = compute_model_weights(
        accuracy_log,
        model_names=model_names,
        temperature=temperature,
        window=window,
    )
    weight_map = dict(zip(model_names, weights))

    ensemble_entries = list((accuracy_log or {}).get("ensemble", []))
    if window is not None:
        ensemble_entries = ensemble_entries[-window:]
    ensemble_sample = len(ensemble_entries)
    ensemble_accuracy = (
        sum(1 for entry in ensemble_entries if entry.get("correct")) / ensemble_sample
        if ensemble_sample else None
    )
    ensemble_log_losses = [
        float(entry.get("log_loss"))
        for entry in ensemble_entries
        if entry.get("log_loss") is not None
    ]
    ensemble_log_loss = (
        sum(ensemble_log_losses) / len(ensemble_log_losses)
        if ensemble_log_losses else None
    )

    models = {}
    disable_candidates = []
    for model_name in model_names:
        entries = list((accuracy_log or {}).get(model_name, []))
        if window is not None:
            entries = entries[-window:]
        sample_size = len(entries)
        accuracy = (
            sum(1 for entry in entries if entry.get("correct")) / sample_size
            if sample_size else None
        )
        log_losses = [
            float(entry.get("log_loss"))
            for entry in entries
            if entry.get("log_loss") is not None
        ]
        avg_log_loss = (
            sum(log_losses) / len(log_losses)
            if log_losses else None
        )

        reasons = []
        status = "active"
        disable_candidate = False
        if sample_size < min_samples:
            status = "watch"
            reasons.append(f"sample below model-health floor ({sample_size}/{min_samples})")
        elif (
            sample_size >= disable_sample_min
            and avg_log_loss is not None
            and ensemble_log_loss is not None
            and accuracy is not None
        ):
            worse_log_loss = avg_log_loss > (ensemble_log_loss + disable_log_loss_margin)
            accuracy_floor = float(disable_accuracy_floor)
            worse_accuracy = accuracy < accuracy_floor
            if worse_log_loss and worse_accuracy:
                disable_candidate = True
                status = "disable_candidate"
                reasons.append(
                    f"log loss trails ensemble by {avg_log_loss - ensemble_log_loss:.3f}"
                )
                reasons.append(
                    f"accuracy trails floor {accuracy_floor:.3f} at {accuracy:.3f}"
                )

        models[model_name] = {
            "sample_size": sample_size,
            "accuracy": None if accuracy is None else round(accuracy, 4),
            "avg_log_loss": None if avg_log_loss is None else round(avg_log_loss, 4),
            "weight": round(float(weight_map.get(model_name, 0.0)), 4),
            "status": status,
            "disable_candidate": disable_candidate,
            "reasons": reasons,
        }
        if disable_candidate:
            disable_candidates.append(model_name)

    return {
        "ensemble": {
            "sample_size": ensemble_sample,
            "accuracy": None if ensemble_accuracy is None else round(ensemble_accuracy, 4),
            "avg_log_loss": None if ensemble_log_loss is None else round(ensemble_log_loss, 4),
        },
        "models": models,
        "disable_candidates": disable_candidates,
    }


def build_model_health_report(data_dir: str = "data", sports: Optional[list[str]] = None) -> dict:
    """Build a per-sport model health report from stored accuracy logs."""
    selected_sports = sports or list(SPORTS.keys())
    report = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sports": {},
    }
    for sport_key in selected_sports:
        accuracy_path = os.path.join(data_dir, sport_key, "model_accuracy.json")
        accuracy_log = {}
        if os.path.exists(accuracy_path):
            with open(accuracy_path) as handle:
                accuracy_log = json.load(handle) or {}
        sport = SPORTS.get(sport_key, {})
        report["sports"][sport_key] = build_model_health_snapshot(
            accuracy_log=accuracy_log,
            model_names=list(sport.get("models", [])),
            temperature=sport.get("accuracy_softmax_temperature", 2.0),
            window=sport.get("accuracy_window"),
        )
        report["sports"][sport_key]["disabled_by_config"] = list(sport.get("disabled_models", []))
    return report


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

    total_return = 0.0
    valid_bets = [b for b in bets if b.get("odds") is not None]
    if not valid_bets:
        return 0.0
        
    total_staked = sum(b["stake"] for b in valid_bets)
    for bet in valid_bets:
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
    """Summarize CLV-style movement from saved pick records.

    Moneyline CLV is tracked in implied-probability points. Totals CLV is tracked
    in line points. We keep those summaries separate and only expose the legacy
    ``avg_clv`` alias when the tracked sample is one unit family.
    """
    tracked = [p for p in picks if p.get("closing_line_value") is not None]
    if not tracked:
        return {
            "tracked": 0,
            "avg_clv": None,
            "positive_rate": None,
            "non_negative_rate": None,
            "moneyline_tracked": 0,
            "moneyline_avg_clv": None,
            "moneyline_positive_rate": None,
            "moneyline_non_negative_rate": None,
            "totals_tracked": 0,
            "totals_avg_clv": None,
            "totals_positive_rate": None,
            "totals_non_negative_rate": None,
        }

    def _summary(values: list[float]) -> dict[str, Union[float, int, None]]:
        if not values:
            return {
                "tracked": 0,
                "avg_clv": None,
                "positive_rate": None,
                "non_negative_rate": None,
            }
        positives = sum(1 for value in values if value > 0)
        non_negative = sum(1 for value in values if value >= 0)
        return {
            "tracked": len(values),
            "avg_clv": round(sum(values) / len(values), 4),
            "positive_rate": round(positives / len(values), 4),
            "non_negative_rate": round(non_negative / len(values), 4),
        }

    moneyline_values = [
        float(p["closing_line_value"])
        for p in tracked
        if str(p.get("market_type") or "moneyline") != "total"
    ]
    totals_values = [
        float(p["closing_line_value"])
        for p in tracked
        if str(p.get("market_type") or "moneyline") == "total"
    ]
    moneyline_summary = _summary(moneyline_values)
    totals_summary = _summary(totals_values)

    comparable_summary = None
    if moneyline_summary["tracked"]:
        comparable_summary = moneyline_summary
    elif totals_summary["tracked"]:
        comparable_summary = totals_summary

    return {
        "tracked": len(tracked),
        "avg_clv": None if comparable_summary is None else comparable_summary["avg_clv"],
        "positive_rate": None if comparable_summary is None else comparable_summary["positive_rate"],
        "non_negative_rate": None if comparable_summary is None else comparable_summary["non_negative_rate"],
        "moneyline_tracked": moneyline_summary["tracked"],
        "moneyline_avg_clv": moneyline_summary["avg_clv"],
        "moneyline_positive_rate": moneyline_summary["positive_rate"],
        "moneyline_non_negative_rate": moneyline_summary["non_negative_rate"],
        "totals_tracked": totals_summary["tracked"],
        "totals_avg_clv": totals_summary["avg_clv"],
        "totals_positive_rate": totals_summary["positive_rate"],
        "totals_non_negative_rate": totals_summary["non_negative_rate"],
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


def _parse_pick_date(pick: dict) -> Optional[date]:
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


def summarize_pick_window(picks, days: int, as_of: Optional[str] = None):
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


def summarize_lane_health(
    picks: list[dict],
    market_type: Optional[str] = None,
    pick_type: Optional[str] = None,
    recent_count: int = 8,
) -> dict:
    """Summarize one pick lane with recent ROI, CLV, and calibration context."""
    lane_picks = []
    for pick in picks:
        current_market = str(pick.get("market_type") or "moneyline")
        current_type = str(pick.get("type") or "slop_lock")
        if market_type is not None and current_market != market_type:
            continue
        if pick_type is not None and current_type != pick_type:
            continue
        lane_picks.append(pick)

    lane_picks.sort(key=lambda item: _parse_pick_date(item) or date.min)
    overall = summarize_pick_history(lane_picks)
    overall_clv = summarize_closing_line_value(lane_picks)

    evaluated = [pick for pick in lane_picks if pick.get("evaluated")]
    recent = evaluated[-recent_count:] if recent_count > 0 else list(evaluated)
    recent_summary = summarize_pick_history(recent)
    recent_clv = summarize_closing_line_value(recent)

    resolved = [
        pick for pick in recent
        if pick.get("model_prob") is not None and not pick.get("push")
    ]
    avg_model_prob = (
        sum(float(pick["model_prob"]) for pick in resolved) / len(resolved)
        if resolved else None
    )
    actual_win_rate = (
        sum(1.0 for pick in resolved if pick.get("won")) / len(resolved)
        if resolved else None
    )
    calibration_gap = (
        avg_model_prob - actual_win_rate
        if avg_model_prob is not None and actual_win_rate is not None else None
    )
    overconfidence_gap = max(0.0, calibration_gap) if calibration_gap is not None else None

    return {
        "market_type": market_type,
        "pick_type": pick_type,
        "overall": {
            **overall,
            "clv": overall_clv,
        },
        "recent": {
            **recent_summary,
            "clv": recent_clv,
            "recent_count": recent_count,
            "avg_model_prob": None if avg_model_prob is None else round(avg_model_prob, 4),
            "actual_win_rate": None if actual_win_rate is None else round(actual_win_rate, 4),
            "calibration_gap": None if calibration_gap is None else round(calibration_gap, 4),
            "overconfidence_gap": None if overconfidence_gap is None else round(overconfidence_gap, 4),
        },
    }


def evaluate_lane_health(
    lane_summary: dict,
    *,
    enabled: bool = True,
    lane_label: str = "lane",
    min_evaluated: int = 0,
    min_recent_evaluated: int = 0,
    min_tracked_clv: int = 0,
    min_avg_clv: Optional[float] = 0.0,
    min_recent_roi: Optional[float] = None,
    max_overconfidence_gap: Optional[float] = None,
) -> dict:
    """Evaluate whether one lane is healthy enough to publish live."""
    overall = lane_summary.get("overall", {})
    recent = lane_summary.get("recent", {})
    recent_clv = recent.get("clv", {})
    reasons = []

    if not enabled:
        reasons.append(f"{lane_label} disabled by config")
    if int(overall.get("evaluated") or 0) < int(min_evaluated or 0):
        reasons.append(
            f"need more settled {lane_label} picks ({int(overall.get('evaluated') or 0)}/{int(min_evaluated or 0)})"
        )
    if int(recent.get("evaluated") or 0) < int(min_recent_evaluated or 0):
        reasons.append(
            f"need more recent settled {lane_label} picks ({int(recent.get('evaluated') or 0)}/{int(min_recent_evaluated or 0)})"
        )
    if int(recent_clv.get("tracked") or 0) < int(min_tracked_clv or 0):
        reasons.append(
            f"need more tracked {lane_label} CLV ({int(recent_clv.get('tracked') or 0)}/{int(min_tracked_clv or 0)})"
        )
    if (
        min_avg_clv is not None
        and recent_clv.get("avg_clv") is not None
        and float(recent_clv.get("avg_clv")) < float(min_avg_clv)
    ):
        reasons.append(
            f"recent {lane_label} CLV {float(recent_clv.get('avg_clv')):.4f} is below {float(min_avg_clv):.4f}"
        )
    if (
        min_recent_roi is not None
        and recent.get("roi") is not None
        and float(recent.get("roi")) < float(min_recent_roi)
    ):
        reasons.append(
            f"recent {lane_label} ROI {float(recent.get('roi')):.4f} is below {float(min_recent_roi):.4f}"
        )
    if (
        max_overconfidence_gap is not None
        and recent.get("overconfidence_gap") is not None
        and float(recent.get("overconfidence_gap")) > float(max_overconfidence_gap)
    ):
        reasons.append(
            f"recent {lane_label} overconfidence gap {float(recent.get('overconfidence_gap')):.4f} exceeds {float(max_overconfidence_gap):.4f}"
        )

    if not enabled:
        status = "disabled"
    elif reasons:
        insufficiency_only = all(reason.startswith("need more") for reason in reasons)
        status = "research" if insufficiency_only else "hold"
    else:
        status = "live"

    score = 0
    checks = 0
    for passed in (
        enabled,
        int(overall.get("evaluated") or 0) >= int(min_evaluated or 0),
        int(recent.get("evaluated") or 0) >= int(min_recent_evaluated or 0),
        int(recent_clv.get("tracked") or 0) >= int(min_tracked_clv or 0),
        min_avg_clv is None or recent_clv.get("avg_clv") is None or float(recent_clv.get("avg_clv")) >= float(min_avg_clv),
        min_recent_roi is None or recent.get("roi") is None or float(recent.get("roi")) >= float(min_recent_roi),
        max_overconfidence_gap is None or recent.get("overconfidence_gap") is None or float(recent.get("overconfidence_gap")) <= float(max_overconfidence_gap),
    ):
        checks += 1
        score += 1 if passed else 0

    return {
        **lane_summary,
        "allow": status == "live",
        "status": status,
        "reasons": reasons,
        "health_score": round(score / checks, 4) if checks else None,
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
    manifest_sport: Optional[dict] = None,
    current_output: Optional[dict] = None,
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
            "sense": int(diagnostics.get("matches_with_qualitative") or 0),
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


def build_dashboard_data(data_dir: str = "data", sports:Optional[ list[str] ] = None, as_of: Optional[str] = None) -> dict:
    """Build a site-friendly reporting dashboard payload."""
    selected_sports = sports or list(SPORTS.keys())
    report = build_backtest_report(data_dir=data_dir, sports=selected_sports)
    walkforward = build_walkforward_report(data_dir=data_dir, sports=selected_sports, as_of=as_of)
    snapshot_replay = build_snapshot_replay_report(data_dir=data_dir, sports=selected_sports)
    decision_replay = build_pick_decision_replay_report(data_dir=data_dir, sports=selected_sports)
    lane_health = build_lane_health_report(data_dir=data_dir, sports=selected_sports)
    model_health = build_model_health_report(data_dir=data_dir, sports=selected_sports)

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
        "sense": sum(int(((manifest.get("sports") or {}).get(s, {}).get("diagnostics", {}) or {}).get("matches_with_qualitative") or 0) for s in selected_sports),
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
        "decision_replay": decision_replay,
        "snapshot_replay": snapshot_replay,
        "lane_health": lane_health,
        "model_health": model_health,
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


def build_lane_health_report(data_dir: str = "data", sports: Optional[list[str]] = None) -> dict:
    """Build lane-health summaries for each sport and publish lane."""
    selected_sports = sports or list(SPORTS.keys())
    report = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sports": {},
    }

    for sport_key in selected_sports:
        sport = SPORTS.get(sport_key, {})
        pick_history_path = os.path.join(data_dir, sport_key, "pick_history.json")
        picks = []
        if os.path.exists(pick_history_path):
            with open(pick_history_path) as handle:
                picks = (json.load(handle) or {}).get("picks", [])

        moneyline = evaluate_lane_health(
            summarize_lane_health(
                picks,
                market_type="moneyline",
                recent_count=int(sport.get("moneyline_health_recent_window", 8) or 8),
            ),
            enabled=True,
            lane_label="moneyline",
            min_evaluated=int(sport.get("publication_min_evaluated_picks", 0) or 0),
            min_recent_evaluated=int(sport.get("moneyline_health_min_recent_evaluated", 0) or 0),
            min_tracked_clv=int(sport.get("moneyline_clv_guard_min_tracked", 0) or 0),
            min_avg_clv=float(sport.get("moneyline_clv_guard_min_avg", 0.0) or 0.0),
            min_recent_roi=float(sport.get("moneyline_health_min_recent_roi", 0.0) or 0.0),
            max_overconfidence_gap=float(sport.get("moneyline_health_max_overconfidence_gap", 0.12) or 0.12),
        )
        totals_enabled = int(sport.get("totals_max_picks", 0) or 0) > 0
        totals = evaluate_lane_health(
            summarize_lane_health(
                picks,
                market_type="total",
                recent_count=int(sport.get("totals_health_recent_window", 8) or 8),
            ),
            enabled=totals_enabled,
            lane_label="totals",
            min_evaluated=int(sport.get("publication_min_evaluated_totals_picks", 0) or 0),
            min_recent_evaluated=int(sport.get("totals_health_min_recent_evaluated", 0) or 0),
            min_tracked_clv=int(sport.get("totals_clv_guard_min_tracked", 0) or 0),
            min_avg_clv=float(sport.get("totals_clv_guard_min_avg", 0.0) or 0.0),
            min_recent_roi=float(sport.get("totals_health_min_recent_roi", 0.0) or 0.0),
            max_overconfidence_gap=float(sport.get("totals_health_max_overconfidence_gap", 0.1) or 0.1),
        )
        longslop = evaluate_lane_health(
            summarize_lane_health(picks, pick_type="longslop", recent_count=5),
            enabled=bool(sport.get("enable_longslop", False)),
            lane_label="longslop",
            min_evaluated=5,
            min_recent_evaluated=3,
            min_tracked_clv=3,
            min_avg_clv=0.0,
            min_recent_roi=0.0,
            max_overconfidence_gap=0.1,
        )
        slimegrinder = evaluate_lane_health(
            summarize_lane_health(picks, pick_type="slimegrinder", recent_count=5),
            enabled=bool(sport.get("enable_slimegrinder", False)),
            lane_label="slimegrinder",
            min_evaluated=5,
            min_recent_evaluated=3,
            min_tracked_clv=3,
            min_avg_clv=0.0,
            min_recent_roi=0.0,
            max_overconfidence_gap=0.1,
        )

        report["sports"][sport_key] = {
            "moneyline": moneyline,
            "total": totals,
            "slop_lock": evaluate_lane_health(
                summarize_lane_health(
                    picks,
                    pick_type="slop_lock",
                    recent_count=int(sport.get("moneyline_health_recent_window", 8) or 8),
                ),
                enabled=int(sport.get("slop_lock_max_picks", 0) or 0) > 0,
                lane_label="slop_lock",
                min_evaluated=min(10, int(sport.get("publication_min_evaluated_picks", 0) or 0)),
                min_recent_evaluated=int(sport.get("moneyline_health_min_recent_evaluated", 0) or 0),
                min_tracked_clv=int(sport.get("moneyline_clv_guard_min_tracked", 0) or 0),
                min_avg_clv=float(sport.get("moneyline_clv_guard_min_avg", 0.0) or 0.0),
                min_recent_roi=float(sport.get("moneyline_health_min_recent_roi", 0.0) or 0.0),
                max_overconfidence_gap=float(sport.get("moneyline_health_max_overconfidence_gap", 0.12) or 0.12),
            ),
            "total_lock": evaluate_lane_health(
                summarize_lane_health(
                    picks,
                    pick_type="total_lock",
                    recent_count=int(sport.get("totals_health_recent_window", 8) or 8),
                ),
                enabled=totals_enabled,
                lane_label="total_lock",
                min_evaluated=min(8, int(sport.get("publication_min_evaluated_totals_picks", 0) or 0)),
                min_recent_evaluated=int(sport.get("totals_health_min_recent_evaluated", 0) or 0),
                min_tracked_clv=int(sport.get("totals_clv_guard_min_tracked", 0) or 0),
                min_avg_clv=float(sport.get("totals_clv_guard_min_avg", 0.0) or 0.0),
                min_recent_roi=float(sport.get("totals_health_min_recent_roi", 0.0) or 0.0),
                max_overconfidence_gap=float(sport.get("totals_health_max_overconfidence_gap", 0.1) or 0.1),
            ),
            "longslop": longslop,
            "slimegrinder": slimegrinder,
        }

    return report


def build_backtest_report(data_dir: str = "data", sports:Optional[ list[str] ] = None) -> dict:
    """Build a historical performance report from saved data files."""
    selected_sports = sports or list(SPORTS.keys())
    report = {
        "data_dir": data_dir,
        "sports": {},
        "aggregate": {
            "predictions": {"evaluated": 0, "accuracy": None, "avg_log_loss": None, "avg_brier": None},
            "picks": {"evaluated": 0, "hit_rate": None, "roi": None},
        },
        "decision_replay": {
            "generated_at": None,
            "aggregate": {
                "logged_picks": 0,
                "settled_picks": {"evaluated": 0, "hit_rate": None, "roi": None},
                "unsettled_logged_picks": 0,
            },
            "sports": {},
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
    report["decision_replay"] = build_pick_decision_replay_report(data_dir=data_dir, sports=selected_sports)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize historical model and pick performance.")
    parser.add_argument("sports", nargs="*", help="Optional sport keys to limit the report.")
    parser.add_argument("--data-dir", default="data", help="Directory containing per-sport pipeline outputs.")
    parser.add_argument("--walkforward", action="store_true", help="Emit the walk-forward replay report instead of the aggregate summary.")
    parser.add_argument("--raw-walkforward", action="store_true", help="Replay models day by day from raw historical inputs.")
    parser.add_argument("--snapshot-replay", action="store_true", help="Replay pick selection from saved live-state snapshots.")
    parser.add_argument("--decision-replay", action="store_true", help="Grade settled picks from the immutable pick-decision ledger.")
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
    elif args.snapshot_replay:
        report = build_snapshot_replay_report(data_dir=args.data_dir, sports=args.sports or None)
    elif args.decision_replay:
        report = build_pick_decision_replay_report(data_dir=args.data_dir, sports=args.sports or None)
    elif args.walkforward:
        report = build_walkforward_report(data_dir=args.data_dir, sports=args.sports or None)
    else:
        report = build_backtest_report(data_dir=args.data_dir, sports=args.sports or None)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
