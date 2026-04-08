"""Pipeline orchestrator for SLOP LOCKS.

Ties together data fetching, model fitting, ensemble blending, backtesting,
and JSON output into a single ``run_pipeline()`` entry point.
"""

import json
import os
import csv
import logging
import datetime as dt
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

from pipeline.config import (
    ANTHROPIC_API_KEY,
    DATA_DIR,
    ENABLE_QUALITATIVE,
    QUALITATIVE_DEFAULT_WEIGHT,
    NBA_B2B_PENALTY,
    NBA_3IN4_PENALTY,
    TRACKING_DIRNAME,
    RESULTS_LOG_FILENAME,
    RESULTS_AUDIT_LOG_FILENAME,
    ODDS_HISTORY_FILENAME,
    PICK_DECISION_LOG_FILENAME,
    SLOP_LOCK_MIN_ODDS,
    SLOP_LOCK_MAX_ODDS,
    SLOP_LOCK_FALLBACK_MIN_ODDS,
    SLIMEGRINDER_MIN_ODDS,
    SLIMEGRINDER_MAX_ODDS,
    SPORTS,
)
try:
    from pipeline.qualitative_analysis import analyze_game_qualitative
    from pipeline.context_scraper import get_game_context
except ImportError:
    analyze_game_qualitative = None

    def get_game_context(*_args, **_kwargs):
        """Fallback when optional qualitative dependencies are unavailable."""
        return ""
from pipeline.fetch_data import fetch_odds
from pipeline.fetch_nba import fetch_nba_games, fetch_nba_schedule, normalize_nba_team_name, fetch_nba_espn_games, fetch_nba_espn_schedule
from pipeline.fetch_nhl import fetch_nhl_games, fetch_nhl_schedule, normalize_nhl_team_name
from pipeline.fetch_mlb import fetch_mlb_games, fetch_mlb_schedule, normalize_mlb_team_name
from pipeline.models import (
    BullpenMatchupModel,
    EloRatings,
    HandednessMatchupModel,
    MlbTotalsModel,
    NbaMatchupModel,
    NbaTotalsModel,
    NhlMatchupModel,
    PitcherMatchupModel,
    RecentBoxScoreModel,
    bullpen_matchup_predict,
    elo_predict,
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
    build_dashboard_data,
    build_model_health_snapshot,
    compute_model_weights,
    compute_roi,
    evaluate_lane_health,
    evaluate_prediction,
    summarize_lane_health,
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


def _lookup_match_odds(odds_lookup: dict, sport_key: str, home_team: str, away_team: str) -> Optional[dict]:
    """Return odds for a fixture."""
    return odds_lookup.get((home_team, away_team))


def _is_live_public_output(base_dir: str) -> bool:
    """Return whether this run is updating the default live data directory.

    Fallback: custom output dirs are treated as research/staging runs so tests
    and offline experiments can still inspect candidate picks before a sport is
    allowed to publish them live.
    """
    try:
        return os.path.abspath(base_dir) == os.path.abspath(DATA_DIR)
    except OSError:
        return False


def _parse_prediction_date(prediction: dict) -> Optional[date]:
    """Parse the best available calendar day from one saved prediction."""
    for key in ("date", "match_date", "snapshot_timestamp", "generated_at"):
        value = prediction.get(key)
        if not value:
            continue
        text = str(value).strip()
        try:
            return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        except ValueError:
            pass
        try:
            return dt.datetime.strptime(text[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
    return None


def _select_calibration_predictions(
    predictions: list[dict],
    lookback_days: Optional[int],
    holdout_days: int,
    as_of: date,
) -> list[dict]:
    """Return the dedicated trailing slice used for probability calibration."""
    cutoff = as_of - timedelta(days=max(holdout_days, 0))
    floor_day = None
    if lookback_days:
        floor_day = cutoff - timedelta(days=max(int(lookback_days) - 1, 0))

    selected = []
    for prediction in predictions:
        if not prediction.get("evaluated"):
            continue
        pred_day = _parse_prediction_date(prediction)
        if pred_day is None:
            continue
        if pred_day > cutoff:
            continue
        if floor_day is not None and pred_day < floor_day:
            continue
        selected.append(prediction)
    return selected


def _build_publication_guard(
    past_picks: list[dict],
    sport: dict,
    enforce_live_guard: bool,
) -> dict:
    """Return whether a sport has enough settled evidence to publish live picks."""
    evaluated_moneylines = [
        pick for pick in past_picks
        if pick.get("evaluated") and str(pick.get("market_type") or "moneyline") == "moneyline"
    ]
    evaluated_totals = [
        pick for pick in past_picks
        if pick.get("evaluated") and str(pick.get("market_type") or "moneyline") == "total"
    ]
    min_picks = int(sport.get("publication_min_evaluated_picks", 0) or 0)
    min_totals = int(sport.get("publication_min_evaluated_totals_picks", 0) or 0)
    totals_enabled = int(sport.get("totals_max_picks", 0) or 0) > 0
    longslop_enabled = bool(sport.get("enable_longslop", False))
    slimegrinder_enabled = bool(sport.get("enable_slimegrinder", False))

    moneyline_guard = evaluate_lane_health(
        summarize_lane_health(
            past_picks,
            market_type="moneyline",
            recent_count=int(sport.get("moneyline_health_recent_window", 8) or 8),
        ),
        enabled=True,
        lane_label="moneylines",
        min_evaluated=min_picks,
        min_recent_evaluated=int(sport.get("moneyline_health_min_recent_evaluated", 0) or 0),
        min_tracked_clv=int(sport.get("moneyline_clv_guard_min_tracked", 0) or 0),
        min_avg_clv=float(sport.get("moneyline_clv_guard_min_avg", 0.0) or 0.0),
        min_recent_roi=float(sport.get("moneyline_health_min_recent_roi", 0.0) or 0.0),
        max_overconfidence_gap=float(sport.get("moneyline_health_max_overconfidence_gap", 0.12) or 0.12),
    )
    totals_guard = evaluate_lane_health(
        summarize_lane_health(
            past_picks,
            market_type="total",
            recent_count=int(sport.get("totals_health_recent_window", 8) or 8),
        ),
        enabled=totals_enabled,
        lane_label="totals",
        min_evaluated=min_totals,
        min_recent_evaluated=int(sport.get("totals_health_min_recent_evaluated", 0) or 0),
        min_tracked_clv=int(sport.get("totals_clv_guard_min_tracked", 0) or 0),
        min_avg_clv=float(sport.get("totals_clv_guard_min_avg", 0.0) or 0.0),
        min_recent_roi=float(sport.get("totals_health_min_recent_roi", 0.0) or 0.0),
        max_overconfidence_gap=float(sport.get("totals_health_max_overconfidence_gap", 0.1) or 0.1),
    )
    longslop_guard = evaluate_lane_health(
        summarize_lane_health(past_picks, pick_type="longslop", recent_count=5),
        enabled=longslop_enabled,
        lane_label="longslop",
        min_evaluated=5,
        min_recent_evaluated=3,
        min_tracked_clv=3,
        min_avg_clv=0.0,
        min_recent_roi=0.0,
        max_overconfidence_gap=0.1,
    )
    slimegrinder_guard = evaluate_lane_health(
        summarize_lane_health(past_picks, pick_type="slimegrinder", recent_count=5),
        enabled=slimegrinder_enabled,
        lane_label="slimegrinder",
        min_evaluated=5,
        min_recent_evaluated=3,
        min_tracked_clv=3,
        min_avg_clv=0.0,
        min_recent_roi=0.0,
        max_overconfidence_gap=0.1,
    )

    if not enforce_live_guard:
        return {
            "enforced": False,
            "allow_moneyline": True,
            "allow_totals": totals_enabled,
            "allow_longslop": longslop_enabled,
            "allow_slimegrinder": slimegrinder_enabled,
            "evaluated_picks": len(evaluated_moneylines),
            "evaluated_totals_picks": len(evaluated_totals),
            "min_evaluated_picks": min_picks,
            "min_evaluated_totals_picks": min_totals,
            "status": "research",
            "reason": None,
            "lane_guards": {
                "moneyline": {**moneyline_guard, "allow": True},
                "totals": {**totals_guard, "allow": totals_enabled},
                "longslop": {**longslop_guard, "allow": longslop_enabled},
                "slimegrinder": {**slimegrinder_guard, "allow": slimegrinder_enabled},
            },
        }

    allow_moneyline = moneyline_guard.get("allow", True)
    allow_totals = totals_guard.get("allow", totals_enabled)
    allow_longslop = allow_moneyline and longslop_guard.get("allow", False)
    allow_slimegrinder = allow_moneyline and slimegrinder_guard.get("allow", False)
    reasons = []
    if not allow_moneyline:
        reasons.extend(moneyline_guard.get("reasons", []))
    if not allow_totals:
        reasons.extend(totals_guard.get("reasons", []))

    if allow_moneyline and allow_totals:
        status = "live"
    elif allow_moneyline or allow_totals:
        status = "partial"
    else:
        status = "suppressed"

    return {
        "enforced": True,
        "allow_moneyline": allow_moneyline,
        "allow_totals": allow_totals,
        "allow_longslop": allow_longslop,
        "allow_slimegrinder": allow_slimegrinder,
        "evaluated_picks": len(evaluated_moneylines),
        "evaluated_totals_picks": len(evaluated_totals),
        "min_evaluated_picks": min_picks,
        "min_evaluated_totals_picks": min_totals,
        "status": status,
        "reason": "; ".join(reason for reason in reasons if reason) or None,
        "lane_guards": {
            "moneyline": moneyline_guard,
            "totals": totals_guard,
            "longslop": longslop_guard,
            "slimegrinder": slimegrinder_guard,
            "totals_enabled": totals_enabled,
        },
    }


def _resolve_start_time(fixture: dict, match_odds: Optional[dict]) -> Optional[str]:
    """Return the best available start time for a fixture."""
    start_time = fixture.get("start_time")
    if start_time:
        return start_time
    if match_odds:
        return match_odds.get("commence_time")
    return None


_SNAPSHOT_VERSION = 1


_RESULTS_LOG_FIELDS = [
    "logged_at",
    "run_id",
    "run_type",
    "snapshot_timestamp",
    "snapshot_path",
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
    "market_source",
    "market_books",
    "hold",
    "market_snapshot_json",
]

_PICK_DECISION_FIELDS = [
    "logged_at",
    "run_id",
    "run_type",
    "snapshot_timestamp",
    "snapshot_path",
    "sport",
    "pick_type",
    "market_type",
    "home_team",
    "away_team",
    "match_date",
    "start_time",
    "pick",
    "total_line",
    "expected_total",
    "total_stddev",
    "model_prob",
    "implied_prob",
    "market_implied_prob",
    "edge",
    "expected_value",
    "american_odds",
    "decimal_odds",
    "confidence_score",
    "kelly_fraction",
    "fractional_kelly",
    "market_source",
    "market_books",
    "hold",
    "publication_guard_status",
    "publication_guard_reason",
    "publication_guard_enforced",
    "publication_guard_evaluated_picks",
    "publication_guard_evaluated_totals_picks",
    "calibration_sample_size",
    "selection_min_expected_value",
    "selection_edge_floor",
    "selection_probability_floor",
    "selection_confidence_floor",
    "selection_confidence_dropoff",
    "selection_max_picks",
    "market_snapshot_json",
    "model_probs_json",
    "individual_models_json",
    "decision_context_json",
    "gate_context_json",
]


def _results_log_path(base_dir: str) -> str:
    """Return the CSV path for the persistent resolved-results log."""
    return os.path.join(base_dir, TRACKING_DIRNAME, RESULTS_LOG_FILENAME)


def _odds_history_path(base_dir: str) -> str:
    """Return the CSV path for tracked odds snapshots."""
    return os.path.join(base_dir, TRACKING_DIRNAME, ODDS_HISTORY_FILENAME)


def _results_audit_log_path(base_dir: str) -> str:
    """Return the CSV path for the append-only results ledger."""
    return os.path.join(base_dir, TRACKING_DIRNAME, RESULTS_AUDIT_LOG_FILENAME)


def _pick_decision_log_path(base_dir: str) -> str:
    """Return the CSV path for the append-only pick decision ledger."""
    return os.path.join(base_dir, TRACKING_DIRNAME, PICK_DECISION_LOG_FILENAME)


def _snapshot_root_dir(base_dir: str) -> str:
    """Return the directory that stores persisted live-state snapshots."""
    return os.path.join(base_dir, TRACKING_DIRNAME, "snapshots")


def _build_run_context(run_type: str = "manual", now: Optional[datetime] = None) -> dict:
    """Build metadata for one pipeline or refresh execution."""
    now = now or datetime.now(timezone.utc)
    return {
        "run_id": f"{run_type}-{now.strftime('%Y%m%dT%H%M%S%fZ')}",
        "run_type": run_type,
        "run_timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _snapshot_relative_path(sport_key: str, run_context: dict) -> str:
    """Return the relative JSON path for one sport snapshot."""
    run_date = str(run_context.get("run_timestamp") or "")[:10] or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return os.path.join("tracking", "snapshots", run_date, sport_key, f"{run_context['run_id']}.json")


def _snapshot_full_path(base_dir: str, sport_key: str, run_context: dict) -> str:
    """Return the absolute JSON path for one sport snapshot."""
    return os.path.join(base_dir, _snapshot_relative_path(sport_key, run_context))


def _attach_run_metadata(record: dict, run_context: dict, snapshot_path: Optional[str]) -> dict:
    """Attach stable run metadata to one record."""
    if not isinstance(record, dict):
        return record
    record["run_id"] = run_context.get("run_id")
    record["run_type"] = run_context.get("run_type")
    record["snapshot_timestamp"] = run_context.get("run_timestamp")
    record["snapshot_path"] = snapshot_path
    return record


def _attach_run_metadata_list(records: list[dict], run_context: dict, snapshot_path: Optional[str]) -> list[dict]:
    """Attach run metadata to each record in a list."""
    for record in records:
        _attach_run_metadata(record, run_context, snapshot_path)
    return records


def _selection_snapshot_config(sport: dict, outcomes: list[str], min_expected_value: float) -> dict:
    """Capture the selection gates used by this run."""
    return {
        "outcomes": list(outcomes),
        "slop_locks": {
            "min_expected_value": min_expected_value,
            "edge_floor": sport.get("slop_lock_edge_threshold", 0.03),
            "probability_floor": sport.get("slop_lock_probability_floor", 0.45),
            "additional_confidence_floor": sport.get("slop_lock_confidence_threshold", 65.0),
            "confidence_dropoff": sport.get("slop_lock_confidence_dropoff", 0.0),
            "max_picks": sport.get("slop_lock_max_picks", 3),
        },
        "longslop": {
            "enabled": bool(sport.get("enable_longslop", False)),
            "min_expected_value": min_expected_value,
            "confidence_floor": sport.get("longslop_confidence_threshold", 65.0),
        },
        "slimegrinder": {
            "enabled": bool(sport.get("enable_slimegrinder", False)),
            "min_expected_value": min_expected_value,
            "confidence_floor": sport.get("slimegrinder_confidence_threshold", 65.0),
        },
        "totals_locks": {
            "enabled": int(sport.get("totals_max_picks", 0) or 0) > 0,
            "min_expected_value": sport.get("totals_min_expected_value", min_expected_value),
            "edge_floor": sport.get("totals_edge_threshold", 0.02),
            "probability_floor": sport.get("totals_probability_floor", 0.53),
            "confidence_floor": sport.get("totals_confidence_threshold", 54.0),
            "max_picks": sport.get("totals_max_picks", 3),
        },
    }


def _write_run_snapshot(base_dir: str, sport_key: str, run_context: dict, snapshot_payload: dict) -> str:
    """Persist a live-state snapshot and return its relative path."""
    snapshot_path = _snapshot_full_path(base_dir, sport_key, run_context)
    _save_json(snapshot_path, snapshot_payload)
    return _snapshot_relative_path(sport_key, run_context)


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
        "run_id": record.get("run_id"),
        "run_type": record.get("run_type"),
        "snapshot_timestamp": record.get("snapshot_timestamp"),
        "snapshot_path": record.get("snapshot_path"),
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


def _append_results_audit_log(path: str, rows: list[dict]) -> None:
    """Append rows to the immutable audit ledger.

    The audit log is intentionally append-only. If a maintenance command later
    rewrites live summary files, this ledger still preserves the original
    settled-result stream for reporting and forensic checks.
    """
    if not rows:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_RESULTS_LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in _RESULTS_LOG_FIELDS})


def _pick_decision_key(row: dict) -> tuple:
    """Stable dedupe key for one published pick-decision row."""
    return (
        row["sport"],
        row["pick_type"],
        row["market_type"],
        row["home_team"],
        row["away_team"],
        row["match_date"],
        row["pick"],
    )


def _append_pick_decision_log(path: str, rows: list[dict]) -> None:
    """Append deduped pick-decision rows to the immutable ledger."""
    if not rows:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing_keys = set()
    if os.path.exists(path):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_keys.add(_pick_decision_key(row))

    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_PICK_DECISION_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            key = _pick_decision_key(row)
            if key in existing_keys:
                continue
            writer.writerow({field: row.get(field, "") for field in _PICK_DECISION_FIELDS})
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


def _json_compact(value):
    """Serialize one payload into stable compact JSON for CSV storage."""
    if value in (None, "", [], {}):
        return ""
    return json.dumps(value, cls=_NumpyEncoder, sort_keys=True, separators=(",", ":"))


def _record_lookup_key(record: dict, market_type: str) -> tuple:
    """Build a stable lookup key for one modeled fixture record."""
    return (
        market_type,
        record.get("home_team"),
        record.get("away_team"),
        str(record.get("date") or record.get("match_date") or "")[:10],
    )


def _build_record_lookup(records: list[dict], market_type: str) -> dict[tuple, dict]:
    """Index modeled records by fixture and market type."""
    return {
        _record_lookup_key(record, market_type): record
        for record in records
        if record.get("home_team") and record.get("away_team")
    }


def _selection_config_for_pick(selection_config: dict, pick_type: str) -> dict:
    """Return the gate config that produced one published pick."""
    key_map = {
        "slop_lock": "slop_locks",
        "longslop": "longslop",
        "total_lock": "totals_locks",
    }
    return (selection_config or {}).get(key_map.get(pick_type, ""), {}) or {}


def _build_pick_decision_row(
    sport_key: str,
    pick_type: str,
    pick_record: dict,
    source_record: Optional[dict],
    publication_guard: dict,
    selection_config: dict,
    calibration_sample_size: int,
) -> dict:
    """Normalize one newly published pick into an append-only decision row."""
    market_type = str(pick_record.get("market_type") or "moneyline")
    source_record = source_record or {}
    model_probs = source_record.get("model_probs") or {}
    edge_data = ((source_record.get("edges") or {}).get(pick_record.get("pick")) or {})
    lane_config = _selection_config_for_pick(selection_config, pick_type)

    # Fallback: if the source record is missing, persist the published pick row
    # itself so the decision ledger still captures what users actually saw.
    decision_context = source_record or dict(pick_record)
    gate_context = {
        "pick_type": pick_type,
        "selection_config": lane_config,
        "publication_guard": publication_guard,
    }

    return {
        "logged_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": pick_record.get("run_id"),
        "run_type": pick_record.get("run_type"),
        "snapshot_timestamp": pick_record.get("snapshot_timestamp"),
        "snapshot_path": pick_record.get("snapshot_path"),
        "sport": sport_key,
        "pick_type": pick_type,
        "market_type": market_type,
        "home_team": pick_record.get("home_team"),
        "away_team": pick_record.get("away_team"),
        "match_date": str(pick_record.get("match_date") or pick_record.get("date") or "")[:10],
        "start_time": pick_record.get("start_time"),
        "pick": pick_record.get("pick"),
        "total_line": pick_record.get("total_line", source_record.get("total_line")),
        "expected_total": pick_record.get("expected_total", source_record.get("expected_total")),
        "total_stddev": source_record.get("total_stddev"),
        "model_prob": pick_record.get("model_prob"),
        "implied_prob": pick_record.get("implied_prob", edge_data.get("implied_prob")),
        "market_implied_prob": pick_record.get("market_implied_prob", edge_data.get("market_implied_prob")),
        "edge": pick_record.get("edge", edge_data.get("edge")),
        "expected_value": pick_record.get("expected_value", edge_data.get("expected_value")),
        "american_odds": pick_record.get("american_odds", edge_data.get("american_odds")),
        "decimal_odds": pick_record.get("decimal_odds", edge_data.get("decimal_odds")),
        "confidence_score": pick_record.get("confidence_score", edge_data.get("confidence_score")),
        "kelly_fraction": pick_record.get("kelly_fraction", edge_data.get("kelly_fraction")),
        "fractional_kelly": pick_record.get("fractional_kelly", edge_data.get("fractional_kelly")),
        "market_source": edge_data.get("market_source"),
        "market_books": edge_data.get("market_books"),
        "hold": edge_data.get("hold"),
        "publication_guard_status": publication_guard.get("status"),
        "publication_guard_reason": publication_guard.get("reason"),
        "publication_guard_enforced": publication_guard.get("enforced"),
        "publication_guard_evaluated_picks": publication_guard.get("evaluated_picks"),
        "publication_guard_evaluated_totals_picks": publication_guard.get("evaluated_totals_picks"),
        "calibration_sample_size": calibration_sample_size,
        "selection_min_expected_value": lane_config.get("min_expected_value"),
        "selection_edge_floor": lane_config.get("edge_floor"),
        "selection_probability_floor": lane_config.get("probability_floor"),
        "selection_confidence_floor": lane_config.get("additional_confidence_floor", lane_config.get("confidence_floor")),
        "selection_confidence_dropoff": lane_config.get("confidence_dropoff"),
        "selection_max_picks": lane_config.get("max_picks"),
        "market_snapshot_json": _json_compact(source_record.get("market_snapshot")),
        "model_probs_json": _json_compact(model_probs),
        "individual_models_json": _json_compact(source_record.get("individual_models")),
        "decision_context_json": _json_compact(decision_context),
        "gate_context_json": _json_compact(gate_context),
    }


def _backfill_pick_decision_log_from_snapshots(base_dir: str, sports:Optional[ list[str] ] = None) -> int:
    """Backfill the pick-decision ledger from immutable saved snapshots.

    Fallback: older snapshots may not have embedded run metadata on each pick,
    so the snapshot header is treated as the authoritative source for those
    fields during ledger reconstruction.
    """
    snapshot_root = _snapshot_root_dir(base_dir)
    if not os.path.exists(snapshot_root):
        return 0

    selected_sports = set(sports or SPORTS.keys())
    ledger_path = _pick_decision_log_path(base_dir)
    rows = []
    lane_specs = (
        ("slop_lock", "moneyline", "slop_locks"),
        ("longslop", "moneyline", "longslop"),
        ("total_lock", "total", "totals_locks"),
    )

    for current_root, _, files in os.walk(snapshot_root):
        for filename in files:
            if not filename.endswith(".json"):
                continue
            path = os.path.join(current_root, filename)
            try:
                with open(path) as f:
                    snapshot = json.load(f) or {}
            except (OSError, json.JSONDecodeError):
                continue

            sport_key = snapshot.get("sport")
            if sport_key not in selected_sports:
                continue

            records = snapshot.get("records") or {}
            record_lookup = {}
            record_lookup.update(_build_record_lookup(records.get("matches") or [], "moneyline"))
            record_lookup.update(_build_record_lookup(records.get("totals_matches") or [], "total"))
            selection_config = snapshot.get("selection_config") or {}
            publication_guard = snapshot.get("publication_guard") or {}
            calibration_sample_size = int(((snapshot.get("inputs") or {}).get("calibration_sample_size") or 0) or 0)
            snapshot_relpath = os.path.relpath(path, base_dir)
            outputs = snapshot.get("outputs") or {}

            for pick_type, market_type, output_key in lane_specs:
                raw_items = outputs.get(output_key)
                if isinstance(raw_items, dict):
                    items = [raw_items]
                else:
                    items = list(raw_items or [])
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    pick_record = dict(item)
                    match_date = str(item.get("match_date") or item.get("date") or "")[:10]
                    pick_record.setdefault("market_type", market_type)
                    pick_record.setdefault("match_date", match_date)
                    pick_record.setdefault("run_id", snapshot.get("run_id"))
                    pick_record.setdefault("run_type", snapshot.get("run_type"))
                    pick_record.setdefault("snapshot_timestamp", snapshot.get("snapshot_timestamp"))
                    pick_record.setdefault("snapshot_path", snapshot.get("snapshot_path") or snapshot_relpath)
                    source_record = record_lookup.get((
                        market_type,
                        pick_record.get("home_team"),
                        pick_record.get("away_team"),
                        match_date,
                    ))
                    rows.append(
                        _build_pick_decision_row(
                            sport_key,
                            pick_type,
                            pick_record,
                            source_record,
                            publication_guard,
                            selection_config,
                            calibration_sample_size,
                        )
                    )

    _append_pick_decision_log(ledger_path, rows)
    return len(rows)


def _build_odds_snapshot_rows(sport_key: str, odds_list: list[dict]) -> list[dict]:
    """Normalize current odds into per-outcome snapshot rows."""
    logged_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []

    for odds in odds_list:
        match_date = str(odds.get("commence_time", ""))[:10]

        moneyline_decimals = {
            outcome: odds.get(f"{outcome}_odds", 0.0)
            for outcome in ("home", "away", "draw")
            if odds.get(f"{outcome}_odds", 0.0) and odds.get(f"{outcome}_odds", 0.0) > 1.0
        }
        fair_probs = {}
        raw_probs = {}
        if moneyline_decimals:
            raw_probs, fair_probs, _ = no_vig_probabilities(moneyline_decimals)
            moneyline_benchmark = odds.get("moneyline_benchmark") or {}
            fair_probs = moneyline_benchmark.get("fair_probs") or fair_probs
            raw_probs = moneyline_benchmark.get("raw_probs") or raw_probs
            market_snapshot = odds.get("moneyline_market_snapshot") or {}
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
                    "market_source": moneyline_benchmark.get("source"),
                    "market_books": moneyline_benchmark.get("books_tracked"),
                    "hold": moneyline_benchmark.get("hold"),
                    "market_snapshot_json": _json_compact(market_snapshot),
                })

        totals_decimals = {
            outcome: odds.get(f"{outcome}_odds", 0.0)
            for outcome in ("over", "under")
            if odds.get(f"{outcome}_odds", 0.0) and odds.get(f"{outcome}_odds", 0.0) > 1.0
        }
        if totals_decimals and odds.get("total_line") is not None:
            raw_probs, fair_probs, _ = no_vig_probabilities(totals_decimals)
            totals_benchmark = odds.get("totals_benchmark") or {}
            fair_probs = totals_benchmark.get("fair_probs") or fair_probs
            raw_probs = totals_benchmark.get("raw_probs") or raw_probs
            market_snapshot = odds.get("totals_market_snapshot") or {}
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
                    "market_source": totals_benchmark.get("source"),
                    "market_books": totals_benchmark.get("books_tracked"),
                    "hold": totals_benchmark.get("hold"),
                    "market_snapshot_json": _json_compact(market_snapshot),
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
        pick["closing_market_source"] = snapshot.get("market_source")
        pick["closing_market_books"] = _safe_int(snapshot.get("market_books"))
        pick["closing_hold"] = _safe_float(snapshot.get("hold"))
        pick["closing_market_snapshot_json"] = snapshot.get("market_snapshot_json")

        if market_type == "total":
            closing_total_line = _safe_float(snapshot.get("total_line"))
            pick["closing_total_line"] = closing_total_line
            opening_total_line = _safe_float(pick.get("total_line"))
            if closing_total_line is not None and opening_total_line is not None:
                if pick.get("pick") == "over":
                    pick["closing_line_value"] = round(closing_total_line - opening_total_line, 3)
                elif pick.get("pick") == "under":
                    pick["closing_line_value"] = round(opening_total_line - closing_total_line, 3)
                pick["closing_line_value_unit"] = "total_points"
        else:
            opening_prob = pick.get("market_implied_prob")
            if opening_prob is None:
                opening_prob = pick.get("implied_prob")
            opening_prob = _safe_float(opening_prob)
            closing_prob = _safe_float(snapshot.get("implied_prob"))
            if opening_prob is not None and closing_prob is not None:
                pick["closing_line_value"] = round(closing_prob - opening_prob, 4)
                pick["closing_line_value_unit"] = "implied_probability_points"


def _backfill_pick_history_market_snapshots(base_dir: str, sports:Optional[ list[str] ] = None) -> int:
    """Backfill missing closing-line fields in saved pick history from odds snapshots."""
    odds_history_path = _odds_history_path(base_dir)
    selected_sports = list(sports or SPORTS.keys())
    updated = 0
    for sport_key in selected_sports:
        pick_history_path = os.path.join(base_dir, sport_key, "pick_history.json")
        pick_history = _load_json(pick_history_path)
        if not isinstance(pick_history, dict):
            continue
        picks = pick_history.get("picks")
        if not isinstance(picks, list) or not picks:
            continue
        latest_snapshot_lookup = _load_latest_odds_snapshots(odds_history_path, sport_key)
        before = [
            (
                pick.get("closing_line_value"),
                pick.get("closing_american_odds"),
                pick.get("closing_total_line"),
            )
            for pick in picks
        ]
        _apply_latest_market_snapshots(picks, latest_snapshot_lookup)
        after = [
            (
                pick.get("closing_line_value"),
                pick.get("closing_american_odds"),
                pick.get("closing_total_line"),
            )
            for pick in picks
        ]
        if after != before:
            pick_history["picks"] = picks
            _save_json(pick_history_path, pick_history)
            updated += 1
    return updated


def _basic_market_snapshot_from_odds_row(odds_row: dict, market_type: str) -> dict:
    """Build a minimal market snapshot from one saved odds payload."""
    if not isinstance(odds_row, dict):
        return {}
    if market_type == "total":
        total_line = odds_row.get("total_line")
        if total_line in (None, "", "None"):
            return {}
        snapshot = {
            "line": _safe_float(total_line),
            "execution_prices": {},
        }
        for outcome in ("over", "under"):
            price = _safe_float(odds_row.get(f"{outcome}_odds"))
            if price and price > 1.0:
                snapshot["execution_prices"][outcome] = round(float(price), 4)
        return snapshot if snapshot["execution_prices"] else {}

    snapshot = {"execution_prices": {}}
    for outcome in ("home", "away", "draw"):
        price = _safe_float(odds_row.get(f"{outcome}_odds"))
        if price and price > 1.0:
            snapshot["execution_prices"][outcome] = round(float(price), 4)
    return snapshot if snapshot["execution_prices"] else {}


def _hydrate_pick_decision_log_market_snapshots(base_dir: str, sports:Optional[ list[str] ] = None) -> int:
    """Fill missing market snapshot JSON in the decision ledger from saved snapshots."""
    ledger_path = _pick_decision_log_path(base_dir)
    if not os.path.exists(ledger_path):
        return 0

    selected_sports = set(sports or SPORTS.keys())
    updated = 0
    snapshot_cache: dict[str, Optional[dict]] = {}

    with open(ledger_path, newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        if row.get("sport") not in selected_sports:
            continue
        if row.get("market_snapshot_json"):
            continue
        snapshot_relpath = row.get("snapshot_path")
        if not snapshot_relpath:
            continue
        snapshot_full_path = os.path.join(base_dir, snapshot_relpath)
        if snapshot_full_path not in snapshot_cache:
            try:
                with open(snapshot_full_path) as f:
                    snapshot_cache[snapshot_full_path] = json.load(f) or {}
            except (OSError, json.JSONDecodeError):
                snapshot_cache[snapshot_full_path] = None
        snapshot = snapshot_cache.get(snapshot_full_path) or {}
        odds_rows = ((snapshot.get("inputs") or {}).get("odds") or [])
        match_date = str(row.get("match_date") or "")[:10]
        odds_row = next((
            item for item in odds_rows
            if item.get("home_team") == row.get("home_team")
            and item.get("away_team") == row.get("away_team")
            and str(item.get("commence_time") or "")[:10] == match_date
        ), None)
        market_snapshot = _basic_market_snapshot_from_odds_row(
            odds_row or {},
            str(row.get("market_type") or "moneyline"),
        )
        if not market_snapshot:
            continue
        row["market_snapshot_json"] = _json_compact(market_snapshot)
        updated += 1

    if updated:
        with open(ledger_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_PICK_DECISION_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in _PICK_DECISION_FIELDS})
    return updated



def _days_since_last_game(team: str, before_date: str, matches: pd.DataFrame) -> Optional[int]:
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


def _apply_qualitative_adjustment(
    blended: dict[str, float],
    qualitative_data: dict,
    weight: float = 0.5,
) -> dict[str, float]:
    """Apply qualitative impact scores as a probability adjustment."""
    home_impact = float(qualitative_data.get("home_impact", 0.0))
    away_impact = float(qualitative_data.get("away_impact", 0.0))
    
    # Differential: positive means home team has qualitative edge
    differential = home_impact - away_impact
    
    # Scale: QUALITATIVE_DEFAULT_WEIGHT (0.005) * weight per impact point
    delta = differential * QUALITATIVE_DEFAULT_WEIGHT * weight
    
    adjusted = {
        "home": min(0.99, max(0.01, blended.get("home", 0.5) + delta)),
        "away": min(0.99, max(0.01, blended.get("away", 0.5) - delta)),
    }
    return _normalize_two_way_probs(adjusted)


def _normalize_two_way_probs(probs: dict[str, float]) -> dict[str, float]:
    """Renormalize two-way probabilities after a heuristic adjustment."""
    total = max(1e-9, probs.get("home", 0.0) + probs.get("away", 0.0))
    return {
        "home": probs.get("home", 0.0) / total,
        "away": probs.get("away", 0.0) / total,
    }


def _apply_mlb_weather_adjustment(
    blended: dict[str, float],
    run_environment_probs: dict[str,Optional[ float] ],
    weather: Optional[dict],
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
    weather: Optional[dict],
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


def _compute_mlb_lineup_index(lineup_profile: Optional[dict], opposing_pitcher_hand: Optional[str]) -> float:
    """Score a current MLB roster profile for the handedness of today's matchup."""
    if not lineup_profile:
        return 0.0

    if lineup_profile.get("confirmed_lineup"):
        lefty_share = float(lineup_profile.get("confirmed_lefty_share", lineup_profile.get("lefty_share", 0.0)) or 0.0)
        righty_share = float(lineup_profile.get("confirmed_righty_share", lineup_profile.get("righty_share", 0.0)) or 0.0)
        switch_share = float(lineup_profile.get("confirmed_switch_share", lineup_profile.get("switch_share", 0.0)) or 0.0)
    else:
        lefty_share = float(lineup_profile.get("lefty_share", 0.0) or 0.0)
        righty_share = float(lineup_profile.get("righty_share", 0.0) or 0.0)
        switch_share = float(lineup_profile.get("switch_share", 0.0) or 0.0)
    active_hitters = float(lineup_profile.get("active_hitters", 0.0) or 0.0)
    available_hitters = float(lineup_profile.get("available_hitters", 0.0) or 0.0)
    injured_hitters = float(lineup_profile.get("injured_hitters", 0.0) or 0.0)
    key_bat_absence_score = float(lineup_profile.get("key_bat_absence_score", 0.0) or 0.0)
    leader_absence_burden = float(lineup_profile.get("leader_absence_burden", 0.0) or 0.0)
    confirmed_hitters = float(lineup_profile.get("confirmed_hitters", 0.0) or 0.0)
    confirmed_top_order_score = float(lineup_profile.get("confirmed_top_order_score", 0.0) or 0.0)
    confirmed_leader_absence_burden = float(lineup_profile.get("confirmed_leader_absence_burden", 0.0) or 0.0)

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
    confirmed_depth_term = 0.0
    confirmed_top_order_term = 0.0
    if lineup_profile.get("confirmed_lineup"):
        confirmed_depth_term = max(-1.0, min(1.0, (confirmed_hitters - 9.0) / 2.0))
        confirmed_top_order_term = max(-1.0, min(1.0, (confirmed_top_order_score - 0.82) / 0.18))

    return (
        (0.6 * platoon_term)
        + (0.2 * depth_term)
        + (0.15 * availability_rate)
        - (0.18 * injury_term)
        - (0.28 * key_bat_absence_score)
        - (0.24 * leader_absence_burden)
        + (0.12 * confirmed_depth_term)
        + (0.16 * confirmed_top_order_term)
        - (0.16 * confirmed_leader_absence_burden)
    )


def _team_bullpen_rows(team: str, before_date: str, matches: pd.DataFrame, recent_days: int = 3) -> list[dict]:
    """Return recent bullpen workload rows for one MLB team before a date."""
    if matches is None or matches.empty or recent_days <= 0:
        return []

    cutoff = pd.to_datetime(before_date[:10])
    window_start = cutoff - timedelta(days=recent_days)
    rows = []
    for _, row in matches.iterrows():
        row_date = pd.to_datetime(row["date"])
        if row_date >= cutoff or row_date < window_start:
            continue
        if row["home_team"] == team:
            rows.append({
                "date": row["date"],
                "innings": float(row.get("home_bullpen_ip", 0.0) or 0.0),
                "earned_runs": float(row.get("home_bullpen_earned_runs", 0.0) or 0.0),
            })
        elif row["away_team"] == team:
            rows.append({
                "date": row["date"],
                "innings": float(row.get("away_bullpen_ip", 0.0) or 0.0),
                "earned_runs": float(row.get("away_bullpen_earned_runs", 0.0) or 0.0),
            })
    rows.sort(key=lambda item: item["date"])
    return rows


def _compute_mlb_bullpen_tax(
    team: str,
    before_date: str,
    matches: pd.DataFrame,
    recent_days: int = 3,
    usage_baseline: float = 6.5,
    last_game_baseline: float = 3.5,
) -> float:
    """Estimate a same-day bullpen fatigue tax from recent innings load."""
    rows = _team_bullpen_rows(team, before_date, matches, recent_days=recent_days)
    if not rows:
        return 0.0

    total_ip = sum(item["innings"] for item in rows)
    last_game_ip = rows[-1]["innings"]
    earned_runs = sum(item["earned_runs"] for item in rows)
    usage_term = max(0.0, total_ip - usage_baseline) / max(usage_baseline, 1.0)
    last_game_term = max(0.0, last_game_ip - last_game_baseline) / max(last_game_baseline, 1.0)
    runs_term = max(0.0, earned_runs - 2.0) / 6.0
    return max(0.0, min(1.5, (0.55 * usage_term) + (0.3 * last_game_term) + (0.15 * runs_term)))


def _apply_mlb_bullpen_availability_adjustment(
    blended: dict[str, float],
    home_tax: float = 0.0,
    away_tax: float = 0.0,
    max_delta: float = 0.012,
) -> dict[str, float]:
    """Shade MLB sides away from the team carrying heavier recent bullpen tax."""
    differential = float(away_tax or 0.0) - float(home_tax or 0.0)
    if abs(differential) < 0.05:
        return blended
    delta = max(-max_delta, min(max_delta, differential * max_delta * 0.75))
    if abs(delta) < 0.001:
        return blended
    adjusted = {
        "home": min(0.99, max(0.01, blended.get("home", 0.5) + delta)),
        "away": min(0.99, max(0.01, blended.get("away", 0.5) - delta)),
    }
    return _normalize_two_way_probs(adjusted)


def _apply_mlb_bullpen_total_adjustment(
    expected_total: float,
    home_tax: float = 0.0,
    away_tax: float = 0.0,
    max_runs_delta: float = 0.3,
) -> float:
    """Push MLB totals upward modestly when both bullpens are carrying fatigue."""
    combined_tax = float(home_tax or 0.0) + float(away_tax or 0.0)
    if combined_tax < 0.1:
        return expected_total
    delta = max(0.0, min(max_runs_delta, combined_tax * max_runs_delta * 0.6))
    return max(4.5, min(16.0, expected_total + delta))


def _apply_mlb_lineup_adjustment(
    blended: dict[str, float],
    home_lineup_profile: Optional[dict],
    away_lineup_profile: Optional[dict],
    home_pitcher_hand: Optional[str],
    away_pitcher_hand: Optional[str],
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
    home_lineup_profile: Optional[dict],
    away_lineup_profile: Optional[dict],
    home_pitcher_hand: Optional[str],
    away_pitcher_hand: Optional[str],
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


def _compute_nba_availability_index(profile: Optional[dict]) -> float:
    """Score a live NBA availability profile."""
    if not profile:
        return 0.0
    active_players = float(profile.get("active_players", 0.0) or 0.0)
    available_core_players = float(profile.get("available_core_players", 0.0) or 0.0)
    injury_burden = float(profile.get("injury_burden", 0.0) or 0.0)
    key_absence_score = float(profile.get("key_absence_score", 0.0) or 0.0)
    leader_absence_burden = float(profile.get("leader_absence_burden", 0.0) or 0.0)
    event_injury_burden = float(profile.get("event_injury_burden", 0.0) or 0.0)
    event_key_absence_score = float(profile.get("event_key_absence_score", 0.0) or 0.0)
    event_leader_absence_burden = float(profile.get("event_leader_absence_burden", 0.0) or 0.0)

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
        - (0.2 * event_injury_burden)
        - (0.3 * event_key_absence_score)
        - (0.55 * event_leader_absence_burden)
    )


def _nba_tipoff_urgency(
    start_time: Optional[str],
    partial_hours: float = 12.0,
    full_hours: float = 2.0,
) -> float:
    """Scale late-news sensitivity upward as tipoff approaches."""
    if not start_time:
        return 0.6
    try:
        tipoff = datetime.fromisoformat(str(start_time).replace("Z", "+00:00"))
    except ValueError:
        return 0.6

    hours_to_tip = (tipoff - datetime.now(timezone.utc)).total_seconds() / 3600.0
    if hours_to_tip <= full_hours:
        return 1.0
    if hours_to_tip <= max(full_hours + 4.0, 6.0):
        return 0.85
    if hours_to_tip <= partial_hours:
        return 0.7
    return 0.55


def _apply_nba_availability_adjustment(
    blended: dict[str, float],
    home_profile: Optional[dict],
    away_profile: Optional[dict],
    start_time: Optional[str] = None,
    max_delta: float = 0.02,
    uncertainty_weight: float = 0.35,
    leader_uncertainty_weight: float = 0.35,
    tipoff_partial_hours: float = 12.0,
    tipoff_full_hours: float = 2.0,
) -> dict[str, float]:
    """Apply a small live NBA availability adjustment from roster status."""
    differential = _compute_nba_availability_index(home_profile) - _compute_nba_availability_index(away_profile)
    urgency = _nba_tipoff_urgency(
        start_time,
        partial_hours=tipoff_partial_hours,
        full_hours=tipoff_full_hours,
    )
    home_uncertainty = float((home_profile or {}).get("uncertainty_burden", 0.0) or 0.0)
    away_uncertainty = float((away_profile or {}).get("uncertainty_burden", 0.0) or 0.0)
    home_leader_uncertainty = float((home_profile or {}).get("leader_uncertainty_burden", 0.0) or 0.0)
    away_leader_uncertainty = float((away_profile or {}).get("leader_uncertainty_burden", 0.0) or 0.0)
    home_event_uncertainty = float((home_profile or {}).get("event_uncertainty_burden", 0.0) or 0.0)
    away_event_uncertainty = float((away_profile or {}).get("event_uncertainty_burden", 0.0) or 0.0)
    home_event_leader_uncertainty = float((home_profile or {}).get("event_leader_uncertainty_burden", 0.0) or 0.0)
    away_event_leader_uncertainty = float((away_profile or {}).get("event_leader_uncertainty_burden", 0.0) or 0.0)
    differential -= (
        ((home_uncertainty - away_uncertainty) * uncertainty_weight)
        + ((home_leader_uncertainty - away_leader_uncertainty) * leader_uncertainty_weight)
        + ((home_event_uncertainty - away_event_uncertainty) * (uncertainty_weight * 0.85))
        + ((home_event_leader_uncertainty - away_event_leader_uncertainty) * (leader_uncertainty_weight * 0.9))
    ) * urgency
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
    home_profile: Optional[dict],
    away_profile: Optional[dict],
    start_time: Optional[str] = None,
    max_points_delta: float = 2.2,
    uncertainty_weight: float = 0.12,
    leader_uncertainty_weight: float = 0.28,
    tipoff_partial_hours: float = 12.0,
    tipoff_full_hours: float = 2.0,
) -> float:
    """Adjust an NBA total modestly for current roster depletion."""
    if not home_profile and not away_profile:
        return expected_total

    urgency = _nba_tipoff_urgency(
        start_time,
        partial_hours=tipoff_partial_hours,
        full_hours=tipoff_full_hours,
    )

    def _offense_drag(profile: Optional[dict]) -> float:
        if not profile:
            return 0.0
        leader_absence = float(profile.get("leader_absence_burden", 0.0) or 0.0)
        leader_uncertainty = float(profile.get("leader_uncertainty_burden", 0.0) or 0.0)
        injury_burden = float(profile.get("injury_burden", 0.0) or 0.0)
        uncertainty_burden = float(profile.get("uncertainty_burden", 0.0) or 0.0)
        active_players = float(profile.get("active_players", 0.0) or 0.0)
        available_core = float(profile.get("available_core_players", 0.0) or 0.0)
        depth_gap = max(0.0, 10.0 - available_core)
        active_gap = 0.0
        if active_players > 0:
            active_gap = max(0.0, 0.75 - (available_core / active_players))
        return (
            (0.65 * leader_absence)
            + (0.25 * injury_burden)
            + (0.2 * depth_gap)
            + (0.35 * active_gap)
            + (leader_uncertainty_weight * leader_uncertainty * urgency)
            + (uncertainty_weight * uncertainty_burden * urgency)
        )

    combined_drag = _offense_drag(home_profile) + _offense_drag(away_profile)
    if combined_drag <= 0.05:
        return expected_total

    delta = max(-max_points_delta, min(0.0, -combined_drag * max_points_delta * 0.45))
    return max(180.0, min(270.0, expected_total + delta))


def _apply_nhl_goalie_status_adjustment(
    blended: dict[str, float],
    home_status: Optional[str],
    away_status: Optional[str],
    max_delta: float = 0.012,
) -> dict[str, float]:
    """Apply a tiny NHL live adjustment from goalie confirmation certainty."""
    status_scale = {
        "confirmed": 1.0,
        "expected": 0.45,
        "projected": 0.35,
        "likely": 0.55,
        "unconfirmed": 0.2,
    }
    home_score = status_scale.get(str(home_status or "").strip().lower(), 0.0)
    away_score = status_scale.get(str(away_status or "").strip().lower(), 0.0)
    differential = home_score - away_score
    if abs(differential) < 0.05:
        return blended

    delta = max(-max_delta, min(max_delta, differential * max_delta * 0.8))
    adjusted = {
        "home": min(0.99, max(0.01, blended.get("home", 0.5) + delta)),
        "away": min(0.99, max(0.01, blended.get("away", 0.5) - delta)),
    }
    return _normalize_two_way_probs(adjusted)


def _compute_nhl_injury_index(profile: Optional[dict]) -> float:
    """Score an NHL skater injury profile from event-specific news."""
    if not profile:
        return 0.0
    return (
        - (0.35 * float(profile.get("injury_burden", 0.0) or 0.0))
        - (0.45 * float(profile.get("key_absence_score", 0.0) or 0.0))
        - (0.7 * float(profile.get("leader_absence_burden", 0.0) or 0.0))
        - (0.2 * float(profile.get("uncertainty_burden", 0.0) or 0.0))
        - (0.3 * float(profile.get("leader_uncertainty_burden", 0.0) or 0.0))
    )


def _apply_nhl_injury_adjustment(
    blended: dict[str, float],
    home_profile: Optional[dict],
    away_profile: Optional[dict],
    max_delta: float = 0.01,
) -> dict[str, float]:
    """Apply a modest NHL side adjustment from same-day skater injury news."""
    differential = _compute_nhl_injury_index(home_profile) - _compute_nhl_injury_index(away_profile)
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
    """Extract SLOP LOCKS with a guaranteed 3-pick minimum fallback."""
    if max_picks <= 0:
        return []

    candidates = []
    all_eligible = []

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
            
            formatted_pick = {
                "home_team": rec["home_team"],
                "away_team": rec["away_team"],
                "date": rec["date"],
                "start_time": rec.get("start_time"),
                "pick": outcome,
                "model_prob": round(prob, 4),
                "implied_prob": round(e.get("implied_prob") or 0.0, 4),
                "market_implied_prob": round(e.get("market_implied_prob") or 0.0, 4),
                "edge": round(edge, 4),
                "expected_value": round(ev, 4),
                "american_odds": e.get("american_odds"),
                "decimal_odds": e.get("decimal_odds"),
                "kelly_fraction": round(e.get("kelly_fraction") or 0.0, 4),
                "fractional_kelly": round(e.get("fractional_kelly") or 0.0, 4),
                "confidence_score": conf,
                "individual_models": rec.get("individual_models", {}),
                "qualitative_analysis": rec.get("qualitative_analysis"),
                "qualitative_summary": rec.get("qualitative_summary"),
            }
            
            all_eligible.append(formatted_pick)

            # Strict Lock Criteria
            if (
                edge >= edge_floor
                and prob >= probability_floor
                and ev >= min_expected_value
                and formatted_pick["american_odds"] is not None
            ):
                candidates.append(formatted_pick)

    # Sort strict candidates by edge and EV
    candidates.sort(key=lambda x: (x["edge"], x["expected_value"], x["model_prob"]), reverse=True)
    
    # Selection using strict criteria (limited by max_picks)
    selected = _exclude_opponent_conflicts(candidates[:max_picks])

    # MANDATE: If we have < 3 picks, fill with "Best Leans" from all_eligible
    if len(selected) < 3:
        already_selected_teams = set()
        for s in selected:
            already_selected_teams.add(s["home_team"])
            already_selected_teams.add(s["away_team"])
            
        remaining = [p for p in all_eligible if id(p) not in [id(s) for s in selected]]
        # Sort by: Has Odds (True first), then Model Prob
        remaining.sort(key=lambda x: (x["american_odds"] is not None, x["model_prob"]), reverse=True)
        
        for r in remaining:
            if len(selected) >= 3:
                break
            # Avoid duplicate team pairings
            if r["home_team"] in already_selected_teams or r["away_team"] in already_selected_teams:
                continue
                
            # Tag as a Lean
            if r["american_odds"] is None:
                r["blurb"] = "[UNPRICED LEAN] Model projections only; market odds unavailable."
            else:
                r["blurb"] = "[SYSTEM LEAN] High-value model projection."
            
            selected.append(r)
            already_selected_teams.add(r["home_team"])
            already_selected_teams.add(r["away_team"])

    return selected



def _compute_longslop(
    prediction_records,
    outcomes,
    min_expected_value: float = 0.0,
    confidence_floor: float = 65.0,
):
    """Extract LONGSLOP using edge/EV filters, with confidence as metadata."""
    longslop_candidates = []
    for rec in prediction_records:
        if rec.get("completed"):
            continue
        edges = rec.get("edges", {})
        
        for outcome in outcomes:
            e = edges.get(outcome)
            if not e:
                continue
            
            american = e.get("american_odds")
            if american is None or american < 500:
                continue
            
            # Phase 3 Thresholds
            conf = e.get("confidence_score", 0)
            edge = e.get("edge", 0)
            prob = e.get("model_prob", 0)
            ev = e.get("expected_value", 0.0)
            
            if edge >= 0 and ev >= min_expected_value:
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

    longslop_candidates.sort(
        key=lambda x: (x["edge"], x["expected_value"], x["model_prob"], x["confidence_score"]),
        reverse=True,
    )
    return longslop_candidates[0] if longslop_candidates else None


def _compute_totals_locks(
    total_records,
    min_expected_value: float = 0.0,
    edge_floor: float = 0.02,
    probability_floor: float = 0.53,
    confidence_floor: float = 54.0,
    max_picks: int = 3,
):
    """Extract totals picks using market edge, not confidence thresholds."""
    if max_picks <= 0:
        return []

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
                and edge_data.get("american_odds") is not None
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
                    "implied_prob": round(edge_data.get("implied_prob") or 0.0, 4),
                    "market_implied_prob": round(edge_data.get("market_implied_prob") or 0.0, 4),
                    "edge": round(edge_data["edge"], 4),
                    "expected_value": round(edge_data["expected_value"], 4),
                    "american_odds": edge_data["american_odds"],
                    "decimal_odds": edge_data["decimal_odds"],
                    "kelly_fraction": round(edge_data.get("kelly_fraction") or 0.0, 4),
                    "fractional_kelly": round(edge_data.get("fractional_kelly") or 0.0, 4),
                    "confidence_score": edge_data.get("confidence_score", 0.0),
                })

    candidates.sort(
        key=lambda item: (
            item.get("edge", 0.0),
            item.get("expected_value", 0.0),
            item.get("model_prob", 0.0),
            item.get("confidence_score", 0.0),
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
    """Extract SLIMEGRINDER likely winners, with confidence as metadata."""
    candidates = []
    for rec in prediction_records:
        if rec.get("completed"):
            continue
        edges = rec.get("edges", {})

        for outcome in outcomes:
            e = edges.get(outcome)
            if not e:
                continue

            american = e.get("american_odds")
            if american is None:
                continue
            if not (SLIMEGRINDER_MIN_ODDS <= american <= SLIMEGRINDER_MAX_ODDS):
                continue

            conf = e.get("confidence_score", 0)
            edge = e.get("edge", 0)
            prob = e.get("model_prob", 0)
            ev = e.get("expected_value", 0.0)

            if edge > 0 and ev >= min_expected_value:
                candidates.append({
                    "home_team": rec["home_team"],
                    "away_team": rec["away_team"],
                    "date": rec["date"],
                    "start_time": rec.get("start_time"),
                    "matchday": rec.get("matchday"),
                    "pick": outcome,
                    "model_prob": round(prob, 4),
                    "implied_prob": round(e.get("implied_prob") or 0.0, 4),
                    "edge": round(edge, 4),
                    "expected_value": round(ev, 4),
                    "american_odds": american,
                    "decimal_odds": e["decimal_odds"],
                    "kelly_fraction": round(e.get("kelly_fraction") or 0.0, 4),
                    "fractional_kelly": round(e.get("fractional_kelly") or 0.0, 4),
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
    longslop: Optional[dict],
    slimegrinder: list[dict],
    publication_guard: Optional[dict] = None,
) -> dict:
    """Summarize why a slate did or did not produce picks."""
    min_expected_value = sport.get("min_expected_value", 0.0)
    edge_floor = sport.get("slop_lock_edge_threshold", 0.03)
    probability_floor = sport.get("slop_lock_probability_floor", 0.45)

    fixtures_with_odds = 0
    coverage_gap_examples = []
    for fixture in fixtures_in_window:
        if _lookup_match_odds(odds_lookup, sport_key, fixture["home_team"], fixture["away_team"]):
            fixtures_with_odds += 1
        elif len(coverage_gap_examples) < 3:
            coverage_gap_examples.append(f"{fixture['away_team']} @ {fixture['home_team']}")

    matches_with_market = 0
    matches_with_positive_ev = set()
    lock_eligible_matches = set()
    lock_eligible_outcomes = 0
    matches_with_qualitative = 0

    for record in prediction_records:
        edges = record.get("edges") or {}
        if edges:
            matches_with_market += 1

        if record.get("qualitative_summary") and record.get("qualitative_summary") != "No qualitative impact.":
            matches_with_qualitative += 1

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
        "coverage_gap_examples": coverage_gap_examples,
        "matches_modeled": len(prediction_records),
        "matches_with_market": matches_with_market,
        "matches_with_positive_ev": len(matches_with_positive_ev),
        "matches_with_qualitative": matches_with_qualitative,
        "lock_eligible_matches": len(lock_eligible_matches),
        "lock_eligible_outcomes": lock_eligible_outcomes,
        "slop_locks_posted": len(slop_locks),
        "longslop_posted": 1 if longslop else 0,
        "slimegrinders_posted": len(slimegrinder),
        "publication_guard": publication_guard or {},
    }
    diagnostics["summary"] = (
        f"modeled={diagnostics['matches_modeled']} | "
        f"odds={diagnostics['fixtures_with_odds']}/{diagnostics['fixtures_in_window']} | "
        f"+ev={diagnostics['matches_with_positive_ev']} | "
        f"sense={diagnostics['matches_with_qualitative']} | "
        f"eligible={diagnostics['lock_eligible_matches']} | "
        f"locks={diagnostics['slop_locks_posted']}"
    )
    if publication_guard and publication_guard.get("reason"):
        diagnostics["summary"] += f" | {publication_guard['reason']}"
    return diagnostics


def _print_pipeline_diagnostics(sport_key: str, diagnostics: dict) -> None:
    """Emit a concise per-sport diagnostics line for CLI / Actions logs."""
    print(f"[{sport_key}] {diagnostics.get('summary', '')}")


# ---------------------------------------------------------------------------
# Per-sport pipeline
# ---------------------------------------------------------------------------

def _format_qualitative_summary(blended_pre_qual, qualitative_data):
    """Return a human-readable summary of qualitative impact and its effect."""
    if not qualitative_data or qualitative_data.get("summary") == "No significant qualitative factors identified or API error.":
        return "No qualitative impact."

    home_impact = qualitative_data.get("home_impact", 0.0)
    away_impact = qualitative_data.get("away_impact", 0.0)
    diff = home_impact - away_impact
    
    # Check if qualitative signal agreed with pre-qualitative pick
    pre_pick = max(blended_pre_qual.keys(), key=lambda k: blended_pre_qual[k])
    
    impact_direction = "none"
    if diff > 0.5:
        impact_direction = "home"
    elif diff < -0.5:
        impact_direction = "away"
        
    agreement = "no effect"
    if impact_direction != "none":
        if impact_direction == pre_pick:
            agreement = "agreed"
        else:
            agreement = "disagreed"
            
    scores = f"H:{home_impact} A:{away_impact}"
    factors = [f["description"] for f in qualitative_data.get("individual_factors", [])]
    factors_str = "; ".join(factors[:2]) # Top 2 factors
    
    return f"Qualitative ({scores}): {agreement}. {factors_str}"


def run_sport_pipeline(sport_key, output_dir=None, run_context=None):
    """Run prediction pipeline for a single sport.

    Parameters
    ----------
    sport_key : str
        Key into ``SPORTS`` config dict (e.g. "nba", "ncaam").
    output_dir : str or None
        Override the sport's data directory.
    """
    print(f"[*] Starting {sport_key.upper()} pipeline run...")
    run_context = dict(run_context or _build_run_context())
    sport = SPORTS[sport_key]
    sport_dir = output_dir or sport["data_dir"]
    base_dir = os.path.dirname(sport_dir)
    outcomes = sport["outcomes"]

    # ... (paths)
    predictions_path = os.path.join(sport_dir, "predictions.json")
    history_path = os.path.join(sport_dir, "history.json")
    accuracy_path = os.path.join(sport_dir, "model_accuracy.json")
    results_log_path = _results_log_path(base_dir)
    results_audit_log_path = _results_audit_log_path(base_dir)
    pick_decision_log_path = _pick_decision_log_path(base_dir)
    odds_history_path = _odds_history_path(base_dir)

    # ------------------------------------------------------------------
    # 1. Fetch data (sport-specific)
    # ------------------------------------------------------------------
    print(f"[*] Fetching {sport_key.upper()} data and schedule...")
    box_scores_df = None

    if sport_key == "nba":
        games_df, box_scores_df = fetch_nba_espn_games(
            cache_path=os.path.join(sport_dir, "espn_cache.json")
        )
        fixtures = fetch_nba_espn_schedule(
            cache_path=os.path.join(sport_dir, "espn_cache.json")
        )
        matches = games_df
    # ...
    elif sport_key == "nhl":
        games_df, box_scores_df = fetch_nhl_games(
            cache_path=os.path.join(sport_dir, "espn_cache.json")
        )
        fixtures = fetch_nhl_schedule(
            cache_path=os.path.join(sport_dir, "espn_cache.json")
        )
        matches = games_df
    elif sport_key == "mlb":
        games_df, box_scores_df = fetch_mlb_games(
            cache_path=os.path.join(sport_dir, "espn_cache.json")
        )
        fixtures = fetch_mlb_schedule(
            cache_path=os.path.join(sport_dir, "espn_cache.json")
        )
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

    accuracy_log = _load_json(accuracy_path)
    if not isinstance(accuracy_log, dict):
        accuracy_log = {}
    model_health = build_model_health_snapshot(
        accuracy_log=accuracy_log,
        model_names=list(sport.get("models", [])),
        temperature=sport.get("accuracy_softmax_temperature", 2.0),
        window=sport.get("accuracy_window"),
    )
    runtime_disabled_models = set(sport.get("disabled_models", []))
    configured_models = [
        model_name
        for model_name in sport.get("models", [])
        if model_name not in runtime_disabled_models
    ]

    # ------------------------------------------------------------------
    # 2. Fit models
    # ------------------------------------------------------------------
    # Elo ratings (with sport-specific parameters)
    elo = None
    if "elo" in configured_models:
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

    # Results-feature logistic model (uses only historical game outcomes)
    results_feature_model = None
    if "results_features" in configured_models and matches is not None and not matches.empty:
        results_feature_model = ResultsFeatureModel(
            matches,
            feature_window=sport.get("results_feature_window", 8),
            min_games=sport.get("results_feature_min_games", 30),
        )

    recent_boxscore_model = None
    if "recent_boxscore" in configured_models and box_scores_df is not None and matches is not None:
        recent_boxscore_model = RecentBoxScoreModel(
            box_scores_df,
            matches,
            feature_window=sport.get("recent_boxscore_window", 8),
            min_games=sport.get("recent_boxscore_min_games", 30),
        )

    nba_matchup_model = None
    if sport_key == "nba" and "nba_matchup" in configured_models and box_scores_df is not None and matches is not None:
        nba_matchup_model = NbaMatchupModel(
            box_scores_df,
            matches,
            feature_window=sport.get("nba_matchup_window", 8),
            min_games=sport.get("nba_matchup_min_games", 30),
        )

    nhl_matchup_model = None
    if sport_key == "nhl" and "nhl_matchup" in configured_models and matches is not None and not matches.empty:
        nhl_matchup_model = NhlMatchupModel(
            matches,
            feature_window=sport.get("nhl_matchup_window", 10),
            min_games=sport.get("nhl_matchup_min_games", 40),
        )

    pitcher_feature_model = None
    if "pitcher_features" in configured_models and matches is not None and not matches.empty:
        pitcher_feature_model = PitcherMatchupModel(
            matches,
            feature_window=sport.get("pitcher_feature_window", 8),
            min_games=sport.get("pitcher_feature_min_games", 20),
        )

    bullpen_feature_model = None
    if "bullpen_features" in configured_models and matches is not None and not matches.empty:
        bullpen_feature_model = BullpenMatchupModel(
            matches,
            feature_window=sport.get("bullpen_feature_window", 12),
            recent_usage_window=sport.get("bullpen_recent_usage_window", 5),
            min_games=sport.get("bullpen_feature_min_games", 20),
        )

    run_environment_model = None
    if "run_environment" in configured_models and matches is not None and not matches.empty:
        run_environment_model = RunEnvironmentModel(
            matches,
            feature_window=sport.get("run_environment_window", 12),
            min_games=sport.get("run_environment_min_games", 20),
        )

    handedness_feature_model = None
    if "handedness_features" in configured_models and matches is not None and not matches.empty:
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
    # Build list of active models for this run
    model_names = []
    if elo is not None:
        model_names.append("elo")
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
    weights = compute_model_weights(
        accuracy_log,
        model_names=model_names,
        temperature=sport.get("accuracy_softmax_temperature", 2.0),
        window=accuracy_window,
    )
    model_weight_dict = dict(zip(model_names, weights))

    history = _load_json(history_path)
    if not isinstance(history, dict):
        history = {}
    past_predictions = history.get("predictions", [])
    calibration_predictions = _select_calibration_predictions(
        past_predictions,
        lookback_days=sport.get("probability_calibration_window_days"),
        holdout_days=int(sport.get("probability_calibration_holdout_days", 0) or 0),
        as_of=datetime.now(timezone.utc).date(),
    )
    probability_calibrators = fit_probability_calibrators(
        calibration_predictions,
        outcomes,
        min_samples=sport.get("probability_calibration_min_samples", 20),
        lookback_days=sport.get("probability_calibration_window_days"),
        holdout_days=int(sport.get("probability_calibration_holdout_days", 0) or 0),
        as_of=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
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

    print(f"--- Running projections for {len(fixtures)} {sport_key.upper()} games ---")

    for fix in fixtures:
        home = fix["home_team"]
        away = fix["away_team"]
        print(f"  - {away} @ {home}...", end=" ", flush=True)
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
        home_bullpen_tax = 0.0
        away_bullpen_tax = 0.0
        if sport_key == "mlb":
            home_bullpen_tax = _compute_mlb_bullpen_tax(
                home,
                fix["date"],
                matches,
                recent_days=sport.get("bullpen_fatigue_window_days", 3),
                usage_baseline=sport.get("bullpen_fatigue_usage_baseline", 6.5),
                last_game_baseline=sport.get("bullpen_last_game_usage_baseline", 3.5),
            )
            away_bullpen_tax = _compute_mlb_bullpen_tax(
                away,
                fix["date"],
                matches,
                recent_days=sport.get("bullpen_fatigue_window_days", 3),
                usage_baseline=sport.get("bullpen_fatigue_usage_baseline", 6.5),
                last_game_baseline=sport.get("bullpen_last_game_usage_baseline", 3.5),
            )
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
            blended = _apply_mlb_bullpen_availability_adjustment(
                blended,
                home_tax=home_bullpen_tax,
                away_tax=away_bullpen_tax,
                max_delta=sport.get("bullpen_availability_adjustment_max_delta", 0.012),
            )
        if sport_key == "nba":
            blended = _apply_nba_availability_adjustment(
                blended,
                fix.get("home_availability_profile"),
                fix.get("away_availability_profile"),
                start_time=fix.get("start_time"),
                max_delta=sport.get("availability_adjustment_max_delta", 0.02),
                uncertainty_weight=sport.get("availability_uncertainty_weight", 0.35),
                leader_uncertainty_weight=sport.get("availability_leader_uncertainty_weight", 0.35),
                tipoff_partial_hours=sport.get("availability_tipoff_partial_hours", 12.0),
                tipoff_full_hours=sport.get("availability_tipoff_full_hours", 2.0),
            )
        if sport_key == "nhl":
            blended = _apply_nhl_injury_adjustment(
                blended,
                fix.get("home_injury_profile"),
                fix.get("away_injury_profile"),
                max_delta=sport.get("injury_adjustment_max_delta", 0.01),
            )
            blended = _apply_nhl_goalie_status_adjustment(
                blended,
                fix.get("home_goalie_status"),
                fix.get("away_goalie_status"),
                max_delta=sport.get("goalie_status_adjustment_max_delta", 0.012),
            )
        blended = apply_probability_calibration(
            blended,
            probability_calibrators,
            blend=sport.get("probability_calibration_blend", 0.5),
        )

        # ------------------------------------------------------------------
        # Qualitative Gemini Integration
        # ------------------------------------------------------------------
        qualitative_data = None
        if (
            ENABLE_QUALITATIVE
            and sport.get("enable_qualitative", False)
            and analyze_game_qualitative is not None
        ):
            context_text = get_game_context(sport_key, fix)
            game_for_ai = {
                "sport": sport_key,
                "home_team": home,
                "away_team": away,
                "date": fix["date"],
                "start_time": fix.get("start_time"),
            }
            qualitative_data = analyze_game_qualitative(game_for_ai, context_text)
            qualitative_summary = _format_qualitative_summary(blended, qualitative_data)
            blended = _apply_qualitative_adjustment(
                blended,
                qualitative_data,
                weight=sport.get("qualitative_weight", 0.5)
            )
        else:
            qualitative_summary = None

        # Edges and best odds
        match_odds = _lookup_match_odds(odds_lookup, sport_key, home, away)
        
        # Always compute edges, even if odds are missing, to get model_probs and baseline stats
        edges = compute_edges(
            blended,
            match_odds or {}, # Pass empty dict if no odds
            individual_probs=individual_preds,
            fractional_kelly=sport.get("kelly_fraction", 0.25),
        )
        
        best_odds = {}
        if match_odds:
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
        
        # Fallback confidence if edge data is sparse (no odds)
        # Use a 0-100 scale based on probability (e.g., 0.6 prob -> 60 conf)
        confidence_score = edge_data.get("confidence_score")
        if confidence_score is None:
            confidence_score = round(model_prob * 100, 1)
        
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
            "home_injury_profile": fix.get("home_injury_profile"),
            "away_injury_profile": fix.get("away_injury_profile"),
            "home_pitcher": fix.get("home_pitcher"),
            "home_pitcher_hand": fix.get("home_pitcher_hand"),
            "away_pitcher": fix.get("away_pitcher"),
            "away_pitcher_hand": fix.get("away_pitcher_hand"),
            "home_lineup_profile": fix.get("home_lineup_profile"),
            "away_lineup_profile": fix.get("away_lineup_profile"),
            "home_bullpen_tax": round(home_bullpen_tax, 3) if sport_key == "mlb" else None,
            "away_bullpen_tax": round(away_bullpen_tax, 3) if sport_key == "mlb" else None,
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
            "market_snapshot": (match_odds or {}).get("moneyline_market_snapshot"),
            "qualitative_analysis": qualitative_data,
            "qualitative_summary": qualitative_summary,
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
                total_projection["expected_total"] = _apply_mlb_bullpen_total_adjustment(
                    total_projection["expected_total"],
                    home_tax=home_bullpen_tax,
                    away_tax=away_bullpen_tax,
                    max_runs_delta=sport.get("bullpen_total_adjustment_max_delta", 0.3),
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
                    start_time=fix.get("start_time"),
                    max_points_delta=sport.get("availability_total_adjustment_max_points", 2.2),
                    tipoff_partial_hours=sport.get("availability_tipoff_partial_hours", 12.0),
                    tipoff_full_hours=sport.get("availability_tipoff_full_hours", 2.0),
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
                "home_bullpen_tax": round(home_bullpen_tax, 3) if sport_key == "mlb" else None,
                "away_bullpen_tax": round(away_bullpen_tax, 3) if sport_key == "mlb" else None,
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
                "market_snapshot": match_odds.get("totals_market_snapshot"),
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
    longslop = None
    if sport.get("enable_longslop", False):
        longslop = _compute_longslop(
            prediction_records,
            outcomes,
            min_expected_value=min_expected_value,
            confidence_floor=sport.get("longslop_confidence_threshold", 65.0),
        )
    slimegrinder = []
    if sport.get("enable_slimegrinder", False):
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

    pick_history_path = os.path.join(sport_dir, "pick_history.json")
    pick_history = _load_json(pick_history_path)
    if not isinstance(pick_history, dict):
        pick_history = {}
    past_picks = pick_history.get("picks", [])
    publication_guard = _build_publication_guard(
        past_picks,
        sport,
        enforce_live_guard=_is_live_public_output(base_dir),
    )
    if not publication_guard.get("allow_moneyline", True):
        # Fallback: keep match projections and diagnostics visible even when the
        # sport is not yet allowed to publish official moneyline lanes live.
        # MANDATE: Always keep at least 3 picks if they exist, even in research mode.
        if len(slop_locks) > 3:
            slop_locks = slop_locks[:3]
        
        # Tag as research leans
        for lock in slop_locks:
            if "blurb" not in lock:
                lock["blurb"] = "[RESEARCH LEAN] This sport is in a model-warming phase; these are non-official leans."
        
        longslop = None
        slimegrinder = []
    elif not publication_guard.get("allow_longslop", False):
        longslop = None
    if publication_guard.get("allow_moneyline", True) and not publication_guard.get("allow_slimegrinder", False):
        slimegrinder = []
    if not publication_guard.get("allow_totals", True):
        totals_locks = []

    # Generate analysis blurbs via Claude
    slop_locks = _generate_blurbs(slop_locks, pick_type="lock")
    longslop = _generate_blurbs(longslop, pick_type="longslop")

    snapshot_relpath = _snapshot_relative_path(sport_key, run_context)
    selection_config = _selection_snapshot_config(sport, outcomes, min_expected_value)
    _attach_run_metadata_list(prediction_records, run_context, snapshot_relpath)
    _attach_run_metadata_list(totals_prediction_records, run_context, snapshot_relpath)
    _attach_run_metadata_list(slop_locks, run_context, snapshot_relpath)
    _attach_run_metadata_list(slimegrinder, run_context, snapshot_relpath)
    _attach_run_metadata_list(totals_locks, run_context, snapshot_relpath)
    if isinstance(longslop, dict):
        _attach_run_metadata(longslop, run_context, snapshot_relpath)

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
    record_lookup = {}
    record_lookup.update(_build_record_lookup(prediction_records, "moneyline"))
    record_lookup.update(_build_record_lookup(totals_prediction_records, "total"))
    new_pick_decision_rows = []
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
            new_pick = {
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
                "run_id": run_context.get("run_id"),
                "run_type": run_context.get("run_type"),
                "snapshot_timestamp": run_context.get("run_timestamp"),
                "snapshot_path": snapshot_relpath,
                "evaluated": False,
            }
            past_picks.append(new_pick)
            source_record = record_lookup.get(
                ("moneyline", lock["home_team"], lock["away_team"], str(lock["date"])[:10])
            )
            new_pick_decision_rows.append(
                _build_pick_decision_row(
                    sport_key,
                    "slop_lock",
                    new_pick,
                    source_record,
                    publication_guard,
                    selection_config,
                    len(calibration_predictions),
                )
            )

    if longslop:
        pk = ("longslop", longslop["home_team"], longslop["away_team"],
              str(longslop["date"])[:10], longslop["pick"])
        if pk not in existing_keys:
            new_pick = {
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
                "run_id": run_context.get("run_id"),
                "run_type": run_context.get("run_type"),
                "snapshot_timestamp": run_context.get("run_timestamp"),
                "snapshot_path": snapshot_relpath,
                "evaluated": False,
            }
            past_picks.append(new_pick)
            source_record = record_lookup.get(
                ("moneyline", longslop["home_team"], longslop["away_team"], str(longslop["date"])[:10])
            )
            new_pick_decision_rows.append(
                _build_pick_decision_row(
                    sport_key,
                    "longslop",
                    new_pick,
                    source_record,
                    publication_guard,
                    selection_config,
                    len(calibration_predictions),
                )
            )

    for total_lock in totals_locks:
        pk = ("total_lock", total_lock["home_team"], total_lock["away_team"],
              str(total_lock["date"])[:10], total_lock["pick"])
        if pk not in existing_keys:
            new_pick = {
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
                "run_id": run_context.get("run_id"),
                "run_type": run_context.get("run_type"),
                "snapshot_timestamp": run_context.get("run_timestamp"),
                "snapshot_path": snapshot_relpath,
                "evaluated": False,
            }
            past_picks.append(new_pick)
            source_record = record_lookup.get(
                ("total", total_lock["home_team"], total_lock["away_team"], str(total_lock["date"])[:10])
            )
            new_pick_decision_rows.append(
                _build_pick_decision_row(
                    sport_key,
                    "total_lock",
                    new_pick,
                    source_record,
                    publication_guard,
                    selection_config,
                    len(calibration_predictions),
                )
            )

    _apply_latest_market_snapshots(past_picks, latest_snapshot_lookup)

    pick_history = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": run_context.get("run_id"),
        "run_type": run_context.get("run_type"),
        "snapshot_timestamp": run_context.get("run_timestamp"),
        "snapshot_path": snapshot_relpath,
        "picks": past_picks,
    }
    _save_json(pick_history_path, pick_history)
    _append_results_log(results_log_path, resolved_results_rows)
    _append_results_audit_log(results_audit_log_path, resolved_results_rows)
    _append_pick_decision_log(pick_decision_log_path, new_pick_decision_rows)

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
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    diagnostics = _build_pipeline_diagnostics(
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
        publication_guard=publication_guard,
    )
    runtime_model_health = {
        **model_health,
        "disabled_by_config": sorted(runtime_disabled_models),
        "active_models": list(model_weight_dict.keys()),
    }

    predictions_output = {
        "generated_at": generated_at,
        "run_id": run_context.get("run_id"),
        "run_type": run_context.get("run_type"),
        "snapshot_timestamp": run_context.get("run_timestamp"),
        "snapshot_path": snapshot_relpath,
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
        "publication_guard": publication_guard,
        "lane_health": publication_guard.get("lane_guards", {}),
        "model_health": runtime_model_health,
        "calibration_sample_size": len(calibration_predictions),
        "selection_config": selection_config,
        "diagnostics": diagnostics,
    }
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
        "updated_at": generated_at,
        "run_id": run_context.get("run_id"),
        "run_type": run_context.get("run_type"),
        "snapshot_timestamp": run_context.get("run_timestamp"),
        "snapshot_path": snapshot_relpath,
        "predictions": updated_past + new_predictions,
    }
    _save_json(history_path, history_output)

    _save_json(accuracy_path, accuracy_log)

    _write_run_snapshot(
        base_dir,
        sport_key,
        run_context,
        {
            "snapshot_version": _SNAPSHOT_VERSION,
            "generated_at": generated_at,
            "sport": sport_key,
            "sport_name": sport["display_name"],
            "run_id": run_context.get("run_id"),
            "run_type": run_context.get("run_type"),
            "snapshot_timestamp": run_context.get("run_timestamp"),
            "selection_config": selection_config,
            "publication_guard": publication_guard,
            "outcomes": outcomes,
            "inputs": {
                "fixtures_fetched": fixtures_fetched,
                "fixtures_in_window": fixtures,
                "odds": odds_list,
                "model_weights": {k: round(v, 4) for k, v in model_weight_dict.items()},
                "models": list(model_weight_dict.keys()),
                "model_health": runtime_model_health,
                "calibration_sample_size": len(calibration_predictions),
            },
            "records": {
                "matches": prediction_records,
                "totals_matches": totals_prediction_records,
            },
            "outputs": {
                "slop_locks": slop_locks,
                "totals_locks": totals_locks,
                "slimegrinder": slimegrinder,
                "longslop": longslop,
            },
            "diagnostics": diagnostics,
        },
    )

    _print_pipeline_diagnostics(sport_key, predictions_output["diagnostics"])

    return predictions_output


# ---------------------------------------------------------------------------
# Main pipeline — runs all sports
# ---------------------------------------------------------------------------

def _update_global_metadata(base_dir):
    """Update the global manifest.json and dashboard.json."""
    manifest_path = os.path.join(base_dir, "manifest.json")
    manifest = _load_json(manifest_path)
    
    # Update timestamp and ensure structure exists
    manifest["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if "sports" not in manifest:
        manifest["sports"] = {}
        
    # Re-build sports status from current directory state
    for sk in SPORTS:
        pred_path = os.path.join(base_dir, sk, "predictions.json")
        if os.path.exists(pred_path):
            with open(pred_path) as f:
                data = json.load(f)
                manifest["sports"][sk] = {
                    "name": SPORTS[sk]["display_name"],
                    "status": "ok",
                    "updated_at": data.get("generated_at", manifest["updated_at"]),
                    "diagnostics": data.get("diagnostics", {}),
                }

    _backfill_pick_decision_log_from_snapshots(base_dir)
    _hydrate_pick_decision_log_market_snapshots(base_dir)
    _backfill_pick_history_market_snapshots(base_dir)
    _save_json(manifest_path, manifest)
    _save_json(os.path.join(base_dir, "dashboard.json"), build_dashboard_data(base_dir))


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
    run_context = _build_run_context(run_type="daily")

    manifest = {
        "updated_at": now,
        "sports": {},
    }

    for sport_key in SPORTS:
        sport_dir = os.path.join(base_dir, sport_key) if output_dir else None
        try:
            sport_output = run_sport_pipeline(sport_key, output_dir=sport_dir, run_context=run_context)
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

    # Post-run cleanup and dashboard refresh
    _update_global_metadata(base_dir)

    return manifest


def _main(argv=None):
    """CLI entry point for the pipeline module."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run the SLOP LOCKS pipeline.")
    parser.add_argument("--sport", choices=sorted(SPORTS.keys()))
    parser.add_argument("--output-dir")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)

    if args.sport:
        run_sport_pipeline(args.sport, output_dir=args.output_dir)
        # Update manifest and dashboard even for single sport runs
        base_dir = os.path.dirname(args.output_dir) if args.output_dir else DATA_DIR
        _update_global_metadata(base_dir)
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
