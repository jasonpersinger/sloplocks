"""Quick refresh of odds and slop locks without re-training models.

Fetches fresh odds from The Odds API, patches edges onto existing match
records, and recomputes slop locks using the current filter logic.  Runs in
seconds — no historical data fetch, no model fitting.

Usage:
    python -m pipeline.refresh_picks            # all sports
    python -m pipeline.refresh_picks nba ncaam  # specific sports
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

from scipy.stats import norm
import pandas as pd

from pipeline.backtest import build_dashboard_data
from pipeline.config import (
    SLOP_LOCK_MIN_ODDS,
    SLOP_LOCK_MAX_ODDS,
    SPORTS,
    DATA_DIR,
)

from pipeline.ensemble import (
    compute_edges,
    compute_totals_edges,
    decimal_to_american,
    compute_confidence_stars,
)
from pipeline.fetch_data import fetch_odds
from pipeline.fetch_ncaam import normalize_ncaam_team_name
from pipeline.fetch_nba import normalize_nba_team_name, fetch_nba_espn_schedule
from pipeline.fetch_nhl import normalize_nhl_team_name, fetch_nhl_schedule
from pipeline.fetch_mlb import normalize_mlb_team_name
from pipeline.fetch_mlb import fetch_mlb_schedule
from pipeline.fetch_mma import normalize_mma_name
from pipeline.run import (
    _append_odds_snapshot_log,
    _apply_mlb_lineup_adjustment,
    _apply_mlb_lineup_total_adjustment,
    _apply_mlb_weather_adjustment,
    _apply_mlb_weather_total_adjustment,
    _apply_nba_availability_adjustment,
    _apply_nba_availability_total_adjustment,
    _apply_nhl_goalie_status_adjustment,
    _apply_latest_market_snapshots,
    _build_pipeline_diagnostics,
    _build_odds_snapshot_rows,
    _compute_slimegrinder as _run_compute_slimegrinder,
    _compute_slop_locks as _run_compute_slop_locks,
    _compute_totals_locks,
    _load_json,
    _load_latest_odds_snapshots,
    _lookup_match_odds,
    _save_json,
)

_NORMALIZERS = {
    "nba": normalize_nba_team_name,
    "nhl": normalize_nhl_team_name,
    "ncaam": normalize_ncaam_team_name,
    "mlb": normalize_mlb_team_name,
    "mma": normalize_mma_name,
}


def _fixture_key(item: dict) -> tuple[str, str, str]:
    """Return a stable matchup key for refreshed fixture metadata."""
    return (
        item["home_team"],
        item["away_team"],
        str(item.get("date", ""))[:10],
    )


def _load_live_fixtures(sport_key: str, data_path: Path) -> list[dict]:
    """Fetch current live schedule metadata for refreshable sports."""
    cache_path = str(data_path.parent / "espn_cache.json")
    if sport_key == "nba":
        return fetch_nba_espn_schedule(cache_path=cache_path)
    if sport_key == "nhl":
        return fetch_nhl_schedule(cache_path=cache_path)
    if sport_key == "mlb":
        return fetch_mlb_schedule(cache_path=cache_path)
    return []


def _merge_live_fixture(record: dict, live_fixture: dict | None) -> dict:
    """Patch a stored match or totals record with fresh fixture metadata."""
    if not live_fixture:
        return record
    merged = dict(record)
    for key in (
        "start_time",
        "completed",
        "neutral",
        "home_availability_profile",
        "away_availability_profile",
        "home_lineup_profile",
        "away_lineup_profile",
        "weather",
        "home_pitcher",
        "away_pitcher",
        "home_pitcher_hand",
        "away_pitcher_hand",
        "home_goalie",
        "away_goalie",
        "home_goalie_status",
        "away_goalie_status",
    ):
        if live_fixture.get(key) is not None:
            merged[key] = live_fixture.get(key)
    return merged


def refresh_sport(sport_key: str) -> None:
    """Refresh odds and slop locks for one sport.

    Returns None if predictions.json doesn't exist for this sport.
    """
    sport = SPORTS[sport_key]
    data_path = Path(f"data/{sport_key}/predictions.json")
    if not data_path.exists():
        print(f"  {sport_key}: no predictions.json, skipping")
        return None

    with data_path.open() as f:
        data = json.load(f)

    outcomes: list[str] = data["outcomes"]
    matches: list[dict] = data["matches"]
    totals_matches: list[dict] = data.get("totals_matches", [])
    normalizer = _NORMALIZERS.get(sport_key, lambda x: x)

    try:
        live_fixtures = _load_live_fixtures(sport_key, data_path)
    except Exception as exc:
        print(f"  {sport_key}: live fixture refresh failed ({exc}), using stored metadata")
        live_fixtures = []
    fixture_lookup = {_fixture_key(fix): fix for fix in live_fixtures}
    matches = [_merge_live_fixture(match, fixture_lookup.get(_fixture_key(match))) for match in matches]
    totals_matches = [_merge_live_fixture(match, fixture_lookup.get(_fixture_key(match))) for match in totals_matches]

    # Fetch fresh odds and build lookup
    odds_list = []
    odds_lookup: dict[tuple, dict] = {}
    try:
        odds_list = fetch_odds(
            sport_key=sport["odds_sport"],
            include_totals=(sport_key in {"nba", "mlb"}),
        )
        for o in odds_list:
            o["home_team"] = normalizer(o["home_team"])
            o["away_team"] = normalizer(o["away_team"])
        odds_lookup = {(o["home_team"], o["away_team"]): o for o in odds_list}
        print(f"  {sport_key}: fetched odds for {len(odds_list)} games")
    except Exception as exc:
        print(f"  {sport_key}: odds fetch failed ({exc}), using cached edges")

    tracking_dir = data_path.parent.parent / "tracking"
    odds_history_path = str(tracking_dir / "odds_history.csv")
    _append_odds_snapshot_log(odds_history_path, _build_odds_snapshot_rows(sport_key, odds_list))

    # Patch edges onto matches where we have fresh odds and refreshed live inputs.
    model_weights = data.get("model_weights") or {}
    for match in matches:
        base_probs = match.get("base_model_probs") or dict(match.get("model_probs") or {})
        match["base_model_probs"] = {k: round(v, 4) for k, v in base_probs.items()}
        refreshed_probs = dict(base_probs)
        if sport_key == "nba":
            refreshed_probs = _apply_nba_availability_adjustment(
                refreshed_probs,
                match.get("home_availability_profile"),
                match.get("away_availability_profile"),
                start_time=match.get("start_time"),
                max_delta=sport.get("availability_adjustment_max_delta", 0.02),
                uncertainty_weight=sport.get("availability_uncertainty_weight", 0.35),
                leader_uncertainty_weight=sport.get("availability_leader_uncertainty_weight", 0.35),
                tipoff_partial_hours=sport.get("availability_tipoff_partial_hours", 12.0),
                tipoff_full_hours=sport.get("availability_tipoff_full_hours", 2.0),
            )
        elif sport_key == "mlb":
            refreshed_probs = _apply_mlb_weather_adjustment(
                refreshed_probs,
                (match.get("individual_models") or {}).get("run_environment"),
                match.get("weather"),
                max_delta=sport.get("weather_adjustment_max_delta", 0.02),
            )
            refreshed_probs = _apply_mlb_lineup_adjustment(
                refreshed_probs,
                match.get("home_lineup_profile"),
                match.get("away_lineup_profile"),
                match.get("home_pitcher_hand"),
                match.get("away_pitcher_hand"),
                max_delta=sport.get("lineup_adjustment_max_delta", 0.015),
            )
        elif sport_key == "nhl":
            refreshed_probs = _apply_nhl_goalie_status_adjustment(
                refreshed_probs,
                match.get("home_goalie_status"),
                match.get("away_goalie_status"),
                max_delta=sport.get("goalie_status_adjustment_max_delta", 0.012),
            )
        match["model_probs"] = {k: round(v, 4) for k, v in refreshed_probs.items()}
        top_pick = max(refreshed_probs, key=refreshed_probs.get)
        match["pick"] = top_pick
        match["model_prob"] = round(refreshed_probs[top_pick], 4)

        match_odds = _lookup_match_odds(
            odds_lookup,
            sport_key,
            match["home_team"],
            match["away_team"],
        )
        if match_odds:
            edges = compute_edges(
                refreshed_probs,
                match_odds,
                fractional_kelly=sport.get("kelly_fraction", 0.25),
            )
            match["edges"] = edges
            match["best_odds"] = {
                outcome: decimal_to_american(match_odds[f"{outcome}_odds"])
                for outcome in outcomes
                if match_odds.get(f"{outcome}_odds", 0) > 0
            }
            # Recompute top-level pick/stars based on new odds
            pick = max(match["model_probs"].keys(), key=lambda k: match["model_probs"][k])
            model_prob = refreshed_probs[pick]
            edge = edges.get(pick, {}).get("edge", 0.0)
            match["pick"] = pick
            match["model_prob"] = round(model_prob, 4)
            match["edge"] = round(edge, 4)
            match["confidence_stars"] = compute_confidence_stars(model_prob, edge)
            match["american_odds"] = match["best_odds"].get(pick)

    for total_match in totals_matches:
        base_expected_total = float(total_match.get("base_expected_total", total_match.get("expected_total", 0.0)) or 0.0)
        total_match["base_expected_total"] = round(base_expected_total, 3)
        match_odds = _lookup_match_odds(
            odds_lookup,
            sport_key,
            total_match["home_team"],
            total_match["away_team"],
        )
        if not match_odds or match_odds.get("total_line") is None:
            continue
        expected_total = base_expected_total or float(match_odds["total_line"])
        if sport_key == "nba":
            expected_total = _apply_nba_availability_total_adjustment(
                expected_total,
                total_match.get("home_availability_profile"),
                total_match.get("away_availability_profile"),
                start_time=total_match.get("start_time"),
                max_points_delta=sport.get("availability_total_adjustment_max_points", 2.2),
                tipoff_partial_hours=sport.get("availability_tipoff_partial_hours", 12.0),
                tipoff_full_hours=sport.get("availability_tipoff_full_hours", 2.0),
            )
        elif sport_key == "mlb":
            expected_total = _apply_mlb_weather_total_adjustment(
                expected_total,
                total_match.get("weather"),
                max_runs_delta=sport.get("weather_total_adjustment_max_runs", 0.8),
            )
            expected_total = _apply_mlb_lineup_total_adjustment(
                expected_total,
                total_match.get("home_lineup_profile"),
                total_match.get("away_lineup_profile"),
                total_match.get("home_pitcher_hand"),
                total_match.get("away_pitcher_hand"),
                max_runs_delta=sport.get("lineup_total_adjustment_max_runs", 0.35),
            )
        sigma = max(1.5, float(total_match.get("total_stddev", sport.get("totals_default_stddev", 3.1))))
        current_line = float(match_odds["total_line"])
        over_prob = float(1.0 - norm.cdf(current_line, loc=expected_total, scale=sigma))
        over_prob = max(0.01, min(0.99, over_prob))
        total_probs = {"over": over_prob, "under": 1.0 - over_prob}
        total_edges = compute_totals_edges(
            total_probs,
            match_odds,
            individual_probs=[total_probs],
            fractional_kelly=sport.get("kelly_fraction", 0.25),
        )
        total_pick = max(total_probs, key=total_probs.get)
        total_match["start_time"] = match_odds.get("commence_time", total_match.get("start_time"))
        total_match["total_line"] = current_line
        total_match["expected_total"] = round(expected_total, 3)
        total_match["pick"] = total_pick
        total_match["model_prob"] = round(total_probs[total_pick], 4)
        total_match["confidence_score"] = total_edges.get(total_pick, {}).get("confidence_score", 0.0)
        total_match["american_odds"] = total_edges.get(total_pick, {}).get("american_odds")
        total_match["model_probs"] = {k: round(v, 4) for k, v in total_probs.items()}
        total_match["edges"] = total_edges

    matches_with_odds = sum(1 for m in matches if m.get("edges"))
    min_expected_value = sport.get("min_expected_value", 0.0)
    slop_locks = _run_compute_slop_locks(
        matches,
        outcomes,
        min_expected_value=min_expected_value,
        edge_floor=sport.get("slop_lock_edge_threshold", 0.03),
        probability_floor=sport.get("slop_lock_probability_floor", 0.45),
        additional_confidence_floor=sport.get("slop_lock_confidence_threshold", 65.0),
        confidence_dropoff=sport.get("slop_lock_confidence_dropoff", 0.0),
        max_picks=sport.get("slop_lock_max_picks", 3),
    )
    slimegrinder = _run_compute_slimegrinder(
        matches,
        outcomes,
        min_expected_value=min_expected_value,
        confidence_floor=sport.get("slimegrinder_confidence_threshold", 65.0),
    )
    totals_locks = _compute_totals_locks(
        totals_matches,
        min_expected_value=sport.get("totals_min_expected_value", min_expected_value),
        edge_floor=sport.get("totals_edge_threshold", 0.02),
        probability_floor=sport.get("totals_probability_floor", 0.53),
        confidence_floor=sport.get("totals_confidence_threshold", 54.0),
        max_picks=sport.get("totals_max_picks", 3),
    ) if totals_matches else []
    in_window = sum(
        1 for s in slop_locks
        if SLOP_LOCK_MIN_ODDS <= s["american_odds"] <= SLOP_LOCK_MAX_ODDS
    )
    fallback = len(slop_locks) - in_window

    data["matches"] = matches
    data["totals_matches"] = totals_matches
    data["slop_locks"] = slop_locks
    data["totals_locks"] = totals_locks
    data["slimegrinder"] = slimegrinder
    data["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    diagnostics = _build_pipeline_diagnostics(
        matches=pd.DataFrame(),
        fixtures_fetched=live_fixtures,
        fixtures_in_window=live_fixtures,
        odds_list=odds_list,
        odds_lookup=odds_lookup,
        prediction_records=matches,
        outcomes=outcomes,
        sport_key=sport_key,
        sport=sport,
        slop_locks=slop_locks,
        longslop=data.get("longslop"),
        slimegrinder=slimegrinder,
    )
    previous_diagnostics = data.get("diagnostics") or {}
    if previous_diagnostics.get("historical_matches") is not None:
        diagnostics["historical_matches"] = previous_diagnostics.get("historical_matches")
    data["diagnostics"] = diagnostics

    pick_history_path = data_path.parent / "pick_history.json"
    pick_history = _load_json(str(pick_history_path))
    if isinstance(pick_history, dict):
        latest_snapshots = _load_latest_odds_snapshots(odds_history_path, sport_key)
        picks = pick_history.get("picks", [])
        if isinstance(picks, list):
            _apply_latest_market_snapshots(picks, latest_snapshots)
            pick_history["picks"] = picks
            _save_json(str(pick_history_path), pick_history)

    with data_path.open("w") as f:
        json.dump(data, f, indent=2)

    print(
        f"  {sport_key}: {matches_with_odds} matches with odds → "
        f"{len(slop_locks)} locks ({in_window} in window, {fallback} fallback), "
        f"{len(totals_locks)} totals"
    )


def main() -> None:
    sports = sys.argv[1:] if len(sys.argv) > 1 else list(SPORTS.keys())
    print(f"Refreshing picks for: {', '.join(sports)}")
    for sport_key in sports:
        refresh_sport(sport_key)
    dashboard = build_dashboard_data(str(DATA_DIR))
    _save_json(str(Path(DATA_DIR) / "dashboard.json"), dashboard)
    print("Done.")


if __name__ == "__main__":
    main()
