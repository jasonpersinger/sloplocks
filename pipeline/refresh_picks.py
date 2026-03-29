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

from scipy.stats import norm

from pipeline.backtest import build_dashboard_data
from pipeline.config import (
    SLOP_LOCK_MIN_ODDS,
    SLOP_LOCK_MAX_ODDS,
    SLOP_LOCK_FALLBACK_MIN_ODDS,
    SLIMEGRINDER_MIN_ODDS,
    SLIMEGRINDER_MAX_ODDS,
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
from pipeline.fetch_nba import normalize_nba_team_name
from pipeline.fetch_mlb import normalize_mlb_team_name
from pipeline.fetch_mma import normalize_mma_name
from pipeline.run import (
    _append_odds_snapshot_log,
    _apply_latest_market_snapshots,
    _build_odds_snapshot_rows,
    _compute_slimegrinder as _run_compute_slimegrinder,
    _compute_slop_locks as _run_compute_slop_locks,
    _compute_totals_locks,
    _load_json,
    _load_latest_odds_snapshots,
    _lookup_match_odds,
    _odds_history_path,
    _save_json,
)

_NORMALIZERS = {
    "nba": normalize_nba_team_name,
    "ncaam": normalize_ncaam_team_name,
    "mlb": normalize_mlb_team_name,
    "mma": normalize_mma_name,
}

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
    normalizer = _NORMALIZERS.get(sport_key, lambda x: x)

    # Fetch fresh odds and build lookup
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

    # Patch edges onto matches where we have fresh odds
    for match in matches:
        match_odds = _lookup_match_odds(
            odds_lookup,
            sport_key,
            match["home_team"],
            match["away_team"],
        )
        if match_odds:
            edges = compute_edges(
                match["model_probs"],
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
            model_prob = match["model_probs"][pick]
            edge = edges.get(pick, {}).get("edge", 0.0)
            match["pick"] = pick
            match["model_prob"] = round(model_prob, 4)
            match["edge"] = round(edge, 4)
            match["confidence_stars"] = compute_confidence_stars(model_prob, edge)
            match["american_odds"] = match["best_odds"].get(pick)

    totals_matches: list[dict] = data.get("totals_matches", [])
    for total_match in totals_matches:
        match_odds = _lookup_match_odds(
            odds_lookup,
            sport_key,
            total_match["home_team"],
            total_match["away_team"],
        )
        if not match_odds or match_odds.get("total_line") is None:
            continue
        expected_total = float(total_match.get("expected_total", match_odds["total_line"]))
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
