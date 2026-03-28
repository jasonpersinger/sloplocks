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

from pipeline.config import (
    SLOP_LOCK_MIN_ODDS,
    SLOP_LOCK_MAX_ODDS,
    SLOP_LOCK_FALLBACK_MIN_ODDS,
    SLIMEGRINDER_MIN_ODDS,
    SLIMEGRINDER_MAX_ODDS,
    SPORTS,
    DATA_DIR,
)

from pipeline.ensemble import compute_edges, decimal_to_american, compute_confidence_stars
from pipeline.fetch_data import fetch_odds
from pipeline.fetch_ncaam import normalize_ncaam_team_name
from pipeline.fetch_nba import normalize_nba_team_name
from pipeline.fetch_mlb import normalize_mlb_team_name
from pipeline.fetch_mma import normalize_mma_name
from pipeline.run import _exclude_opponent_conflicts

_NORMALIZERS = {
    "nba": normalize_nba_team_name,
    "ncaam": normalize_ncaam_team_name,
    "mlb": normalize_mlb_team_name,
    "mma": normalize_mma_name,
}

_MAX_LOCKS = 5


def _recompute_slop_locks(matches: list[dict], outcomes: list[str], min_expected_value: float = 0.0) -> list[dict]:
    """Rebuild slop locks from current match edges.

    Picks within the -150/+195 window are preferred; remaining slots are
    filled from outside the window so there are always picks when odds exist.
    """
    in_window: list[dict] = []
    outside_window: list[dict] = []

    for m in matches:
        if m.get("completed"):
            continue
        edges = m.get("edges", {})
        best_odds = m.get("best_odds", {})
        game_candidates = []
        for outcome in outcomes:
            e = edges.get(outcome)
            if not e:
                continue
            american = best_odds.get(outcome, e.get("american_odds"))
            if american is None:
                continue
            game_candidates.append({
                "home_team": m["home_team"],
                "away_team": m["away_team"],
                "date": m["date"],
                "matchday": m.get("matchday"),
                "pick": outcome,
                "model_prob": round(e["model_prob"], 4),
                "implied_prob": round(e["implied_prob"], 4),
                "edge": round(e["edge"], 4),
                "expected_value": round(e.get("expected_value", 0.0), 4),
                "american_odds": american,
                "decimal_odds": e["decimal_odds"],
                "kelly_fraction": round(e.get("kelly_fraction", 0.0), 4),
                "fractional_kelly": round(e.get("fractional_kelly", 0.0), 4),
                "confidence_stars": compute_confidence_stars(e["model_prob"], e["edge"]),
                "individual_models": m.get("individual_models", {}),
                "blurb": "",
            })
        game_candidates = [c for c in game_candidates if c["edge"] >= 0 and c["expected_value"] >= min_expected_value]
        if not game_candidates:
            continue
        best = max(game_candidates, key=lambda x: x["model_prob"])
        if SLOP_LOCK_MIN_ODDS <= best["american_odds"] <= SLOP_LOCK_MAX_ODDS:
            in_window.append(best)
        elif best["american_odds"] >= SLOP_LOCK_FALLBACK_MIN_ODDS:
            outside_window.append(best)

    in_window.sort(key=lambda x: x["model_prob"], reverse=True)
    outside_window.sort(key=lambda x: x["model_prob"], reverse=True)
    result = in_window[:_MAX_LOCKS]
    if len(result) < _MAX_LOCKS:
        result.extend(outside_window[:_MAX_LOCKS - len(result)])
    return _exclude_opponent_conflicts(result)


def _compute_slimegrinder(matches: list[dict], outcomes: list[str], min_expected_value: float = 0.0) -> list[dict]:
    """Extract SLIMEGRINDER: Top 3 likely winners with positive edge.
    Odds window: -250 to +165.
    """
    candidates = []
    for m in matches:
        if m.get("completed"):
            continue
        edges = m.get("edges", {})
        best_odds = m.get("best_odds", {})

        for outcome in outcomes:
            e = edges.get(outcome)
            if not e or not best_odds.get(outcome):
                continue

            american = best_odds[outcome]
            if not (SLIMEGRINDER_MIN_ODDS <= american <= SLIMEGRINDER_MAX_ODDS):
                continue

            if e["edge"] <= 0:
                continue
            if e.get("expected_value", 0.0) < min_expected_value:
                continue

            candidates.append({
                "home_team": m["home_team"],
                "away_team": m["away_team"],
                "date": m["date"],
                "matchday": m.get("matchday"),
                "pick": outcome,
                "model_prob": round(e["model_prob"], 4),
                "implied_prob": round(e["implied_prob"], 4),
                "edge": round(e["edge"], 4),
                "expected_value": round(e.get("expected_value", 0.0), 4),
                "american_odds": american,
                "decimal_odds": e["decimal_odds"],
                "kelly_fraction": round(e.get("kelly_fraction", 0.0), 4),
                "fractional_kelly": round(e.get("fractional_kelly", 0.0), 4),
                "confidence_stars": m.get("confidence_stars", 1),
            })

    candidates.sort(key=lambda x: x["model_prob"], reverse=True)
    return _exclude_opponent_conflicts(candidates)[:3]


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
        odds_list = fetch_odds(sport_key=sport["odds_sport"])
        for o in odds_list:
            o["home_team"] = normalizer(o["home_team"])
            o["away_team"] = normalizer(o["away_team"])
        odds_lookup = {(o["home_team"], o["away_team"]): o for o in odds_list}
        print(f"  {sport_key}: fetched odds for {len(odds_list)} games")
    except Exception as exc:
        print(f"  {sport_key}: odds fetch failed ({exc}), using cached edges")

    # Patch edges onto matches where we have fresh odds
    for match in matches:
        match_odds = odds_lookup.get((match["home_team"], match["away_team"]))
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

    matches_with_odds = sum(1 for m in matches if m.get("edges"))
    min_expected_value = sport.get("min_expected_value", 0.0)
    slop_locks = _recompute_slop_locks(matches, outcomes, min_expected_value=min_expected_value)
    slimegrinder = _compute_slimegrinder(matches, outcomes, min_expected_value=min_expected_value)
    in_window = sum(
        1 for s in slop_locks
        if SLOP_LOCK_MIN_ODDS <= s["american_odds"] <= SLOP_LOCK_MAX_ODDS
    )
    fallback = len(slop_locks) - in_window

    data["matches"] = matches
    data["slop_locks"] = slop_locks
    data["slimegrinder"] = slimegrinder

    with data_path.open("w") as f:
        json.dump(data, f, indent=2)

    print(
        f"  {sport_key}: {matches_with_odds} matches with odds → "
        f"{len(slop_locks)} locks ({in_window} in window, {fallback} fallback)"
    )


def main() -> None:
    sports = sys.argv[1:] if len(sys.argv) > 1 else list(SPORTS.keys())
    print(f"Refreshing picks for: {', '.join(sports)}")
    for sport_key in sports:
        refresh_sport(sport_key)
    print("Done.")


if __name__ == "__main__":
    main()
