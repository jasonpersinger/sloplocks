"""Quick refresh of odds and slop locks without re-training models.

Fetches fresh odds from The Odds API, patches edges onto existing match
records, and recomputes slop locks using the current filter logic.  Runs in
seconds — no historical data fetch, no model fitting.

Usage:
    python -m pipeline.refresh_picks            # all sports
    python -m pipeline.refresh_picks nba epl    # specific sports
"""

import json
import sys
from pathlib import Path

from pipeline.config import SLOP_LOCK_MIN_ODDS, SLOP_LOCK_MAX_ODDS, SPORTS
from pipeline.ensemble import compute_edges, decimal_to_american
from pipeline.fetch_data import fetch_odds, normalize_team_name
from pipeline.fetch_ncaam import normalize_ncaam_team_name
from pipeline.fetch_nba import normalize_nba_team_name
from pipeline.run import _compute_best_candidate, compute_sotd

_NORMALIZERS = {
    "epl": normalize_team_name,
    "nba": normalize_nba_team_name,
    "ncaam": normalize_ncaam_team_name,
}

_MAX_LOCKS = 5


def _recompute_slop_locks(matches: list[dict], outcomes: list[str]) -> list[dict]:
    """Rebuild slop locks from current match edges.

    Picks within the -150/+195 window are preferred; remaining slots are
    filled from outside the window so there are always picks when odds exist.
    """
    in_window: list[dict] = []
    outside_window: list[dict] = []

    for m in matches:
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
                "american_odds": american,
                "decimal_odds": e["decimal_odds"],
                "individual_models": m.get("individual_models", {}),
                "blurb": "",
            })
        if not game_candidates:
            continue
        best = max(game_candidates, key=lambda x: x["model_prob"])
        if SLOP_LOCK_MIN_ODDS <= best["american_odds"] <= SLOP_LOCK_MAX_ODDS:
            in_window.append(best)
        else:
            outside_window.append(best)

    in_window.sort(key=lambda x: x["model_prob"], reverse=True)
    outside_window.sort(key=lambda x: x["model_prob"], reverse=True)
    result = in_window[:_MAX_LOCKS]
    if len(result) < _MAX_LOCKS:
        result.extend(outside_window[:_MAX_LOCKS - len(result)])
    return result


def refresh_sport(sport_key: str) -> dict | None:
    """Refresh odds and slop locks for one sport.

    Returns a dict with ``best_candidate`` and ``sport_name`` keys for SOTD
    computation, or None if predictions.json doesn't exist for this sport.
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
            match["edges"] = compute_edges(match["model_probs"], match_odds)
            match["best_odds"] = {
                outcome: decimal_to_american(match_odds[f"{outcome}_odds"])
                for outcome in outcomes
                if match_odds.get(f"{outcome}_odds", 0) > 0
            }

    matches_with_odds = sum(1 for m in matches if m.get("edges"))
    slop_locks = _recompute_slop_locks(matches, outcomes)
    best_candidate = _compute_best_candidate(matches, outcomes)
    in_window = sum(
        1 for s in slop_locks
        if SLOP_LOCK_MIN_ODDS <= s["american_odds"] <= SLOP_LOCK_MAX_ODDS
    )
    fallback = len(slop_locks) - in_window

    data["matches"] = matches
    data["slop_locks"] = slop_locks

    with data_path.open("w") as f:
        json.dump(data, f, indent=2)

    print(
        f"  {sport_key}: {matches_with_odds} matches with odds → "
        f"{len(slop_locks)} locks ({in_window} in window, {fallback} fallback)"
    )

    return {
        "best_candidate": best_candidate,
        "sport_name": sport["display_name"],
    }


def main() -> None:
    sports = sys.argv[1:] if len(sys.argv) > 1 else list(SPORTS.keys())
    print(f"Refreshing picks for: {', '.join(sports)}")
    sport_candidates: dict[str, dict] = {}
    for sport_key in sports:
        result = refresh_sport(sport_key)
        if result and result.get("best_candidate"):
            sport_candidates[sport_key] = result
    compute_sotd(sport_candidates, "data")
    print("Done.")


if __name__ == "__main__":
    main()
