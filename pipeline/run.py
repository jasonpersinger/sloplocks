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
    ANTHROPIC_API_KEY,
    DATA_DIR,
    NBA_B2B_PENALTY,
    NBA_3IN4_PENALTY,
    SLOP_LOCK_MIN_ODDS,
    SLOP_LOCK_MAX_ODDS,
    SLOP_LOCK_FALLBACK_MIN_ODDS,
    SLIMEGRINDER_MIN_ODDS,
    SLIMEGRINDER_MAX_ODDS,
    SPORTS,
)
from pipeline.fetch_data import fetch_odds
from pipeline.fetch_nba import fetch_nba_games, fetch_nba_schedule, normalize_nba_team_name, fetch_nba_espn_games, fetch_nba_espn_schedule
from pipeline.fetch_ncaam import fetch_ncaam_games, fetch_ncaam_schedule, normalize_ncaam_team_name
from pipeline.fetch_mlb import fetch_mlb_games, fetch_mlb_schedule, normalize_mlb_team_name
from pipeline.fetch_mma import fetch_mma_games, fetch_mma_schedule, normalize_mma_name
from pipeline.models import (
    AdjustedEfficiency,
    EloRatings,
    FourFactorsModel,
    efficiency_predict,
    elo_predict,
    four_factors_predict,
)
from pipeline.ensemble import blend_predictions, compute_edges, decimal_to_american, compute_confidence_stars
from pipeline.backtest import (
    compute_model_weights,
    compute_roi,
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
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)



def _days_since_last_game(team: str, before_date: str, matches: pd.DataFrame) -> int | None:
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


def _compute_slop_locks(prediction_records, outcomes):
    """Extract SLOP LOCKS: High-confidence picks meeting Phase 3 criteria.

    Strict Requirements:
    - POTD (Pick of the Day): Always the #1 highest confidence pick.
    - Additional Locks: Must meet Confidence Score >= 65.
    - All must have Edge >= 3% (reduced from 5% to allow for POTD flexibility).
    - Win Prob > 45% (Must be competitive).
    - Max 3 picks total per sport.
    """
    candidates = []
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
            
            # Basic eligibility for any pick
            if edge >= 0.03 and prob > 0.45:
                candidates.append({
                    "home_team": rec["home_team"],
                    "away_team": rec["away_team"],
                    "date": rec["date"],
                    "matchday": rec.get("matchday"),
                    "pick": outcome,
                    "model_prob": round(prob, 4),
                    "implied_prob": round(e["implied_prob"], 4),
                    "edge": round(edge, 4),
                    "american_odds": e["american_odds"],
                    "decimal_odds": e["decimal_odds"],
                    "confidence_score": conf,
                    "individual_models": rec.get("individual_models", {}),
                })

    # Sort all eligible candidates by confidence score
    candidates.sort(key=lambda x: (x["confidence_score"], x["edge"]), reverse=True)
    
    if not candidates:
        return []

    # 1. The Pick of the Day (Always the #1 candidate)
    selected = [candidates[0]]
    
    # 2. Additional Locks (Only if they meet the strict 65+ Sniper threshold)
    for c in candidates[1:3]:
        if c["confidence_score"] >= 65:
            selected.append(c)

    return _exclude_opponent_conflicts(selected)



def _compute_longslop(prediction_records, outcomes):
    """Extract LONGSLOP: High-upside longshots (+500 or better) that pass Phase 3.

    Must meet the same 65+ confidence threshold as standard locks.
    """
    longslop_candidates = []
    for rec in prediction_records:
        if rec.get("completed"):
            continue
        edges = rec.get("edges", {})
        
        for outcome in outcomes:
            e = edges.get(outcome)
            if not e:
                continue
            
            american = e.get("american_odds", 0)
            if american < 500:
                continue
            
            # Phase 3 Thresholds
            conf = e.get("confidence_score", 0)
            edge = e.get("edge", 0)
            prob = e.get("model_prob", 0)
            
            # For longslop, we require high confidence and positive edge
            if conf >= 65 and edge >= 0:
                longslop_candidates.append({
                    "home_team": rec["home_team"],
                    "away_team": rec["away_team"],
                    "date": rec["date"],
                    "matchday": rec.get("matchday"),
                    "pick": outcome,
                    "model_prob": round(prob, 4),
                    "implied_prob": round(e["implied_prob"], 4),
                    "edge": round(edge, 4),
                    "american_odds": american,
                    "decimal_odds": e["decimal_odds"],
                    "confidence_score": conf,
                    "individual_models": rec.get("individual_models", {}),
                })

    longslop_candidates.sort(key=lambda x: (x["confidence_score"], x["edge"]), reverse=True)
    return longslop_candidates[0] if longslop_candidates else None


def _compute_slimegrinder(prediction_records, outcomes):
    """Extract SLIMEGRINDER: High-confidence, likely winners (odds -250 to +165).
    
    Must meet the same 65+ confidence threshold as Locks.
    """
    candidates = []
    for rec in prediction_records:
        if rec.get("completed"):
            continue
        edges = rec.get("edges", {})

        for outcome in outcomes:
            e = edges.get(outcome)
            if not e:
                continue

            american = e.get("american_odds", 0)
            if not (SLIMEGRINDER_MIN_ODDS <= american <= SLIMEGRINDER_MAX_ODDS):
                continue

            conf = e.get("confidence_score", 0)
            edge = e.get("edge", 0)
            prob = e.get("model_prob", 0)

            # Strict Phase 3 filter
            if conf >= 65 and edge > 0:
                candidates.append({
                    "home_team": rec["home_team"],
                    "away_team": rec["away_team"],
                    "date": rec["date"],
                    "matchday": rec.get("matchday"),
                    "pick": outcome,
                    "model_prob": round(prob, 4),
                    "implied_prob": round(e["implied_prob"], 4),
                    "edge": round(edge, 4),
                    "american_odds": american,
                    "decimal_odds": e["decimal_odds"],
                    "confidence_score": conf,
                })

    # Sort by raw model probability (descending) to find the likeliest winners
    candidates.sort(key=lambda x: x["model_prob"], reverse=True)
    return _exclude_opponent_conflicts(candidates)[:3]


def _compute_pick_stats(picks):
    """Compute aggregate stats from evaluated picks.

    Returns a dict with stats broken out by pick type (slop_lock, longslop, all).
    """
    stats = {}
    for pick_type in ("slop_lock", "longslop", "all"):
        subset = [p for p in picks if pick_type == "all" or p["type"] == pick_type]
        evaluated = [p for p in subset if p.get("evaluated")]
        wins = [p for p in evaluated if p.get("won")]
        losses = [p for p in evaluated if not p.get("won")]

        # ROI: flat $100 bet per pick
        bets = []
        for p in evaluated:
            bets.append({
                "stake": 100.0,
                "odds": p.get("decimal_odds", 0),
                "won": p.get("won", False),
            })
        roi = compute_roi(bets)

        stats[pick_type] = {
            "total": len(subset),
            "evaluated": len(evaluated),
            "wins": len(wins),
            "losses": len(losses),
            "pending": len(subset) - len(evaluated),
            "hit_rate": round(len(wins) / max(len(evaluated), 1), 3) if evaluated else None,
            "roi": round(roi, 3) if evaluated else None,
        }

    return stats


# ---------------------------------------------------------------------------
# Per-sport pipeline
# ---------------------------------------------------------------------------

def run_sport_pipeline(sport_key, output_dir=None):
    """Run prediction pipeline for a single sport.

    Parameters
    ----------
    sport_key : str
        Key into ``SPORTS`` config dict (e.g. "nba", "ncaam").
    output_dir : str or None
        Override the sport's data directory.
    """
    sport = SPORTS[sport_key]
    sport_dir = output_dir or sport["data_dir"]
    outcomes = sport["outcomes"]

    predictions_path = os.path.join(sport_dir, "predictions.json")
    history_path = os.path.join(sport_dir, "history.json")
    accuracy_path = os.path.join(sport_dir, "model_accuracy.json")

    # ------------------------------------------------------------------
    # 1. Fetch data (sport-specific)
    # ------------------------------------------------------------------
    box_scores_df = None

    if sport_key == "nba":
        games_df, box_scores_df = fetch_nba_espn_games(
            cache_path=os.path.join(sport_dir, "espn_cache.json")
        )
        fixtures = fetch_nba_espn_schedule()
        matches = games_df
    elif sport_key == "ncaam":
        games_df, box_scores_df = fetch_ncaam_games(
            cache_path=os.path.join(sport_dir, "espn_cache.json")
        )
        fixtures = fetch_ncaam_schedule()
        # Fallback: if scoreboard is empty, check cache for any games matching the allowed window
        if not fixtures and games_df is not None:
            today_utc = datetime.now(timezone.utc).date()
            allowed = {
                (today_utc - timedelta(days=1)).strftime("%Y-%m-%d"),
                today_utc.strftime("%Y-%m-%d"),
                (today_utc + timedelta(days=1)).strftime("%Y-%m-%d")
            }
            upcoming = games_df[games_df["date"].isin(allowed)]
            for _, row in upcoming.iterrows():
                fixtures.append({
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "date": row["date"],
                    "completed": row.get("completed", False)
                })
        matches = games_df
    elif sport_key == "mlb":
        games_df, box_scores_df = fetch_mlb_games(
            cache_path=os.path.join(sport_dir, "espn_cache.json")
        )
        fixtures = fetch_mlb_schedule()
        matches = games_df
    elif sport_key == "mma":
        games_df, box_scores_df = fetch_mma_games(
            cache_path=os.path.join(sport_dir, "espn_cache.json")
        )
        fixtures = fetch_mma_schedule()
        matches = games_df
    else:
        raise ValueError(f"Unknown sport: {sport_key}")

    # Odds are generic — just pass the right sport key
    odds_list = []
    try:
        odds_list = fetch_odds(sport_key=sport["odds_sport"])
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 2. Fit models
    # ------------------------------------------------------------------
    # Elo ratings (with sport-specific parameters)
    elo = None
    if "elo" in sport["models"]:
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

    # Adjusted Efficiency model (NCAAM)
    efficiency_model = None
    if "efficiency" in sport["models"] and box_scores_df is not None:
        efficiency_model = AdjustedEfficiency(box_scores_df, matches)

    # Four Factors model (NCAAM)
    four_factors_model = None
    if "four_factors" in sport["models"] and box_scores_df is not None:
        four_factors_model = FourFactorsModel(box_scores_df, matches)

    # ------------------------------------------------------------------
    # 3. Load accuracy log and compute model weights
    # ------------------------------------------------------------------
    accuracy_log = _load_json(accuracy_path)
    if not isinstance(accuracy_log, dict):
        accuracy_log = {}

    # Build list of active models for this run
    model_names = []
    if elo is not None:
        model_names.append("elo")
    if efficiency_model is not None:
        model_names.append("efficiency")
    if four_factors_model is not None:
        model_names.append("four_factors")

    accuracy_window = sport.get("accuracy_window", None)
    accuracies = [get_rolling_accuracy(accuracy_log, name, window=accuracy_window) for name in model_names]
    weights = compute_model_weights(accuracies)
    model_weight_dict = dict(zip(model_names, weights))

    # ------------------------------------------------------------------
    # 4. Normalize odds team names and build lookup
    # ------------------------------------------------------------------
    if sport_key == "nba":
        normalizer = normalize_nba_team_name
    elif sport_key == "ncaam":
        normalizer = normalize_ncaam_team_name
    elif sport_key == "mlb":
        normalizer = normalize_mlb_team_name
    elif sport_key == "mma":
        normalizer = normalize_mma_name
    else:
        normalizer = lambda x: x
    for o in odds_list:
        o["home_team"] = normalizer(o["home_team"])
        o["away_team"] = normalizer(o["away_team"])

    odds_lookup = {}
    for o in odds_list:
        key = (o["home_team"], o["away_team"])
        odds_lookup[key] = o

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
    fixtures = [f for f in fixtures if str(f.get("date", ""))[:10] in allowed_dates]

    prediction_records = []

    for fix in fixtures:
        home = fix["home_team"]
        away = fix["away_team"]
        is_neutral = fix.get("neutral", False)

        individual_preds = []
        blend_weights = []
        individual_models = {}

        # Elo (all sports)
        if elo is not None:
            home_rest_adj = 0.0
            away_rest_adj = 0.0
            if sport_key == "nba":
                home_rest = _days_since_last_game(home, fix["date"], matches)
                away_rest = _days_since_last_game(away, fix["date"], matches)
                if home_rest == 1:
                    home_rest_adj = -NBA_B2B_PENALTY
                if away_rest == 1:
                    away_rest_adj = -NBA_B2B_PENALTY
                # 3-games-in-4-nights fatigue
                if _games_in_window(home, fix["date"], matches, days=4) >= 3:
                    home_rest_adj -= NBA_3IN4_PENALTY
                if _games_in_window(away, fix["date"], matches, days=4) >= 3:
                    away_rest_adj -= NBA_3IN4_PENALTY
            
            # Disable home advantage for neutral sites
            current_home_adv = 0.0 if is_neutral else elo.home_advantage
            
            elo_probs = elo_predict(elo, home, away, outcomes=outcomes,
                                    home_rest_adj=home_rest_adj,
                                    away_rest_adj=away_rest_adj,
                                    home_advantage_override=current_home_adv)
            individual_preds.append(elo_probs)
            blend_weights.append(model_weight_dict["elo"])
            individual_models["elo"] = elo_probs

        # Adjusted Efficiency (NCAAM)
        if efficiency_model is not None and home in efficiency_model.off_efficiency and away in efficiency_model.off_efficiency:
            # Disable home bonus for neutral sites
            current_home_bonus = 0.0 if is_neutral else sport.get("efficiency_home_bonus", 3.5)
            
            eff_probs = efficiency_predict(
                efficiency_model, home, away,
                home_bonus=current_home_bonus,
            )
            individual_preds.append(eff_probs)
            blend_weights.append(model_weight_dict["efficiency"])
            individual_models["efficiency"] = eff_probs

        # Four Factors (NCAAM)
        if four_factors_model is not None and four_factors_model.model is not None:
            if home in four_factors_model.team_stats and away in four_factors_model.team_stats:
                ff_probs = four_factors_predict(four_factors_model, home, away)
                individual_preds.append(ff_probs)
                blend_weights.append(model_weight_dict["four_factors"])
                individual_models["four_factors"] = ff_probs

        if not individual_preds:
            continue

        # Blend
        blended = blend_predictions(individual_preds, blend_weights)

        # Edges and best odds
        match_odds = odds_lookup.get((home, away))
        edges = {}
        best_odds = {}
        if match_odds:
            # Pass individual_preds to enable Agreement Score
            edges = compute_edges(blended, match_odds, individual_probs=individual_preds)
            best_odds = {}
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
        confidence_score = edge_data.get("confidence_score", 0)
        
        # Legacy stars for frontend compatibility
        stars = compute_confidence_stars(model_prob, edge)

        record = {
            "home_team": home,
            "away_team": away,
            "date": fix["date"],
            "matchday": fix.get("matchday"),
            "completed": fix.get("completed", False),
            "neutral": is_neutral,
            "home_pitcher": fix.get("home_pitcher"),
            "away_pitcher": fix.get("away_pitcher"),
            "pick": pick,
            "model_prob": round(model_prob, 4),
            "edge": round(edge, 4),
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
        }
        prediction_records.append(record)

    # ------------------------------------------------------------------
    # 5b. SLOP LOCKS + LONGSLOP
    # ------------------------------------------------------------------
    slop_locks = _compute_slop_locks(prediction_records, outcomes)
    longslop = _compute_longslop(prediction_records, outcomes)
    slimegrinder = _compute_slimegrinder(prediction_records, outcomes)

    # Generate analysis blurbs via Claude
    slop_locks = _generate_blurbs(slop_locks, pick_type="lock")
    longslop = _generate_blurbs(longslop, pick_type="longslop")

    history = _load_json(history_path)
    if not isinstance(history, dict):
        history = {}
    past_predictions = history.get("predictions", [])

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

        updated_past.append(pred)

    # ------------------------------------------------------------------
    # 6b. Track and evaluate picks
    # ------------------------------------------------------------------
    pick_history_path = os.path.join(sport_dir, "pick_history.json")
    pick_history = _load_json(pick_history_path)
    if not isinstance(pick_history, dict):
        pick_history = {}
    past_picks = pick_history.get("picks", [])

    # Evaluate unevaluated past picks against results
    for pick in past_picks:
        if pick.get("evaluated"):
            continue
        match_date = str(pick.get("match_date", ""))[:10]
        key = (pick["home_team"], pick["away_team"], match_date)
        result = result_lookup.get(key)
        if result is not None:
            hg, ag = result
            if hg > ag:
                actual = "home"
            elif hg == ag:
                actual = "draw"
            else:
                actual = "away"
            pick["evaluated"] = True
            pick["actual"] = actual
            pick["won"] = pick["pick"] == actual
            pick["home_goals"] = hg
            pick["away_goals"] = ag

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
            past_picks.append({
                "pick_date": today_str,
                "type": "slop_lock",
                "home_team": lock["home_team"],
                "away_team": lock["away_team"],
                "match_date": str(lock["date"])[:10],
                "pick": lock["pick"],
                "model_prob": lock["model_prob"],
                "implied_prob": lock["implied_prob"],
                "edge": lock["edge"],
                "american_odds": lock["american_odds"],
                "decimal_odds": lock["decimal_odds"],
                "evaluated": False,
            })

    if longslop:
        pk = ("longslop", longslop["home_team"], longslop["away_team"],
              str(longslop["date"])[:10], longslop["pick"])
        if pk not in existing_keys:
            past_picks.append({
                "pick_date": today_str,
                "type": "longslop",
                "home_team": longslop["home_team"],
                "away_team": longslop["away_team"],
                "match_date": str(longslop["date"])[:10],
                "pick": longslop["pick"],
                "model_prob": longslop["model_prob"],
                "implied_prob": longslop["implied_prob"],
                "edge": longslop["edge"],
                "american_odds": longslop["american_odds"],
                "decimal_odds": longslop["decimal_odds"],
                "evaluated": False,
            })

    pick_history = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "picks": past_picks,
    }
    _save_json(pick_history_path, pick_history)

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
    predictions_output = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sport": sport_key,
        "sport_name": sport["display_name"],
        "outcomes": outcomes,
        "slop_locks": slop_locks,
        "slimegrinder": slimegrinder,
        "longslop": longslop,
        "matches": prediction_records,
        "season_stats": season_stats,
        "model_weights": {k: round(v, 4) for k, v in model_weight_dict.items()},
        "pick_stats": pick_stats,
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
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "predictions": updated_past + new_predictions,
    }
    _save_json(history_path, history_output)

    _save_json(accuracy_path, accuracy_log)

    return predictions_output


# ---------------------------------------------------------------------------
# Main pipeline — runs all sports
# ---------------------------------------------------------------------------

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

    manifest = {
        "updated_at": now,
        "sports": {},
    }

    for sport_key in SPORTS:
        sport_dir = os.path.join(base_dir, sport_key) if output_dir else None
        try:
            run_sport_pipeline(sport_key, output_dir=sport_dir)
            manifest["sports"][sport_key] = {
                "name": SPORTS[sport_key]["display_name"],
                "status": "ok",
                "updated_at": now,
            }
        except Exception as exc:
            manifest["sports"][sport_key] = {
                "name": SPORTS[sport_key]["display_name"],
                "status": "error",
                "error": str(exc),
                "updated_at": now,
            }

    _save_json(os.path.join(base_dir, "manifest.json"), manifest)

    return manifest


if __name__ == "__main__":
    import sys
    manifest = run_pipeline()
    # Check if any sport has an error
    errors = [s["error"] for s in manifest["sports"].values() if s["status"] == "error"]
    if errors:
        for err in errors:
            print(f"Error in pipeline: {err}", file=sys.stderr)
        sys.exit(1)
