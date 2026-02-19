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
    CONGESTION_THRESHOLD_DAYS,
    DATA_DIR,
    NBA_B2B_PENALTY,
    SLOP_LOCK_MIN_ODDS,
    SLOP_LOCK_MAX_ODDS,
    SPORTS,
)
from pipeline.fetch_data import fetch_epl_matches, fetch_epl_fixtures, fetch_odds, normalize_team_name
from pipeline.fetch_nba import fetch_nba_games, fetch_nba_schedule, normalize_nba_team_name, fetch_nba_espn_games, fetch_nba_espn_schedule
from pipeline.fetch_ncaam import fetch_ncaam_games, fetch_ncaam_schedule, normalize_ncaam_team_name
from pipeline.fetch_xg import fetch_understat_xg
from pipeline.models import (
    AdjustedEfficiency,
    EloRatings,
    FourFactorsModel,
    dixon_coles_predict,
    efficiency_predict,
    elo_predict,
    fit_dixon_coles,
    four_factors_predict,
    scoreline_to_probabilities,
)
from pipeline.ensemble import blend_predictions, compute_edges, decimal_to_american
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


def _check_congestion(team, fixtures, matches, threshold_days=CONGESTION_THRESHOLD_DAYS):
    """Return True if *team* has a fixture within *threshold_days* of another."""
    team_dates = []

    for _, row in matches.iterrows():
        if row["home_team"] == team or row["away_team"] == team:
            dt = pd.to_datetime(row["date"])
            if dt.tzinfo is not None:
                dt = dt.tz_localize(None)
            team_dates.append(dt)

    for fix in fixtures:
        if fix["home_team"] == team or fix["away_team"] == team:
            dt = pd.to_datetime(fix["date"])
            if dt.tzinfo is not None:
                dt = dt.tz_localize(None)
            team_dates.append(dt)

    if len(team_dates) < 2:
        return False

    team_dates.sort()
    for i in range(1, len(team_dates)):
        gap = (team_dates[i] - team_dates[i - 1]).days
        if 0 < gap <= threshold_days:
            return True

    return False


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
    cutoff = pd.to_datetime(before_date)
    team_mask = (matches["home_team"] == team) | (matches["away_team"] == team)
    team_games = matches[team_mask].copy()
    team_games["_dt"] = pd.to_datetime(team_games["date"])
    past_games = team_games[team_games["_dt"] < cutoff]

    if past_games.empty:
        return None

    last_game = past_games["_dt"].max()
    return (cutoff - last_game).days


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
                    f"Write exactly 1-2 sentences explaining why this is a value pick. Be direct, "
                    f"confident, concise. No hedging. Reference specific model data or edge.\n\n"
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


def _compute_slop_locks(prediction_records, outcomes):
    """Extract SLOP LOCKS: top 5 most confident picks at -150 to +195 odds."""
    lock_candidates = []
    for rec in prediction_records:
        edges = rec.get("edges", {})
        best_odds = rec.get("best_odds", {})
        for outcome in outcomes:
            e = edges.get(outcome)
            if not e:
                continue
            american = best_odds.get(outcome, e.get("american_odds"))
            if american is None:
                continue
            if not (SLOP_LOCK_MIN_ODDS <= american <= SLOP_LOCK_MAX_ODDS):
                continue
            lock_candidates.append({
                "home_team": rec["home_team"],
                "away_team": rec["away_team"],
                "date": rec["date"],
                "matchday": rec.get("matchday"),
                "pick": outcome,
                "model_prob": round(e["model_prob"], 4),
                "implied_prob": round(e["implied_prob"], 4),
                "edge": round(e["edge"], 4),
                "american_odds": american,
                "decimal_odds": e["decimal_odds"],
                "individual_models": rec.get("individual_models", {}),
            })

    lock_candidates.sort(key=lambda x: x["model_prob"], reverse=True)
    return lock_candidates[:5]


def _compute_longslop(prediction_records, outcomes):
    """Extract LONGSLOP (best longshot the model believes may hit, +500 or better).

    Ranked by model probability among +500 lines where the model is at
    least as optimistic as the books (model_prob >= implied_prob).
    """
    longslop_candidates = []
    for rec in prediction_records:
        edges = rec.get("edges", {})
        best_odds = rec.get("best_odds", {})
        model_probs = rec.get("model_probs", {})
        for outcome in outcomes:
            e = edges.get(outcome)
            american = best_odds.get(outcome, e.get("american_odds") if e else None)
            if american is None or american < 500:
                continue
            # Use model probability from edges if available, else from blended probs
            model_prob = e["model_prob"] if e else model_probs.get(outcome, 0)
            implied_prob = e["implied_prob"] if e else 0
            edge = e["edge"] if e else 0
            decimal_odds = e["decimal_odds"] if e else 0
            # Model must believe in this at least as much as the books do
            if model_prob <= 0 or model_prob < implied_prob:
                continue
            longslop_candidates.append({
                "home_team": rec["home_team"],
                "away_team": rec["away_team"],
                "date": rec["date"],
                "matchday": rec.get("matchday"),
                "pick": outcome,
                "model_prob": round(model_prob, 4),
                "implied_prob": round(implied_prob, 4),
                "edge": round(edge, 4),
                "american_odds": american,
                "decimal_odds": decimal_odds,
                "individual_models": rec.get("individual_models", {}),
            })

    longslop_candidates.sort(key=lambda x: x["model_prob"], reverse=True)
    return longslop_candidates[0] if longslop_candidates else None


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
        Key into ``SPORTS`` config dict (e.g. "epl", "nba").
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

    if sport_key == "epl":
        matches = fetch_epl_matches()
        fixtures = fetch_epl_fixtures()

        xg_data = None
        try:
            xg_data = fetch_understat_xg()
        except Exception:
            pass
    elif sport_key == "nba":
        games_df, box_scores_df = fetch_nba_espn_games()
        fixtures = fetch_nba_espn_schedule()
        matches = games_df
        xg_data = None
    elif sport_key == "ncaam":
        games_df, box_scores_df = fetch_ncaam_games()
        fixtures = fetch_ncaam_schedule()
        matches = games_df
        xg_data = None
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
    dc_params = None
    xg_params = None

    if "dixon_coles" in sport["models"]:
        dc_params = fit_dixon_coles(matches)

    if "xg" in sport["models"] and xg_data is not None and len(xg_data) >= 20:
        xg_params = fit_dixon_coles(
            xg_data,
            goals_col_home="home_xg",
            goals_col_away="away_xg",
        )

    # Elo ratings (with sport-specific parameters)
    elo = None
    if "elo" in sport["models"]:
        all_teams = sorted(
            set(matches["home_team"].unique()) | set(matches["away_team"].unique())
        )
        elo = EloRatings(
            all_teams,
            k_factor=sport["elo_k_factor"],
            home_advantage=sport["elo_home_advantage"],
        )
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
    if dc_params is not None:
        model_names.append("dixon_coles")
    if xg_params is not None:
        model_names.append("xg")
    if elo is not None:
        model_names.append("elo")
    if efficiency_model is not None:
        model_names.append("efficiency")
    if four_factors_model is not None:
        model_names.append("four_factors")

    accuracies = [get_rolling_accuracy(accuracy_log, name) for name in model_names]
    weights = compute_model_weights(accuracies)
    model_weight_dict = dict(zip(model_names, weights))

    # ------------------------------------------------------------------
    # 4. Normalize odds team names and build lookup
    # ------------------------------------------------------------------
    if sport_key == "epl":
        normalizer = normalize_team_name
    elif sport_key == "nba":
        normalizer = normalize_nba_team_name
    elif sport_key == "ncaam":
        normalizer = normalize_ncaam_team_name
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
    # ------------------------------------------------------------------
    prediction_records = []

    for fix in fixtures:
        home = fix["home_team"]
        away = fix["away_team"]

        # For models requiring fitted params, check team is known
        if dc_params is not None:
            if home not in dc_params["attack"] or away not in dc_params["attack"]:
                continue
        if elo is not None:
            if home not in elo.ratings and away not in elo.ratings:
                continue

        individual_preds = []
        blend_weights = []
        individual_models = {}

        # Dixon-Coles (EPL only)
        if dc_params is not None and home in dc_params["attack"] and away in dc_params["attack"]:
            cong_home = _check_congestion(home, fixtures, matches)
            cong_away = _check_congestion(away, fixtures, matches)

            dc_matrix = dixon_coles_predict(home, away, dc_params,
                                            congestion_home=cong_home,
                                            congestion_away=cong_away)
            dc_probs = scoreline_to_probabilities(dc_matrix)
            individual_preds.append(dc_probs)
            blend_weights.append(model_weight_dict["dixon_coles"])
            individual_models["dixon_coles"] = dc_probs

        # xG (EPL only)
        if xg_params is not None and home in xg_params["attack"] and away in xg_params["attack"]:
            cong_home = _check_congestion(home, fixtures, matches)
            cong_away = _check_congestion(away, fixtures, matches)

            xg_matrix = dixon_coles_predict(home, away, xg_params,
                                            congestion_home=cong_home,
                                            congestion_away=cong_away)
            xg_probs = scoreline_to_probabilities(xg_matrix)
            individual_preds.append(xg_probs)
            blend_weights.append(model_weight_dict["xg"])
            individual_models["xg"] = xg_probs

        # Elo (all sports)
        if elo is not None and home in elo.ratings and away in elo.ratings:
            home_rest_adj = 0.0
            away_rest_adj = 0.0
            if sport_key == "nba":
                home_rest = _days_since_last_game(home, fix["date"], matches)
                away_rest = _days_since_last_game(away, fix["date"], matches)
                if home_rest == 1:
                    home_rest_adj = -NBA_B2B_PENALTY
                if away_rest == 1:
                    away_rest_adj = -NBA_B2B_PENALTY
            elo_probs = elo_predict(elo, home, away, outcomes=outcomes,
                                    home_rest_adj=home_rest_adj,
                                    away_rest_adj=away_rest_adj)
            individual_preds.append(elo_probs)
            blend_weights.append(model_weight_dict["elo"])
            individual_models["elo"] = elo_probs

        # Adjusted Efficiency (NCAAM)
        if efficiency_model is not None and home in efficiency_model.off_efficiency and away in efficiency_model.off_efficiency:
            eff_probs = efficiency_predict(
                efficiency_model, home, away,
                home_bonus=sport.get("efficiency_home_bonus", 3.5),
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
            edges = compute_edges(blended, match_odds)
            best_odds = {}
            for out in outcomes:
                odds_key = f"{out}_odds"
                dec = match_odds.get(odds_key, 0)
                if dec > 0:
                    best_odds[out] = decimal_to_american(dec)

        record = {
            "home_team": home,
            "away_team": away,
            "date": fix["date"],
            "matchday": fix.get("matchday"),
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

    # Generate analysis blurbs via Claude
    slop_locks = _generate_blurbs(slop_locks, pick_type="lock")
    longslop = _generate_blurbs(longslop, pick_type="longslop")

    # ------------------------------------------------------------------
    # 6. Evaluate past predictions
    # ------------------------------------------------------------------
    history = _load_json(history_path)
    if not isinstance(history, dict):
        history = {}
    past_predictions = history.get("predictions", [])

    result_lookup = {}
    for _, row in matches.iterrows():
        key = (row["home_team"], row["away_team"], str(row["date"])[:10])
        result_lookup[key] = (int(row["home_goals"]), int(row["away_goals"]))

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
                update_accuracy_log(accuracy_log, model_name, eval_result)

            if "model_probs" in pred:
                eval_result = evaluate_prediction(pred["model_probs"], hg, ag)
                update_accuracy_log(accuracy_log, "ensemble", eval_result)

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

    # Append today's new picks (deduplicate by match + pick + type)
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    existing_keys = set()
    for p in past_picks:
        existing_keys.add((p.get("pick_date"), p["type"], p["home_team"],
                           p["away_team"], p["pick"]))

    for lock in slop_locks:
        pk = (today_str, "slop_lock", lock["home_team"],
              lock["away_team"], lock["pick"])
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
        pk = (today_str, "longslop", longslop["home_team"],
              longslop["away_team"], longslop["pick"])
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
        "longslop": longslop,
        "matches": prediction_records,
        "season_stats": season_stats,
        "model_weights": {k: round(v, 4) for k, v in model_weight_dict.items()},
        "pick_stats": pick_stats,
    }
    _save_json(predictions_path, predictions_output)

    history_output = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "predictions": updated_past + prediction_records,
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
    run_pipeline()
