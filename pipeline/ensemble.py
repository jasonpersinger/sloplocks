from typing import Optional, Union
"""Ensemble blending and edge detection for SLOP LOCKS pipeline."""

from pipeline.config import VALUE_EDGE_THRESHOLD

from datetime import datetime, timedelta

import numpy as np
from sklearn.isotonic import IsotonicRegression

def decimal_to_american(decimal_odds: float) -> Optional[int]:
    """Convert decimal odds to American format."""
    if decimal_odds <= 1.0:
        return None
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1) * 100)
    else:
        return round(-100 / (decimal_odds - 1))


def implied_probability(decimal_odds: float) -> float:
    """Return the implied probability from decimal odds: 1 / decimal_odds."""
    return 1.0 / decimal_odds


def no_vig_probabilities(odds_by_outcome: dict[str, float]) -> tuple[dict[str, float], dict[str, float], float]:
    """Return raw implied probs, no-vig implied probs, and total market hold.

    ``odds_by_outcome`` should already contain only active outcomes with valid
    decimal odds. The hold is ``sum(raw_implied) - 1``.
    """
    raw = {outcome: implied_probability(decimal) for outcome, decimal in odds_by_outcome.items()}
    total = sum(raw.values())
    if total <= 0:
        return raw, raw.copy(), 0.0
    fair = {outcome: prob / total for outcome, prob in raw.items()}
    return raw, fair, total - 1.0


def expected_value(probability: float, decimal_odds: float, stake: float = 1.0) -> float:
    """Return expected profit for a bet at decimal odds."""
    if decimal_odds <= 1.0:
        return -stake
    net_payout = (decimal_odds - 1.0) * stake
    return (probability * net_payout) - ((1.0 - probability) * stake)


def kelly_fraction(probability: float, decimal_odds: float, fraction: float = 1.0) -> float:
    """Return the capped Kelly stake fraction for decimal odds."""
    if decimal_odds <= 1.0:
        return 0.0
    b = decimal_odds - 1.0
    q = 1.0 - probability
    full_kelly = ((b * probability) - q) / b
    full_kelly = max(0.0, full_kelly)
    return full_kelly * max(0.0, fraction)


def blend_predictions(
    predictions: list[dict[str, float]],
    weights: list[float],
) -> dict[str, float]:
    """Weighted-average blend of multiple prediction dicts."""
    total_weight = sum(weights)
    norm_weights = [w / total_weight for w in weights]

    keys = predictions[0].keys()
    blended = {k: 0.0 for k in keys}
    for pred, w in zip(predictions, norm_weights):
        for key in blended:
            blended[key] += pred[key] * w

    return blended


def fit_probability_calibrators(
    historical_predictions: list[dict],
    outcomes: list[str],
    min_samples: int = 20,
    lookback_days: Optional[int] = None,
    holdout_days: int = 0,
    as_of: Optional[str] = None,
) -> dict[str, IsotonicRegression]:
    """Fit per-outcome isotonic calibrators from resolved historical predictions."""
    calibrators: dict[str, IsotonicRegression] = {}
    anchor = None
    if as_of:
        try:
            anchor = datetime.fromisoformat(str(as_of).replace("Z", "+00:00")).date()
        except ValueError:
            anchor = None
    cutoff = None if anchor is None else anchor - timedelta(days=max(holdout_days, 0))
    floor_day = None
    if cutoff is not None and lookback_days:
        floor_day = cutoff - timedelta(days=max(int(lookback_days) - 1, 0))

    def _prediction_day(pred: dict):
        for key in ("date", "match_date", "snapshot_timestamp", "generated_at"):
            value = pred.get(key)
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

    for outcome in outcomes:
        xs = []
        ys = []
        for pred in historical_predictions:
            if not pred.get("evaluated"):
                continue
            pred_day = _prediction_day(pred)
            if cutoff is not None and pred_day is not None and pred_day > cutoff:
                continue
            if floor_day is not None and pred_day is not None and pred_day < floor_day:
                continue
            probs = pred.get("model_probs", {})
            if outcome not in probs:
                continue
            if pred.get("home_goals") is None or pred.get("away_goals") is None:
                continue

            hg = int(pred["home_goals"])
            ag = int(pred["away_goals"])
            if hg > ag:
                actual = "home"
            elif hg == ag:
                actual = "draw"
            else:
                actual = "away"

            xs.append(float(probs[outcome]))
            ys.append(1 if actual == outcome else 0)

        if len(xs) < min_samples or len(set(ys)) < 2:
            continue

        reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        reg.fit(xs, ys)
        calibrators[outcome] = reg

    return calibrators


def apply_probability_calibration(
    model_probs: dict[str, float],
    calibrators: dict[str, IsotonicRegression],
    blend: float = 0.5,
) -> dict[str, float]:
    """Apply fitted per-outcome calibrators and renormalize the distribution."""
    if not calibrators:
        return model_probs

    blend = min(1.0, max(0.0, blend))
    adjusted = {}
    for outcome, prob in model_probs.items():
        if outcome in calibrators:
            calibrated = float(calibrators[outcome].predict([prob])[0])
            adjusted[outcome] = ((1.0 - blend) * prob) + (blend * calibrated)
        else:
            adjusted[outcome] = prob

    total = sum(max(0.0, prob) for prob in adjusted.values())
    if total <= 0:
        return model_probs

    return {outcome: max(0.0, prob) / total for outcome, prob in adjusted.items()}


def compute_confidence_score(
    model_probs_list: list[float],
    blended_prob: float,
    edge: float,
    implied_prob: float,
    expected_value: float = 0.0,
) -> float:
    """
    Calculate a 0-100 confidence score based on:
    1. Model Agreement (30%): Low variance between Elo, Efficiency, etc.
    2. Raw Win Probability (40%): Higher probability = higher score.
    3. Edge Quality (30%): Measured edge over market.
    4. Market Divergence Penalty: Severe penalty if model is >15% from books.
    """
    if not model_probs_list:
        return 0.0

    # 1. Agreement Score (0-1.0)
    # Single-model sports should not be treated as "perfect agreement", but they
    # also should not get zeroed out just because only one signal is available.
    if len(model_probs_list) == 1:
        agreement = 0.75
    else:
        std_dev = np.std(model_probs_list)
        agreement = max(0, 1.0 - (std_dev / 0.12))

    # 2. Probability Score (0-1.0)
    # Scale the useful range more tightly around realistic pick probabilities.
    prob_score = max(0.0, min(1.0, (blended_prob - 0.45) / 0.25))

    # 3. Edge Score (0-1.0)
    # A 6% edge is already meaningful in efficient markets.
    edge_score = max(0.0, min(1.0, edge / 0.06))

    # 4. EV Score (0-1.0)
    # Reward asymmetric payout spots directly instead of only via probability gap.
    ev_score = max(0.0, min(1.0, expected_value / 0.10))

    score = (agreement * 25) + (prob_score * 35) + (edge_score * 20) + (ev_score * 20)

    # 4. Market Divergence Penalty
    # If the model is too far from the market, it's likely missing info (injuries, etc.)
    divergence = abs(blended_prob - implied_prob)
    if divergence > 0.18:
        score *= 0.75
    elif divergence > 0.12:
        score *= 0.88

    # 5. Underdog Penalty for "Locks"
    # Softer than before: underdogs can still be sharp if the EV is there.
    if blended_prob < 0.40:
        score *= 0.75
    elif blended_prob < 0.45:
        score *= 0.90

    return round(float(score), 1)


from pipeline.config import VALUE_EDGE_THRESHOLD, MARKET_RESPECT_FACTOR, MAX_ALLOWED_DIVERGENCE


def calibrate_probability(model_prob: float, implied_prob: float) -> float:
    """
    Phase 4: Market Respect Calibration.
    Blends the model probability with the market's implied probability to 
    account for market efficiency (injuries, late scratches, motivation).
    
    P_calibrated = (w_model * P_model) + (w_market * P_market)
    """
    # 1. Unrealistic Check: If model is too far from market, it's probably wrong
    divergence = abs(model_prob - implied_prob)
    if divergence > MAX_ALLOWED_DIVERGENCE:
        # Heavily shrink back toward market if divergence is 'unrealistic'
        return (0.3 * model_prob) + (0.7 * implied_prob)
    
    # 2. Standard Shrinkage: Respect the market's efficiency
    w_market = MARKET_RESPECT_FACTOR  # 0.30

    # Add extra market weight for tail probabilities (> 0.15 from midline).
    # Threshold raised from 0.10 to 0.15 so moderate favourites (60-65%) are
    # not penalised; cap halved to 0.10 to limit total market pull to 40%.
    tail_excess = max(0.0, abs(model_prob - 0.5) - 0.15)
    if tail_excess > 0.0:
        w_market = min(0.40, w_market + min(0.10, tail_excess))
    w_model = 1.0 - w_market

    calibrated = (w_model * model_prob) + (w_market * implied_prob)

    # Mild compression of extreme confidence bands. Threshold raised from 0.60
    # to 0.65; multiplier eased from 0.80 to 0.85 so strong favourites survive.
    if calibrated > 0.65:
        calibrated = 0.65 + ((calibrated - 0.65) * 0.85)
    elif calibrated < 0.35:
        calibrated = 0.35 - ((0.35 - calibrated) * 0.85)

    return calibrated


def compute_edges(
    model_probs: dict[str, float],
    odds: dict[str, float],
    individual_probs: list[dict[str, float]] = None,
    fractional_kelly: float = 0.25,
) -> dict[str, dict]:
    """Compute calibrated value edges and confidence scores."""
    outcome_map = {
        "home": "home_odds",
        "draw": "draw_odds",
        "away": "away_odds",
    }

    active_odds = {}
    for outcome, odds_key in outcome_map.items():
        if outcome not in model_probs:
            continue
        dec_odds = odds.get(odds_key, 0)
        if dec_odds > 0:
            active_odds[outcome] = dec_odds

    raw_probs, fair_probs, hold = no_vig_probabilities(active_odds)
    benchmark = odds.get("moneyline_benchmark") or {}
    benchmark_fair = benchmark.get("fair_probs") or {}
    benchmark_raw = benchmark.get("raw_probs") or {}
    benchmark_hold = benchmark.get("hold")

    edges: dict[str, dict] = {}
    
    # If no odds are active, return pure model projections as a baseline
    if not active_odds:
        for outcome, prob in model_probs.items():
            outcome_model_probs = []
            if individual_probs:
                for p in individual_probs:
                    if outcome in p:
                        outcome_model_probs.append(p[outcome])
            
            # Confidence without market context is lower but not zero
            conf_score = compute_confidence_score(
                outcome_model_probs, prob, 0.0, prob, expected_value=0.0
            )
            
            edges[outcome] = {
                "model_prob": round(prob, 4),
                "raw_model_prob": round(prob, 4),
                "implied_prob": None,
                "market_implied_prob": None,
                "edge": 0.0,
                "expected_value": 0.0,
                "decimal_odds": None,
                "american_odds": None,
                "kelly_fraction": 0.0,
                "fractional_kelly": 0.0,
                "confidence_score": conf_score,
                "is_value": False,
                "unrealistic_flag": False,
                "hold": None,
                "market_source": "none",
                "market_books": 0,
            }
        return edges

    for outcome, dec_odds in active_odds.items():
        imp_prob = float(benchmark_fair.get(outcome, fair_probs[outcome]))
        raw_imp_prob = float(benchmark_raw.get(outcome, raw_probs[outcome]))
        raw_mod_prob = model_probs[outcome]
        
        # Phase 4: Calibrate probability (shrink toward market)
        # MANDATE: Only trust the market benchmark if we have a consensus (at least 3 books).
        # Single-book outliers (like 1.01 prices) can poison the model via calibration.
        market_books = int(benchmark.get("books_tracked", 0) or 0)
        if market_books >= 3:
            calibrated_mod_prob = calibrate_probability(raw_mod_prob, imp_prob)
        else:
            # Low confidence in market benchmark; stick closer to raw model
            calibrated_mod_prob = raw_mod_prob
        
        # Calculate edge using calibrated probability
        edge = calibrated_mod_prob - imp_prob
        ev = expected_value(calibrated_mod_prob, dec_odds)
        full_kelly = kelly_fraction(calibrated_mod_prob, dec_odds, fraction=1.0)
        fractional = kelly_fraction(calibrated_mod_prob, dec_odds, fraction=fractional_kelly)

        # Extract individual model probabilities for agreement score
        outcome_model_probs = []
        if individual_probs:
            for p in individual_probs:
                if outcome in p:
                    outcome_model_probs.append(p[outcome])

        conf_score = compute_confidence_score(
            outcome_model_probs, calibrated_mod_prob, edge, imp_prob, expected_value=ev
        )

        edges[outcome] = {
            "model_prob": round(calibrated_mod_prob, 4),
            "raw_model_prob": round(raw_mod_prob, 4),
            "implied_prob": round(imp_prob, 4),
            "market_implied_prob": round(raw_imp_prob, 4),
            "edge": round(edge, 4),
            "expected_value": round(ev, 4),
            "decimal_odds": dec_odds,
            "american_odds": decimal_to_american(dec_odds),
            "kelly_fraction": round(full_kelly, 4),
            "fractional_kelly": round(fractional, 4),
            "confidence_score": conf_score,
            "is_value": edge >= 0.05 and conf_score >= 65,
            "unrealistic_flag": abs(raw_mod_prob - imp_prob) > MAX_ALLOWED_DIVERGENCE,
            "hold": round(float(benchmark_hold if benchmark_hold is not None else hold), 4),
            "market_source": benchmark.get("source", "execution_line_no_vig"),
            "market_books": int(benchmark.get("books_tracked", 0) or 0),
        }

    return edges


def compute_totals_edges(
    model_probs: dict[str, float],
    odds: dict[str, float],
    individual_probs: list[dict[str,Optional[ float]] ] = None,
    fractional_kelly: float = 0.25,
) -> dict[str, dict]:
    """Compute over/under edges for totals markets."""
    active_odds = {}
    for outcome in ("over", "under"):
        odds_key = f"{outcome}_odds"
        dec_odds = odds.get(odds_key, 0)
        if dec_odds > 0 and outcome in model_probs:
            active_odds[outcome] = dec_odds

    raw_probs, fair_probs, hold = no_vig_probabilities(active_odds)
    benchmark = odds.get("totals_benchmark") or {}
    benchmark_fair = benchmark.get("fair_probs") or {}
    benchmark_raw = benchmark.get("raw_probs") or {}
    benchmark_hold = benchmark.get("hold")
    edges: dict[str, dict] = {}

    # If no odds are active, return pure model projections as a baseline
    if not active_odds:
        for outcome, prob in model_probs.items():
            outcome_model_probs = []
            if individual_probs:
                for p in individual_probs:
                    if outcome in p:
                        outcome_model_probs.append(p[outcome])
            
            conf_score = compute_confidence_score(
                outcome_model_probs, prob, 0.0, prob, expected_value=0.0
            )
            
            edges[outcome] = {
                "model_prob": round(prob, 4),
                "raw_model_prob": round(prob, 4),
                "implied_prob": None,
                "market_implied_prob": None,
                "edge": 0.0,
                "expected_value": 0.0,
                "decimal_odds": None,
                "american_odds": None,
                "kelly_fraction": 0.0,
                "fractional_kelly": 0.0,
                "confidence_score": conf_score,
                "is_value": False,
                "unrealistic_flag": False,
                "hold": None,
                "market_source": "none",
                "market_books": 0,
            }
        return edges

    for outcome, dec_odds in active_odds.items():
        imp_prob = float(benchmark_fair.get(outcome, fair_probs[outcome]))
        raw_imp_prob = float(benchmark_raw.get(outcome, raw_probs[outcome]))
        raw_mod_prob = model_probs[outcome]
        
        # Phase 4: Calibrate probability (shrink toward market)
        # MANDATE: Only trust the market benchmark if we have a consensus (at least 3 books).
        # Single-book outliers (like 1.01 prices) can poison the model via calibration.
        market_books = int(benchmark.get("books_tracked", 0) or 0)
        if market_books >= 3:
            calibrated_mod_prob = calibrate_probability(raw_mod_prob, imp_prob)
        else:
            # Low confidence in market benchmark; stick closer to raw model
            calibrated_mod_prob = raw_mod_prob
            
        edge = calibrated_mod_prob - imp_prob
        ev = expected_value(calibrated_mod_prob, dec_odds)
        full_kelly = kelly_fraction(calibrated_mod_prob, dec_odds, fraction=1.0)
        fractional = kelly_fraction(calibrated_mod_prob, dec_odds, fraction=fractional_kelly)

        outcome_model_probs = []
        if individual_probs:
            for p in individual_probs:
                if outcome in p:
                    outcome_model_probs.append(p[outcome])

        conf_score = compute_confidence_score(
            outcome_model_probs, calibrated_mod_prob, edge, imp_prob, expected_value=ev
        )
        edges[outcome] = {
            "model_prob": round(calibrated_mod_prob, 4),
            "raw_model_prob": round(raw_mod_prob, 4),
            "implied_prob": round(imp_prob, 4),
            "market_implied_prob": round(raw_imp_prob, 4),
            "edge": round(edge, 4),
            "expected_value": round(ev, 4),
            "decimal_odds": dec_odds,
            "american_odds": decimal_to_american(dec_odds),
            "kelly_fraction": round(full_kelly, 4),
            "fractional_kelly": round(fractional, 4),
            "confidence_score": conf_score,
            "is_value": edge >= VALUE_EDGE_THRESHOLD and conf_score >= 65,
            "unrealistic_flag": abs(raw_mod_prob - imp_prob) > MAX_ALLOWED_DIVERGENCE,
            "hold": round(float(benchmark_hold if benchmark_hold is not None else hold), 4),
            "market_source": benchmark.get("source", "execution_line_no_vig"),
            "market_books": int(benchmark.get("books_tracked", 0) or 0),
        }

    return edges


def compute_confidence_stars(model_prob: float, edge: float) -> int:
    """Legacy helper (keeping for compatibility, but moving to score)."""
    stars = 1
    if edge >= 0.10: stars += 2
    elif edge >= 0.05: stars += 1
    if model_prob >= 0.75: stars += 2
    elif model_prob >= 0.60: stars += 1
    return min(5, stars)
