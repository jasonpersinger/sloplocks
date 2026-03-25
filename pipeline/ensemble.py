"""Ensemble blending and edge detection for SLOP LOCKS pipeline."""

from pipeline.config import VALUE_EDGE_THRESHOLD


import numpy as np

def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to American format."""
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1) * 100)
    else:
        return round(-100 / (decimal_odds - 1))


def implied_probability(decimal_odds: float) -> float:
    """Return the implied probability from decimal odds: 1 / decimal_odds."""
    return 1.0 / decimal_odds


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


def compute_confidence_score(
    model_probs_list: list[float],
    blended_prob: float,
    edge: float,
    implied_prob: float
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
    # std_dev of 0.15 is considered "high disagreement"
    std_dev = np.std(model_probs_list)
    agreement = max(0, 1.0 - (std_dev / 0.15))

    # 2. Probability Score (0-1.0)
    # Scaled such that 50% is 0 and 100% is 1.0
    prob_score = max(0, (blended_prob - 0.4) / 0.6)

    # 3. Edge Score (0-1.0)
    # 15% edge is a "perfect" score
    edge_score = max(0, min(1.0, edge / 0.15))

    score = (agreement * 30) + (prob_score * 40) + (edge_score * 30)

    # 4. Market Divergence Penalty
    # If the model is too far from the market, it's likely missing info (injuries, etc.)
    divergence = abs(blended_prob - implied_prob)
    if divergence > 0.18:
        score *= 0.4  # Severe penalty (Red Flag)
    elif divergence > 0.12:
        score *= 0.7  # Moderate penalty

    # 5. Underdog Penalty for "Locks"
    # If it's a lock, it should probably be expected to win
    if blended_prob < 0.45:
        score *= 0.5

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
    w_market = MARKET_RESPECT_FACTOR
    w_model = 1.0 - w_market
    
    return (w_model * model_prob) + (w_market * implied_prob)


def compute_edges(
    model_probs: dict[str, float],
    odds: dict[str, float],
    individual_probs: list[dict[str, float]] = None
) -> dict[str, dict]:
    """Compute calibrated value edges and confidence scores."""
    outcome_map = {
        "home": "home_odds",
        "draw": "draw_odds",
        "away_odds": "away_odds",
    }

    edges: dict[str, dict] = {}
    for outcome, odds_key in outcome_map.items():
        if outcome not in model_probs:
            continue
        dec_odds = odds.get(odds_key, 0)
        if dec_odds <= 0:
            continue
        
        imp_prob = implied_probability(dec_odds)
        raw_mod_prob = model_probs[outcome]
        
        # Phase 4: Calibrate probability (shrink toward market)
        calibrated_mod_prob = calibrate_probability(raw_mod_prob, imp_prob)
        
        # Calculate edge using calibrated probability
        edge = calibrated_mod_prob - imp_prob

        # Extract individual model probabilities for agreement score
        outcome_model_probs = []
        if individual_probs:
            for p in individual_probs:
                if outcome in p:
                    outcome_model_probs.append(p[outcome])

        conf_score = compute_confidence_score(
            outcome_model_probs, calibrated_mod_prob, edge, imp_prob
        )

        edges[outcome] = {
            "model_prob": round(calibrated_mod_prob, 4),
            "raw_model_prob": round(raw_mod_prob, 4),
            "implied_prob": round(imp_prob, 4),
            "edge": round(edge, 4),
            "decimal_odds": dec_odds,
            "american_odds": decimal_to_american(dec_odds),
            "confidence_score": conf_score,
            "is_value": edge >= 0.05 and conf_score >= 65,
            "unrealistic_flag": abs(raw_mod_prob - imp_prob) > MAX_ALLOWED_DIVERGENCE,
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
