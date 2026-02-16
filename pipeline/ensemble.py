"""Ensemble blending and edge detection for SLOP LOCKS pipeline."""

from pipeline.config import VALUE_EDGE_THRESHOLD


def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to American format.

    Decimal >= 2.0 -> positive American: (decimal - 1) * 100
    Decimal < 2.0  -> negative American: -100 / (decimal - 1)
    Decimal == 2.0 -> +100
    """
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
    """Weighted-average blend of multiple prediction dicts.

    Keys are inferred from the first prediction dict, so this works
    for both 3-way (home/draw/away) and 2-way (home/away) outcomes.
    Weights are normalised to sum to 1.
    """
    total_weight = sum(weights)
    norm_weights = [w / total_weight for w in weights]

    keys = predictions[0].keys()
    blended = {k: 0.0 for k in keys}
    for pred, w in zip(predictions, norm_weights):
        for key in blended:
            blended[key] += pred[key] * w

    return blended


def compute_edges(
    model_probs: dict[str, float],
    odds: dict[str, float],
) -> dict[str, dict]:
    """Compute value edges for each outcome.

    Parameters
    ----------
    model_probs : dict with keys home, draw, away (probabilities).
    odds : dict with keys home_odds, draw_odds, away_odds (decimal).

    Returns
    -------
    Dict keyed by outcome (home, draw, away), each containing:
        model_prob, implied_prob, edge, decimal_odds, american_odds, is_value.
    """
    outcome_map = {
        "home": "home_odds",
        "draw": "draw_odds",
        "away": "away_odds",
    }

    edges: dict[str, dict] = {}
    for outcome, odds_key in outcome_map.items():
        if outcome not in model_probs:
            continue
        dec_odds = odds.get(odds_key, 0)
        if dec_odds <= 0:
            continue
        imp_prob = implied_probability(dec_odds)
        mod_prob = model_probs[outcome]
        edge = mod_prob - imp_prob

        edges[outcome] = {
            "model_prob": mod_prob,
            "implied_prob": imp_prob,
            "edge": edge,
            "decimal_odds": dec_odds,
            "american_odds": decimal_to_american(dec_odds),
            "is_value": edge >= VALUE_EDGE_THRESHOLD,
        }

    return edges
