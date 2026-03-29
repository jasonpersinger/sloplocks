"""Tests for pipeline.ensemble — odds conversion, blending, and edge detection."""

import pytest
from pipeline.ensemble import (
    apply_probability_calibration,
    compute_totals_edges,
    decimal_to_american,
    implied_probability,
    blend_predictions,
    compute_edges,
    expected_value,
    fit_probability_calibrators,
    kelly_fraction,
)
from pipeline.backtest import compute_model_weights


# ── Odds conversion ────────────────────────────────────────────────


class TestOddsConversion:
    """decimal_to_american and implied_probability."""

    def test_favorite_decimal_to_american(self):
        # 1.67 -> -100 / (1.67 - 1) = -100 / 0.67 ≈ -149
        american = decimal_to_american(1.67)
        assert american == -149

    def test_underdog_decimal_to_american(self):
        # 3.0 -> (3.0 - 1) * 100 = +200
        assert decimal_to_american(3.0) == 200

    def test_even_money_decimal_to_american(self):
        # 2.0 -> exactly +100
        assert decimal_to_american(2.0) == 100

    def test_invalid_decimal_placeholder_returns_none(self):
        assert decimal_to_american(1.0) is None

    def test_implied_probability_even(self):
        assert implied_probability(2.0) == pytest.approx(0.50)

    def test_implied_probability_favorite(self):
        assert implied_probability(1.5) == pytest.approx(0.6667, abs=1e-3)


# ── Blend predictions ──────────────────────────────────────────────


class TestBlendPredictions:
    """blend_predictions weighted averaging."""

    def test_equal_weights(self):
        p1 = {"home": 0.50, "draw": 0.30, "away": 0.20}
        p2 = {"home": 0.40, "draw": 0.30, "away": 0.30}
        blended = blend_predictions([p1, p2], [1.0, 1.0])

        assert blended["home"] == pytest.approx(0.45)
        assert blended["draw"] == pytest.approx(0.30)
        assert blended["away"] == pytest.approx(0.25)

    def test_weighted_blend(self):
        p1 = {"home": 0.60, "draw": 0.20, "away": 0.20}
        p2 = {"home": 0.40, "draw": 0.30, "away": 0.30}
        # weights 3:1 -> 75% p1, 25% p2
        blended = blend_predictions([p1, p2], [3.0, 1.0])

        assert blended["home"] == pytest.approx(0.55)
        assert blended["draw"] == pytest.approx(0.225)
        assert blended["away"] == pytest.approx(0.225)

    def test_blended_probs_sum_to_one(self):
        p1 = {"home": 0.50, "draw": 0.30, "away": 0.20}
        p2 = {"home": 0.40, "draw": 0.35, "away": 0.25}
        p3 = {"home": 0.45, "draw": 0.25, "away": 0.30}
        blended = blend_predictions([p1, p2, p3], [2.0, 1.0, 1.0])

        total = blended["home"] + blended["draw"] + blended["away"]
        assert total == pytest.approx(1.0)

    def test_two_way_blend(self):
        """2-way (NBA) predictions should blend without a draw key."""
        p1 = {"home": 0.60, "away": 0.40}
        p2 = {"home": 0.50, "away": 0.50}
        blended = blend_predictions([p1, p2], [1.0, 1.0])

        assert "draw" not in blended
        assert set(blended.keys()) == {"home", "away"}
        assert blended["home"] == pytest.approx(0.55)
        assert blended["away"] == pytest.approx(0.45)
        assert blended["home"] + blended["away"] == pytest.approx(1.0)


# ── Edge detection ─────────────────────────────────────────────────


class TestComputeEdges:
    """compute_edges value-edge identification."""

    def test_positive_edge_flagged_as_value(self):
        model_probs = {"home": 0.70, "draw": 0.15, "away": 0.15}
        odds = {"home_odds": 1.67, "draw_odds": 3.80, "away_odds": 4.50}
        # The current code removes vig first, then calibrates toward the
        # fair market probability before computing edge.
        edges = compute_edges(model_probs, odds)

        assert edges["home"]["raw_model_prob"] == pytest.approx(0.70)
        assert edges["home"]["market_implied_prob"] == pytest.approx(0.5988, abs=1e-4)
        assert edges["home"]["implied_prob"] == pytest.approx(0.5523, abs=1e-4)
        assert edges["home"]["hold"] == pytest.approx(0.0842, abs=1e-4)
        assert edges["home"]["model_prob"] == pytest.approx(0.6409, abs=1e-4)
        assert edges["home"]["edge"] == pytest.approx(0.0886, abs=1e-2)
        assert edges["home"]["expected_value"] == pytest.approx(0.0704, abs=1e-3)
        assert edges["home"]["kelly_fraction"] > 0
        assert edges["home"]["fractional_kelly"] == pytest.approx(
            edges["home"]["kelly_fraction"] * 0.25, abs=1e-4
        )
        assert edges["home"]["is_value"] is False
        assert edges["home"]["american_odds"] == -149

    def test_negative_edge_not_flagged(self):
        model_probs = {"home": 0.55, "draw": 0.25, "away": 0.20}
        odds = {"home_odds": 1.67, "draw_odds": 3.80, "away_odds": 4.50}
        # implied away prob = 1/4.50 ≈ 0.222, model away = 0.20
        # edge ≈ -0.022 -> is_value False
        edges = compute_edges(model_probs, odds)

        assert edges["away"]["edge"] < 0
        assert edges["away"]["is_value"] is False

    def test_two_way_edges_skip_draw(self):
        """2-way model probs (NBA) should produce edges for home/away only."""
        model_probs = {"home": 0.60, "away": 0.40}
        odds = {"home_odds": 2.10, "draw_odds": 0.0, "away_odds": 1.75}
        edges = compute_edges(model_probs, odds)

        assert "draw" not in edges
        assert "home" in edges
        assert "away" in edges
        # For 2-way markets, the implied probabilities are normalized to no-vig
        # fair probabilities before calibration and edge computation.
        assert edges["home"]["raw_model_prob"] == pytest.approx(0.60)
        assert edges["home"]["market_implied_prob"] == pytest.approx(0.4762, abs=1e-4)
        assert edges["home"]["implied_prob"] == pytest.approx(0.4545, abs=1e-4)
        assert edges["home"]["hold"] == pytest.approx(0.0476, abs=1e-4)
        assert edges["home"]["edge"] == pytest.approx(0.0873, abs=1e-2)


class TestComputeTotalsEdges:
    def test_totals_edges_calculate_over_under_value(self):
        edges = compute_totals_edges(
            {"over": 0.58, "under": 0.42},
            {"over_odds": 1.95, "under_odds": 1.91},
        )

        assert "over" in edges
        assert "under" in edges
        assert edges["over"]["market_implied_prob"] == pytest.approx(1 / 1.95, abs=1e-4)
        assert edges["over"]["expected_value"] > 0
        assert edges["over"]["american_odds"] == decimal_to_american(1.95)


class TestExpectedValue:
    def test_expected_value_handles_plus_money(self):
        assert expected_value(0.55, 2.10) == pytest.approx(0.155, abs=1e-6)

    def test_expected_value_handles_negative_case(self):
        assert expected_value(0.40, 1.75) < 0


class TestKellyFraction:
    def test_kelly_fraction_scales_by_fraction(self):
        full = kelly_fraction(0.55, 2.10, fraction=1.0)
        quarter = kelly_fraction(0.55, 2.10, fraction=0.25)
        assert full > 0
        assert quarter == pytest.approx(full * 0.25, abs=1e-6)

    def test_kelly_fraction_is_clamped_at_zero(self):
        assert kelly_fraction(0.40, 1.75, fraction=1.0) == 0.0


class TestComputeModelWeights:
    def test_higher_temperature_sharpens_weighting(self):
        cool = compute_model_weights([0.60, 0.55], temperature=1.0)
        hot = compute_model_weights([0.60, 0.55], temperature=4.0)
        assert hot[0] > cool[0]
        assert sum(hot) == pytest.approx(1.0)


class TestProbabilityCalibration:
    def test_fit_probability_calibrators_returns_models_with_history(self):
        historical = [
            {"evaluated": True, "home_goals": 1, "away_goals": 0, "model_probs": {"home": 0.20, "away": 0.80}},
            {"evaluated": True, "home_goals": 1, "away_goals": 0, "model_probs": {"home": 0.30, "away": 0.70}},
            {"evaluated": True, "home_goals": 1, "away_goals": 0, "model_probs": {"home": 0.40, "away": 0.60}},
            {"evaluated": True, "home_goals": 0, "away_goals": 1, "model_probs": {"home": 0.80, "away": 0.20}},
            {"evaluated": True, "home_goals": 0, "away_goals": 1, "model_probs": {"home": 0.90, "away": 0.10}},
            {"evaluated": True, "home_goals": 0, "away_goals": 1, "model_probs": {"home": 0.95, "away": 0.05}},
        ]

        calibrators = fit_probability_calibrators(historical, ["home", "away"], min_samples=4)

        assert set(calibrators.keys()) == {"home", "away"}

    def test_apply_probability_calibration_shrinks_bad_probabilities(self):
        historical = [
            {"evaluated": True, "home_goals": 1, "away_goals": 0, "model_probs": {"home": 0.20, "away": 0.80}},
            {"evaluated": True, "home_goals": 1, "away_goals": 0, "model_probs": {"home": 0.30, "away": 0.70}},
            {"evaluated": True, "home_goals": 1, "away_goals": 0, "model_probs": {"home": 0.40, "away": 0.60}},
            {"evaluated": True, "home_goals": 0, "away_goals": 1, "model_probs": {"home": 0.80, "away": 0.20}},
            {"evaluated": True, "home_goals": 0, "away_goals": 1, "model_probs": {"home": 0.90, "away": 0.10}},
            {"evaluated": True, "home_goals": 0, "away_goals": 1, "model_probs": {"home": 0.95, "away": 0.05}},
        ]

        calibrators = fit_probability_calibrators(historical, ["home", "away"], min_samples=4)
        calibrated = apply_probability_calibration({"home": 0.90, "away": 0.10}, calibrators, blend=1.0)

        assert calibrated["home"] < 0.90
        assert calibrated["away"] > 0.10
        assert sum(calibrated.values()) == pytest.approx(1.0)
