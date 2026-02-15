"""Tests for pipeline.ensemble — odds conversion, blending, and edge detection."""

import pytest
from pipeline.ensemble import (
    decimal_to_american,
    implied_probability,
    blend_predictions,
    compute_edges,
)


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


# ── Edge detection ─────────────────────────────────────────────────


class TestComputeEdges:
    """compute_edges value-edge identification."""

    def test_positive_edge_flagged_as_value(self):
        model_probs = {"home": 0.70, "draw": 0.15, "away": 0.15}
        odds = {"home_odds": 1.67, "draw_odds": 3.80, "away_odds": 4.50}
        # implied home prob = 1/1.67 ≈ 0.599, edge ≈ 0.101 -> is_value True
        edges = compute_edges(model_probs, odds)

        assert edges["home"]["edge"] == pytest.approx(0.101, abs=1e-2)
        assert edges["home"]["is_value"] is True
        assert edges["home"]["american_odds"] == -149

    def test_negative_edge_not_flagged(self):
        model_probs = {"home": 0.55, "draw": 0.25, "away": 0.20}
        odds = {"home_odds": 1.67, "draw_odds": 3.80, "away_odds": 4.50}
        # implied away prob = 1/4.50 ≈ 0.222, model away = 0.20
        # edge ≈ -0.022 -> is_value False
        edges = compute_edges(model_probs, odds)

        assert edges["away"]["edge"] < 0
        assert edges["away"]["is_value"] is False
