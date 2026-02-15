"""Tests for the Dixon-Coles and Elo models."""

import math
import numpy as np
import pytest

from pipeline.config import MAX_GOALS
from pipeline.models import (
    EloRatings,
    dixon_coles_predict,
    elo_predict,
    fit_dixon_coles,
    scoreline_to_probabilities,
)


# ---------------------------------------------------------------------------
# scoreline_to_probabilities
# ---------------------------------------------------------------------------

class TestScorelineToProbabilities:
    """Tests for collapsing a scoreline matrix to 1x3 probabilities."""

    def test_probabilities_sum_to_one(self):
        """Home + draw + away should sum to 1."""
        # Uniform (ish) matrix
        matrix = np.ones((7, 7)) / 49.0
        probs = scoreline_to_probabilities(matrix)
        assert math.isclose(
            probs["home"] + probs["draw"] + probs["away"], 1.0, abs_tol=1e-9
        )

    def test_returns_three_keys(self):
        matrix = np.ones((7, 7)) / 49.0
        probs = scoreline_to_probabilities(matrix)
        assert set(probs.keys()) == {"home", "draw", "away"}

    def test_all_positive(self):
        matrix = np.ones((7, 7)) / 49.0
        probs = scoreline_to_probabilities(matrix)
        for key in ("home", "draw", "away"):
            assert probs[key] > 0


# ---------------------------------------------------------------------------
# Dixon-Coles end-to-end
# ---------------------------------------------------------------------------

class TestDixonColes:
    """Integration tests for the Dixon-Coles model."""

    def test_fit_returns_params_for_all_teams(self, sample_matches):
        params = fit_dixon_coles(sample_matches)
        teams_in_data = sorted(
            set(sample_matches["home_team"]) | set(sample_matches["away_team"])
        )
        for team in teams_in_data:
            assert team in params["attack"], f"{team} missing from attack"
            assert team in params["defense"], f"{team} missing from defense"

    def test_strong_team_has_higher_attack(self, sample_matches):
        """Liverpool scored 5 goals in 2 matches — should rank high."""
        params = fit_dixon_coles(sample_matches)
        teams_in_data = sorted(
            set(sample_matches["home_team"]) | set(sample_matches["away_team"])
        )
        # Liverpool's attack should be among the highest
        liverpool_attack = params["attack"]["Liverpool"]
        median_attack = np.median(
            [params["attack"][t] for t in teams_in_data]
        )
        assert liverpool_attack > median_attack

    def test_predict_returns_correct_shape(self, sample_matches):
        params = fit_dixon_coles(sample_matches)
        matrix = dixon_coles_predict("Arsenal", "Wolves", params)
        assert matrix.shape == (MAX_GOALS + 1, MAX_GOALS + 1)

    def test_predict_sums_to_one(self, sample_matches):
        params = fit_dixon_coles(sample_matches)
        matrix = dixon_coles_predict("Arsenal", "Wolves", params)
        assert math.isclose(matrix.sum(), 1.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Elo rating system
# ---------------------------------------------------------------------------

class TestElo:
    """Tests for the Elo rating class and prediction function."""

    def test_initial_ratings_are_1500(self, teams):
        elo = EloRatings(teams)
        for team in teams:
            assert elo.get_rating(team) == 1500.0

    def test_winner_gains_loser_loses(self, teams):
        elo = EloRatings(teams)
        r_before_home = elo.get_rating("Arsenal")
        r_before_away = elo.get_rating("Wolves")
        elo.update("Arsenal", "Wolves", 3, 0)
        assert elo.get_rating("Arsenal") > r_before_home
        assert elo.get_rating("Wolves") < r_before_away

    def test_draw_adjusts_toward_parity(self, teams):
        """When ratings differ, a draw should push them closer together."""
        elo = EloRatings(teams)
        # Artificially set different ratings
        elo.ratings["Arsenal"] = 1600.0
        elo.ratings["Wolves"] = 1400.0
        elo.update("Arsenal", "Wolves", 1, 1)
        # The higher-rated team should drop, the lower should rise
        assert elo.get_rating("Arsenal") < 1600.0
        assert elo.get_rating("Wolves") > 1400.0

    def test_predict_returns_three_probs_summing_to_one(self, teams):
        elo = EloRatings(teams)
        probs = elo_predict(elo, "Arsenal", "Wolves")
        assert set(probs.keys()) == {"home", "draw", "away"}
        assert math.isclose(
            probs["home"] + probs["draw"] + probs["away"], 1.0, abs_tol=1e-9
        )
