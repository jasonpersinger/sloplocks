"""Tests for the Dixon-Coles and Elo models."""

import math
import numpy as np
import pytest

from pipeline.config import MAX_GOALS
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

    def test_two_way_prediction_no_draw(self, teams):
        """2-way mode (NBA) should return only home/away, summing to 1."""
        elo = EloRatings(teams)
        probs = elo_predict(elo, "Arsenal", "Wolves", outcomes=["home", "away"])
        assert set(probs.keys()) == {"home", "away"}
        assert "draw" not in probs
        assert math.isclose(probs["home"] + probs["away"], 1.0, abs_tol=1e-9)

    def test_custom_k_factor_and_home_advantage(self, teams):
        """EloRatings should respect custom k_factor and home_advantage."""
        elo = EloRatings(teams, k_factor=25, home_advantage=100)
        assert elo.k_factor == 25
        assert elo.home_advantage == 100

        # Update should use the custom k_factor — winner gains more with k=25
        r_before = elo.get_rating("Arsenal")
        elo.update("Arsenal", "Wolves", 3, 0)
        gain_custom = elo.get_rating("Arsenal") - r_before

        elo2 = EloRatings(teams, k_factor=10, home_advantage=100)
        r_before2 = elo2.get_rating("Arsenal")
        elo2.update("Arsenal", "Wolves", 3, 0)
        gain_small_k = elo2.get_rating("Arsenal") - r_before2

        assert gain_custom > gain_small_k

    def test_two_way_uses_elo_home_advantage(self, teams):
        """In 2-way mode, home team with equal ratings should have >50%."""
        elo = EloRatings(teams, home_advantage=100)
        probs = elo_predict(elo, "Arsenal", "Wolves", outcomes=["home", "away"])
        assert probs["home"] > 0.5

    def test_rest_adjustment_reduces_home_win_prob(self):
        """B2B penalty on home team should lower their win probability."""
        teams = ["A", "B"]
        elo = EloRatings(teams, k_factor=20, home_advantage=65)

        prob_fresh = elo_predict(elo, "A", "B", outcomes=["home", "away"])
        prob_b2b   = elo_predict(elo, "A", "B", outcomes=["home", "away"],
                                 home_rest_adj=-30.0)

        assert prob_b2b["home"] < prob_fresh["home"]

    def test_rest_adjustment_zero_is_unchanged(self):
        elo = EloRatings(["A", "B"], k_factor=20, home_advantage=65)
        prob_default = elo_predict(elo, "A", "B", outcomes=["home", "away"])
        prob_zero    = elo_predict(elo, "A", "B", outcomes=["home", "away"],
                                   home_rest_adj=0.0, away_rest_adj=0.0)
        assert prob_default == prob_zero


# ---------------------------------------------------------------------------
# Adjusted Efficiency (KenPom-style)
# ---------------------------------------------------------------------------

NCAAM_TEAMS = ["Duke", "North Carolina", "Kansas", "Kentucky", "Gonzaga", "UCLA"]


class TestAdjustedEfficiency:
    """Tests for the KenPom-style adjusted efficiency model."""

    def test_all_teams_have_ratings(self, ncaam_box_scores, ncaam_games):
        model = AdjustedEfficiency(ncaam_box_scores, ncaam_games)
        for team in NCAAM_TEAMS:
            assert team in model.off_efficiency, f"{team} missing from off_efficiency"
            assert team in model.def_efficiency, f"{team} missing from def_efficiency"
            assert team in model.tempo, f"{team} missing from tempo"

    def test_efficiencies_are_positive(self, ncaam_box_scores, ncaam_games):
        model = AdjustedEfficiency(ncaam_box_scores, ncaam_games)
        for team in NCAAM_TEAMS:
            assert model.off_efficiency[team] > 0
            assert model.def_efficiency[team] > 0
            assert model.tempo[team] > 0

    def test_predict_returns_two_way_probs(self, ncaam_box_scores, ncaam_games):
        model = AdjustedEfficiency(ncaam_box_scores, ncaam_games)
        probs = efficiency_predict(model, "Duke", "Kansas")
        assert set(probs.keys()) == {"home", "away"}
        assert math.isclose(probs["home"] + probs["away"], 1.0, abs_tol=1e-9)

    def test_home_team_gets_bonus(self, ncaam_box_scores, ncaam_games):
        """With equal forced ratings, home team should have >50% win prob."""
        model = AdjustedEfficiency(ncaam_box_scores, ncaam_games)
        # Force equal ratings
        for team in NCAAM_TEAMS:
            model.off_efficiency[team] = 100.0
            model.def_efficiency[team] = 100.0
            model.tempo[team] = 68.0
        probs = efficiency_predict(model, "Duke", "Kansas", home_bonus=3.5)
        assert probs["home"] > 0.5


# ---------------------------------------------------------------------------
# Four Factors Logistic Regression
# ---------------------------------------------------------------------------

class TestFourFactorsModel:
    """Tests for the Four Factors logistic regression model."""

    def test_all_teams_have_stats(self, ncaam_box_scores, ncaam_games):
        model = FourFactorsModel(ncaam_box_scores, ncaam_games)
        for team in NCAAM_TEAMS:
            assert team in model.team_stats, f"{team} missing from team_stats"

    def test_team_stats_have_expected_keys(self, ncaam_box_scores, ncaam_games):
        model = FourFactorsModel(ncaam_box_scores, ncaam_games)
        expected_keys = {
            "off_efg", "off_to_rate", "off_orb_pct", "off_ft_rate",
            "def_efg", "def_to_rate", "def_orb_pct", "def_ft_rate",
        }
        for team in NCAAM_TEAMS:
            assert set(model.team_stats[team].keys()) == expected_keys

    def test_predict_returns_two_way_probs(self, ncaam_box_scores, ncaam_games):
        model = FourFactorsModel(ncaam_box_scores, ncaam_games)
        probs = four_factors_predict(model, "Duke", "Kansas")
        assert set(probs.keys()) == {"home", "away"}
        assert math.isclose(probs["home"] + probs["away"], 1.0, abs_tol=1e-9)

    def test_probabilities_are_reasonable(self, ncaam_box_scores, ncaam_games):
        model = FourFactorsModel(ncaam_box_scores, ncaam_games)
        probs = four_factors_predict(model, "Duke", "Kansas")
        assert 0.1 <= probs["home"] <= 0.9
        assert 0.1 <= probs["away"] <= 0.9
