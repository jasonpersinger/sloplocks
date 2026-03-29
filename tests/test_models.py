"""Tests for the Dixon-Coles and Elo models."""

import math
import numpy as np
import pandas as pd
import pytest

from pipeline.config import MAX_GOALS
from pipeline.models import (
    AdjustedEfficiency,
    BullpenMatchupModel,
    HandednessMatchupModel,
    EloRatings,
    FourFactorsModel,
    MlbTotalsModel,
    NbaMatchupModel,
    NbaTotalsModel,
    NhlMatchupModel,
    PitcherMatchupModel,
    RecentBoxScoreModel,
    ResultsFeatureModel,
    RunEnvironmentModel,
    bullpen_matchup_predict,
    dixon_coles_predict,
    efficiency_predict,
    elo_predict,
    fit_dixon_coles,
    four_factors_predict,
    handedness_matchup_predict,
    mlb_totals_predict,
    nba_matchup_predict,
    nba_totals_predict,
    nhl_matchup_predict,
    pitcher_matchup_predict,
    recent_boxscore_predict,
    run_environment_predict,
    results_features_predict,
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


# ---------------------------------------------------------------------------
# Results Feature Logistic Regression
# ---------------------------------------------------------------------------

class TestResultsFeatureModel:
    def _sample_results_games(self):
        rows = []
        base = pd.Timestamp("2026-01-01")
        games = [
            ("A", "B", 110, 100),
            ("C", "D", 102, 99),
            ("A", "C", 108, 101),
            ("B", "D", 95, 104),
            ("A", "D", 112, 103),
            ("B", "C", 99, 101),
            ("C", "A", 97, 105),
            ("D", "B", 101, 94),
            ("A", "B", 111, 98),
            ("C", "D", 103, 96),
        ]
        for idx, (home, away, hg, ag) in enumerate(games):
            rows.append({
                "date": (base + pd.Timedelta(days=idx)).strftime("%Y-%m-%d"),
                "home_team": home,
                "away_team": away,
                "home_goals": hg,
                "away_goals": ag,
            })
        return pd.DataFrame(rows)

    def test_fit_produces_model_when_enough_games_exist(self):
        model = ResultsFeatureModel(self._sample_results_games(), feature_window=4, min_games=5)
        assert model.model is not None

    def test_predict_returns_two_way_probs(self):
        model = ResultsFeatureModel(self._sample_results_games(), feature_window=4, min_games=5)
        probs = results_features_predict(
            model,
            "A",
            "B",
            neutral_site=False,
            game_date="2026-01-20",
        )
        assert set(probs.keys()) == {"home", "away"}
        assert math.isclose(probs["home"] + probs["away"], 1.0, abs_tol=1e-9)

    def test_unknown_or_untrained_model_falls_back_to_coin_flip(self):
        model = ResultsFeatureModel(
            pd.DataFrame(
                columns=["date", "home_team", "away_team", "home_goals", "away_goals"]
            ),
            feature_window=4,
            min_games=5,
        )
        probs = results_features_predict(model, "A", "B", game_date="2026-03-28")
        assert probs == {"home": 0.5, "away": 0.5}


class TestPitcherMatchupModel:
    def _sample_pitcher_games(self):
        rows = []
        base = pd.Timestamp("2026-03-01")
        games = [
            ("A", "B", 5, 2, "Ace A", "Scrub B", 6.0, 2, 2, 1, 8, 5.0, 5, 5, 3, 3),
            ("C", "D", 4, 1, "Ace C", "Scrub D", 6.0, 1, 1, 2, 7, 5.0, 4, 4, 2, 4),
            ("A", "C", 3, 2, "Ace A", "Scrub D", 6.0, 2, 2, 1, 7, 4.0, 3, 3, 2, 3),
            ("B", "D", 2, 6, "Scrub B", "Ace C", 5.0, 4, 4, 3, 4, 6.0, 2, 2, 1, 8),
            ("A", "D", 7, 1, "Ace A", "Scrub B", 7.0, 1, 1, 1, 9, 4.0, 6, 6, 3, 2),
            ("B", "C", 1, 5, "Scrub D", "Ace C", 4.0, 5, 5, 2, 3, 6.0, 1, 1, 1, 8),
        ]
        for idx, game in enumerate(games):
            (home, away, hg, ag, hp, ap,
             hip, hpr, hper, hpw, hpk,
             aip, apr, aper, apw, apk) = game
            rows.append({
                "date": (base + pd.Timedelta(days=idx)).strftime("%Y-%m-%d"),
                "home_team": home,
                "away_team": away,
                "home_goals": hg,
                "away_goals": ag,
                "home_pitcher": hp,
                "away_pitcher": ap,
                "home_pitcher_hand": "R" if "Ace" in hp else "L",
                "away_pitcher_hand": "R" if "Ace" in ap else "L",
                "home_pitcher_ip": hip,
                "home_pitcher_runs_allowed": hpr,
                "home_pitcher_earned_runs": hper,
                "home_pitcher_walks": hpw,
                "home_pitcher_strikeouts": hpk,
                "home_bullpen_ip": 3.0 if home in {"A", "C"} else 4.0,
                "home_bullpen_earned_runs": 1 if home in {"A", "C"} else 3,
                "home_bullpen_walks": 1 if home in {"A", "C"} else 3,
                "home_bullpen_strikeouts": 4 if home in {"A", "C"} else 2,
                "away_pitcher_ip": aip,
                "away_pitcher_runs_allowed": apr,
                "away_pitcher_earned_runs": aper,
                "away_pitcher_walks": apw,
                "away_pitcher_strikeouts": apk,
                "away_bullpen_ip": 3.0 if away in {"A", "C"} else 4.0,
                "away_bullpen_earned_runs": 1 if away in {"A", "C"} else 3,
                "away_bullpen_walks": 1 if away in {"A", "C"} else 3,
                "away_bullpen_strikeouts": 4 if away in {"A", "C"} else 2,
            })
        return pd.DataFrame(rows)

    def test_fit_produces_model_with_pitcher_history(self):
        model = PitcherMatchupModel(self._sample_pitcher_games(), feature_window=4, min_games=4)
        assert model.model is not None

    def test_predict_prefers_stronger_starter(self):
        model = PitcherMatchupModel(self._sample_pitcher_games(), feature_window=4, min_games=4)
        probs = pitcher_matchup_predict(model, "Ace A", "Scrub B")
        assert probs["home"] > 0.5
        assert math.isclose(probs["home"] + probs["away"], 1.0, abs_tol=1e-9)

    def test_tracks_pitcher_rest_days(self):
        model = PitcherMatchupModel(self._sample_pitcher_games(), feature_window=4, min_games=4)
        features = model._pitcher_features(
            [
                {
                    "date": "2026-03-01",
                    "innings": 6.0,
                    "earned_runs": 2.0,
                    "walks": 1.0,
                    "strikeouts": 8.0,
                    "margin": 3.0,
                }
            ],
            game_date="2026-03-06",
        )

        assert features["days_rest"] == 5.0


class TestBullpenMatchupModel:
    def _sample_bullpen_games(self):
        rows = []
        base = pd.Timestamp("2026-03-01")
        games = [
            ("A", "B", 5, 2, 3.0, 1, 1, 4, 4.0, 3, 3, 2),
            ("C", "D", 4, 1, 3.0, 1, 1, 5, 4.0, 4, 3, 2),
            ("A", "D", 6, 3, 3.0, 0, 1, 4, 4.0, 2, 3, 2),
            ("C", "B", 5, 2, 3.0, 1, 1, 5, 4.0, 3, 4, 2),
            ("A", "C", 4, 3, 3.0, 1, 1, 4, 3.0, 1, 1, 5),
            ("B", "D", 2, 6, 4.0, 4, 3, 2, 3.0, 1, 1, 5),
        ]
        for idx, game in enumerate(games):
            home, away, hg, ag, h_ip, h_er, h_bb, h_k, a_ip, a_er, a_bb, a_k = game
            rows.append({
                "date": (base + pd.Timedelta(days=idx)).strftime("%Y-%m-%d"),
                "home_team": home,
                "away_team": away,
                "home_goals": hg,
                "away_goals": ag,
                "home_bullpen_ip": h_ip,
                "home_bullpen_earned_runs": h_er,
                "home_bullpen_walks": h_bb,
                "home_bullpen_strikeouts": h_k,
                "away_bullpen_ip": a_ip,
                "away_bullpen_earned_runs": a_er,
                "away_bullpen_walks": a_bb,
                "away_bullpen_strikeouts": a_k,
            })
        return pd.DataFrame(rows)

    def test_fit_and_predict_prefers_stronger_bullpen(self):
        model = BullpenMatchupModel(self._sample_bullpen_games(), feature_window=4, recent_usage_window=3, min_games=4)
        probs = bullpen_matchup_predict(model, "A", "B")

        assert model.model is not None
        assert probs["home"] > 0.5
        assert math.isclose(probs["home"] + probs["away"], 1.0, abs_tol=1e-9)


class TestRunEnvironmentModel:
    def _sample_games(self):
        rows = []
        base = pd.Timestamp("2026-03-01")
        games = [
            ("Reds", "Giants", 7, 3),
            ("Red Sox", "Cardinals", 6, 2),
            ("Reds", "Cardinals", 5, 2),
            ("Giants", "Red Sox", 3, 4),
            ("Reds", "Giants", 8, 4),
            ("Cardinals", "Red Sox", 2, 5),
        ]
        for idx, (home, away, hg, ag) in enumerate(games):
            rows.append({
                "date": (base + pd.Timedelta(days=idx)).strftime("%Y-%m-%d"),
                "home_team": home,
                "away_team": away,
                "home_goals": hg,
                "away_goals": ag,
            })
        return pd.DataFrame(rows)

    def test_fit_and_predict_uses_run_environment_context(self):
        model = RunEnvironmentModel(self._sample_games(), feature_window=4, min_games=4)
        probs = run_environment_predict(model, "Reds", "Giants")

        assert model.model is not None
        assert probs["home"] > 0.5
        assert math.isclose(probs["home"] + probs["away"], 1.0, abs_tol=1e-9)


class TestHandednessMatchupModel:
    def _sample_games(self):
        rows = []
        base = pd.Timestamp("2026-03-01")
        games = [
            ("Mashers", "Gloves", 7, 3, "R", "L"),
            ("Mashers", "Gloves", 6, 2, "R", "L"),
            ("Mashers", "Gloves", 3, 4, "L", "R"),
            ("Mashers", "Gloves", 2, 5, "L", "R"),
            ("Mashers", "Gloves", 8, 4, "R", "L"),
            ("Mashers", "Gloves", 3, 6, "L", "R"),
        ]
        for idx, (home, away, hg, ag, hh, ah) in enumerate(games):
            rows.append({
                "date": (base + pd.Timedelta(days=idx)).strftime("%Y-%m-%d"),
                "home_team": home,
                "away_team": away,
                "home_goals": hg,
                "away_goals": ag,
                "home_pitcher_hand": hh,
                "away_pitcher_hand": ah,
            })
        return pd.DataFrame(rows)

    def test_fit_and_predict_uses_pitcher_hand_splits(self):
        model = HandednessMatchupModel(self._sample_games(), feature_window=4, min_games=4)
        probs = handedness_matchup_predict(model, "Mashers", "Gloves", home_pitcher_hand="R", away_pitcher_hand="L")

        assert model.model is not None
        assert probs["home"] > 0.5
        assert math.isclose(probs["home"] + probs["away"], 1.0, abs_tol=1e-9)


class TestMlbTotalsModel:
    def _sample_games(self):
        rows = []
        base = pd.Timestamp("2026-03-01")
        games = [
            ("Reds", "Red Sox", 6, 5, "Ace R", "Ace B", 5.0, 3, 6.0, 2, 4.0, 3, 3.0, 2),
            ("Reds", "Giants", 7, 3, "Ace R", "Ace G", 5.0, 2, 6.0, 2, 4.0, 2, 3.0, 1),
            ("Red Sox", "Cardinals", 5, 4, "Ace B", "Ace C", 6.0, 2, 6.0, 3, 3.0, 1, 3.0, 2),
            ("Reds", "Cardinals", 8, 4, "Ace R", "Ace C", 5.0, 3, 6.0, 3, 4.0, 2, 3.0, 2),
            ("Giants", "Cardinals", 3, 2, "Ace G", "Ace C", 6.0, 1, 6.0, 2, 3.0, 1, 3.0, 1),
            ("Red Sox", "Giants", 6, 3, "Ace B", "Ace G", 6.0, 2, 6.0, 2, 3.0, 1, 3.0, 1),
        ]
        for idx, game in enumerate(games):
            home, away, hg, ag, hp, ap, hip, her, aip, aer, hbip, hber, abip, aber = game
            rows.append({
                "date": (base + pd.Timedelta(days=idx)).strftime("%Y-%m-%d"),
                "home_team": home,
                "away_team": away,
                "home_goals": hg,
                "away_goals": ag,
                "home_pitcher": hp,
                "away_pitcher": ap,
                "home_pitcher_ip": hip,
                "home_pitcher_earned_runs": her,
                "away_pitcher_ip": aip,
                "away_pitcher_earned_runs": aer,
                "home_bullpen_ip": hbip,
                "home_bullpen_earned_runs": hber,
                "away_bullpen_ip": abip,
                "away_bullpen_earned_runs": aber,
            })
        return pd.DataFrame(rows)

    def test_predicts_expected_total_and_market_probabilities(self):
        model = MlbTotalsModel(self._sample_games(), feature_window=4, min_games=4)
        result = mlb_totals_predict(
            model,
            {
                "home_team": "Reds",
                "away_team": "Red Sox",
                "home_pitcher": "Ace R",
                "away_pitcher": "Ace B",
            },
            total_line=9.5,
        )

        assert model.model is not None
        assert result["expected_total"] > 7.0
        assert 0.0 < result["over"] < 1.0
        assert math.isclose(result["over"] + result["under"], 1.0, abs_tol=1e-9)


class TestRecentBoxScoreModel:
    def test_fit_and_predict_from_recent_box_score_form(self):
        games = []
        box_rows = []
        for idx in range(24):
            game_id = f"g{idx}"
            home_team = "A" if idx % 2 == 0 else "B"
            away_team = "B" if idx % 2 == 0 else "A"
            home_win = home_team == "A"

            if home_win:
                home_pts, away_pts = 88, 74
                home_fgm, away_fgm = 31, 26
                home_fg3m, away_fg3m = 10, 6
                home_fta, away_fta = 20, 14
                home_orb, away_orb = 12, 8
                home_drb, away_drb = 26, 22
                home_to, away_to = 9, 14
                home_poss, away_poss = 72.0, 71.0
            else:
                home_pts, away_pts = 74, 88
                home_fgm, away_fgm = 26, 31
                home_fg3m, away_fg3m = 6, 10
                home_fta, away_fta = 14, 20
                home_orb, away_orb = 8, 12
                home_drb, away_drb = 22, 26
                home_to, away_to = 14, 9
                home_poss, away_poss = 71.0, 72.0

            date_str = f"2026-01-{idx + 1:02d}"
            games.append({
                "game_id": game_id,
                "date": date_str,
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": home_pts,
                "away_goals": away_pts,
            })
            box_rows.extend([
                {
                    "game_id": game_id,
                    "team": home_team,
                    "date": date_str,
                    "pts": home_pts,
                    "fgm": home_fgm,
                    "fga": 60,
                    "fg3m": home_fg3m,
                    "fg3a": 24,
                    "ftm": max(home_fta - 3, 0),
                    "fta": home_fta,
                    "orb": home_orb,
                    "drb": home_drb,
                    "to": home_to,
                    "possessions": home_poss,
                },
                {
                    "game_id": game_id,
                    "team": away_team,
                    "date": date_str,
                    "pts": away_pts,
                    "fgm": away_fgm,
                    "fga": 60,
                    "fg3m": away_fg3m,
                    "fg3a": 24,
                    "ftm": max(away_fta - 3, 0),
                    "fta": away_fta,
                    "orb": away_orb,
                    "drb": away_drb,
                    "to": away_to,
                    "possessions": away_poss,
                },
            ])

        model = RecentBoxScoreModel(pd.DataFrame(box_rows), pd.DataFrame(games), feature_window=6, min_games=10)
        probs = recent_boxscore_predict(model, "A", "B")

        assert model.model is not None
        assert probs["home"] > 0.5
        assert math.isclose(probs["home"] + probs["away"], 1.0, abs_tol=1e-9)


class TestNbaMatchupModel:
    def test_fit_and_predict_uses_venue_rest_and_style_context(self):
        games = []
        box_rows = []
        base = pd.Timestamp("2026-01-01")
        matchups = [
            ("A", "B", 114, 101, 100.0),
            ("C", "D", 107, 95, 97.0),
            ("B", "A", 99, 110, 99.0),
            ("D", "C", 94, 103, 96.0),
        ]
        for idx in range(24):
            game_id = f"nba-{idx}"
            home_team, away_team, home_pts, away_pts, pace = matchups[idx % len(matchups)]
            date_str = (base + pd.Timedelta(days=idx * 2)).strftime("%Y-%m-%d")
            games.append({
                "game_id": game_id,
                "date": date_str,
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": home_pts,
                "away_goals": away_pts,
            })
            home_strong = home_team in {"A", "C"}
            away_strong = away_team in {"A", "C"}
            box_rows.extend([
                {
                    "game_id": game_id,
                    "team": home_team,
                    "date": date_str,
                    "pts": home_pts,
                    "fgm": 40 if home_strong else 34,
                    "fga": 82,
                    "fg3m": 13 if home_strong else 9,
                    "fg3a": 33,
                    "ftm": 21 if home_strong else 17,
                    "fta": 25 if home_strong else 21,
                    "orb": 11 if home_strong else 8,
                    "drb": 30 if home_strong else 26,
                    "to": 10 if home_strong else 14,
                    "possessions": pace,
                },
                {
                    "game_id": game_id,
                    "team": away_team,
                    "date": date_str,
                    "pts": away_pts,
                    "fgm": 34 if away_strong else 31,
                    "fga": 82,
                    "fg3m": 9 if away_strong else 7,
                    "fg3a": 31,
                    "ftm": 17 if away_strong else 14,
                    "fta": 21 if away_strong else 18,
                    "orb": 8 if away_strong else 7,
                    "drb": 26 if away_strong else 24,
                    "to": 14 if away_strong else 16,
                    "possessions": pace - 1.0,
                },
            ])

        model = NbaMatchupModel(pd.DataFrame(box_rows), pd.DataFrame(games), feature_window=6, min_games=10)
        probs = nba_matchup_predict(model, "A", "B", game_date="2026-03-15")

        assert model.model is not None
        assert probs["home"] > 0.5
        assert math.isclose(probs["home"] + probs["away"], 1.0, abs_tol=1e-9)


class TestNbaTotalsModel:
    def test_predicts_expected_total_and_market_probabilities(self):
        games = []
        box_rows = []
        base = pd.Timestamp("2026-01-01")
        for idx in range(24):
            game_id = f"nt-{idx}"
            home_team = "A" if idx % 2 == 0 else "B"
            away_team = "B" if idx % 2 == 0 else "A"
            home_pts = 118 if home_team == "A" else 108
            away_pts = 108 if away_team == "B" else 118
            date_str = (base + pd.Timedelta(days=idx)).strftime("%Y-%m-%d")
            games.append({
                "game_id": game_id,
                "date": date_str,
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": home_pts,
                "away_goals": away_pts,
            })
            box_rows.extend([
                {
                    "game_id": game_id,
                    "team": home_team,
                    "date": date_str,
                    "pts": home_pts,
                    "fgm": 42,
                    "fga": 87,
                    "fg3m": 14,
                    "fg3a": 37,
                    "ftm": 20,
                    "fta": 24,
                    "orb": 11,
                    "drb": 31,
                    "to": 11,
                    "possessions": 100.0,
                },
                {
                    "game_id": game_id,
                    "team": away_team,
                    "date": date_str,
                    "pts": away_pts,
                    "fgm": 39,
                    "fga": 84,
                    "fg3m": 12,
                    "fg3a": 34,
                    "ftm": 18,
                    "fta": 22,
                    "orb": 9,
                    "drb": 29,
                    "to": 12,
                    "possessions": 99.0,
                },
            ])

        model = NbaTotalsModel(pd.DataFrame(box_rows), pd.DataFrame(games), feature_window=6, min_games=10)
        result = nba_totals_predict(
            model,
            {"home_team": "A", "away_team": "B"},
            total_line=225.5,
        )

        assert model.model is not None
        assert result["expected_total"] > 210.0
        assert 0.0 < result["over"] < 1.0
        assert math.isclose(result["over"] + result["under"], 1.0, abs_tol=1e-9)


class TestNhlMatchupModel:
    def test_predicts_home_edge_for_stronger_team(self):
        games = []
        base = pd.Timestamp("2026-01-01")
        for idx in range(48):
            home_team = "Bruins" if idx % 2 == 0 else "Canadiens"
            away_team = "Canadiens" if idx % 2 == 0 else "Bruins"
            strong_home = home_team == "Bruins"
            date_str = (base + pd.Timedelta(days=idx)).strftime("%Y-%m-%d")
            home_goals = 4 if strong_home else 2
            away_goals = 2 if strong_home else 4
            games.append({
                "game_id": f"nhl-{idx}",
                "date": date_str,
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_saves": 28 if strong_home else 24,
                "away_saves": 24 if strong_home else 28,
                "home_save_pct": 0.925 if strong_home else 0.895,
                "away_save_pct": 0.895 if strong_home else 0.925,
                "home_shots": 32 if strong_home else 27,
                "away_shots": 27 if strong_home else 32,
                "home_goalie": "Bruins Starter" if strong_home else "Canadiens Starter",
                "away_goalie": "Canadiens Starter" if strong_home else "Bruins Starter",
                "home_goalie_save_pct": 0.928 if strong_home else 0.892,
                "away_goalie_save_pct": 0.892 if strong_home else 0.928,
                "home_goalie_goals_allowed": 2 if strong_home else 4,
                "away_goalie_goals_allowed": 4 if strong_home else 2,
                "home_faceoff_pct": 0.56 if strong_home else 0.46,
                "away_faceoff_pct": 0.46 if strong_home else 0.56,
                "home_power_play_pct": 0.24 if strong_home else 0.14,
                "away_power_play_pct": 0.14 if strong_home else 0.24,
                "home_blocked_shots": 17 if strong_home else 12,
                "away_blocked_shots": 12 if strong_home else 17,
                "home_takeaways": 7 if strong_home else 4,
                "away_takeaways": 4 if strong_home else 7,
                "home_giveaways": 4 if strong_home else 7,
                "away_giveaways": 7 if strong_home else 4,
                "home_penalty_minutes": 6.0 if strong_home else 10.0,
                "away_penalty_minutes": 10.0 if strong_home else 6.0,
            })

        model = NhlMatchupModel(pd.DataFrame(games), feature_window=8, min_games=20)
        probs = nhl_matchup_predict(
            model,
            "Bruins",
            "Canadiens",
            game_date="2026-03-10",
            home_goalie="Bruins Starter",
            away_goalie="Canadiens Starter",
        )

        assert model.model is not None
        assert "recent_penalty_kill_pct_diff" in model.feature_names
        assert "recent_blocked_shot_diff" in model.feature_names
        assert probs["home"] > 0.5
        assert math.isclose(probs["home"] + probs["away"], 1.0, abs_tol=1e-9)
