"""Integration tests for the pipeline orchestrator."""

import csv
import json
import os
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
import pandas as pd
from pipeline.run import run_pipeline, run_sport_pipeline

_TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_nba_matches():
    """Minimal NBA match results for testing."""
    base_date = datetime(2024, 10, 22)
    results = [
        ("Lakers", "Celtics", 112, 108),
        ("Warriors", "Heat", 120, 115),
        ("Bucks", "Knicks", 105, 110),
        ("Suns", "Nuggets", 98, 103),
        ("76ers", "Bulls", 115, 102),
        ("Celtics", "Lakers", 130, 125),
        ("Heat", "Warriors", 99, 105),
        ("Knicks", "Bucks", 108, 100),
        ("Nuggets", "Suns", 118, 110),
        ("Bulls", "76ers", 95, 102),
    ]
    matches = []
    for i, (home, away, hg, ag) in enumerate(results):
        matches.append({
            "game_id": str(1000 + i),
            "date": (base_date + timedelta(days=i)).isoformat(),
            "home_team": home,
            "away_team": away,
            "home_goals": hg,
            "away_goals": ag,
        })
    return pd.DataFrame(matches)


@pytest.fixture
def sample_nba_box_scores(sample_nba_matches):
    """Box scores matching the sample_nba_matches fixture."""
    from tests.conftest import _make_box_score
    rows = []
    for _, g in sample_nba_matches.iterrows():
        rows.append(_make_box_score(
            g["game_id"], g["date"], g["home_team"], g["home_goals"], True,
        ))
        rows.append(_make_box_score(
            g["game_id"], g["date"], g["away_team"], g["away_goals"], False,
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# NBA pipeline
# ---------------------------------------------------------------------------

class TestRunNBAPipeline:
    @patch("pipeline.run.fetch_odds")
    @patch("pipeline.run.fetch_nba_espn_schedule")
    @patch("pipeline.run.fetch_nba_espn_games")
    def test_produces_valid_nba_predictions(
        self, mock_games, mock_schedule, mock_odds,
        sample_nba_matches, sample_nba_box_scores, tmp_path
    ):
        mock_games.return_value = (sample_nba_matches, sample_nba_box_scores)
        mock_schedule.return_value = [
            {
                "home_team": "Lakers",
                "away_team": "Warriors",
                "date": _TODAY,
            }
        ]
        mock_odds.return_value = [
            {
                "home_team": "Lakers",
                "away_team": "Warriors",
                "commence_time": f"{_TODAY}T00:30:00Z",
                "home_odds": 2.10,
                "draw_odds": 0.0,
                "away_odds": 1.75,
            }
        ]

        output_dir = str(tmp_path / "nba")
        run_sport_pipeline("nba", output_dir=output_dir)

        predictions_path = os.path.join(output_dir, "predictions.json")
        assert os.path.exists(predictions_path)

        with open(predictions_path) as f:
            data = json.load(f)

        assert data["sport"] == "nba"
        assert data["outcomes"] == ["home", "away"]
        assert len(data["matches"]) >= 1

        match = data["matches"][0]
        assert "draw" not in match["model_probs"]
        assert set(match["model_probs"].keys()) == {"home", "away"}
        assert abs(sum(match["model_probs"].values()) - 1.0) < 0.01

        # Season stats should not have draw fields
        stats = data["season_stats"]
        assert "draws" not in stats
        assert "draw_pct" not in stats

        # Only elo model for NBA
        assert "elo" in data["model_weights"]
        assert "dixon_coles" not in data["model_weights"]


# ---------------------------------------------------------------------------
# Multi-sport orchestrator
# ---------------------------------------------------------------------------

class TestRunPipeline:
    @patch("pipeline.run.fetch_odds")
    @patch("pipeline.run.fetch_ncaam_schedule")
    @patch("pipeline.run.fetch_ncaam_games")
    @patch("pipeline.run.fetch_nba_espn_schedule")
    @patch("pipeline.run.fetch_nba_espn_games")
    def test_produces_per_sport_files_and_manifest(
        self, mock_nba_games, mock_nba_schedule, mock_ncaam_games, mock_ncaam_schedule, mock_odds,
        sample_nba_matches, sample_nba_box_scores,
        ncaam_games, ncaam_box_scores, tmp_path
    ):
        mock_nba_games.return_value = (sample_nba_matches, sample_nba_box_scores)
        mock_nba_schedule.return_value = [
            {
                "home_team": "Lakers",
                "away_team": "Warriors",
                "date": "2026-02-19",
            }
        ]
        mock_ncaam_games.return_value = (ncaam_games, ncaam_box_scores)
        mock_ncaam_schedule.return_value = [
            {
                "home_team": "Duke",
                "away_team": "Kansas",
                "date": "2026-02-19",
            }
        ]
        mock_odds.return_value = []

        output_dir = str(tmp_path)
        manifest = run_pipeline(output_dir=output_dir)

        # Manifest
        manifest_path = os.path.join(output_dir, "manifest.json")
        assert os.path.exists(manifest_path)
        assert "nba" in manifest["sports"]
        assert "ncaam" in manifest["sports"]
        assert manifest["sports"]["nba"]["status"] == "ok"
        assert manifest["sports"]["ncaam"]["status"] == "ok"

        # Per-sport prediction files
        assert os.path.exists(os.path.join(output_dir, "nba", "predictions.json"))
        assert os.path.exists(os.path.join(output_dir, "ncaam", "predictions.json"))


# ---------------------------------------------------------------------------
# _days_since_last_game helper
# ---------------------------------------------------------------------------

class TestDaysSinceLastGame:
    def _make_matches(self, rows):
        return pd.DataFrame(rows, columns=["date", "home_team", "away_team",
                                           "home_goals", "away_goals"])

    def test_returns_correct_days_for_known_team(self):
        matches = self._make_matches([
            {"date": "2026-02-10", "home_team": "Lakers", "away_team": "Celtics",
             "home_goals": 110, "away_goals": 105},
        ])
        from pipeline.run import _days_since_last_game
        result = _days_since_last_game("Lakers", "2026-02-11", matches)
        assert result == 1

    def test_returns_none_for_unknown_team(self):
        matches = self._make_matches([
            {"date": "2026-02-10", "home_team": "Lakers", "away_team": "Celtics",
             "home_goals": 110, "away_goals": 105},
        ])
        from pipeline.run import _days_since_last_game
        result = _days_since_last_game("Thunder", "2026-02-11", matches)
        assert result is None

    def test_ignores_future_games(self):
        matches = self._make_matches([
            {"date": "2026-02-10", "home_team": "Lakers", "away_team": "Celtics",
             "home_goals": 110, "away_goals": 105},
            {"date": "2026-02-12", "home_team": "Lakers", "away_team": "Nuggets",
             "home_goals": 100, "away_goals": 98},
        ])
        from pipeline.run import _days_since_last_game
        # Asking "as of Feb 11", the Feb 12 game is in the future
        result = _days_since_last_game("Lakers", "2026-02-11", matches)
        assert result == 1  # only Feb 10 counts

    def test_handles_empty_matches(self):
        matches = pd.DataFrame(
            columns=["date", "home_team", "away_team", "home_goals", "away_goals"]
        )
        from pipeline.run import _days_since_last_game
        result = _days_since_last_game("Lakers", "2026-02-11", matches)
        assert result is None

    def test_returns_none_for_same_day_game(self):
        """Games on the same date as before_date are excluded (strict <)."""
        matches = self._make_matches([
            {"date": "2026-02-11", "home_team": "Lakers", "away_team": "Celtics",
             "home_goals": 110, "away_goals": 105},
        ])
        from pipeline.run import _days_since_last_game
        result = _days_since_last_game("Lakers", "2026-02-11", matches)
        assert result is None


class TestRecentFormAdjustment:
    def test_rewards_recent_wins_and_scales_to_window(self):
        matches = pd.DataFrame([
            {"date": "2026-02-01", "home_team": "Lakers", "away_team": "Heat", "home_goals": 110, "away_goals": 100},
            {"date": "2026-02-03", "home_team": "Bulls", "away_team": "Lakers", "home_goals": 99, "away_goals": 105},
            {"date": "2026-02-05", "home_team": "Lakers", "away_team": "Celtics", "home_goals": 102, "away_goals": 108},
        ])

        from pipeline.run import _recent_form_adjustment

        adj = _recent_form_adjustment("Lakers", "2026-02-10", matches, window=4, max_adjustment=20)

        # Scores: win, win, loss -> 0.667 recent form.
        # Centered and sample-scaled adjustment = ((0.667 - 0.5) * 2) * 20 * (3/4) ~= 5.0
        assert adj == pytest.approx(5.0, abs=0.1)

    def test_returns_zero_without_prior_games(self):
        matches = pd.DataFrame([
            {"date": "2026-02-11", "home_team": "Lakers", "away_team": "Heat", "home_goals": 110, "away_goals": 100},
        ])

        from pipeline.run import _recent_form_adjustment

        assert _recent_form_adjustment("Lakers", "2026-02-11", matches, window=6, max_adjustment=30) == 0.0


class TestRestAdjustment:
    def test_applies_back_to_back_and_fatigue_penalties(self):
        matches = pd.DataFrame([
            {"date": "2026-02-06", "home_team": "Lakers", "away_team": "Heat", "home_goals": 110, "away_goals": 100},
            {"date": "2026-02-08", "home_team": "Celtics", "away_team": "Lakers", "home_goals": 101, "away_goals": 99},
            {"date": "2026-02-09", "home_team": "Bulls", "away_team": "Lakers", "home_goals": 95, "away_goals": 97},
        ])
        sport = {
            "back_to_back_penalty": 20,
            "fatigue_window_days": 4,
            "fatigue_threshold_games": 3,
            "fatigue_penalty": 10,
            "rest_bonus_days": 3,
            "rest_bonus_points": 5,
        }

        from pipeline.run import _rest_adjustment

        assert _rest_adjustment("Lakers", "2026-02-10", matches, sport) == -30

    def test_applies_rest_bonus_when_team_has_multiple_days_off(self):
        matches = pd.DataFrame([
            {"date": "2026-02-03", "home_team": "Lakers", "away_team": "Heat", "home_goals": 110, "away_goals": 100},
        ])
        sport = {
            "back_to_back_penalty": 0,
            "fatigue_window_days": 0,
            "fatigue_threshold_games": 0,
            "fatigue_penalty": 0,
            "rest_bonus_days": 3,
            "rest_bonus_points": 7,
        }

        from pipeline.run import _rest_adjustment

        assert _rest_adjustment("Lakers", "2026-02-07", matches, sport) == 7


# ---------------------------------------------------------------------------
# _compute_slop_locks helper
# ---------------------------------------------------------------------------

class TestComputeSlopLocks:
    """Tests for the _compute_slop_locks helper."""

    def _make_record(
        self,
        home,
        away,
        outcome,
        model_prob,
        american_odds,
        edge=0.0,
        confidence_score=0,
        expected_value=0.0,
    ):
        implied_prob = 1 / (1 + abs(american_odds) / 100) if american_odds < 0 else 100 / (american_odds + 100)
        return {
            "home_team": home,
            "away_team": away,
            "date": "2026-03-01",
            "matchday": None,
            "edges": {
                outcome: {
                    "model_prob": model_prob,
                    "implied_prob": implied_prob,
                    "edge": edge,
                    "expected_value": expected_value,
                    "decimal_odds": 0.0,
                    "american_odds": american_odds,
                    "confidence_score": confidence_score,
                    "is_value": edge >= 0.05,
                }
            },
            "best_odds": {outcome: american_odds},
            "model_probs": {outcome: model_prob},
            "individual_models": {},
        }

    def test_pick_of_day_and_slate_aware_additional_locks(self):
        """Later picks can qualify by floor or by staying close to the top score."""
        from pipeline.run import _compute_slop_locks
        records = [
            self._make_record("A", "B", "home", 0.70, -200, edge=0.04, confidence_score=58, expected_value=0.03),
            self._make_record("C", "D", "home", 0.65, -140, edge=0.06, confidence_score=61, expected_value=0.06),
            self._make_record("E", "F", "away", 0.55, 190, edge=0.05, confidence_score=53, expected_value=0.08),
            self._make_record("G", "H", "away", 0.61, 250, edge=0.05, confidence_score=49, expected_value=0.04),
        ]
        locks = _compute_slop_locks(
            records,
            ["home", "away"],
            additional_confidence_floor=52,
            confidence_dropoff=8,
            max_picks=5,
        )

        assert len(locks) == 3
        assert locks[0]["home_team"] == "C"
        assert locks[0]["confidence_score"] == 61
        assert locks[1]["home_team"] == "A"
        assert locks[1]["confidence_score"] == 58
        assert locks[2]["away_team"] == "F"
        assert locks[2]["confidence_score"] == 53

    def test_ranked_by_confidence_then_edge(self):
        """Candidates are ordered by confidence score, breaking ties on edge."""
        from pipeline.run import _compute_slop_locks
        records = [
            self._make_record("A", "B", "home", 0.80, 100, edge=0.03, confidence_score=70, expected_value=0.02),
            self._make_record("C", "D", "home", 0.55, -130, edge=0.08, confidence_score=70, expected_value=0.06),
            self._make_record("E", "F", "away", 0.65, 110, edge=0.06, confidence_score=90, expected_value=0.09),
        ]
        locks = _compute_slop_locks(records, ["home", "away"], max_picks=5, additional_confidence_floor=52, confidence_dropoff=8)
        picked = [(l["home_team"], l["away_team"], l["confidence_score"], l["edge"]) for l in locks]
        assert picked == [
            ("E", "F", 90, 0.06),
            ("C", "D", 70, 0.08),
            ("A", "B", 70, 0.03),
        ]

    def test_below_threshold_picks_excluded(self):
        """Picks must clear both the edge and win-probability floors."""
        from pipeline.run import _compute_slop_locks
        records = [
            self._make_record("A", "B", "home", 0.52, -130, edge=-0.045, confidence_score=95, expected_value=-0.02),
            self._make_record("C", "D", "home", 0.44, 120, edge=0.09, confidence_score=95, expected_value=0.04),
            self._make_record("E", "F", "away", 0.60, 110, edge=0.029, confidence_score=95, expected_value=0.03),
        ]
        locks = _compute_slop_locks(records, ["home", "away"], edge_floor=0.03, probability_floor=0.45)
        assert len(locks) == 0

    def test_negative_ev_picks_excluded(self):
        from pipeline.run import _compute_slop_locks
        records = [
            self._make_record("A", "B", "home", 0.62, -110, edge=0.04, confidence_score=90, expected_value=-0.01),
        ]
        locks = _compute_slop_locks(records, ["home", "away"])
        assert locks == []

    def test_opponent_conflict_excluded(self):
        """If Team A is picked over Team B, Team B must not also appear as a pick."""
        from pipeline.run import _compute_slop_locks
        records = [
            # Game 1: Brentford beats Brighton — higher edge
            self._make_record("Brentford", "Brighton", "home", 0.60, 114, edge=0.05, confidence_score=85, expected_value=0.07),
            # Game 2: Brighton beats NF — lower edge; Brighton is loser in game 1
            self._make_record("Brighton", "NF", "home", 0.55, 110, edge=0.04, confidence_score=70, expected_value=0.05),
            # Unrelated game — should still appear
            self._make_record("Wolves", "Villa", "away", 0.62, -108, edge=0.049, confidence_score=80, expected_value=0.06),
        ]
        locks = _compute_slop_locks(records, ["home", "away"], max_picks=5, additional_confidence_floor=52, confidence_dropoff=8)
        picked_teams = [
            l["home_team"] if l["pick"] == "home" else l["away_team"]
            for l in locks
        ]
        # Brentford and Wolves picked; Brighton excluded (opponent of Brentford pick)
        assert "Brentford" in picked_teams
        assert "Brighton" not in picked_teams
        assert "Villa" in picked_teams

    def test_returns_at_most_three(self):
        """At most the configured number of locks are returned."""
        from pipeline.run import _compute_slop_locks
        records = [
            self._make_record(
                f"T{i}",
                f"T{i+1}",
                "home",
                0.60,
                -100,
                edge=0.05 + i * 0.001,
                confidence_score=90 - i,
                expected_value=0.04 + i * 0.001,
            )
            for i in range(10)
        ]
        locks = _compute_slop_locks(records, ["home", "away"], max_picks=5, additional_confidence_floor=52, confidence_dropoff=8)
        assert len(locks) <= 5

    def test_later_candidate_is_considered_if_earlier_ones_miss_threshold(self):
        """The selector should scan beyond ranks 2-3 when filling the card."""
        from pipeline.run import _compute_slop_locks
        records = [
            self._make_record("A", "B", "home", 0.70, -120, edge=0.05, confidence_score=90, expected_value=0.06),
            self._make_record("C", "D", "home", 0.60, -110, edge=0.04, confidence_score=43, expected_value=0.05),
            self._make_record("E", "F", "home", 0.59, -105, edge=0.04, confidence_score=41, expected_value=0.05),
            self._make_record("G", "H", "home", 0.58, -102, edge=0.04, confidence_score=54, expected_value=0.05),
        ]

        locks = _compute_slop_locks(
            records,
            ["home", "away"],
            additional_confidence_floor=52,
            confidence_dropoff=8,
            max_picks=5,
        )

        assert [(lock["home_team"], lock["confidence_score"]) for lock in locks] == [
            ("A", 90),
            ("G", 54),
        ]


class TestResultsLog:
    def test_append_results_log_creates_file_and_dedupes_rows(self, tmp_path):
        from pipeline.run import _append_results_log

        path = str(tmp_path / "tracking" / "results_log.csv")
        row = {
            "logged_at": "2026-03-28T12:00:00Z",
            "sport": "nba",
            "entry_type": "prediction",
            "home_team": "Lakers",
            "away_team": "Celtics",
            "match_date": "2026-03-28",
            "pick": "home",
            "actual": "home",
            "won": "true",
            "model_prob": 0.61,
            "home_prob": 0.61,
            "away_prob": 0.39,
            "draw_prob": "",
            "implied_prob": 0.52,
            "market_implied_prob": 0.54,
            "edge": 0.09,
            "expected_value": 0.07,
            "american_odds": 110,
            "decimal_odds": 2.1,
            "confidence_score": 72,
            "kelly_fraction": 0.06,
            "fractional_kelly": 0.015,
        }

        _append_results_log(path, [row, row])

        assert os.path.exists(path)
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["sport"] == "nba"
        assert rows[0]["expected_value"] == "0.07"
