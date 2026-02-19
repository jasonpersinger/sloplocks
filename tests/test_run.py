"""Integration tests for the pipeline orchestrator."""

import json
import os
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import pandas as pd
from pipeline.run import run_pipeline, run_sport_pipeline


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
            "date": (base_date + timedelta(days=i)).isoformat(),
            "home_team": home,
            "away_team": away,
            "home_goals": hg,
            "away_goals": ag,
        })
    return pd.DataFrame(matches)


# ---------------------------------------------------------------------------
# EPL pipeline
# ---------------------------------------------------------------------------

class TestRunEPLPipeline:
    @patch("pipeline.run.fetch_odds")
    @patch("pipeline.run.fetch_understat_xg")
    @patch("pipeline.run.fetch_epl_fixtures")
    @patch("pipeline.run.fetch_epl_matches")
    def test_produces_valid_epl_predictions(
        self, mock_matches, mock_fixtures, mock_xg, mock_odds,
        sample_matches, sample_xg, sample_odds, tmp_path
    ):
        mock_matches.return_value = sample_matches
        mock_fixtures.return_value = [
            {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "date": "2026-02-22T15:00:00Z",
                "matchday": 26,
            }
        ]
        mock_xg.return_value = sample_xg
        mock_odds.return_value = [
            {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "commence_time": "2026-02-22T15:00:00Z",
                "home_odds": 1.67,
                "draw_odds": 3.80,
                "away_odds": 4.50,
            }
        ]

        output_dir = str(tmp_path / "epl")
        run_sport_pipeline("epl", output_dir=output_dir)

        predictions_path = os.path.join(output_dir, "predictions.json")
        assert os.path.exists(predictions_path)

        with open(predictions_path) as f:
            data = json.load(f)

        assert data["sport"] == "epl"
        assert data["outcomes"] == ["home", "draw", "away"]
        assert "generated_at" in data
        assert "matches" in data
        assert len(data["matches"]) >= 1

        match = data["matches"][0]
        assert match["home_team"] == "Arsenal"
        assert match["away_team"] == "Chelsea"
        assert "model_probs" in match
        assert "draw" in match["model_probs"]
        assert abs(sum(match["model_probs"].values()) - 1.0) < 0.01

        # Slop Locks
        assert "slop_locks" in data
        assert isinstance(data["slop_locks"], list)
        assert len(data["slop_locks"]) <= 5
        for lock in data["slop_locks"]:
            assert lock["pick"] in ("home", "draw", "away")
            assert "american_odds" in lock
            assert "model_prob" in lock


# ---------------------------------------------------------------------------
# NBA pipeline
# ---------------------------------------------------------------------------

class TestRunNBAPipeline:
    @patch("pipeline.run.fetch_odds")
    @patch("pipeline.run.fetch_nba_schedule")
    @patch("pipeline.run.fetch_nba_games")
    def test_produces_valid_nba_predictions(
        self, mock_games, mock_schedule, mock_odds,
        sample_nba_matches, tmp_path
    ):
        mock_games.return_value = sample_nba_matches
        mock_schedule.return_value = [
            {
                "home_team": "Lakers",
                "away_team": "Warriors",
                "date": "2026-02-19",
            }
        ]
        mock_odds.return_value = [
            {
                "home_team": "Lakers",
                "away_team": "Warriors",
                "commence_time": "2026-02-19T00:30:00Z",
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
    @patch("pipeline.run.fetch_nba_schedule")
    @patch("pipeline.run.fetch_nba_games")
    @patch("pipeline.run.fetch_understat_xg")
    @patch("pipeline.run.fetch_epl_fixtures")
    @patch("pipeline.run.fetch_epl_matches")
    def test_produces_per_sport_files_and_manifest(
        self, mock_epl_matches, mock_epl_fixtures, mock_xg,
        mock_nba_games, mock_nba_schedule, mock_odds,
        sample_matches, sample_xg, sample_nba_matches, tmp_path
    ):
        mock_epl_matches.return_value = sample_matches
        mock_epl_fixtures.return_value = [
            {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "date": "2026-02-22T15:00:00Z",
                "matchday": 26,
            }
        ]
        mock_xg.return_value = sample_xg
        mock_nba_games.return_value = sample_nba_matches
        mock_nba_schedule.return_value = [
            {
                "home_team": "Lakers",
                "away_team": "Warriors",
                "date": "2026-02-19",
            }
        ]
        mock_odds.return_value = []

        output_dir = str(tmp_path)
        manifest = run_pipeline(output_dir=output_dir)

        # Manifest
        manifest_path = os.path.join(output_dir, "manifest.json")
        assert os.path.exists(manifest_path)
        assert "epl" in manifest["sports"]
        assert "nba" in manifest["sports"]
        assert manifest["sports"]["epl"]["status"] == "ok"
        assert manifest["sports"]["nba"]["status"] == "ok"

        # Per-sport prediction files
        assert os.path.exists(os.path.join(output_dir, "epl", "predictions.json"))
        assert os.path.exists(os.path.join(output_dir, "nba", "predictions.json"))


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


# ---------------------------------------------------------------------------
# _compute_slop_locks helper
# ---------------------------------------------------------------------------

class TestComputeSlopLocks:
    """Tests for the _compute_slop_locks helper."""

    def _make_record(self, home, away, outcome, model_prob, american_odds, edge=0.0):
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
                    "decimal_odds": 0.0,
                    "american_odds": american_odds,
                    "is_value": edge >= 0.05,
                }
            },
            "best_odds": {outcome: american_odds},
            "model_probs": {outcome: model_prob},
            "individual_models": {},
        }

    def test_prefers_odds_window_with_fallback(self):
        """In-window picks come first; outside-window picks backfill when fewer than 5."""
        from pipeline.run import _compute_slop_locks
        records = [
            self._make_record("A", "B", "home", 0.70, -200),   # outside window (short)
            self._make_record("C", "D", "home", 0.65, -140),   # in window
            self._make_record("E", "F", "away", 0.55, 190),    # in window
            self._make_record("G", "H", "away", 0.45, 250),    # outside window (long)
        ]
        locks = _compute_slop_locks(records, ["home", "away"])
        # All 4 returned because in-window count (2) < 5, so fallback fills the rest
        assert len(locks) == 4
        # In-window picks come first, ranked by model_prob
        assert locks[0]["american_odds"] == -140   # highest prob in-window
        assert locks[1]["american_odds"] == 190    # second in-window
        # Fallback picks follow, also ranked by model_prob
        assert locks[2]["american_odds"] == -200   # highest prob outside window
        assert locks[3]["american_odds"] == 250

    def test_ranked_by_model_probability(self):
        """Locks are sorted by model_prob descending."""
        from pipeline.run import _compute_slop_locks
        records = [
            self._make_record("A", "B", "home", 0.55, 100),
            self._make_record("C", "D", "home", 0.75, -130),
            self._make_record("E", "F", "away", 0.65, 110),
        ]
        locks = _compute_slop_locks(records, ["home", "away"])
        probs = [l["model_prob"] for l in locks]
        assert probs == sorted(probs, reverse=True)

    def test_no_market_agreement_required(self):
        """A pick qualifies even when model_prob < implied_prob (negative edge)."""
        from pipeline.run import _compute_slop_locks
        records = [
            self._make_record("A", "B", "home", 0.52, -130, edge=-0.045),
        ]
        locks = _compute_slop_locks(records, ["home", "away"])
        assert len(locks) == 1

    def test_returns_at_most_five(self):
        """At most 5 locks returned."""
        from pipeline.run import _compute_slop_locks
        records = [
            self._make_record(f"T{i}", f"T{i+1}", "home", 0.60, -100)
            for i in range(10)
        ]
        locks = _compute_slop_locks(records, ["home", "away"])
        assert len(locks) <= 5
