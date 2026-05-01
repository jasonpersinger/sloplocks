"""Tests for NCAAM pipeline integration."""

import json
import os
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from pipeline.run import run_sport_pipeline

pytestmark = pytest.mark.skip(reason="NCAAM pipeline is season-disabled in live runtime.")


@pytest.fixture
def ncaam_pipeline_mocks(ncaam_games, ncaam_box_scores, tmp_path):
    """Mock all external calls for NCAAM pipeline."""
    fixtures = [
        {"home_team": "Duke", "away_team": "North Carolina", "date": "2026-02-20"},
        {"home_team": "Kansas", "away_team": "Kentucky", "date": "2026-02-21"},
    ]
    odds = [
        {
            "home_team": "Duke",
            "away_team": "North Carolina",
            "home_odds": 1.65,
            "away_odds": 2.25,
        },
    ]

    return {
        "games": ncaam_games,
        "box_scores": ncaam_box_scores,
        "fixtures": fixtures,
        "odds": odds,
        "output_dir": str(tmp_path),
    }


class TestNcaamPipeline:
    @patch("pipeline.run.fetch_odds")
    @patch("pipeline.run.fetch_ncaam_schedule")
    @patch("pipeline.run.fetch_ncaam_games")
    def test_produces_predictions_json(self, mock_games, mock_sched, mock_odds,
                                        ncaam_pipeline_mocks):
        mock_games.return_value = (
            ncaam_pipeline_mocks["games"],
            ncaam_pipeline_mocks["box_scores"],
        )
        mock_sched.return_value = ncaam_pipeline_mocks["fixtures"]
        mock_odds.return_value = ncaam_pipeline_mocks["odds"]

        result = run_sport_pipeline("ncaam", output_dir=ncaam_pipeline_mocks["output_dir"])

        assert result is not None
        assert result["sport"] == "ncaam"
        assert "matches" in result
        assert "slop_locks" in result
        assert "model_weights" in result

        # All three models should have weights
        weights = result["model_weights"]
        assert "elo" in weights
        assert "efficiency" in weights
        assert "four_factors" in weights
        assert "results_features" in weights
        assert "recent_boxscore" in weights

    @patch("pipeline.run.fetch_odds")
    @patch("pipeline.run.fetch_ncaam_schedule")
    @patch("pipeline.run.fetch_ncaam_games")
    def test_predictions_have_two_way_probs(self, mock_games, mock_sched, mock_odds,
                                             ncaam_pipeline_mocks):
        mock_games.return_value = (
            ncaam_pipeline_mocks["games"],
            ncaam_pipeline_mocks["box_scores"],
        )
        mock_sched.return_value = ncaam_pipeline_mocks["fixtures"]
        mock_odds.return_value = ncaam_pipeline_mocks["odds"]

        result = run_sport_pipeline("ncaam", output_dir=ncaam_pipeline_mocks["output_dir"])

        for match in result.get("matches", []):
            probs = match["model_probs"]
            assert "home" in probs
            assert "away" in probs
            assert "draw" not in probs

    @patch("pipeline.run.fetch_odds")
    @patch("pipeline.run.fetch_ncaam_schedule")
    @patch("pipeline.run.fetch_ncaam_games")
    def test_writes_output_files(self, mock_games, mock_sched, mock_odds,
                                  ncaam_pipeline_mocks):
        mock_games.return_value = (
            ncaam_pipeline_mocks["games"],
            ncaam_pipeline_mocks["box_scores"],
        )
        mock_sched.return_value = ncaam_pipeline_mocks["fixtures"]
        mock_odds.return_value = ncaam_pipeline_mocks["odds"]

        output_dir = ncaam_pipeline_mocks["output_dir"]
        run_sport_pipeline("ncaam", output_dir=output_dir)

        assert os.path.exists(os.path.join(output_dir, "predictions.json"))
        assert os.path.exists(os.path.join(output_dir, "history.json"))
        assert os.path.exists(os.path.join(output_dir, "model_accuracy.json"))
