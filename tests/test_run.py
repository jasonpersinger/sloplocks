"""Integration tests for the pipeline orchestrator."""

import json
import os
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from pipeline.run import run_pipeline


class TestRunPipeline:
    @patch("pipeline.run.fetch_odds")
    @patch("pipeline.run.fetch_understat_xg")
    @patch("pipeline.run.fetch_epl_fixtures")
    @patch("pipeline.run.fetch_epl_matches")
    def test_produces_valid_predictions_json(
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

        output_dir = str(tmp_path)
        run_pipeline(output_dir=output_dir)

        predictions_path = os.path.join(output_dir, "predictions.json")
        assert os.path.exists(predictions_path)

        with open(predictions_path) as f:
            data = json.load(f)

        assert "generated_at" in data
        assert "matches" in data
        assert len(data["matches"]) >= 1

        match = data["matches"][0]
        assert match["home_team"] == "Arsenal"
        assert match["away_team"] == "Chelsea"
        assert "model_probs" in match
        assert "edges" in match
        assert abs(sum(match["model_probs"].values()) - 1.0) < 0.01

        # Slop Locks of the Week
        assert "slop_locks" in data
        assert isinstance(data["slop_locks"], list)
        assert len(data["slop_locks"]) <= 5
        for lock in data["slop_locks"]:
            assert lock["pick"] in ("home", "draw", "away")
            assert lock["edge"] > 0
            assert -200 <= lock["american_odds"] <= 200
