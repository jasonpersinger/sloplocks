"""Tests for the backtesting and accuracy-tracking utilities."""

import json
import math
import os
import pytest

from pipeline.backtest import (
    build_backtest_report,
    build_threshold_guidance,
    compute_brier_score,
    compute_model_weights,
    compute_roi,
    evaluate_prediction,
    get_rolling_accuracy,
    summarize_closing_line_value,
    summarize_pick_breakdowns,
    summarize_pick_history,
    summarize_prediction_history,
    update_accuracy_log,
)
from pipeline.config import ENSEMBLE_ACCURACY_WINDOW


# ---------------------------------------------------------------------------
# evaluate_prediction
# ---------------------------------------------------------------------------

class TestEvaluatePrediction:
    """Tests for scoring a single prediction against an actual result."""

    def test_correct_home_prediction(self):
        """Model predicts home win and home team wins."""
        probs = {"home": 0.6, "draw": 0.2, "away": 0.2}
        result = evaluate_prediction(probs, home_goals=2, away_goals=0)

        assert result["predicted"] == "home"
        assert result["actual"] == "home"
        assert result["correct"] is True
        assert math.isclose(result["actual_prob"], 0.6)
        assert math.isclose(result["log_loss"], -math.log(0.6))

    def test_correct_draw(self):
        """Model predicts draw and the match is drawn."""
        probs = {"home": 0.2, "draw": 0.5, "away": 0.3}
        result = evaluate_prediction(probs, home_goals=1, away_goals=1)

        assert result["predicted"] == "draw"
        assert result["actual"] == "draw"
        assert result["correct"] is True
        assert math.isclose(result["actual_prob"], 0.5)

    def test_incorrect_prediction(self):
        """Model predicts home win but away team wins."""
        probs = {"home": 0.5, "draw": 0.3, "away": 0.2}
        result = evaluate_prediction(probs, home_goals=0, away_goals=1)

        assert result["predicted"] == "home"
        assert result["actual"] == "away"
        assert result["correct"] is False
        assert math.isclose(result["actual_prob"], 0.2)
        assert math.isclose(result["log_loss"], -math.log(0.2))

    def test_two_way_home_win(self):
        """2-way probs (NBA) should work with no draw key."""
        probs = {"home": 0.65, "away": 0.35}
        result = evaluate_prediction(probs, home_goals=112, away_goals=105)

        assert result["predicted"] == "home"
        assert result["actual"] == "home"
        assert result["correct"] is True
        assert math.isclose(result["actual_prob"], 0.65)

    def test_two_way_away_win(self):
        """2-way probs — away team wins."""
        probs = {"home": 0.45, "away": 0.55}
        result = evaluate_prediction(probs, home_goals=98, away_goals=110)

        assert result["predicted"] == "away"
        assert result["actual"] == "away"
        assert result["correct"] is True

    def test_compute_brier_score(self):
        probs = {"home": 0.6, "away": 0.4}
        score = compute_brier_score(probs, home_goals=110, away_goals=100)
        assert math.isclose(score, 0.32)


# ---------------------------------------------------------------------------
# compute_model_weights
# ---------------------------------------------------------------------------

class TestComputeModelWeights:
    """Tests for softmax-style ensemble weight computation."""

    def test_better_model_gets_more_weight(self):
        """A model with higher accuracy should receive a larger weight."""
        weights = compute_model_weights([0.8, 0.5])
        assert weights[0] > weights[1]
        assert math.isclose(sum(weights), 1.0, abs_tol=1e-9)

    def test_equal_accuracies_equal_weights(self):
        """Identical accuracies must produce identical weights."""
        weights = compute_model_weights([0.6, 0.6, 0.6])
        assert math.isclose(weights[0], weights[1], abs_tol=1e-9)
        assert math.isclose(weights[1], weights[2], abs_tol=1e-9)
        assert math.isclose(sum(weights), 1.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# compute_roi
# ---------------------------------------------------------------------------

class TestComputeROI:
    """Tests for ROI calculation."""

    def test_profitable_bets_positive_roi(self):
        """Winning bets at good odds should yield a positive ROI."""
        bets = [
            {"stake": 10.0, "odds": 2.5, "won": True},
            {"stake": 10.0, "odds": 2.0, "won": False},
        ]
        roi = compute_roi(bets)
        # Return = 25, staked = 20, profit = 5, ROI = 0.25
        assert math.isclose(roi, 0.25)

    def test_no_bets_returns_zero(self):
        """An empty bet list should yield 0.0 ROI."""
        assert compute_roi([]) == 0.0


# ---------------------------------------------------------------------------
# update_accuracy_log
# ---------------------------------------------------------------------------

class TestUpdateAccuracyLog:
    """Tests for the rolling accuracy log."""

    def test_appends_entries(self):
        """New entries should be appended to the model's log."""
        log = {}
        result = {"correct": True}
        update_accuracy_log(log, "elo", result)
        update_accuracy_log(log, "elo", {"correct": False})

        assert len(log["elo"]) == 2
        assert log["elo"][0]["correct"] is True
        assert log["elo"][1]["correct"] is False

    def test_respects_window_limit(self):
        """The log should never exceed ENSEMBLE_ACCURACY_WINDOW entries."""
        log = {}
        for i in range(ENSEMBLE_ACCURACY_WINDOW + 5):
            update_accuracy_log(log, "dc", {"correct": i % 2 == 0, "i": i})

        assert len(log["dc"]) == ENSEMBLE_ACCURACY_WINDOW
        # The oldest entries should have been trimmed; the last entry's
        # index marker should equal the total count minus one.
        assert log["dc"][-1]["i"] == ENSEMBLE_ACCURACY_WINDOW + 4


# ---------------------------------------------------------------------------
# get_rolling_accuracy
# ---------------------------------------------------------------------------

class TestGetRollingAccuracy:
    """Tests for computing rolling accuracy from the log."""

    def test_returns_correct_fraction(self):
        """Accuracy should equal correct / total entries."""
        log = {
            "elo": [
                {"correct": True},
                {"correct": True},
                {"correct": False},
                {"correct": True},
            ]
        }
        assert math.isclose(get_rolling_accuracy(log, "elo"), 0.75)

    def test_returns_default_for_no_data(self):
        """With no log entries the function should return 0.5."""
        assert get_rolling_accuracy({}, "missing_model") == 0.5


class TestBacktestSummary:
    def test_summarize_pick_history_includes_confidence_and_ev(self):
        picks = [
            {
                "evaluated": True,
                "won": True,
                "decimal_odds": 2.1,
                "confidence_score": 70,
                "expected_value": 0.12,
            },
            {
                "evaluated": True,
                "won": False,
                "decimal_odds": 1.8,
                "confidence_score": 50,
                "expected_value": 0.04,
            },
        ]

        summary = summarize_pick_history(picks)

        assert summary["total_picks"] == 2
        assert summary["wins"] == 1
        assert summary["losses"] == 1
        assert summary["avg_confidence"] == pytest.approx(60.0)
        assert summary["avg_expected_value"] == pytest.approx(0.08)

    def test_summarize_closing_line_value(self):
        picks = [
            {"closing_line_value": 0.03},
            {"closing_line_value": -0.02},
            {"closing_line_value": 0.00},
            {},
        ]

        summary = summarize_closing_line_value(picks)

        assert summary["tracked"] == 3
        assert summary["avg_clv"] == pytest.approx(0.0033, abs=1e-4)
        assert summary["positive_rate"] == pytest.approx(1 / 3, abs=1e-4)
        assert summary["non_negative_rate"] == pytest.approx(2 / 3, abs=1e-4)

    def test_summarize_pick_breakdowns(self):
        picks = [
            {"type": "slop_lock", "market_type": "moneyline", "evaluated": True, "won": True, "decimal_odds": 2.1, "closing_line_value": 0.01},
            {"type": "total_lock", "market_type": "total", "evaluated": True, "won": False, "decimal_odds": 1.9, "closing_line_value": -0.02},
        ]

        breakdowns = summarize_pick_breakdowns(picks)

        assert breakdowns["type"]["slop_lock"]["evaluated"] == 1
        assert breakdowns["type"]["slop_lock"]["clv"]["avg_clv"] == pytest.approx(0.01)
        assert breakdowns["market_type"]["total"]["evaluated"] == 1
        assert breakdowns["market_type"]["moneyline"]["hit_rate"] == 1.0

    def test_build_threshold_guidance_handles_small_sample(self):
        guidance = build_threshold_guidance({
            "evaluated": 6,
            "roi": 0.12,
            "clv": {"tracked": 3, "avg_clv": 0.01},
        })

        assert guidance
        assert "Insufficient settled pick volume" in guidance[0]

    def test_summarize_prediction_history(self):
        predictions = [
            {
                "evaluated": True,
                "home_goals": 110,
                "away_goals": 100,
                "model_probs": {"home": 0.7, "away": 0.3},
            },
            {
                "evaluated": True,
                "home_goals": 95,
                "away_goals": 102,
                "model_probs": {"home": 0.6, "away": 0.4},
            },
        ]

        summary = summarize_prediction_history(predictions)

        assert summary["evaluated"] == 2
        assert summary["accuracy"] == pytest.approx(0.5)
        assert summary["avg_log_loss"] is not None
        assert summary["avg_brier"] is not None

    def test_build_backtest_report(self, tmp_path):
        data_dir = tmp_path / "data"
        nba_dir = data_dir / "nba"
        nba_dir.mkdir(parents=True)

        with open(nba_dir / "history.json", "w") as f:
            json.dump({
                "predictions": [
                    {
                        "evaluated": True,
                        "home_goals": 110,
                        "away_goals": 100,
                        "model_probs": {"home": 0.7, "away": 0.3},
                    }
                ]
            }, f)

        with open(nba_dir / "pick_history.json", "w") as f:
            json.dump({
                "picks": [
                    {"type": "slop_lock", "market_type": "moneyline", "evaluated": True, "won": True, "decimal_odds": 2.1, "closing_line_value": 0.02},
                    {"type": "total_lock", "market_type": "total", "evaluated": True, "won": False, "decimal_odds": 1.8, "closing_line_value": -0.03},
                ]
            }, f)

        report = build_backtest_report(str(data_dir), sports=["nba"])

        assert report["sports"]["nba"]["predictions"]["evaluated"] == 1
        assert report["sports"]["nba"]["picks"]["total_picks"] == 2
        assert report["sports"]["nba"]["picks"]["evaluated"] == 2
        assert report["sports"]["nba"]["picks"]["clv"]["tracked"] == 2
        assert "type" in report["sports"]["nba"]["picks"]["breakdowns"]
        assert report["sports"]["nba"]["picks"]["breakdowns"]["market_type"]["total"]["evaluated"] == 1
        assert report["sports"]["nba"]["threshold_guidance"]
        assert report["aggregate"]["picks"]["roi"] is not None
