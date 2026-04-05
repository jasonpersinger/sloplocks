"""Tests for the backtesting and accuracy-tracking utilities."""

import csv
import json
import math
import os
import pandas as pd
import pytest

from pipeline.backtest import (
    build_dashboard_data,
    build_backtest_report,
    build_pick_decision_replay_report,
    build_raw_walkforward_report,
    build_snapshot_replay_report,
    build_walkforward_report,
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
    run_raw_walkforward_for_sport,
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
        assert report["decision_replay"]["aggregate"]["logged_picks"] == 0

    def test_build_dashboard_data(self, tmp_path):
        data_dir = tmp_path / "data"
        nba_dir = data_dir / "nba"
        nhl_dir = data_dir / "nhl"
        nba_dir.mkdir(parents=True)
        nhl_dir.mkdir(parents=True)

        with open(data_dir / "manifest.json", "w") as f:
            json.dump({
                "updated_at": "2026-03-29T18:25:18Z",
                "sports": {
                    "nba": {"status": "ok", "diagnostics": {"matches_modeled": 4, "fixtures_in_window": 4, "fixtures_with_odds": 4, "matches_with_positive_ev": 2, "lock_eligible_matches": 1, "slop_locks_posted": 1}},
                    "nhl": {"status": "ok", "diagnostics": {"matches_modeled": 3, "fixtures_in_window": 3, "fixtures_with_odds": 2, "matches_with_positive_ev": 1, "lock_eligible_matches": 1, "slop_locks_posted": 1}},
                },
            }, f)

        with open(nba_dir / "history.json", "w") as f:
            json.dump({"predictions": []}, f)
        with open(nhl_dir / "history.json", "w") as f:
            json.dump({"predictions": []}, f)

        with open(nba_dir / "pick_history.json", "w") as f:
            json.dump({
                "picks": [
                    {"pick_date": "2026-03-29", "type": "slop_lock", "market_type": "moneyline", "evaluated": True, "won": True, "decimal_odds": 2.1, "closing_line_value": 0.02},
                    {"pick_date": "2026-03-23", "type": "total_lock", "market_type": "total", "evaluated": True, "won": False, "decimal_odds": 1.8, "closing_line_value": -0.03},
                ]
            }, f)
        with open(nhl_dir / "pick_history.json", "w") as f:
            json.dump({
                "picks": [
                    {"pick_date": "2026-03-29", "type": "slop_lock", "market_type": "moneyline", "evaluated": True, "won": True, "decimal_odds": 1.9, "closing_line_value": 0.01},
                ]
            }, f)

        dashboard = build_dashboard_data(str(data_dir), sports=["nba", "nhl"], as_of="2026-03-29")

        assert dashboard["aggregate"]["record"]["record"] == "2-1"
        assert dashboard["aggregate"]["slate"]["modeled"] == 7
        assert dashboard["windows"]["7d"]["evaluated"] == 3
        assert dashboard["windows"]["30d"]["evaluated"] == 3
        assert dashboard["sports"][0]["current"]["summary"] is None or "summary" in dashboard["sports"][0]["current"]
        assert dashboard["leaders"]["best_roi_sport"]["sport"] in {"nba", "nhl"}
        assert dashboard["recommended_actions"]
        assert dashboard["insights"]
        assert "walkforward" in dashboard
        assert "decision_replay" in dashboard
        assert "calibration" in dashboard["walkforward"]
        assert dashboard["snapshot_replay"]["aggregate"]["snapshots"] == 0

    def test_build_snapshot_replay_report(self, tmp_path):
        data_dir = tmp_path / "data"
        snapshot_dir = data_dir / "tracking" / "snapshots" / "2026-03-29" / "nba"
        snapshot_dir.mkdir(parents=True)

        payload = {
            "sport": "nba",
            "run_id": "daily-20260329T120000000000Z",
            "run_type": "daily",
            "snapshot_timestamp": "2026-03-29T12:00:00Z",
            "outcomes": ["home", "away"],
            "selection_config": {
                "outcomes": ["home", "away"],
                "slop_locks": {
                    "min_expected_value": 0.0,
                    "edge_floor": 0.03,
                    "probability_floor": 0.45,
                    "additional_confidence_floor": 65.0,
                    "confidence_dropoff": 0.0,
                    "max_picks": 3,
                },
                "longslop": {"min_expected_value": 0.0, "confidence_floor": 65.0},
                "totals_locks": {
                    "min_expected_value": 0.0,
                    "edge_floor": 0.02,
                    "probability_floor": 0.53,
                    "confidence_floor": 54.0,
                    "max_picks": 3,
                },
            },
            "records": {
                "matches": [
                    {
                        "home_team": "Lakers",
                        "away_team": "Warriors",
                        "date": "2026-03-29",
                        "edges": {
                            "home": {
                                "edge": 0.07,
                                "model_prob": 0.6,
                                "expected_value": 0.08,
                                "confidence_score": 70.0,
                            }
                        },
                    }
                ],
                "totals_matches": [
                    {
                        "home_team": "Lakers",
                        "away_team": "Warriors",
                        "date": "2026-03-29",
                        "total_line": 224.5,
                        "edges": {
                            "over": {
                                "edge": 0.03,
                                "model_prob": 0.56,
                                "expected_value": 0.04,
                                "confidence_score": 58.0,
                            }
                        },
                    }
                ],
            },
            "outputs": {
                "slop_locks": [
                    {
                        "market_type": "moneyline",
                        "home_team": "Lakers",
                        "away_team": "Warriors",
                        "date": "2026-03-29",
                        "pick": "home",
                    }
                ],
                "totals_locks": [
                    {
                        "market_type": "total",
                        "home_team": "Lakers",
                        "away_team": "Warriors",
                        "date": "2026-03-29",
                        "pick": "over",
                        "total_line": 224.5,
                    }
                ],
                "longslop": None,
            },
        }

        with open(snapshot_dir / "daily-20260329T120000000000Z.json", "w") as f:
            json.dump(payload, f)

        report = build_snapshot_replay_report(str(data_dir), sports=["nba"])

        assert report["aggregate"]["snapshots"] == 1
        assert report["aggregate"]["exact_matches"] == 1
        assert report["aggregate"]["exact_match_rate"] == pytest.approx(1.0)
        assert report["sports"]["nba"]["snapshots"] == 1

    def test_build_pick_decision_replay_report(self, tmp_path):
        data_dir = tmp_path / "data"
        tracking_dir = data_dir / "tracking"
        tracking_dir.mkdir(parents=True)

        with open(tracking_dir / "pick_decisions.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "logged_at",
                    "sport",
                    "pick_type",
                    "market_type",
                    "home_team",
                    "away_team",
                    "match_date",
                    "pick",
                    "model_prob",
                    "market_implied_prob",
                    "expected_value",
                    "confidence_score",
                    "american_odds",
                    "decimal_odds",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "logged_at": "2026-03-29T12:00:00Z",
                "sport": "nba",
                "pick_type": "slop_lock",
                "market_type": "moneyline",
                "home_team": "Lakers",
                "away_team": "Warriors",
                "match_date": "2026-03-29",
                "pick": "home",
                "model_prob": 0.61,
                "market_implied_prob": 0.48,
                "expected_value": 0.07,
                "confidence_score": 72,
                "american_odds": 105,
                "decimal_odds": 2.05,
            })

        with open(tracking_dir / "results_log.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "logged_at",
                    "sport",
                    "entry_type",
                    "market_type",
                    "home_team",
                    "away_team",
                    "match_date",
                    "pick",
                    "actual",
                    "won",
                    "push",
                    "decimal_odds",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "logged_at": "2026-03-30T12:00:00Z",
                "sport": "nba",
                "entry_type": "slop_lock",
                "market_type": "moneyline",
                "home_team": "Lakers",
                "away_team": "Warriors",
                "match_date": "2026-03-29",
                "pick": "home",
                "actual": "home",
                "won": "true",
                "push": "false",
                "decimal_odds": 2.05,
            })

        with open(tracking_dir / "odds_history.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "logged_at",
                    "sport",
                    "market_type",
                    "home_team",
                    "away_team",
                    "match_date",
                    "start_time",
                    "outcome",
                    "total_line",
                    "decimal_odds",
                    "american_odds",
                    "implied_prob",
                    "market_implied_prob",
                    "market_source",
                    "market_books",
                    "hold",
                    "market_snapshot_json",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "logged_at": "2026-03-29T20:00:00Z",
                "sport": "nba",
                "market_type": "moneyline",
                "home_team": "Lakers",
                "away_team": "Warriors",
                "match_date": "2026-03-29",
                "start_time": "2026-03-29T23:00:00Z",
                "outcome": "home",
                "total_line": "",
                "decimal_odds": "1.95",
                "american_odds": "-105",
                "implied_prob": "0.512",
                "market_implied_prob": "0.525",
                "market_source": "median_complete_book_no_vig",
                "market_books": "8",
                "hold": "0.0234",
                "market_snapshot_json": "{\"execution_prices\":{\"home\":1.95,\"away\":1.9}}",
            })

        report = build_pick_decision_replay_report(str(data_dir), sports=["nba"])

        assert report["aggregate"]["logged_picks"] == 1
        assert report["aggregate"]["settled_picks"]["evaluated"] == 1
        assert report["aggregate"]["settled_picks"]["wins"] == 1
        assert report["aggregate"]["settled_picks"]["clv"]["tracked"] == 1
        assert report["aggregate"]["settled_picks"]["clv"]["avg_clv"] == pytest.approx(0.032, abs=1e-6)
        assert report["sports"]["nba"]["settled_picks"]["breakdowns"]["type"]["slop_lock"]["evaluated"] == 1

    def test_build_walkforward_report(self, tmp_path):
        data_dir = tmp_path / "data"
        tracking_dir = data_dir / "tracking"
        tracking_dir.mkdir(parents=True)

        with open(tracking_dir / "results_log.csv", "w") as f:
            f.write(
                "logged_at,sport,entry_type,home_team,away_team,match_date,pick,actual,won,model_prob,home_prob,away_prob,draw_prob,implied_prob,market_implied_prob,edge,expected_value,american_odds,decimal_odds,confidence_score,kelly_fraction,fractional_kelly\n"
                "2026-03-28T10:00:00Z,nba,prediction,Celtics,Hawks,2026-03-28,home,home,true,0.65,0.65,0.35,,,,0.03,,-120,,58,,\n"
                "2026-03-28T10:00:00Z,nba,slop_lock,Celtics,Hawks,2026-03-28,home,home,true,0.65,,,,0.54,,0.11,0.08,-120,1.83,58,0.02,0.005\n"
                "2026-03-29T10:00:00Z,nhl,prediction,Rangers,Panthers,2026-03-29,away,home,false,0.58,0.42,0.58,,,,0.01,,135,,41,,\n"
                "2026-03-29T10:00:00Z,nhl,total_lock,Rangers,Panthers,2026-03-29,under,under,true,0.56,,,,0.51,,0.05,0.06,-102,1.98,55,0.03,0.007\n"
            )

        report = build_walkforward_report(str(data_dir), sports=["nba", "nhl"], as_of="2026-03-29")

        assert report["aggregate"]["predictions"]["evaluated"] == 2
        assert report["aggregate"]["predictions"]["accuracy"] == pytest.approx(0.5)
        assert report["aggregate"]["picks"]["evaluated"] == 2
        assert report["aggregate"]["picks"]["record"] == "2-0"
        assert len(report["daily"]) == 2
        assert report["daily"][0]["date"] == "2026-03-28"
        assert report["daily"][1]["cumulative_picks"]["evaluated"] == 2
        assert report["sports"]["nba"]["predictions"]["evaluated"] == 1
        assert report["sports"]["nhl"]["picks"]["wins"] == 1
        assert report["aggregate"]["calibration"]
        assert report["aggregate"]["calibration"][0]["sample"] >= 1

    def test_build_walkforward_report_skips_malformed_legacy_rows(self, tmp_path):
        data_dir = tmp_path / "data"
        tracking_dir = data_dir / "tracking"
        tracking_dir.mkdir(parents=True)

        with open(tracking_dir / "results_log.csv", "w") as f:
            f.write(
                "logged_at,sport,entry_type,home_team,away_team,match_date,pick,actual,won,model_prob,home_prob,away_prob,draw_prob,implied_prob,market_implied_prob,edge,expected_value,american_odds,decimal_odds,confidence_score,kelly_fraction,fractional_kelly\n"
                "2026-03-29T10:00:00Z,mlb,slop_lock,moneyline,Reds,Red Sox,2026-03-29,home,home,true,false,0.6026,,,,0.5,0.5128,0.1026,0.1751,-105,1.95\n"
                "2026-03-29T10:00:00Z,nba,prediction,Celtics,Hawks,2026-03-29,home,home,true,0.65,0.65,0.35,,,,0.03,,-120,,58,,\n"
            )

        report = build_walkforward_report(str(data_dir), sports=["mlb", "nba"], as_of="2026-03-29")

        assert report["aggregate"]["predictions"]["evaluated"] == 1
        assert report["aggregate"]["picks"]["evaluated"] == 0

    def test_run_raw_walkforward_for_sport(self):
        matches = pd.DataFrame([
            {"date": "2026-01-01", "home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 0},
            {"date": "2026-01-02", "home_team": "A", "away_team": "C", "home_goals": 1, "away_goals": 0},
            {"date": "2026-01-03", "home_team": "B", "away_team": "C", "home_goals": 0, "away_goals": 1},
            {"date": "2026-01-04", "home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 0},
        ])

        report = run_raw_walkforward_for_sport(
            "nhl",
            matches,
            box_scores_df=None,
            min_training_games=2,
            model_names=["elo"],
        )

        assert report["dates_evaluated"] == 2
        assert report["predictions"]["evaluated"] == 2
        assert "elo" in report["models"]
        assert report["daily"][-1]["cumulative_predictions"]["evaluated"] == 2

    def test_build_raw_walkforward_report(self, monkeypatch):
        matches = pd.DataFrame([
            {"date": "2026-01-01", "home_team": "A", "away_team": "B", "home_goals": 110, "away_goals": 100},
            {"date": "2026-01-02", "home_team": "A", "away_team": "C", "home_goals": 108, "away_goals": 101},
            {"date": "2026-01-03", "home_team": "B", "away_team": "C", "home_goals": 99, "away_goals": 105},
            {"date": "2026-01-04", "home_team": "A", "away_team": "B", "home_goals": 111, "away_goals": 102},
        ])
        monkeypatch.setattr(
            "pipeline.backtest._load_raw_walkforward_inputs",
            lambda sport_key, data_dir="data": (matches, None),
        )

        report = build_raw_walkforward_report(
            data_dir="data",
            sports=["nba"],
            max_days=4,
            min_training_games=2,
        )

        assert report["aggregate"]["evaluated"] == 2
        assert "nba" in report["sports"]
        assert report["sports"]["nba"]["predictions"]["evaluated"] == 2
