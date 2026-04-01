"""Integration tests for the pipeline orchestrator."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
import csv
import json
import os

import pandas as pd
import pytest
from pipeline.config import SPORTS
from pipeline.run import (
    _apply_mlb_bullpen_availability_adjustment,
    _apply_mlb_bullpen_total_adjustment,
    _apply_nhl_injury_adjustment,
    _apply_nba_availability_adjustment,
    _apply_nba_availability_total_adjustment,
    _compute_mlb_bullpen_tax,
    _apply_mlb_lineup_adjustment,
    _apply_mlb_lineup_total_adjustment,
    _apply_mlb_weather_adjustment,
    _main,
    run_pipeline,
    run_sport_pipeline,
)

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


@pytest.fixture
def sample_mma_matches():
    """Minimal MMA fight history for testing."""
    base_date = datetime(2026, 1, 4)
    results = [
        ("Fighter A", "Fighter B", 1, 0),
        ("Fighter C", "Fighter D", 1, 0),
        ("Fighter A", "Fighter C", 1, 0),
        ("Fighter B", "Fighter D", 0, 1),
        ("Fighter A", "Fighter D", 1, 0),
        ("Fighter B", "Fighter C", 0, 1),
        ("Fighter A", "Fighter B", 1, 0),
        ("Fighter C", "Fighter D", 1, 0),
        ("Fighter A", "Fighter C", 1, 0),
        ("Fighter D", "Fighter B", 1, 0),
        ("Fighter A", "Fighter D", 1, 0),
        ("Fighter C", "Fighter B", 1, 0),
    ]
    fights = []
    for i, (home, away, hg, ag) in enumerate(results):
        fights.append({
            "game_id": str(2000 + i),
            "date": (base_date + timedelta(days=i * 7)).isoformat(),
            "home_team": home,
            "away_team": away,
            "home_goals": hg,
            "away_goals": ag,
        })
    return pd.DataFrame(fights)


@pytest.fixture
def sample_nhl_matches():
    """Minimal NHL history with shot/save context."""
    base_date = datetime(2026, 1, 1)
    rows = []
    for i in range(24):
        home_team = "Bruins" if i % 2 == 0 else "Canadiens"
        away_team = "Canadiens" if i % 2 == 0 else "Bruins"
        strong_home = home_team == "Bruins"
        rows.append({
            "game_id": str(2500 + i),
            "date": (base_date + timedelta(days=i)).strftime("%Y-%m-%d"),
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": 4 if strong_home else 2,
            "away_goals": 2 if strong_home else 4,
            "home_saves": 29 if strong_home else 24,
            "away_saves": 24 if strong_home else 29,
            "home_save_pct": 0.928 if strong_home else 0.896,
            "away_save_pct": 0.896 if strong_home else 0.928,
            "home_shots": 33 if strong_home else 27,
            "away_shots": 27 if strong_home else 33,
            "home_goalie": "Bruins Starter" if strong_home else "Canadiens Starter",
            "away_goalie": "Canadiens Starter" if strong_home else "Bruins Starter",
            "home_goalie_save_pct": 0.931 if strong_home else 0.891,
            "away_goalie_save_pct": 0.891 if strong_home else 0.931,
            "home_goalie_goals_allowed": 2 if strong_home else 4,
            "away_goalie_goals_allowed": 4 if strong_home else 2,
            "home_faceoff_pct": 0.57 if strong_home else 0.45,
            "away_faceoff_pct": 0.45 if strong_home else 0.57,
            "home_power_play_pct": 0.23 if strong_home else 0.12,
            "away_power_play_pct": 0.12 if strong_home else 0.23,
            "home_takeaways": 8 if strong_home else 4,
            "away_takeaways": 4 if strong_home else 8,
            "home_giveaways": 4 if strong_home else 7,
            "away_giveaways": 7 if strong_home else 4,
            "home_penalty_minutes": 6.0 if strong_home else 10.0,
            "away_penalty_minutes": 10.0 if strong_home else 6.0,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_mlb_matches():
    """Minimal MLB history with starter and bullpen context."""
    base_date = datetime(2026, 3, 1)
    teams = [("Aces", "Bruins"), ("Caps", "Dragons"), ("Aces", "Dragons"), ("Caps", "Bruins")]
    rows = []
    for i in range(12):
        home_team, away_team = teams[i % len(teams)]
        strong_home = home_team in {"Aces", "Caps"}
        strong_away = away_team in {"Aces", "Caps"}
        home_goals = 5 if strong_home else 2
        away_goals = 2 if strong_away else 5
        if strong_home and not strong_away:
            home_goals, away_goals = 5, 2
        elif strong_away and not strong_home:
            home_goals, away_goals = 2, 5
        rows.append({
            "game_id": str(3000 + i),
            "date": (base_date + timedelta(days=i)).strftime("%Y-%m-%d"),
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "home_pitcher": f"{home_team} Starter",
            "away_pitcher": f"{away_team} Starter",
            "home_pitcher_hand": "R" if strong_home else "L",
            "away_pitcher_hand": "R" if strong_away else "L",
            "home_pitcher_ip": 6.0 if strong_home else 5.0,
            "home_pitcher_runs_allowed": 2 if strong_home else 4,
            "home_pitcher_earned_runs": 2 if strong_home else 4,
            "home_pitcher_walks": 1 if strong_home else 3,
            "home_pitcher_strikeouts": 7 if strong_home else 4,
            "home_bullpen_ip": 3.0 if strong_home else 4.0,
            "home_bullpen_runs_allowed": 1 if strong_home else 3,
            "home_bullpen_earned_runs": 1 if strong_home else 3,
            "home_bullpen_walks": 1 if strong_home else 3,
            "home_bullpen_strikeouts": 4 if strong_home else 2,
            "away_pitcher_ip": 6.0 if strong_away else 5.0,
            "away_pitcher_runs_allowed": 2 if strong_away else 4,
            "away_pitcher_earned_runs": 2 if strong_away else 4,
            "away_pitcher_walks": 1 if strong_away else 3,
            "away_pitcher_strikeouts": 7 if strong_away else 4,
            "away_bullpen_ip": 3.0 if strong_away else 4.0,
            "away_bullpen_runs_allowed": 1 if strong_away else 3,
            "away_bullpen_earned_runs": 1 if strong_away else 3,
            "away_bullpen_walks": 1 if strong_away else 3,
            "away_bullpen_strikeouts": 4 if strong_away else 2,
        })
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
        sample_nba_matches, sample_nba_box_scores, tmp_path, monkeypatch
    ):
        mock_games.return_value = (sample_nba_matches, sample_nba_box_scores)
        mock_schedule.return_value = [
            {
                "home_team": "Lakers",
                "away_team": "Warriors",
                "date": _TODAY,
                "start_time": f"{_TODAY}T00:30:00Z",
                "home_availability_profile": {
                    "active_players": 15,
                    "injured_players": 0,
                    "injury_burden": 0.0,
                    "key_absence_score": 0.0,
                    "leader_absence_burden": 0.0,
                    "available_core_players": 12,
                },
                "away_availability_profile": {
                    "active_players": 14,
                    "injured_players": 2,
                    "injury_burden": 1.25,
                    "key_absence_score": 1.0,
                    "leader_absence_burden": 1.25,
                    "available_core_players": 9,
                },
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
                "total_line": 224.5,
                "over_odds": 1.91,
                "under_odds": 1.95,
            }
        ]
        monkeypatch.setitem(SPORTS["nba"], "totals_feature_min_games", 4)
        monkeypatch.setitem(SPORTS["nba"], "totals_edge_threshold", 0.0)
        monkeypatch.setitem(SPORTS["nba"], "totals_probability_floor", 0.5)
        monkeypatch.setitem(SPORTS["nba"], "totals_confidence_threshold", 0.0)

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
        assert match["start_time"] == f"{_TODAY}T00:30:00Z"
        assert match["home_availability_profile"]["available_core_players"] == 12
        assert "draw" not in match["model_probs"]
        assert set(match["model_probs"].keys()) == {"home", "away"}
        assert abs(sum(match["model_probs"].values()) - 1.0) < 0.01
        assert data["totals_matches"][0]["total_line"] == 224.5
        assert data["totals_matches"][0]["pick"] in {"over", "under"}
        assert len(data["totals_locks"]) == 1

        diagnostics = data["diagnostics"]
        assert diagnostics["fixtures_fetched"] == 1
        assert diagnostics["fixtures_in_window"] == 1
        assert diagnostics["odds_events_fetched"] == 1
        assert diagnostics["fixtures_with_odds"] == 1
        assert diagnostics["coverage_gap_examples"] == []
        assert diagnostics["matches_modeled"] >= 1

        # Season stats should not have draw fields
        stats = data["season_stats"]
        assert "draws" not in stats
        assert "draw_pct" not in stats

        assert "elo" in data["model_weights"]
        assert "efficiency" in data["model_weights"]
        assert "four_factors" in data["model_weights"]
        assert "results_features" in data["model_weights"]
        assert "recent_boxscore" in data["model_weights"]
        assert "nba_matchup" in data["model_weights"]
        assert "dixon_coles" not in data["model_weights"]

        with open(os.path.join(output_dir, "pick_history.json")) as f:
            pick_history = json.load(f)
        assert any(pick["type"] == "total_lock" for pick in pick_history["picks"])
        assert data["run_type"] == "manual"
        assert data["run_id"]
        assert data["snapshot_path"].endswith(".json")
        assert os.path.exists(os.path.join(tmp_path, data["snapshot_path"]))
        assert pick_history["run_id"] == data["run_id"]
        assert all(pick.get("snapshot_path") == data["snapshot_path"] for pick in pick_history["picks"])


class TestRunMMAPipeline:
    @patch("pipeline.run.fetch_odds")
    @patch("pipeline.run.fetch_mma_schedule")
    @patch("pipeline.run.fetch_mma_games")
    def test_produces_valid_mma_predictions(
        self, mock_games, mock_schedule, mock_odds, sample_mma_matches, tmp_path
    ):
        mock_games.return_value = (sample_mma_matches, None)
        mock_schedule.return_value = [
            {
                "home_team": "Fighter A",
                "away_team": "Fighter C",
                "date": _TODAY,
                "start_time": f"{_TODAY}T03:00:00Z",
                "neutral": True,
            }
        ]
        mock_odds.return_value = [
            {
                "home_team": "Fighter A",
                "away_team": "Fighter C",
                "commence_time": f"{_TODAY}T00:30:00Z",
                "home_odds": 1.85,
                "away_odds": 2.05,
            }
        ]

        output_dir = str(tmp_path / "mma")
        run_sport_pipeline("mma", output_dir=output_dir)

        predictions_path = os.path.join(output_dir, "predictions.json")
        assert os.path.exists(predictions_path)

        with open(predictions_path) as f:
            data = json.load(f)

        assert data["sport"] == "mma"
        assert data["outcomes"] == ["home", "away"]
        assert data["matches"][0]["start_time"] == f"{_TODAY}T03:00:00Z"
        assert data["diagnostics"]["fixtures_fetched"] == 1
        assert data["diagnostics"]["fixtures_with_odds"] == 1
        assert data["diagnostics"]["coverage_gap_examples"] == []
        assert "elo" in data["model_weights"]
        assert "results_features" in data["model_weights"]

    @patch("pipeline.run.fetch_odds")
    @patch("pipeline.run.fetch_mma_schedule")
    @patch("pipeline.run.fetch_mma_games")
    def test_matches_mma_odds_when_fighter_order_is_reversed(
        self, mock_games, mock_schedule, mock_odds, sample_mma_matches, tmp_path
    ):
        mock_games.return_value = (sample_mma_matches, None)
        mock_schedule.return_value = [
            {
                "home_team": "Fighter A",
                "away_team": "Fighter C",
                "date": _TODAY,
                "neutral": True,
            }
        ]
        mock_odds.return_value = [
            {
                "home_team": "Fighter C",
                "away_team": "Fighter A",
                "commence_time": f"{_TODAY}T00:30:00Z",
                "home_odds": 2.05,
                "away_odds": 1.85,
            }
        ]

        output_dir = str(tmp_path / "mma")
        run_sport_pipeline("mma", output_dir=output_dir)

        with open(os.path.join(output_dir, "predictions.json")) as f:
            data = json.load(f)

        match = data["matches"][0]
        assert match["best_odds"]
        assert match["american_odds"] is not None


class TestRunNHLPipeline:
    @patch("pipeline.run.fetch_odds")
    @patch("pipeline.run.fetch_nhl_schedule")
    @patch("pipeline.run.fetch_nhl_games")
    def test_produces_valid_nhl_predictions(
        self, mock_games, mock_schedule, mock_odds, sample_nhl_matches, tmp_path
    ):
        mock_games.return_value = (sample_nhl_matches, None)
        mock_schedule.return_value = [
            {
                "home_team": "Bruins",
                "away_team": "Canadiens",
                "date": _TODAY,
                "start_time": f"{_TODAY}T23:00:00Z",
                "home_goalie": "Bruins Starter",
                "away_goalie": "Canadiens Starter",
                "home_goalie_status": "confirmed",
                "away_goalie_status": "confirmed",
            }
        ]
        mock_odds.return_value = [
            {
                "home_team": "Bruins",
                "away_team": "Canadiens",
                "commence_time": f"{_TODAY}T23:00:00Z",
                "home_odds": 1.72,
                "draw_odds": 0.0,
                "away_odds": 2.2,
            }
        ]

        output_dir = str(tmp_path / "nhl")
        run_sport_pipeline("nhl", output_dir=output_dir)

        with open(os.path.join(output_dir, "predictions.json")) as f:
            data = json.load(f)

        assert data["sport"] == "nhl"
        assert data["outcomes"] == ["home", "away"]
        assert len(data["matches"]) == 1
        assert data["matches"][0]["pick"] in {"home", "away"}
        assert data["matches"][0]["home_goalie"] == "Bruins Starter"
        assert data["matches"][0]["away_goalie_status"] == "confirmed"
        assert "elo" in data["model_weights"]
        assert "results_features" in data["model_weights"]
        assert "nhl_matchup" in data["model_weights"]


class TestRunMLBPipeline:
    @patch("pipeline.run.fetch_odds")
    @patch("pipeline.run.fetch_mlb_schedule")
    @patch("pipeline.run.fetch_mlb_games")
    def test_includes_bullpen_model_in_mlb_ensemble(
        self, mock_games, mock_schedule, mock_odds, sample_mlb_matches, tmp_path, monkeypatch
    ):
        mock_games.return_value = (sample_mlb_matches, None)
        mock_schedule.return_value = [
            {
                "home_team": "Aces",
                "away_team": "Bruins",
                "date": _TODAY,
                "start_time": f"{_TODAY}T20:10:00Z",
                "home_pitcher": "Aces Starter",
                "home_pitcher_hand": "R",
                "away_pitcher": "Bruins Starter",
                "away_pitcher_hand": "L",
                "home_lineup_profile": {
                    "active_hitters": 13,
                    "available_hitters": 13,
                    "injured_hitters": 0,
                    "key_bat_absence_score": 0.0,
                    "leader_absence_burden": 0.0,
                    "left_handed_batters": 5,
                    "right_handed_batters": 6,
                    "switch_hitters": 2,
                    "lefty_share": 0.3846,
                    "righty_share": 0.4615,
                    "switch_share": 0.1538,
                },
                "away_lineup_profile": {
                    "active_hitters": 12,
                    "available_hitters": 11,
                    "injured_hitters": 1,
                    "key_bat_absence_score": 0.9,
                    "leader_absence_burden": 0.85,
                    "left_handed_batters": 2,
                    "right_handed_batters": 8,
                    "switch_hitters": 1,
                    "lefty_share": 0.1818,
                    "righty_share": 0.7273,
                    "switch_share": 0.0909,
                },
                "weather": {
                    "weather_exposed": True,
                    "temperature_f": 82.0,
                    "wind_mph": 14.0,
                    "precipitation_probability": 0,
                },
            }
        ]
        mock_odds.return_value = [
            {
                "home_team": "Aces",
                "away_team": "Bruins",
                "commence_time": f"{_TODAY}T20:10:00Z",
                "home_odds": 1.85,
                "away_odds": 2.05,
                "total_line": 8.5,
                "over_odds": 1.95,
                "under_odds": 1.91,
            }
        ]

        monkeypatch.setitem(SPORTS["mlb"], "bullpen_feature_min_games", 4)
        monkeypatch.setitem(SPORTS["mlb"], "pitcher_feature_min_games", 4)
        monkeypatch.setitem(SPORTS["mlb"], "run_environment_min_games", 4)
        monkeypatch.setitem(SPORTS["mlb"], "handedness_feature_min_games", 4)
        monkeypatch.setitem(SPORTS["mlb"], "totals_feature_min_games", 4)
        monkeypatch.setitem(SPORTS["mlb"], "results_feature_min_games", 50)

        output_dir = str(tmp_path / "mlb")
        run_sport_pipeline("mlb", output_dir=output_dir)

        with open(os.path.join(output_dir, "predictions.json")) as f:
            data = json.load(f)

        match = data["matches"][0]
        assert "bullpen_features" in match["individual_models"]
        assert "run_environment" in match["individual_models"]
        assert "handedness_features" in match["individual_models"]
        assert match["home_lineup_profile"]["active_hitters"] == 13
        assert match["weather"]["temperature_f"] == 82.0
        total_market = data["totals_matches"][0]
        assert total_market["total_line"] == 8.5
        assert total_market["home_lineup_profile"]["available_hitters"] == 13
        assert total_market["pick"] in {"over", "under"}
        assert "over" in total_market["edges"]


class TestMlbWeatherAdjustment:
    def test_hitter_friendly_weather_amplifies_run_environment_edge(self):
        adjusted = _apply_mlb_weather_adjustment(
            {"home": 0.54, "away": 0.46},
            {"home": 0.60, "away": 0.40},
            {
                "weather_exposed": True,
                "temperature_f": 82.0,
                "wind_mph": 15.0,
                "precipitation_probability": 0,
            },
            max_delta=0.02,
        )

        assert adjusted["home"] > 0.54
        assert pytest.approx(adjusted["home"] + adjusted["away"], abs=1e-9) == 1.0


class TestMlbLineupAdjustment:
    def test_healthy_platoon_friendly_home_lineup_gets_small_boost(self):
        adjusted = _apply_mlb_lineup_adjustment(
            {"home": 0.54, "away": 0.46},
            {
                "active_hitters": 13,
                "available_hitters": 13,
                "injured_hitters": 0,
                "key_bat_absence_score": 0.0,
                "leader_absence_burden": 0.0,
                "lefty_share": 0.42,
                "righty_share": 0.42,
                "switch_share": 0.16,
            },
            {
                "active_hitters": 12,
                "available_hitters": 10,
                "injured_hitters": 2,
                "key_bat_absence_score": 1.0,
                "leader_absence_burden": 0.9,
                "lefty_share": 0.10,
                "righty_share": 0.80,
                "switch_share": 0.10,
            },
            home_pitcher_hand="R",
            away_pitcher_hand="R",
            max_delta=0.015,
        )

        assert adjusted["home"] > 0.54
        assert pytest.approx(adjusted["home"] + adjusted["away"], abs=1e-9) == 1.0

    def test_confirmed_lineup_absence_penalizes_missing_top_bats(self):
        adjusted = _apply_mlb_lineup_adjustment(
            {"home": 0.52, "away": 0.48},
            {
                "active_hitters": 13,
                "available_hitters": 13,
                "injured_hitters": 0,
                "key_bat_absence_score": 0.0,
                "leader_absence_burden": 0.0,
                "lefty_share": 0.4,
                "righty_share": 0.4,
                "switch_share": 0.2,
                "confirmed_lineup": True,
                "confirmed_hitters": 9,
                "confirmed_top_order_score": 1.0,
                "confirmed_lefty_share": 0.44,
                "confirmed_righty_share": 0.44,
                "confirmed_switch_share": 0.11,
                "confirmed_leader_absence_burden": 0.0,
            },
            {
                "active_hitters": 13,
                "available_hitters": 13,
                "injured_hitters": 0,
                "key_bat_absence_score": 0.0,
                "leader_absence_burden": 0.0,
                "lefty_share": 0.4,
                "righty_share": 0.4,
                "switch_share": 0.2,
                "confirmed_lineup": True,
                "confirmed_hitters": 8,
                "confirmed_top_order_score": 0.73,
                "confirmed_lefty_share": 0.25,
                "confirmed_righty_share": 0.75,
                "confirmed_switch_share": 0.0,
                "confirmed_leader_absence_burden": 1.3,
            },
            "R",
            "L",
            max_delta=0.015,
        )

        assert adjusted["home"] > 0.52


class TestMlbBullpenAdjustments:
    def test_recent_heavy_bullpen_usage_creates_tax(self):
        matches = pd.DataFrame([
            {
                "date": "2026-03-25",
                "home_team": "Aces",
                "away_team": "Bruins",
                "home_bullpen_ip": 4.0,
                "home_bullpen_earned_runs": 1,
                "away_bullpen_ip": 2.0,
                "away_bullpen_earned_runs": 0,
            },
            {
                "date": "2026-03-26",
                "home_team": "Aces",
                "away_team": "Bruins",
                "home_bullpen_ip": 3.1,
                "home_bullpen_earned_runs": 2,
                "away_bullpen_ip": 1.2,
                "away_bullpen_earned_runs": 0,
            },
        ])

        tax = _compute_mlb_bullpen_tax("Aces", "2026-03-28", matches)
        assert tax > 0.0

    def test_bullpen_tax_shifts_side_and_total(self):
        side = _apply_mlb_bullpen_availability_adjustment(
            {"home": 0.51, "away": 0.49},
            home_tax=0.1,
            away_tax=0.8,
            max_delta=0.012,
        )
        total = _apply_mlb_bullpen_total_adjustment(
            8.3,
            home_tax=0.6,
            away_tax=0.7,
            max_runs_delta=0.3,
        )

        assert side["home"] > 0.51
        assert total > 8.3

    def test_lineup_adjustment_can_raise_total_for_live_bats(self):
        adjusted = _apply_mlb_lineup_total_adjustment(
            8.2,
            {
                "active_hitters": 13,
                "available_hitters": 13,
                "injured_hitters": 0,
                "key_bat_absence_score": 0.0,
                "leader_absence_burden": 0.0,
                "lefty_share": 0.40,
                "righty_share": 0.44,
                "switch_share": 0.16,
            },
            {
                "active_hitters": 13,
                "available_hitters": 12,
                "injured_hitters": 0,
                "key_bat_absence_score": 0.0,
                "leader_absence_burden": 0.0,
                "lefty_share": 0.18,
                "righty_share": 0.64,
                "switch_share": 0.18,
            },
            home_pitcher_hand="L",
            away_pitcher_hand="R",
            max_runs_delta=0.35,
        )

        assert adjusted > 8.2

    def test_missing_middle_of_order_bats_drag_side_probability(self):
        adjusted = _apply_mlb_lineup_adjustment(
            {"home": 0.5, "away": 0.5},
            {
                "active_hitters": 13,
                "available_hitters": 11,
                "injured_hitters": 1,
                "key_bat_absence_score": 1.0,
                "leader_absence_burden": 0.95,
                "lefty_share": 0.36,
                "righty_share": 0.46,
                "switch_share": 0.18,
            },
            {
                "active_hitters": 13,
                "available_hitters": 13,
                "injured_hitters": 0,
                "key_bat_absence_score": 0.0,
                "leader_absence_burden": 0.0,
                "lefty_share": 0.36,
                "righty_share": 0.46,
                "switch_share": 0.18,
            },
            home_pitcher_hand="R",
            away_pitcher_hand="R",
            max_delta=0.015,
        )

        assert adjusted["home"] < 0.5


class TestNbaAvailabilityAdjustment:
    def test_missing_key_players_pushes_probability_away_from_shorthanded_team(self):
        adjusted = _apply_nba_availability_adjustment(
            {"home": 0.54, "away": 0.46},
            {
                "active_players": 15,
                "injured_players": 0,
                "questionable_players": 0,
                "doubtful_players": 0,
                "injury_burden": 0.0,
                "uncertainty_burden": 0.0,
                "key_absence_score": 0.0,
                "leader_absence_burden": 0.0,
                "leader_uncertainty_burden": 0.0,
                "available_core_players": 12,
            },
            {
                "active_players": 14,
                "injured_players": 2,
                "questionable_players": 1,
                "doubtful_players": 1,
                "injury_burden": 1.25,
                "uncertainty_burden": 0.75,
                "key_absence_score": 1.0,
                "leader_absence_burden": 1.25,
                "leader_uncertainty_burden": 0.75,
                "available_core_players": 9,
            },
            start_time=(datetime.now(timezone.utc) + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            max_delta=0.02,
        )

        assert adjusted["home"] > 0.54
        assert pytest.approx(adjusted["home"] + adjusted["away"], abs=1e-9) == 1.0

    def test_missing_key_scorers_can_pull_total_down(self):
        adjusted_total = _apply_nba_availability_total_adjustment(
            228.5,
            {
                "active_players": 15,
                "injured_players": 0,
                "questionable_players": 0,
                "doubtful_players": 0,
                "injury_burden": 0.0,
                "uncertainty_burden": 0.0,
                "key_absence_score": 0.0,
                "leader_absence_burden": 0.0,
                "leader_uncertainty_burden": 0.0,
                "available_core_players": 12,
            },
            {
                "active_players": 13,
                "injured_players": 2,
                "questionable_players": 1,
                "doubtful_players": 1,
                "injury_burden": 1.25,
                "uncertainty_burden": 0.75,
                "key_absence_score": 1.0,
                "leader_absence_burden": 1.4,
                "leader_uncertainty_burden": 0.75,
                "available_core_players": 8,
            },
            start_time=(datetime.now(timezone.utc) + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            max_points_delta=2.2,
        )

        assert adjusted_total < 228.5

    def test_questionable_star_near_tip_matters_more_than_early_day(self):
        profile = {
            "active_players": 14,
            "injured_players": 1,
            "questionable_players": 1,
            "doubtful_players": 0,
            "injury_burden": 0.35,
            "uncertainty_burden": 0.35,
            "key_absence_score": 0.35,
            "leader_absence_burden": 0.35,
            "leader_uncertainty_burden": 0.35,
            "available_core_players": 10,
        }
        early = _apply_nba_availability_adjustment(
            {"home": 0.5, "away": 0.5},
            profile,
            None,
            start_time=(datetime.now(timezone.utc) + timedelta(hours=16)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            max_delta=0.02,
        )
        late = _apply_nba_availability_adjustment(
            {"home": 0.5, "away": 0.5},
            profile,
            None,
            start_time=(datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            max_delta=0.02,
        )

        assert late["home"] < early["home"]

    def test_event_specific_absence_news_penalizes_team(self):
        adjusted = _apply_nba_availability_adjustment(
            {"home": 0.53, "away": 0.47},
            {
                "active_players": 12,
                "available_core_players": 10,
                "injury_burden": 0.0,
                "uncertainty_burden": 0.0,
                "key_absence_score": 0.0,
                "leader_absence_burden": 0.0,
                "leader_uncertainty_burden": 0.0,
                "event_injury_burden": 1.0,
                "event_key_absence_score": 1.0,
                "event_leader_absence_burden": 1.0,
            },
            {
                "active_players": 12,
                "available_core_players": 10,
                "injury_burden": 0.0,
                "uncertainty_burden": 0.0,
                "key_absence_score": 0.0,
                "leader_absence_burden": 0.0,
                "leader_uncertainty_burden": 0.0,
            },
            start_time=(datetime.now(timezone.utc) + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            max_delta=0.02,
        )

        assert adjusted["home"] < 0.53


class TestNhlInjuryAdjustment:
    def test_missing_key_skater_penalizes_side(self):
        adjusted = _apply_nhl_injury_adjustment(
            {"home": 0.52, "away": 0.48},
            {"injury_burden": 1.0, "key_absence_score": 1.0, "leader_absence_burden": 1.0},
            {"injury_burden": 0.0, "key_absence_score": 0.0, "leader_absence_burden": 0.0},
            max_delta=0.01,
        )

        assert adjusted["home"] < 0.52


# ---------------------------------------------------------------------------
# Multi-sport orchestrator
# ---------------------------------------------------------------------------

class TestRunPipeline:
    @patch("pipeline.run.fetch_odds")
    @patch("pipeline.run.fetch_mma_schedule")
    @patch("pipeline.run.fetch_mma_games")
    @patch("pipeline.run.fetch_mlb_schedule")
    @patch("pipeline.run.fetch_mlb_games")
    @patch("pipeline.run.fetch_nhl_schedule")
    @patch("pipeline.run.fetch_nhl_games")
    @patch("pipeline.run.fetch_ncaam_schedule")
    @patch("pipeline.run.fetch_ncaam_games")
    @patch("pipeline.run.fetch_nba_espn_schedule")
    @patch("pipeline.run.fetch_nba_espn_games")
    def test_produces_per_sport_files_and_manifest(
        self,
        mock_nba_games,
        mock_nba_schedule,
        mock_ncaam_games,
        mock_ncaam_schedule,
        mock_nhl_games,
        mock_nhl_schedule,
        mock_mlb_games,
        mock_mlb_schedule,
        mock_mma_games,
        mock_mma_schedule,
        mock_odds,
        sample_nba_matches, sample_nba_box_scores, sample_nhl_matches,
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
        mock_nhl_games.return_value = (sample_nhl_matches, None)
        mock_nhl_schedule.return_value = [
            {
                "home_team": "Bruins",
                "away_team": "Canadiens",
                "date": "2026-02-19",
            }
        ]
        mock_mlb_games.return_value = (pd.DataFrame(columns=["game_id", "date", "home_team", "away_team", "home_goals", "away_goals"]), None)
        mock_mlb_schedule.return_value = []
        mock_mma_games.return_value = (pd.DataFrame(columns=["game_id", "date", "home_team", "away_team", "home_goals", "away_goals"]), None)
        mock_mma_schedule.return_value = []
        mock_odds.return_value = []

        output_dir = str(tmp_path)
        manifest = run_pipeline(output_dir=output_dir)

        # Manifest
        manifest_path = os.path.join(output_dir, "manifest.json")
        dashboard_path = os.path.join(output_dir, "dashboard.json")
        assert os.path.exists(manifest_path)
        assert os.path.exists(dashboard_path)
        assert "nba" in manifest["sports"]
        assert "nhl" in manifest["sports"]
        assert "ncaam" in manifest["sports"]
        assert manifest["sports"]["nba"]["status"] == "ok"
        assert manifest["sports"]["nhl"]["status"] == "ok"
        assert manifest["sports"]["ncaam"]["status"] == "ok"
        assert "diagnostics" in manifest["sports"]["nba"]
        assert "diagnostics" in manifest["sports"]["nhl"]
        assert "diagnostics" in manifest["sports"]["ncaam"]

        with open(dashboard_path) as f:
            dashboard = json.load(f)
        assert "aggregate" in dashboard
        assert "sports" in dashboard
        assert dashboard["aggregate"]["slate"]["modeled"] >= 0

        # Per-sport prediction files
        assert os.path.exists(os.path.join(output_dir, "nba", "predictions.json"))
        assert os.path.exists(os.path.join(output_dir, "nhl", "predictions.json"))
        assert os.path.exists(os.path.join(output_dir, "ncaam", "predictions.json"))


class TestRunCli:
    @patch("pipeline.run.run_sport_pipeline")
    def test_main_supports_single_sport(self, mock_run_sport_pipeline, tmp_path):
        exit_code = _main(["--sport", "mlb", "--output-dir", str(tmp_path / "mlb")])

        assert exit_code == 0
        mock_run_sport_pipeline.assert_called_once_with("mlb", output_dir=str(tmp_path / "mlb"))


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


class TestOddsTracking:
    def test_append_odds_snapshot_log_dedupes_same_market_state(self, tmp_path):
        from pipeline.run import _append_odds_snapshot_log

        path = str(tmp_path / "tracking" / "odds_history.csv")
        row = {
            "logged_at": "2026-03-29T12:00:00Z",
            "sport": "nba",
            "market_type": "moneyline",
            "home_team": "Lakers",
            "away_team": "Celtics",
            "match_date": "2026-03-29",
            "start_time": "2026-03-29T23:00:00Z",
            "outcome": "home",
            "total_line": "",
            "decimal_odds": 2.1,
            "american_odds": 110,
            "implied_prob": 0.5,
            "market_implied_prob": 0.52,
        }

        _append_odds_snapshot_log(path, [row, row])

        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["outcome"] == "home"

    def test_build_odds_snapshot_rows_skips_invalid_even_money_placeholders(self):
        from pipeline.run import _build_odds_snapshot_rows

        rows = _build_odds_snapshot_rows("ncaam", [{
            "home_team": "Duke",
            "away_team": "UConn",
            "commence_time": "2026-03-29T23:00:00Z",
            "home_odds": 1.91,
            "away_odds": 1.0,
        }])

        assert len(rows) == 1
        assert rows[0]["outcome"] == "home"


class TestSlimegrinderSelection:
    def test_skips_outcomes_with_invalid_american_odds(self):
        from pipeline.run import _compute_slimegrinder

        picks = _compute_slimegrinder(
            [{
                "home_team": "Duke",
                "away_team": "UConn",
                "date": "2026-03-29",
                "edges": {
                    "home": {
                        "american_odds": None,
                        "implied_prob": 0.5,
                        "edge": 0.03,
                        "expected_value": 0.02,
                        "model_prob": 0.53,
                        "decimal_odds": 1.0,
                        "confidence_score": 70,
                    }
                },
            }],
            ["home", "away"],
            min_expected_value=0.0,
            confidence_floor=65.0,
        )

        assert picks == []

    def test_apply_latest_market_snapshots_sets_moneyline_and_total_clv(self):
        from pipeline.run import _apply_latest_market_snapshots

        picks = [
            {
                "market_type": "moneyline",
                "home_team": "Lakers",
                "away_team": "Celtics",
                "match_date": "2026-03-29",
                "pick": "home",
                "market_implied_prob": 0.48,
            },
            {
                "market_type": "total",
                "home_team": "Dodgers",
                "away_team": "Padres",
                "match_date": "2026-03-29",
                "pick": "over",
                "total_line": 8.5,
            },
        ]
        snapshots = {
            ("moneyline", "Lakers", "Celtics", "2026-03-29", "home"): {
                "logged_at": "2026-03-29T17:00:00Z",
                "decimal_odds": "1.95",
                "american_odds": "-105",
                "implied_prob": "0.512",
                "market_implied_prob": "0.525",
            },
            ("total", "Dodgers", "Padres", "2026-03-29", "over"): {
                "logged_at": "2026-03-29T17:00:00Z",
                "decimal_odds": "1.91",
                "american_odds": "-110",
                "implied_prob": "0.5",
                "market_implied_prob": "0.5236",
                "total_line": "9.0",
            },
        }

        _apply_latest_market_snapshots(picks, snapshots)

        assert picks[0]["closing_american_odds"] == -105
        assert picks[0]["closing_line_value"] == pytest.approx(0.032, abs=1e-6)
        assert picks[1]["closing_total_line"] == 9.0
        assert picks[1]["closing_line_value"] == pytest.approx(0.5, abs=1e-6)
