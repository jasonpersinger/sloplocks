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
    _build_pipeline_diagnostics,
    _compute_slop_locks,
    _compute_mlb_bullpen_tax,
    _apply_mlb_lineup_adjustment,
    _apply_mlb_lineup_total_adjustment,
    _apply_mlb_weather_adjustment,
    _passes_pick_gate,
    _main,
    run_pipeline,
    run_sport_pipeline,
    validate_publishable_picks,
)

_TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")


def test_compute_slop_locks_includes_configured_value_dog_lane():
    records = [
        {
            "home_team": "Yankees",
            "away_team": "Rays",
            "date": "2026-05-22",
            "edges": {
                "away": {
                    "edge": 0.061,
                    "model_prob": 0.53,
                    "expected_value": 0.13,
                    "american_odds": 155,
                    "decimal_odds": 2.55,
                    "confidence_score": 56.0,
                    "implied_prob": 0.38,
                    "market_implied_prob": 0.39,
                },
                "home": {
                    "edge": -0.061,
                    "model_prob": 0.56,
                    "expected_value": -0.08,
                    "american_odds": -170,
                    "decimal_odds": 1.59,
                    "confidence_score": 48.0,
                },
            },
        }
    ]
    lanes = {
        "value_dog": {
            "enabled": True,
            "edge_floor": 0.04,
            "probability_floor": 0.35,
            "min_expected_value": 0.05,
            "american_odds_min": 120,
            "american_odds_max": 500,
        }
    }

    picks = _compute_slop_locks(
        records,
        ["home", "away"],
        edge_floor=0.025,
        probability_floor=0.55,
        additional_confidence_floor=52,
        max_picks=3,
        lane_configs=lanes,
    )

    assert len(picks) == 1
    assert picks[0]["pick"] == "away"
    assert picks[0]["selection_lane"] == "value_dog"


def test_slop_lock_validation_uses_selection_lane_thresholds():
    issues = []
    pick = {
        "home_team": "Yankees",
        "away_team": "Rays",
        "date": "2026-05-22",
        "pick": "away",
        "selection_lane": "value_dog",
        "model_prob": 0.53,
        "edge": 0.061,
        "expected_value": 0.13,
        "american_odds": 155,
        "confidence_score": 56,
    }
    config = {
        "edge_floor": 0.025,
        "probability_floor": 0.55,
        "min_expected_value": 0.0,
        "lanes": {
            "value_dog": {
                "enabled": True,
                "edge_floor": 0.04,
                "probability_floor": 0.35,
                "min_expected_value": 0.05,
                "american_odds_min": 120,
                "american_odds_max": 500,
            }
        },
    }

    assert _passes_pick_gate(pick, "slop_lock", config, issues) is True
    assert issues == []


def test_publish_validation_removes_stale_moneyline_and_total_picks():
    now = datetime(2026, 6, 10, 16, 0, tzinfo=timezone.utc)
    selection_config = {
        "slop_locks": {
            "edge_floor": 0.02,
            "probability_floor": 0.52,
            "min_expected_value": 0.0,
            "lanes": {},
        },
        "totals_locks": {
            "edge_floor": 0.02,
            "probability_floor": 0.53,
            "confidence_floor": 54.0,
            "min_expected_value": 0.0,
        },
    }
    publication_guard = {
        "enforced": True,
        "allow_moneyline": True,
        "allow_totals": True,
        "allow_longslop": False,
        "allow_slimegrinder": False,
    }
    stale_moneyline = {
        "home_team": "Astros",
        "away_team": "Pirates",
        "date": "2026-06-04",
        "start_time": "2026-06-05T00:10Z",
        "pick": "home",
        "model_prob": 0.56,
        "edge": 0.03,
        "expected_value": 0.04,
        "american_odds": -105,
    }
    stale_total = {
        "home_team": "Twins",
        "away_team": "Royals",
        "date": "2026-06-04",
        "start_time": "2026-06-04T23:40Z",
        "pick": "over",
        "model_prob": 0.59,
        "edge": 0.08,
        "expected_value": 0.13,
        "american_odds": -109,
        "confidence_score": 72.6,
    }

    slop, totals, _, _, issues = validate_publishable_picks(
        sport_key="mlb",
        slop_locks=[stale_moneyline],
        totals_locks=[stale_total],
        longslop=None,
        slimegrinder=[],
        publication_guard=publication_guard,
        selection_config=selection_config,
        now=now,
    )

    assert slop == []
    assert totals == []
    assert [issue["reason"] for issue in issues] == ["stale_pick", "stale_pick"]


def test_pipeline_diagnostics_reports_gate_failures_and_lane_candidates():
    sport = {
        "min_expected_value": 0.0,
        "slop_lock_edge_threshold": 0.025,
        "slop_lock_probability_floor": 0.55,
        "slop_lock_lanes": {
            "value_dog": {
                "enabled": True,
                "edge_floor": 0.04,
                "probability_floor": 0.35,
                "min_expected_value": 0.05,
                "american_odds_min": 120,
                "american_odds_max": 500,
            }
        },
    }
    records = [
        {
            "home_team": "Yankees",
            "away_team": "Rays",
            "date": "2026-05-22",
            "edges": {
                "away": {
                    "edge": 0.061,
                    "model_prob": 0.44,
                    "expected_value": 0.13,
                    "american_odds": 155,
                },
                "home": {
                    "edge": -0.061,
                    "model_prob": 0.56,
                    "expected_value": -0.08,
                    "american_odds": -170,
                },
            },
        }
    ]

    diagnostics = _build_pipeline_diagnostics(
        matches=pd.DataFrame(),
        fixtures_fetched=[{"home_team": "Yankees", "away_team": "Rays"}],
        fixtures_in_window=[{"home_team": "Yankees", "away_team": "Rays"}],
        odds_list=[{"home_team": "Yankees", "away_team": "Rays"}],
        odds_lookup={("Yankees", "Rays"): {"home_team": "Yankees", "away_team": "Rays"}},
        prediction_records=records,
        outcomes=["home", "away"],
        sport_key="mlb",
        sport=sport,
        slop_locks=[],
        longslop=None,
        slimegrinder=[],
    )

    assert diagnostics["candidate_lanes"]["value_dog"]["eligible_outcomes"] == 1
    assert diagnostics["gate_failures"]["negative_expected_value"] == 1
    assert diagnostics["gate_failures"]["below_probability_floor"] == 1


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
        monkeypatch.setitem(SPORTS["nba"], "totals_max_picks", 1)

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
        assert "results_features" in data["model_weights"]
        assert "four_factors" not in data["model_weights"]
        assert "results_features" in data["model_weights"]
        assert "recent_boxscore" in data["model_weights"]
        assert "nba_matchup" in data["model_weights"]
        assert "dixon_coles" not in data["model_weights"]

        with open(os.path.join(output_dir, "pick_history.json")) as f:
            pick_history = json.load(f)
        assert any(pick["type"] == "total_lock" for pick in pick_history["picks"])
        decision_log_path = os.path.join(tmp_path, "tracking", "pick_decisions.csv")
        assert os.path.exists(decision_log_path)
        with open(decision_log_path, newline="") as f:
            decision_rows = list(csv.DictReader(f))
        assert decision_rows
        assert any(row["market_type"] == "total" for row in decision_rows)
        assert all(row["decision_context_json"] for row in decision_rows)
        assert data["run_type"] == "manual"
        assert data["run_id"]
        assert data["snapshot_path"].endswith(".json")
        assert os.path.exists(os.path.join(tmp_path, data["snapshot_path"]))
        assert pick_history["run_id"] == data["run_id"]
        assert all(pick.get("snapshot_path") == data["snapshot_path"] for pick in pick_history["picks"])


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
                "home_pitcher_source": "espn",
                "home_pitcher_last_checked": f"{_TODAY}T12:00:00+00:00",
                "home_pitcher_cache_stale": False,
                "away_pitcher": "Bruins Starter",
                "away_pitcher_hand": "L",
                "away_pitcher_source": "mlb_stats_api",
                "away_pitcher_last_checked": f"{_TODAY}T12:00:00+00:00",
                "away_pitcher_cache_stale": True,
                "pitcher_warnings": ["away_pitcher_from_stale_cache"],
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
        monkeypatch.setitem(SPORTS["mlb"], "min_expected_value", -1.0)
        monkeypatch.setitem(SPORTS["mlb"], "slop_lock_edge_threshold", -1.0)
        monkeypatch.setitem(SPORTS["mlb"], "slop_lock_probability_floor", 0.0)
        monkeypatch.setitem(SPORTS["mlb"], "slop_lock_confidence_threshold", 0.0)
        monkeypatch.setattr("pipeline.run.compute_pick_tier", lambda confidence, probability, edge: "LEAN")

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
        assert match["home_pitcher_source"] == "espn"
        assert match["away_pitcher_source"] == "mlb_stats_api"
        assert match["home_pitcher_last_checked"] == f"{_TODAY}T12:00:00+00:00"
        assert match["away_pitcher_cache_stale"] is True
        assert match["pitcher_warnings"] == ["away_pitcher_from_stale_cache"]
        total_market = data["totals_matches"][0]
        assert total_market["total_line"] == 8.5
        assert total_market["home_lineup_profile"]["available_hitters"] == 13
        assert total_market["home_pitcher_source"] == "espn"
        assert total_market["away_pitcher_source"] == "mlb_stats_api"
        assert total_market["away_pitcher_cache_stale"] is True
        assert total_market["pick"] in {"over", "under"}
        assert "over" in total_market["edges"]
        with open(os.path.join(output_dir, "pick_history.json")) as f:
            pick_history = json.load(f)
        moneyline_pick = next(pick for pick in pick_history["picks"] if pick["type"] == "slop_lock")
        assert moneyline_pick["home_pitcher"] == "Aces Starter"
        assert moneyline_pick["home_pitcher_source"] == "espn"
        assert moneyline_pick["home_pitcher_last_checked"] == f"{_TODAY}T12:00:00+00:00"
        assert moneyline_pick["away_pitcher"] == "Bruins Starter"
        assert moneyline_pick["away_pitcher_source"] == "mlb_stats_api"
        assert moneyline_pick["away_pitcher_cache_stale"] is True
        assert moneyline_pick["pitcher_warnings"] == ["away_pitcher_from_stale_cache"]


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
    @patch("pipeline.run.fetch_mlb_schedule")
    @patch("pipeline.run.fetch_mlb_games")
    @patch("pipeline.run.fetch_nhl_schedule")
    @patch("pipeline.run.fetch_nhl_games")
    @patch("pipeline.run.fetch_wnba_espn_schedule")
    @patch("pipeline.run.fetch_wnba_espn_games")
    @patch("pipeline.run.fetch_nba_espn_schedule")
    @patch("pipeline.run.fetch_nba_espn_games")
    def test_produces_per_sport_files_and_manifest(
        self,
        mock_nba_games,
        mock_nba_schedule,
        mock_wnba_games,
        mock_wnba_schedule,
        mock_nhl_games,
        mock_nhl_schedule,
        mock_mlb_games,
        mock_mlb_schedule,
        mock_odds,
        sample_nba_matches, sample_nba_box_scores, sample_nhl_matches,
        tmp_path
    ):
        mock_nba_games.return_value = (sample_nba_matches, sample_nba_box_scores)
        mock_nba_schedule.return_value = [
            {
                "home_team": "Lakers",
                "away_team": "Warriors",
                "date": "2026-02-19",
            }
        ]
        mock_wnba_games.return_value = (sample_nba_matches, sample_nba_box_scores)
        mock_wnba_schedule.return_value = [
            {
                "home_team": "Liberty",
                "away_team": "Fever",
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
        mock_odds.return_value = []

        output_dir = str(tmp_path)
        manifest = run_pipeline(output_dir=output_dir)

        # Manifest
        manifest_path = os.path.join(output_dir, "manifest.json")
        dashboard_path = os.path.join(output_dir, "dashboard.json")
        assert os.path.exists(manifest_path)
        assert os.path.exists(dashboard_path)
        assert "nba" in manifest["sports"]
        assert "wnba" in manifest["sports"]
        assert "nhl" in manifest["sports"]
        assert "ncaam" in manifest["sports"]
        assert "mma" not in manifest["sports"]
        assert manifest["sports"]["nba"]["status"] == "ok"
        assert manifest["sports"]["wnba"]["status"] == "ok"
        assert manifest["sports"]["nhl"]["status"] == "ok"
        assert manifest["sports"]["ncaam"]["status"] == "season_disabled"
        assert manifest["sports"]["ncaam"]["active"] is False
        assert "diagnostics" in manifest["sports"]["nba"]
        assert "diagnostics" in manifest["sports"]["wnba"]
        assert "diagnostics" in manifest["sports"]["nhl"]
        assert "diagnostics" in manifest["sports"]["ncaam"]

        with open(dashboard_path) as f:
            dashboard = json.load(f)
        assert "aggregate" in dashboard
        assert "sports" in dashboard
        assert dashboard["aggregate"]["slate"]["modeled"] >= 0

        # Per-sport prediction files
        assert os.path.exists(os.path.join(output_dir, "nba", "predictions.json"))
        assert os.path.exists(os.path.join(output_dir, "wnba", "predictions.json"))
        assert os.path.exists(os.path.join(output_dir, "nhl", "predictions.json"))
        assert not os.path.exists(os.path.join(output_dir, "ncaam", "predictions.json"))


class TestRunCli:
    @patch("pipeline.run.run_sport_pipeline")
    def test_main_supports_single_sport(self, mock_run_sport_pipeline, tmp_path):
        exit_code = _main(["--sport", "mlb", "--output-dir", str(tmp_path / "mlb")])

        assert exit_code == 0
        mock_run_sport_pipeline.assert_called_once_with("mlb", output_dir=str(tmp_path / "mlb"))

    @patch("pipeline.run._update_global_metadata")
    @patch("pipeline.run.run_sport_pipeline")
    def test_main_skips_season_disabled_sport(self, mock_run_sport_pipeline, mock_update_metadata, tmp_path, capsys):
        exit_code = _main(["--sport", "ncaam", "--output-dir", str(tmp_path / "ncaam")])

        assert exit_code == 0
        mock_run_sport_pipeline.assert_not_called()
        mock_update_metadata.assert_called_once_with(str(tmp_path))
        assert "season-disabled" in capsys.readouterr().out


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
        """Candidates are ranked by confidence, then probability, then edge."""
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
        assert locks[0]["home_team"] == "C"   # conf=61
        assert locks[1]["home_team"] == "A"   # conf=58
        assert locks[2]["away_team"] == "F"   # conf=53

    def test_ranked_by_confidence_then_edge(self):
        """When confidence ties, higher probability wins; edge is final tiebreaker."""
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
            ("A", "B", 70, 0.03),
            ("C", "D", 70, 0.08),
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

    def test_confidence_below_threshold_is_excluded(self):
        from pipeline.run import _compute_slop_locks
        records = [
            self._make_record("A", "B", "home", 0.61, -110, edge=0.05, confidence_score=51.9, expected_value=0.06),
        ]

        locks = _compute_slop_locks(
            records,
            ["home", "away"],
            additional_confidence_floor=52,
            max_picks=5,
        )

        assert locks == []

    def test_confidence_exactly_at_threshold_is_allowed(self):
        from pipeline.run import _compute_slop_locks
        records = [
            self._make_record("A", "B", "home", 0.61, -110, edge=0.05, confidence_score=52.0, expected_value=0.06),
        ]

        locks = _compute_slop_locks(
            records,
            ["home", "away"],
            additional_confidence_floor=52,
            max_picks=5,
        )

        assert len(locks) == 1
        assert locks[0]["confidence_score"] == 52.0

    def test_no_play_tier_is_excluded_from_slop_locks(self):
        from pipeline.run import _compute_slop_locks
        records = [
            self._make_record("A", "B", "away", 0.44, 180, edge=0.08, confidence_score=70, expected_value=0.23),
        ]

        locks = _compute_slop_locks(
            records,
            ["home", "away"],
            additional_confidence_floor=52,
            max_picks=5,
            lane_configs={
                "value_dog": {
                    "enabled": True,
                    "edge_floor": 0.04,
                    "probability_floor": 0.35,
                    "min_expected_value": 0.05,
                    "american_odds_min": 120,
                    "american_odds_max": 500,
                }
            },
        )

        assert locks == []

    def test_valid_slop_lock_candidate_is_selected(self):
        from pipeline.run import _compute_slop_locks
        records = [
            self._make_record("A", "B", "home", 0.58, -110, edge=0.04, confidence_score=56, expected_value=0.05),
        ]

        locks = _compute_slop_locks(
            records,
            ["home", "away"],
            additional_confidence_floor=52,
            max_picks=5,
        )

        assert len(locks) == 1
        assert locks[0]["tier"] != "NO_PLAY"

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
        """Candidates below the confidence floor are skipped while later valid candidates publish."""
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
            ("A", 90.0),
            ("G", 54.0),
        ]

    def test_picks_have_tier_field(self):
        """Each returned pick must include a 'tier' field."""
        from pipeline.run import _compute_slop_locks
        records = [
            self._make_record("A", "B", "home", 0.65, -130, edge=0.04, confidence_score=65, expected_value=0.04),
            self._make_record("C", "D", "home", 0.54, -110, edge=0.012, confidence_score=55, expected_value=0.01),
        ]
        locks = _compute_slop_locks(records, ["home", "away"], max_picks=5)
        assert locks
        for lock in locks:
            assert "tier" in lock
            assert lock["tier"] in ("STRONG", "LEAN", "WATCHLIST")

    def test_strong_pick_tier_assigned(self):
        """A pick meeting STRONG criteria should receive 'STRONG' tier."""
        from pipeline.run import _compute_slop_locks
        records = [
            self._make_record("A", "B", "home", 0.62, -130, edge=0.03, confidence_score=65, expected_value=0.03),
        ]
        locks = _compute_slop_locks(records, ["home", "away"], max_picks=5)
        assert len(locks) == 1
        assert locks[0]["tier"] == "STRONG"


class TestPublishablePickValidation:
    def _pick(self, home="A", away="B", pick="home", **overrides):
        item = {
            "home_team": home,
            "away_team": away,
            "date": "2026-05-01",
            "pick": pick,
            "model_prob": 0.58,
            "implied_prob": 0.52,
            "edge": 0.04,
            "expected_value": 0.08,
            "american_odds": 115,
            "decimal_odds": 2.15,
            "confidence_score": 70,
        }
        item.update(overrides)
        return item

    def test_strips_invalid_and_duplicate_official_moneylines(self):
        from pipeline.run import validate_publishable_picks

        selection_config = {
            "slop_locks": {
                "min_expected_value": 0.0,
                "edge_floor": 0.03,
                "probability_floor": 0.5,
            },
            "totals_locks": {
                "min_expected_value": 0.0,
                "edge_floor": 0.02,
                "probability_floor": 0.53,
                "confidence_floor": 54,
            },
            "slimegrinder": {"min_expected_value": 0.0, "confidence_floor": 55},
        }
        slop_locks = [
            self._pick(home="Pistons", away="Magic", pick="home"),
            self._pick(home="Pistons", away="Magic", pick="away", model_prob=0.57, edge=0.035),
            self._pick(home="Knicks", away="Heat", pick="home", expected_value=-0.01),
        ]
        slimegrinder = [
            self._pick(home="Pistons", away="Magic", pick="away", edge=0.02),
            self._pick(home="Bulls", away="Hawks", pick="home", edge=0.02),
        ]

        cleaned_slop, cleaned_totals, longslop, cleaned_slime, issues = validate_publishable_picks(
            sport_key="nba",
            slop_locks=slop_locks,
            totals_locks=[],
            longslop=None,
            slimegrinder=slimegrinder,
            publication_guard={"allow_moneyline": True, "allow_totals": True, "allow_slimegrinder": True},
            selection_config=selection_config,
        )

        assert [(p["home_team"], p["away_team"], p["pick"]) for p in cleaned_slop] == [
            ("Pistons", "Magic", "home")
        ]
        assert cleaned_totals == []
        assert longslop is None
        assert [(p["home_team"], p["away_team"], p["pick"]) for p in cleaned_slime] == [
            ("Bulls", "Hawks", "home")
        ]
        assert {issue["reason"] for issue in issues} == {
            "duplicate_official_matchup",
            "below_min_expected_value",
            "duplicate_secondary_matchup",
        }

    def test_slop_lock_publish_validation_enforces_confidence_and_tier(self):
        from pipeline.run import validate_publishable_picks

        selection_config = {
            "slop_locks": {
                "min_expected_value": 0.0,
                "edge_floor": 0.03,
                "probability_floor": 0.5,
                "additional_confidence_floor": 52,
            },
        }
        low_confidence = self._pick(home="A", away="B", confidence_score=51.9)
        no_play = self._pick(home="C", away="D", model_prob=0.51, confidence_score=70, tier="NO_PLAY")
        valid = self._pick(home="E", away="F", model_prob=0.56, confidence_score=56, tier="LEAN")

        cleaned_slop, _, _, _, issues = validate_publishable_picks(
            sport_key="nba",
            slop_locks=[low_confidence, no_play, valid],
            totals_locks=[],
            longslop=None,
            slimegrinder=[],
            publication_guard={"allow_moneyline": True, "allow_totals": True},
            selection_config=selection_config,
        )

        assert [(p["home_team"], p["away_team"]) for p in cleaned_slop] == [("E", "F")]
        assert [issue["reason"] for issue in issues] == ["below_confidence_floor", "no_play_tier"]

    def test_publication_guard_suppression_strips_official_lanes(self):
        from pipeline.run import validate_publishable_picks

        cleaned_slop, cleaned_totals, longslop, cleaned_slime, issues = validate_publishable_picks(
            sport_key="nba",
            slop_locks=[self._pick()],
            totals_locks=[self._pick(market_type="total", pick="over", total_line=210.5)],
            longslop=None,
            slimegrinder=[self._pick(home="C", away="D")],
            publication_guard={
                "allow_moneyline": False,
                "allow_totals": False,
                "allow_slimegrinder": False,
            },
            selection_config={
                "slop_locks": {"edge_floor": 0.03, "probability_floor": 0.5},
                "totals_locks": {"edge_floor": 0.02, "probability_floor": 0.53, "confidence_floor": 54},
                "slimegrinder": {"confidence_floor": 55},
            },
        )

        assert cleaned_slop == []
        assert cleaned_totals == []
        assert longslop is None
        assert cleaned_slime == []
        assert [issue["reason"] for issue in issues] == [
            "publication_guard_suppressed",
            "publication_guard_suppressed",
            "publication_guard_suppressed",
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

    def test_append_pick_decision_log_creates_file_and_dedupes_rows(self, tmp_path):
        from pipeline.run import _append_pick_decision_log

        path = str(tmp_path / "tracking" / "pick_decisions.csv")
        row = {
            "logged_at": "2026-03-28T12:00:00Z",
            "run_id": "daily-20260328T120000000000Z",
            "run_type": "daily",
            "snapshot_timestamp": "2026-03-28T12:00:00Z",
            "snapshot_path": "tracking/snapshots/2026-03-28/nba/daily-20260328T120000000000Z.json",
            "sport": "nba",
            "pick_type": "slop_lock",
            "market_type": "moneyline",
            "home_team": "Lakers",
            "away_team": "Celtics",
            "match_date": "2026-03-28",
            "start_time": "2026-03-28T23:00:00Z",
            "pick": "home",
            "model_prob": 0.61,
            "implied_prob": 0.52,
            "market_implied_prob": 0.54,
            "edge": 0.09,
            "expected_value": 0.07,
            "american_odds": 110,
            "decimal_odds": 2.1,
            "confidence_score": 72,
            "decision_context_json": "{\"pick\":\"home\"}",
            "gate_context_json": "{\"selection_config\":{\"edge_floor\":0.03}}",
        }

        _append_pick_decision_log(path, [row, row])

        assert os.path.exists(path)
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["sport"] == "nba"
        assert rows[0]["pick_type"] == "slop_lock"

    def test_backfill_pick_decision_log_from_snapshots_creates_historical_rows(self, tmp_path):
        from pipeline.run import _backfill_pick_decision_log_from_snapshots

        snapshot_dir = tmp_path / "tracking" / "snapshots" / "2026-03-29" / "nba"
        snapshot_dir.mkdir(parents=True)
        snapshot_path = snapshot_dir / "daily-20260329T120000000000Z.json"
        with open(snapshot_path, "w") as f:
            json.dump({
                "sport": "nba",
                "run_id": "daily-20260329T120000000000Z",
                "run_type": "daily",
                "snapshot_timestamp": "2026-03-29T12:00:00Z",
                "selection_config": {"slop_locks": {"edge_floor": 0.03}},
                "publication_guard": {"status": "live", "enforced": True},
                "inputs": {"calibration_sample_size": 14},
                "records": {
                    "matches": [
                        {
                            "home_team": "Lakers",
                            "away_team": "Warriors",
                            "date": "2026-03-29",
                            "model_probs": {"home": 0.61, "away": 0.39},
                            "edges": {
                                "home": {
                                    "edge": 0.07,
                                    "expected_value": 0.08,
                                    "confidence_score": 71,
                                    "american_odds": 105,
                                    "decimal_odds": 2.05,
                                    "implied_prob": 0.5,
                                    "market_implied_prob": 0.54,
                                }
                            },
                        }
                    ],
                    "totals_matches": [],
                },
                "outputs": {
                    "slop_locks": [
                        {
                            "home_team": "Lakers",
                            "away_team": "Warriors",
                            "date": "2026-03-29",
                            "pick": "home",
                            "model_prob": 0.61,
                            "expected_value": 0.08,
                            "confidence_score": 71,
                            "american_odds": 105,
                            "decimal_odds": 2.05,
                        }
                    ],
                    "longslop": None,
                    "totals_locks": [],
                },
            }, f)

        written = _backfill_pick_decision_log_from_snapshots(str(tmp_path), sports=["nba"])

        assert written == 1
        with open(tmp_path / "tracking" / "pick_decisions.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["snapshot_path"].endswith(".json")
        assert rows[0]["calibration_sample_size"] == "14"

    def test_backfill_pick_history_market_snapshots_updates_missing_clv(self, tmp_path):
        from pipeline.run import _backfill_pick_history_market_snapshots

        tracking_dir = tmp_path / "tracking"
        tracking_dir.mkdir(parents=True)
        sport_dir = tmp_path / "nba"
        sport_dir.mkdir(parents=True)

        with open(sport_dir / "pick_history.json", "w") as f:
            json.dump({
                "picks": [
                    {
                        "market_type": "moneyline",
                        "home_team": "Lakers",
                        "away_team": "Celtics",
                        "match_date": "2026-03-29",
                        "pick": "home",
                        "market_implied_prob": 0.48,
                    }
                ]
            }, f)

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
                "logged_at": "2026-03-29T17:00:00Z",
                "sport": "nba",
                "market_type": "moneyline",
                "home_team": "Lakers",
                "away_team": "Celtics",
                "match_date": "2026-03-29",
                "start_time": "2026-03-29T23:00:00Z",
                "outcome": "home",
                "total_line": "",
                "decimal_odds": "1.95",
                "american_odds": "-105",
                "implied_prob": "0.512",
                "market_implied_prob": "0.525",
                "market_source": "median_complete_book_no_vig",
                "market_books": "7",
                "hold": "0.024",
                "market_snapshot_json": "{\"execution_prices\":{\"home\":1.95}}",
            })

        updated = _backfill_pick_history_market_snapshots(str(tmp_path), sports=["nba"])

        assert updated == 1
        with open(sport_dir / "pick_history.json") as f:
            picks = json.load(f)["picks"]
        assert picks[0]["closing_line_value"] == pytest.approx(0.032, abs=1e-6)
        assert picks[0]["closing_market_books"] == 7

    def test_publication_guard_trickles_undersampled_but_healthy_moneylines(self):
        """A lane short only on settled volume (research status) with a strong
        health score trickle-publishes at the hold cap instead of going dark."""
        from pipeline.run import _build_publication_guard

        past_picks = []
        for idx in range(8):
            past_picks.append({
                "evaluated": True,
                "market_type": "moneyline",
                "type": "slop_lock",
                "closing_line_value": 0.02,
                "model_prob": 0.52,
                "won": True,
                "decimal_odds": 2.0,
                "match_date": f"2026-03-{10 + idx:02d}",
            })
        for idx in range(10):
            past_picks.append({
                "evaluated": True,
                "market_type": "total",
                "type": "total_lock",
                "closing_line_value": 0.2,
                "model_prob": 0.53,
                "won": True,
                "decimal_odds": 2.0,
                "match_date": f"2026-03-{10 + idx:02d}",
            })

        sport = {
            "publication_min_evaluated_picks": 10,
            "publication_min_evaluated_totals_picks": 8,
            "totals_max_picks": 2,
            "moneyline_health_recent_window": 8,
            "moneyline_health_min_recent_evaluated": 5,
            "moneyline_health_min_recent_roi": 0.0,
            "moneyline_health_max_overconfidence_gap": 0.12,
            "moneyline_clv_guard_window": 8,
            "moneyline_clv_guard_min_tracked": 5,
            "moneyline_clv_guard_min_avg": 0.0,
            "totals_health_recent_window": 8,
            "totals_health_min_recent_evaluated": 5,
            "totals_health_min_recent_roi": 0.0,
            "totals_health_max_overconfidence_gap": 0.1,
            "totals_clv_guard_window": 8,
            "totals_clv_guard_min_tracked": 8,
            "totals_clv_guard_min_avg": 0.0,
            "enable_longslop": False,
            "enable_slimegrinder": False,
        }

        guard = _build_publication_guard(past_picks, sport, enforce_live_guard=True)

        assert guard["allow_moneyline"] is True
        assert guard["moneyline_publish_cap"] == 1
        assert guard["allow_totals"] is True
        assert guard["status"] == "live"
        assert guard["lane_guards"]["moneyline"]["status"] == "research"
        assert "need more settled moneylines picks (8/10)" in guard["lane_guards"]["moneyline"]["reasons"]
        assert guard["allow_slimegrinder"] is False

    def test_hydrate_pick_decision_log_market_snapshots_uses_saved_snapshot_odds(self, tmp_path):
        from pipeline.run import _append_pick_decision_log, _hydrate_pick_decision_log_market_snapshots

        snapshot_dir = tmp_path / "tracking" / "snapshots" / "2026-03-29" / "nba"
        snapshot_dir.mkdir(parents=True)
        snapshot_relpath = "tracking/snapshots/2026-03-29/nba/daily-20260329T120000000000Z.json"
        with open(tmp_path / snapshot_relpath, "w") as f:
            json.dump({
                "inputs": {
                    "odds": [
                        {
                            "home_team": "Lakers",
                            "away_team": "Celtics",
                            "commence_time": "2026-03-29T23:00:00Z",
                            "home_odds": 1.95,
                            "away_odds": 1.91,
                        }
                    ]
                }
            }, f)

        path = str(tmp_path / "tracking" / "pick_decisions.csv")
        _append_pick_decision_log(path, [{
            "logged_at": "2026-03-29T12:00:00Z",
            "run_id": "daily-20260329T120000000000Z",
            "run_type": "daily",
            "snapshot_timestamp": "2026-03-29T12:00:00Z",
            "snapshot_path": snapshot_relpath,
            "sport": "nba",
            "pick_type": "slop_lock",
            "market_type": "moneyline",
            "home_team": "Lakers",
            "away_team": "Celtics",
            "match_date": "2026-03-29",
            "pick": "home",
            "decision_context_json": "{\"pick\":\"home\"}",
            "gate_context_json": "{\"selection_config\":{}}",
        }])

        updated = _hydrate_pick_decision_log_market_snapshots(str(tmp_path), sports=["nba"])

        assert updated == 1
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["market_snapshot_json"]
        assert "1.95" in rows[0]["market_snapshot_json"]


class TestPublicationGuardTrickle:
    @staticmethod
    def _healthy_moneyline_picks():
        return [
            {
                "evaluated": True,
                "market_type": "moneyline",
                "type": "slop_lock",
                "closing_line_value": 0.02,
                "model_prob": 0.52,
                "won": True,
                "decimal_odds": 2.0,
                "match_date": f"2026-03-{10 + idx:02d}",
            }
            for idx in range(10)
        ]

    @staticmethod
    def _guard_sport_config():
        return {
            "publication_min_evaluated_picks": 10,
            "publication_min_evaluated_totals_picks": 8,
            "totals_max_picks": 3,
            "totals_hold_max_picks": 2,
            "totals_hold_min_health_score": 0.75,
            "moneyline_health_recent_window": 8,
            "moneyline_health_min_recent_evaluated": 5,
            "moneyline_health_min_recent_roi": 0.0,
            "moneyline_health_max_overconfidence_gap": 0.12,
            "moneyline_clv_guard_window": 8,
            "moneyline_clv_guard_min_tracked": 5,
            "moneyline_clv_guard_min_avg": 0.0,
            "totals_health_recent_window": 8,
            "totals_health_min_recent_evaluated": 5,
            "totals_health_min_recent_roi": 0.0,
            "totals_health_max_overconfidence_gap": 0.1,
            "totals_clv_guard_window": 8,
            "totals_clv_guard_min_tracked": 8,
            "totals_clv_guard_min_avg": 0.0,
            "enable_longslop": False,
            "enable_slimegrinder": False,
        }

    def test_research_status_totals_get_capped_trickle(self):
        """A lane failing only on data insufficiency (research status) should
        still publish a capped trickle when its health score clears the floor."""
        from pipeline.run import _build_publication_guard

        past_picks = self._healthy_moneyline_picks()
        for idx in range(10):
            pick = {
                "evaluated": True,
                "market_type": "total",
                "type": "total_lock",
                "closing_line_value": 0.2,
                "model_prob": 0.53,
                "won": True,
                "decimal_odds": 2.0,
                "match_date": f"2026-03-{10 + idx:02d}",
            }
            if idx == 9:
                # Newest pick is missing its closing-line snapshot: 7/8 tracked
                # in the recent window, which is insufficiency, not bad play.
                pick.pop("closing_line_value")
            past_picks.append(pick)

        guard = _build_publication_guard(past_picks, self._guard_sport_config(), enforce_live_guard=True)

        assert guard["allow_moneyline"] is True
        assert guard["allow_totals"] is True
        assert guard["totals_publish_cap"] == 2
        assert guard["lane_guards"]["totals"]["status"] == "research"

    def test_cold_start_lane_stays_suppressed(self):
        """A brand-new sport with no settled evidence must not trickle-publish:
        its health score is too low to clear the hold floor."""
        from pipeline.run import _build_publication_guard

        guard = _build_publication_guard([], self._guard_sport_config(), enforce_live_guard=True)

        assert guard["allow_moneyline"] is False
        assert guard["moneyline_publish_cap"] is None
        assert guard["status"] == "suppressed"


class TestShadowPickRecording:
    def _run_nba(self, mock_games, mock_schedule, mock_odds,
                 sample_nba_matches, sample_nba_box_scores, output_dir, monkeypatch):
        mock_games.return_value = (sample_nba_matches, sample_nba_box_scores)
        mock_schedule.return_value = [
            {
                "home_team": "Lakers",
                "away_team": "Warriors",
                "date": _TODAY,
                "start_time": f"{_TODAY}T00:30:00Z",
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
        # The fixture model is a near coin flip, which the pick-tier policy
        # rejects; tier policy is not under test here, recording is.
        monkeypatch.setattr("pipeline.run.compute_pick_tier", lambda *a, **k: "LEAN")
        monkeypatch.setitem(SPORTS["nba"], "totals_feature_min_games", 4)
        monkeypatch.setitem(SPORTS["nba"], "totals_edge_threshold", 0.0)
        monkeypatch.setitem(SPORTS["nba"], "totals_probability_floor", 0.5)
        monkeypatch.setitem(SPORTS["nba"], "totals_confidence_threshold", 0.0)
        monkeypatch.setitem(SPORTS["nba"], "totals_max_picks", 1)
        monkeypatch.setitem(SPORTS["nba"], "slop_lock_edge_threshold", -1.0)
        monkeypatch.setitem(SPORTS["nba"], "slop_lock_probability_floor", 0.0)
        monkeypatch.setitem(SPORTS["nba"], "slop_lock_confidence_threshold", 0.0)
        monkeypatch.setitem(SPORTS["nba"], "min_expected_value", -1.0)
        monkeypatch.setitem(SPORTS["nba"], "slop_lock_lanes", {
            "near_favorite": {
                "enabled": True,
                "edge_floor": -1.0,
                "probability_floor": 0.0,
                "min_expected_value": -1.0,
                "american_odds_min": -100000,
                "american_odds_max": 100000,
                "max_picks": 5,
            },
        })
        run_sport_pipeline("nba", output_dir=output_dir)

    @patch("pipeline.run.fetch_odds")
    @patch("pipeline.run.fetch_nba_espn_schedule")
    @patch("pipeline.run.fetch_nba_espn_games")
    def test_suppressed_lanes_record_shadow_picks(
        self, mock_games, mock_schedule, mock_odds,
        sample_nba_matches, sample_nba_box_scores, tmp_path, monkeypatch
    ):
        # Force live-guard enforcement with an empty pick history: every lane
        # is suppressed, but candidates must still be recorded as shadows.
        monkeypatch.setattr("pipeline.run._is_live_public_output", lambda base_dir: True)
        output_dir = str(tmp_path / "nba")

        self._run_nba(mock_games, mock_schedule, mock_odds,
                      sample_nba_matches, sample_nba_box_scores, output_dir, monkeypatch)

        with open(os.path.join(output_dir, "predictions.json")) as f:
            data = json.load(f)
        assert data["publication_guard"]["status"] == "suppressed"
        assert data["slop_locks"] == []
        assert data["totals_locks"] == []

        with open(os.path.join(output_dir, "pick_history.json")) as f:
            pick_history = json.load(f)
        picks = pick_history["picks"]
        assert picks, "suppressed run must still record shadow picks"
        assert all(pick["published"] is False for pick in picks)
        assert any(pick["type"] == "slop_lock" for pick in picks)
        assert any(pick["type"] == "total_lock" for pick in picks)

        # Shadow picks must not appear in the public record.
        assert data["pick_stats"]["all"]["total"] == 0

        # Shadow picks must not produce pick-decision ledger rows.
        decision_log_path = os.path.join(tmp_path, "tracking", "pick_decisions.csv")
        if os.path.exists(decision_log_path):
            with open(decision_log_path, newline="") as f:
                assert list(csv.DictReader(f)) == []

    @patch("pipeline.run.fetch_odds")
    @patch("pipeline.run.fetch_nba_espn_schedule")
    @patch("pipeline.run.fetch_nba_espn_games")
    def test_shadow_pick_upgraded_when_lane_goes_live(
        self, mock_games, mock_schedule, mock_odds,
        sample_nba_matches, sample_nba_box_scores, tmp_path, monkeypatch
    ):
        output_dir = str(tmp_path / "nba")

        # Run 1: guard enforced, everything suppressed -> shadows only.
        monkeypatch.setattr("pipeline.run._is_live_public_output", lambda base_dir: True)
        self._run_nba(mock_games, mock_schedule, mock_odds,
                      sample_nba_matches, sample_nba_box_scores, output_dir, monkeypatch)
        with open(os.path.join(output_dir, "pick_history.json")) as f:
            shadow_count = len(json.load(f)["picks"])
        assert shadow_count > 0

        # Run 2: guard not enforced (research mode publishes everything). The
        # same candidates must upgrade in place, not duplicate.
        monkeypatch.setattr("pipeline.run._is_live_public_output", lambda base_dir: False)
        self._run_nba(mock_games, mock_schedule, mock_odds,
                      sample_nba_matches, sample_nba_box_scores, output_dir, monkeypatch)

        with open(os.path.join(output_dir, "pick_history.json")) as f:
            picks = json.load(f)["picks"]
        assert len(picks) == shadow_count
        assert all(pick["published"] is True for pick in picks)

    def test_compute_pick_stats_excludes_shadow_picks(self):
        from pipeline.run import _compute_pick_stats

        picks = [
            {"type": "slop_lock", "evaluated": True, "won": True, "push": False,
             "decimal_odds": 2.0, "published": True},
            {"type": "slop_lock", "evaluated": True, "won": True, "push": False,
             "decimal_odds": 2.0},  # legacy pick without flag counts as published
            {"type": "slop_lock", "evaluated": True, "won": False, "push": False,
             "decimal_odds": 2.0, "published": False},
        ]

        stats = _compute_pick_stats(picks)

        assert stats["slop_lock"]["total"] == 2
        assert stats["slop_lock"]["evaluated"] == 2
        assert stats["slop_lock"]["wins"] == 2
        assert stats["slop_lock"]["losses"] == 0
        assert stats["all"]["total"] == 2


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


class TestConfigConstants:
    """Smoke tests for config constants that gate pick selection."""

    def test_slop_lock_min_odds_allows_heavy_favorites(self):
        from pipeline.config import SLOP_LOCK_MIN_ODDS
        assert SLOP_LOCK_MIN_ODDS == -220

    def test_nba_clv_guard_min_tracked_is_relaxed(self):
        from pipeline.config import SPORTS
        assert SPORTS["nba"]["moneyline_clv_guard_min_tracked"] == 3

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
