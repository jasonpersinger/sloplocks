"""Tests for pipeline.fetch_wnba."""

import json
from unittest.mock import MagicMock, patch

import pandas as pd

from pipeline.fetch_wnba import (
    fetch_wnba_espn_games,
    fetch_wnba_espn_schedule,
    normalize_wnba_team_name,
)


def _mock_response(payload):
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = payload
    return response


def _make_wnba_scoreboard(events):
    return {"events": events}


def _make_wnba_event(
    home_name,
    away_name,
    home_score,
    away_score,
    completed=True,
    event_id="123",
    home_id="9",
    away_id="5",
):
    return {
        "id": event_id,
        "date": "2025-08-01T23:00Z",
        "competitions": [{
            "date": "2025-08-01T23:00Z",
            "neutralSite": False,
            "status": {"type": {"completed": completed}},
            "competitors": [
                {
                    "homeAway": "home",
                    "score": str(home_score),
                    "team": {"id": home_id, "displayName": home_name},
                    "leaders": [{"name": "pointsPerGame", "leaders": [{"athlete": {"id": "11"}}]}],
                },
                {
                    "homeAway": "away",
                    "score": str(away_score),
                    "team": {"id": away_id, "displayName": away_name},
                    "leaders": [{"name": "pointsPerGame", "leaders": [{"athlete": {"id": "22"}}]}],
                },
            ],
        }],
    }


def _make_wnba_summary(home_id="9", away_id="5"):
    def stats(fgm, fga, fg3m, fg3a, ftm, fta, orb, drb, turnovers):
        return [
            {"name": "fieldGoalsMade-fieldGoalsAttempted", "displayValue": f"{fgm}-{fga}"},
            {"name": "threePointFieldGoalsMade-threePointFieldGoalsAttempted", "displayValue": f"{fg3m}-{fg3a}"},
            {"name": "freeThrowsMade-freeThrowsAttempted", "displayValue": f"{ftm}-{fta}"},
            {"name": "offensiveRebounds", "displayValue": str(orb)},
            {"name": "defensiveRebounds", "displayValue": str(drb)},
            {"name": "turnovers", "displayValue": str(turnovers)},
        ]

    return {
        "boxscore": {
            "teams": [
                {"team": {"id": home_id, "displayName": "New York Liberty"}, "statistics": stats(31, 65, 8, 20, 18, 22, 8, 27, 12)},
                {"team": {"id": away_id, "displayName": "Indiana Fever"}, "statistics": stats(28, 62, 7, 18, 12, 15, 7, 24, 14)},
            ]
        },
        "injuries": [
            {
                "team": {"id": home_id},
                "injuries": [{"status": "Questionable", "athlete": {"id": "11"}}],
            }
        ],
    }


def _make_roster(player_id, status="active", injury_status=None):
    athlete = {
        "id": player_id,
        "status": {"type": status},
        "injuries": [],
    }
    if injury_status:
        athlete["injuries"] = [{"status": injury_status}]
    return {"athletes": [athlete]}


class TestNormalizeWnbaTeamName:
    def test_known_full_names(self):
        assert normalize_wnba_team_name("New York Liberty") == "Liberty"
        assert normalize_wnba_team_name("Las Vegas Aces") == "Aces"
        assert normalize_wnba_team_name("Toronto Tempo") == "Tempo"

    def test_unknown_name_passthrough(self):
        assert normalize_wnba_team_name("Liberty") == "Liberty"
        assert normalize_wnba_team_name("Unknown Team") == "Unknown Team"


class TestFetchWnbaEspnGames:
    @patch("pipeline.fetch_wnba.time.sleep", lambda *_args, **_kwargs: None)
    @patch("pipeline.fetch_wnba.requests.get")
    def test_returns_games_and_team_named_boxscores(self, mock_get):
        scoreboard = _make_wnba_scoreboard([
            _make_wnba_event("New York Liberty", "Indiana Fever", 88, 75, completed=True, event_id="101"),
        ])
        mock_get.side_effect = [
            _mock_response(scoreboard),
            _mock_response(_make_wnba_summary()),
        ]

        games_df, box_df = fetch_wnba_espn_games(dates=["2025-08-01"])

        assert isinstance(games_df, pd.DataFrame)
        assert isinstance(box_df, pd.DataFrame)
        assert len(games_df) == 1
        assert games_df.iloc[0]["home_team"] == "Liberty"
        assert games_df.iloc[0]["away_team"] == "Fever"
        assert games_df.iloc[0]["home_goals"] == 88
        assert games_df.iloc[0]["away_goals"] == 75
        assert set(box_df["team"]) == {"Liberty", "Fever"}
        assert box_df.loc[box_df["team"] == "Liberty", "drb"].iloc[0] == 27

    @patch("pipeline.fetch_wnba.time.sleep", lambda *_args, **_kwargs: None)
    @patch("pipeline.fetch_wnba.requests.get")
    def test_skips_incomplete_games(self, mock_get):
        scoreboard = _make_wnba_scoreboard([
            _make_wnba_event("New York Liberty", "Indiana Fever", 0, 0, completed=False),
        ])
        mock_get.return_value = _mock_response(scoreboard)

        games_df, box_df = fetch_wnba_espn_games(dates=["2025-08-01"])

        assert len(games_df) == 0
        assert len(box_df) == 0

    @patch("pipeline.fetch_wnba.time.sleep", lambda *_args, **_kwargs: None)
    @patch("pipeline.fetch_wnba.requests.get")
    def test_cache_is_keyed_by_game_id_for_incremental_refetch(self, mock_get, tmp_path):
        cache_path = tmp_path / "espn_cache.json"
        scoreboard = _make_wnba_scoreboard([
            _make_wnba_event("New York Liberty", "Indiana Fever", 88, 75, completed=True, event_id="101"),
        ])
        mock_get.side_effect = [
            _mock_response(scoreboard),
            _mock_response(_make_wnba_summary()),
        ]

        fetch_wnba_espn_games(dates=["2025-08-01"], cache_path=str(cache_path))

        with cache_path.open() as f:
            cache = json.load(f)
        assert "101" in cache["games"]
        assert cache["games"]["101"]["date"] == "2025-08-01"


class TestFetchWnbaEspnSchedule:
    @patch("pipeline.fetch_wnba.time.sleep", lambda *_args, **_kwargs: None)
    @patch("pipeline.fetch_wnba.requests.get")
    def test_returns_fixture_list_with_nba_shaped_availability(self, mock_get):
        scoreboard = _make_wnba_scoreboard([
            _make_wnba_event("New York Liberty", "Indiana Fever", 0, 0, completed=False, event_id="202"),
        ])
        mock_get.side_effect = [
            _mock_response(scoreboard),
            _mock_response(_make_wnba_summary()),
            _mock_response(_make_roster("11", injury_status="Questionable")),
            _mock_response(_make_roster("22")),
        ]

        fixtures = fetch_wnba_espn_schedule()

        assert isinstance(fixtures, list)
        assert len(fixtures) == 1
        fixture = fixtures[0]
        assert fixture["home_team"] == "Liberty"
        assert fixture["away_team"] == "Fever"
        assert fixture["completed"] is False
        assert fixture["home_availability_profile"]["questionable_players"] == 1
        assert "event_uncertainty_burden" in fixture["home_availability_profile"]
        assert fixture["away_availability_profile"]["available_core_players"] == 1


class TestWnbaConfig:
    def test_wnba_config_has_required_keys(self):
        from pipeline.config import SPORTS

        wnba = SPORTS["wnba"]
        assert wnba["odds_sport"] == "basketball_wnba"
        assert wnba["outcomes"] == ["home", "away"]
        assert wnba["models"] == ["elo", "results_features", "recent_boxscore"]
        assert wnba["data_dir"].endswith("data/wnba")


class TestWnbaPipelineWiring:
    def test_run_module_imports_wnba_fetch_functions(self):
        import pipeline.run as run_mod

        assert hasattr(run_mod, "fetch_wnba_espn_games")
        assert hasattr(run_mod, "fetch_wnba_espn_schedule")
        assert run_mod.normalize_wnba_team_name("New York Liberty") == "Liberty"
