"""Tests for pipeline.fetch_nba — balldontlie.io API client."""

from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

from pipeline.fetch_nba import (
    normalize_nba_team_name,
    fetch_nba_games,
    fetch_nba_schedule,
)


# ---------------------------------------------------------------------------
# normalize_nba_team_name
# ---------------------------------------------------------------------------


class TestNormalizeNbaTeamName:
    def test_standard_names(self):
        assert normalize_nba_team_name("Los Angeles Lakers") == "Lakers"
        assert normalize_nba_team_name("Boston Celtics") == "Celtics"
        assert normalize_nba_team_name("Golden State Warriors") == "Warriors"

    def test_already_short(self):
        assert normalize_nba_team_name("Lakers") == "Lakers"
        assert normalize_nba_team_name("Unknown Team") == "Unknown Team"


# ---------------------------------------------------------------------------
# fetch_nba_games
# ---------------------------------------------------------------------------


def _make_bdl_game(home_name, away_name, home_score, away_score,
                   status="Final", date="2025-01-15T00:00:00.000Z"):
    """Helper: build a balldontlie.io game object."""
    return {
        "date": date,
        "status": status,
        "home_team": {"full_name": home_name},
        "visitor_team": {"full_name": away_name},
        "home_team_score": home_score,
        "visitor_team_score": away_score,
    }


class TestFetchNbaGames:
    @patch("pipeline.fetch_nba.requests.get")
    def test_returns_dataframe_with_required_columns(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                _make_bdl_game("Los Angeles Lakers", "Boston Celtics", 112, 108),
                _make_bdl_game("Golden State Warriors", "Miami Heat", 120, 115),
            ],
            "meta": {"next_cursor": None},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        df = fetch_nba_games(season=2024)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["date", "home_team", "away_team", "home_goals", "away_goals"]
        assert len(df) == 2
        assert df.iloc[0]["home_team"] == "Lakers"
        assert df.iloc[0]["away_team"] == "Celtics"
        assert df.iloc[0]["home_goals"] == 112
        assert df.iloc[0]["away_goals"] == 108

    @patch("pipeline.fetch_nba.requests.get")
    def test_skips_non_final_games(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                _make_bdl_game("Los Angeles Lakers", "Boston Celtics", 112, 108, status="Final"),
                _make_bdl_game("Chicago Bulls", "Miami Heat", None, None, status="Scheduled"),
            ],
            "meta": {"next_cursor": None},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        df = fetch_nba_games(season=2024)
        assert len(df) == 1

    @patch("pipeline.fetch_nba.requests.get")
    def test_handles_pagination(self, mock_get):
        page1_resp = MagicMock()
        page1_resp.json.return_value = {
            "data": [_make_bdl_game("Los Angeles Lakers", "Boston Celtics", 112, 108)],
            "meta": {"next_cursor": 100},
        }
        page1_resp.raise_for_status = MagicMock()

        page2_resp = MagicMock()
        page2_resp.json.return_value = {
            "data": [_make_bdl_game("Miami Heat", "Chicago Bulls", 105, 99)],
            "meta": {"next_cursor": None},
        }
        page2_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [page1_resp, page2_resp]

        df = fetch_nba_games(season=2024)
        assert len(df) == 2


# ---------------------------------------------------------------------------
# fetch_nba_schedule
# ---------------------------------------------------------------------------


class TestFetchNbaSchedule:
    @patch("pipeline.fetch_nba.requests.get")
    def test_returns_upcoming_games(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {
                    "date": "2026-02-19T00:00:00.000Z",
                    "status": "Scheduled",
                    "home_team": {"full_name": "Los Angeles Lakers"},
                    "visitor_team": {"full_name": "Boston Celtics"},
                    "home_team_score": None,
                    "visitor_team_score": None,
                },
            ],
            "meta": {"next_cursor": None},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        fixtures = fetch_nba_schedule()

        assert isinstance(fixtures, list)
        assert len(fixtures) == 1
        assert fixtures[0]["home_team"] == "Lakers"
        assert fixtures[0]["away_team"] == "Celtics"
        assert fixtures[0]["date"] == "2026-02-19"

    @patch("pipeline.fetch_nba.requests.get")
    def test_excludes_final_games(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                _make_bdl_game("Los Angeles Lakers", "Boston Celtics", 112, 108, status="Final"),
                {
                    "date": "2026-02-20T00:00:00.000Z",
                    "status": "Scheduled",
                    "home_team": {"full_name": "Miami Heat"},
                    "visitor_team": {"full_name": "Chicago Bulls"},
                    "home_team_score": None,
                    "visitor_team_score": None,
                },
            ],
            "meta": {"next_cursor": None},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        fixtures = fetch_nba_schedule()
        assert len(fixtures) == 1
        assert fixtures[0]["home_team"] == "Heat"
