"""Tests for pipeline.fetch_nba — balldontlie.io API client."""

import json
import os
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
import requests

from pipeline.fetch_nba import (
    normalize_nba_team_name,
    fetch_nba_games,
    fetch_nba_schedule,
    fetch_nba_espn_games,
    fetch_nba_espn_schedule,
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


# ---- ESPN-based NBA fetcher tests -------------------------------------------

NBA_SAMPLE_TOTALS = [
    "", "110", "44-95", "17-42", "5-7",
    "49", "25", "10", "9", "6", "15", "34", "17", "",
]


def _make_nba_espn_event(event_id, home_name, away_name,
                         home_score, away_score,
                         completed=True, date="2026-01-15T00:00Z"):
    return {
        "id": event_id,
        "date": date,
        "competitions": [{
            "competitors": [
                {
                    "homeAway": "home",
                    "team": {"displayName": home_name},
                    "score": str(home_score),
                },
                {
                    "homeAway": "away",
                    "team": {"displayName": away_name},
                    "score": str(away_score),
                },
            ],
            "status": {"type": {"completed": completed}},
        }],
    }


def _make_nba_summary(home_name, away_name, home_totals, away_totals):
    return {
        "boxscore": {
            "players": [
                {
                    "team": {"displayName": home_name},
                    "statistics": [{"totals": home_totals}],
                },
                {
                    "team": {"displayName": away_name},
                    "statistics": [{"totals": away_totals}],
                },
            ]
        }
    }


class TestFetchNbaEspnGames:
    @patch("pipeline.fetch_nba.requests.get")
    def test_returns_games_and_box_scores(self, mock_get):
        scoreboard_resp = MagicMock()
        scoreboard_resp.raise_for_status = MagicMock()
        scoreboard_resp.json.return_value = {
            "events": [
                _make_nba_espn_event("1", "Los Angeles Lakers", "Boston Celtics",
                                     112, 108),
            ]
        }
        summary_resp = MagicMock()
        summary_resp.raise_for_status = MagicMock()
        summary_resp.json.return_value = _make_nba_summary(
            "Los Angeles Lakers", "Boston Celtics",
            NBA_SAMPLE_TOTALS, NBA_SAMPLE_TOTALS,
        )
        mock_get.side_effect = [scoreboard_resp, summary_resp]

        games_df, box_df = fetch_nba_espn_games(season=2025, dates=["2026-01-15"])

        assert len(games_df) == 1
        assert list(games_df.columns) == [
            "game_id", "date", "home_team", "away_team", "home_goals", "away_goals"
        ]
        assert games_df.iloc[0]["home_team"] == "Lakers"
        assert games_df.iloc[0]["away_team"] == "Celtics"
        assert games_df.iloc[0]["home_goals"] == 112
        assert "game_id" in games_df.columns
        assert len(box_df) == 2
        assert box_df.iloc[0]["pts"] == 110

    @patch("pipeline.fetch_nba.requests.get")
    def test_skips_non_completed_games(self, mock_get):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "events": [
                _make_nba_espn_event("1", "Lakers", "Celtics", 0, 0, completed=False),
            ]
        }
        mock_get.return_value = resp

        games_df, box_df = fetch_nba_espn_games(season=2025, dates=["2026-01-15"])
        assert len(games_df) == 0
        assert len(box_df) == 0

    @patch("pipeline.fetch_nba.requests.get")
    def test_handles_missing_box_score(self, mock_get):
        scoreboard_resp = MagicMock()
        scoreboard_resp.raise_for_status = MagicMock()
        scoreboard_resp.json.return_value = {
            "events": [
                _make_nba_espn_event("1", "Los Angeles Lakers", "Boston Celtics", 112, 108),
            ]
        }
        summary_resp = MagicMock()
        summary_resp.raise_for_status.side_effect = requests.RequestException("timeout")
        mock_get.side_effect = [scoreboard_resp, summary_resp]

        games_df, box_df = fetch_nba_espn_games(season=2025, dates=["2026-01-15"])
        assert len(games_df) == 1   # game still recorded
        assert len(box_df) == 0     # box score silently skipped


class TestFetchNbaEspnSchedule:
    @patch("pipeline.fetch_nba.requests.get")
    def test_returns_upcoming_games(self, mock_get):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "events": [
                _make_nba_espn_event("1", "Los Angeles Lakers", "Boston Celtics",
                                     0, 0, completed=False, date="2026-02-19T00:00Z"),
            ]
        }
        mock_get.return_value = resp

        fixtures = fetch_nba_espn_schedule()
        assert len(fixtures) >= 1
        assert fixtures[0]["home_team"] == "Lakers"
        assert fixtures[0]["away_team"] == "Celtics"
        # Date is now stored as the queried ET date (YYYY-MM-DD), not the raw UTC event timestamp
        import re
        assert re.match(r"\d{4}-\d{2}-\d{2}$", fixtures[0]["date"])

    @patch("pipeline.fetch_nba.requests.get")
    def test_excludes_completed_games(self, mock_get):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "events": [
                _make_nba_espn_event("1", "Lakers", "Celtics", 112, 108,
                                     completed=True),
                _make_nba_espn_event("2", "Heat", "Bulls", 0, 0,
                                     completed=False, date="2026-02-20T00:00Z"),
            ]
        }
        mock_get.return_value = resp

        fixtures = fetch_nba_espn_schedule()
        assert all(f["home_team"] != "Lakers" for f in fixtures)


# ---------------------------------------------------------------------------
# TestFetchNbaEspnGamesCache
# ---------------------------------------------------------------------------


class TestFetchNbaEspnGamesCache:
    @patch("pipeline.fetch_nba.time.sleep")
    @patch("pipeline.fetch_nba.requests.get")
    def test_skips_box_score_for_cached_game(self, mock_get, mock_sleep, tmp_path):
        """Game already in cache with box_scores → summary endpoint NOT called."""
        cache_path = str(tmp_path / "nba_espn_cache.json")
        existing_cache = {
            "games": {
                "401585634": {
                    "date": "2026-01-15",
                    "home_team": "Lakers",
                    "away_team": "Celtics",
                    "home_goals": 112,
                    "away_goals": 108,
                    "box_scores": [
                        {
                            "team": "Lakers",
                            "pts": 112, "fgm": 44, "fga": 95, "fg3m": 17, "fg3a": 42,
                            "ftm": 5, "fta": 7, "orb": 15, "drb": 34, "to": 10,
                            "possessions": 84.08,
                        },
                        {
                            "team": "Celtics",
                            "pts": 108, "fgm": 40, "fga": 90, "fg3m": 15, "fg3a": 38,
                            "ftm": 8, "fta": 10, "orb": 12, "drb": 30, "to": 11,
                            "possessions": 82.4,
                        },
                    ],
                }
            }
        }
        with open(cache_path, "w") as f:
            json.dump(existing_cache, f)

        # Scoreboard returns the same game that is already cached
        scoreboard_resp = MagicMock()
        scoreboard_resp.json.return_value = {
            "events": [
                _make_nba_espn_event("401585634", "Los Angeles Lakers", "Boston Celtics",
                                     112, 108),
            ]
        }
        scoreboard_resp.raise_for_status = MagicMock()

        # Only one HTTP request should be made (the scoreboard); no summary call
        mock_get.side_effect = [scoreboard_resp]

        games_df, box_df = fetch_nba_espn_games(
            season=2025, dates=["2026-01-15"], cache_path=cache_path
        )

        assert mock_get.call_count == 1
        assert len(games_df) == 1
        assert len(box_df) == 2

    @patch("pipeline.fetch_nba.time.sleep")
    @patch("pipeline.fetch_nba.requests.get")
    def test_cache_written_after_fetch(self, mock_get, mock_sleep, tmp_path):
        """Cache file is created after fetching a new game."""
        cache_path = str(tmp_path / "nba_espn_cache.json")

        scoreboard_resp = MagicMock()
        scoreboard_resp.json.return_value = {
            "events": [
                _make_nba_espn_event("401585634", "Los Angeles Lakers", "Boston Celtics",
                                     112, 108),
            ]
        }
        scoreboard_resp.raise_for_status = MagicMock()

        summary_resp = MagicMock()
        summary_resp.json.return_value = _make_nba_summary(
            "Los Angeles Lakers", "Boston Celtics",
            NBA_SAMPLE_TOTALS, NBA_SAMPLE_TOTALS,
        )
        summary_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [scoreboard_resp, summary_resp]

        fetch_nba_espn_games(season=2025, dates=["2026-01-15"], cache_path=cache_path)

        assert os.path.exists(cache_path)
        with open(cache_path) as f:
            cache = json.load(f)
        assert "401585634" in cache["games"]
        assert cache["games"]["401585634"]["home_team"] == "Lakers"
        assert len(cache["games"]["401585634"]["box_scores"]) == 2

    @patch("pipeline.fetch_nba.time.sleep")
    @patch("pipeline.fetch_nba.requests.get")
    def test_only_recent_dates_fetched_when_cache_has_data(self, mock_get, mock_sleep, tmp_path):
        """When cache has max date 2026-01-18, only dates >= 2026-01-16 are fetched."""
        cache_path = str(tmp_path / "nba_espn_cache.json")
        existing_cache = {
            "games": {
                "999": {
                    "date": "2026-01-18",
                    "home_team": "Lakers",
                    "away_team": "Celtics",
                    "home_goals": 110,
                    "away_goals": 105,
                    "box_scores": [],
                }
            }
        }
        with open(cache_path, "w") as f:
            json.dump(existing_cache, f)

        all_dates = [
            "2026-01-14", "2026-01-15", "2026-01-16", "2026-01-17", "2026-01-18",
        ]

        empty_resp = MagicMock()
        empty_resp.json.return_value = {"events": []}
        empty_resp.raise_for_status = MagicMock()

        # 3 dates from 2026-01-16 onward → 3 scoreboard requests
        mock_get.side_effect = [empty_resp] * 3

        fetch_nba_espn_games(season=2025, dates=all_dates, cache_path=cache_path)

        # Only 3 scoreboard requests (2026-01-16, 2026-01-17, 2026-01-18)
        assert mock_get.call_count == 3
