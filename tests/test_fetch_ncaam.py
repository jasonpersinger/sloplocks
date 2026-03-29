"""Tests for pipeline.fetch_ncaam — ESPN API client for NCAAM."""

import json
import os
from unittest.mock import patch, MagicMock, call
import pandas as pd
import pytest

from pipeline.fetch_ncaam import (
    normalize_ncaam_team_name,
    _parse_box_score_totals,
    fetch_ncaam_games,
    fetch_ncaam_schedule,
    _build_team_map,
    _load_espn_cache,
    _save_espn_cache,
    _incremental_dates,
)


# ---------------------------------------------------------------------------
# Sample ESPN API responses
# ---------------------------------------------------------------------------

SAMPLE_TEAMS_RESPONSE = {
    "sports": [{
        "leagues": [{
            "teams": [
                {"team": {"displayName": "Duke Blue Devils", "location": "Duke"}},
                {"team": {"displayName": "North Carolina Tar Heels", "location": "North Carolina"}},
                {"team": {"displayName": "Kansas Jayhawks", "location": "Kansas"}},
                {"team": {"displayName": "UConn Huskies", "location": "UConn"}},
            ]
        }]
    }]
}


def _make_espn_event(event_id, home_name, away_name, home_score, away_score,
                     completed=True, date="2026-02-15T00:00Z"):
    """Helper: build an ESPN scoreboard event object."""
    return {
        "id": event_id,
        "date": date,
        "competitions": [{
            "competitors": [
                {
                    "homeAway": "home",
                    "score": str(home_score),
                    "team": {"displayName": home_name},
                },
                {
                    "homeAway": "away",
                    "score": str(away_score),
                    "team": {"displayName": away_name},
                },
            ],
            "status": {"type": {"completed": completed}},
        }],
    }


def _make_summary_response(home_totals, away_totals,
                           home_name="Duke Blue Devils",
                           away_name="Kansas Jayhawks"):
    """Helper: build an ESPN summary response with box score totals."""
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


SAMPLE_TOTALS = ["", "75", "28-58", "8-20", "11-14", "35", "15", "10", "5", "3", "8", "27", "12"]


# ---------------------------------------------------------------------------
# _build_team_map
# ---------------------------------------------------------------------------


class TestBuildTeamMap:
    @patch("pipeline.fetch_ncaam.requests.get")
    def test_builds_map_from_espn_teams(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = SAMPLE_TEAMS_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        team_map = _build_team_map()

        assert team_map["Duke Blue Devils"] == "Duke"
        assert team_map["North Carolina Tar Heels"] == "North Carolina"
        assert team_map["Kansas Jayhawks"] == "Kansas"
        assert team_map["UConn Huskies"] == "UConn"

    @patch("pipeline.fetch_ncaam.requests.get")
    def test_handles_empty_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"sports": [{"leagues": [{"teams": []}]}]}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        team_map = _build_team_map()
        assert team_map == {}


# ---------------------------------------------------------------------------
# normalize_ncaam_team_name
# ---------------------------------------------------------------------------


class TestNormalizeNcaamTeamName:
    @patch("pipeline.fetch_ncaam._team_map", {"Duke Blue Devils": "Duke", "Kansas Jayhawks": "Kansas"})
    def test_maps_known_teams(self):
        assert normalize_ncaam_team_name("Duke Blue Devils") == "Duke"
        assert normalize_ncaam_team_name("Kansas Jayhawks") == "Kansas"

    @patch("pipeline.fetch_ncaam._team_map", {"Duke Blue Devils": "Duke"})
    def test_returns_input_for_unknown_team(self):
        assert normalize_ncaam_team_name("Unknown Team") == "Unknown Team"

    @patch("pipeline.fetch_ncaam._team_map", None)
    @patch("pipeline.fetch_ncaam._build_team_map")
    def test_lazy_loads_team_map(self, mock_build):
        mock_build.return_value = {"Duke Blue Devils": "Duke"}
        result = normalize_ncaam_team_name("Duke Blue Devils")
        assert result == "Duke"
        mock_build.assert_called_once()


# ---------------------------------------------------------------------------
# _parse_box_score_totals
# ---------------------------------------------------------------------------


class TestParseBoxScoreTotals:
    def test_parses_standard_totals(self):
        result = _parse_box_score_totals(SAMPLE_TOTALS)

        assert result["pts"] == 75
        assert result["fgm"] == 28
        assert result["fga"] == 58
        assert result["fg3m"] == 8
        assert result["fg3a"] == 20
        assert result["ftm"] == 11
        assert result["fta"] == 14
        assert result["orb"] == 8
        assert result["drb"] == 27
        assert result["to"] == 10
        # possessions = FGA - ORB + TO + 0.44 * FTA = 58 - 8 + 10 + 0.44 * 14 = 66.16
        assert abs(result["possessions"] - 66.16) < 0.01

    def test_zero_stats(self):
        totals = ["", "0", "0-0", "0-0", "0-0", "0", "0", "0", "0", "0", "0", "0", "0"]
        result = _parse_box_score_totals(totals)
        assert result["pts"] == 0
        assert result["fgm"] == 0
        assert result["possessions"] == 0.0

    def test_high_scoring_game(self):
        totals = ["", "100", "38-70", "12-30", "12-16", "40", "20", "8", "7", "5", "10", "30", "15"]
        result = _parse_box_score_totals(totals)
        assert result["pts"] == 100
        assert result["fgm"] == 38
        assert result["fga"] == 70
        # possessions = 70 - 10 + 8 + 0.44 * 16 = 75.04
        assert abs(result["possessions"] - 75.04) < 0.01


# ---------------------------------------------------------------------------
# fetch_ncaam_games
# ---------------------------------------------------------------------------


class TestFetchNcaamGames:
    @patch("pipeline.fetch_ncaam.time.sleep")
    @patch("pipeline.fetch_ncaam._team_map", {"Duke Blue Devils": "Duke", "North Carolina Tar Heels": "North Carolina"})
    @patch("pipeline.fetch_ncaam.requests.get")
    def test_returns_games_and_box_scores(self, mock_get, mock_sleep):
        scoreboard_resp = MagicMock()
        scoreboard_resp.json.return_value = {
            "events": [
                _make_espn_event("401825518", "Duke Blue Devils", "North Carolina Tar Heels", 75, 70),
            ]
        }
        scoreboard_resp.raise_for_status = MagicMock()

        empty_scoreboard_resp = MagicMock()
        empty_scoreboard_resp.json.return_value = {"events": []}
        empty_scoreboard_resp.raise_for_status = MagicMock()

        summary_resp = MagicMock()
        summary_resp.json.return_value = _make_summary_response(SAMPLE_TOTALS, SAMPLE_TOTALS)
        summary_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [scoreboard_resp, empty_scoreboard_resp, summary_resp]

        games_df, box_df = fetch_ncaam_games(season=2025, dates=["2026-02-15"])

        assert isinstance(games_df, pd.DataFrame)
        assert isinstance(box_df, pd.DataFrame)
        assert len(games_df) == 1
        assert games_df.iloc[0]["home_team"] == "Duke"
        assert games_df.iloc[0]["away_team"] == "North Carolina"
        assert games_df.iloc[0]["home_goals"] == 75
        assert games_df.iloc[0]["away_goals"] == 70

        # Two box score rows (home + away)
        assert len(box_df) == 2
        assert list(box_df.columns) == [
            "game_id", "team", "date", "pts", "fgm", "fga", "fg3m", "fg3a",
            "ftm", "fta", "orb", "drb", "to", "possessions",
        ]

    @patch("pipeline.fetch_ncaam.time.sleep")
    @patch("pipeline.fetch_ncaam._team_map", {"Duke Blue Devils": "Duke", "Kansas Jayhawks": "Kansas"})
    @patch("pipeline.fetch_ncaam.requests.get")
    def test_skips_non_final_games(self, mock_get, mock_sleep):
        scoreboard_resp = MagicMock()
        scoreboard_resp.json.return_value = {
            "events": [
                _make_espn_event("1", "Duke Blue Devils", "Kansas Jayhawks", 75, 70, completed=True),
                _make_espn_event("2", "Duke Blue Devils", "Kansas Jayhawks", 0, 0, completed=False),
            ]
        }
        scoreboard_resp.raise_for_status = MagicMock()

        empty_scoreboard_resp = MagicMock()
        empty_scoreboard_resp.json.return_value = {"events": []}
        empty_scoreboard_resp.raise_for_status = MagicMock()

        summary_resp = MagicMock()
        summary_resp.json.return_value = _make_summary_response(SAMPLE_TOTALS, SAMPLE_TOTALS)
        summary_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [scoreboard_resp, empty_scoreboard_resp, summary_resp]

        games_df, box_df = fetch_ncaam_games(season=2025, dates=["2026-02-15"])
        assert len(games_df) == 1
        assert len(box_df) == 2  # only box scores for the final game

    @patch("pipeline.fetch_ncaam.time.sleep")
    @patch("pipeline.fetch_ncaam._team_map", {"Duke Blue Devils": "Duke", "Kansas Jayhawks": "Kansas"})
    @patch("pipeline.fetch_ncaam.requests.get")
    def test_multiple_dates(self, mock_get, mock_sleep):
        resp1 = MagicMock()
        resp1.json.return_value = {
            "events": [_make_espn_event("1", "Duke Blue Devils", "Kansas Jayhawks", 80, 70, date="2026-02-15T00:00Z")]
        }
        resp1.raise_for_status = MagicMock()

        empty_resp1 = MagicMock()
        empty_resp1.json.return_value = {"events": []}
        empty_resp1.raise_for_status = MagicMock()

        resp2 = MagicMock()
        resp2.json.return_value = {
            "events": [_make_espn_event("2", "Kansas Jayhawks", "Duke Blue Devils", 90, 85, date="2026-02-16T00:00Z")]
        }
        resp2.raise_for_status = MagicMock()

        empty_resp2 = MagicMock()
        empty_resp2.json.return_value = {"events": []}
        empty_resp2.raise_for_status = MagicMock()

        summary1 = MagicMock()
        summary1.json.return_value = _make_summary_response(SAMPLE_TOTALS, SAMPLE_TOTALS)
        summary1.raise_for_status = MagicMock()

        summary2 = MagicMock()
        summary2.json.return_value = _make_summary_response(SAMPLE_TOTALS, SAMPLE_TOTALS)
        summary2.raise_for_status = MagicMock()

        mock_get.side_effect = [resp1, empty_resp1, summary1, resp2, empty_resp2, summary2]

        games_df, box_df = fetch_ncaam_games(season=2025, dates=["2026-02-15", "2026-02-16"])
        assert len(games_df) == 2
        assert len(box_df) == 4

    @patch("pipeline.fetch_ncaam.time.sleep")
    @patch("pipeline.fetch_ncaam._team_map", {})
    @patch("pipeline.fetch_ncaam.requests.get")
    def test_empty_scoreboard(self, mock_get, mock_sleep):
        resp = MagicMock()
        resp.json.return_value = {"events": []}
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        games_df, box_df = fetch_ncaam_games(season=2025, dates=["2026-02-15"])
        assert len(games_df) == 0
        assert len(box_df) == 0

    @patch("pipeline.fetch_ncaam.time.sleep")
    @patch("pipeline.fetch_ncaam._team_map", {"Duke Blue Devils": "Duke", "Kansas Jayhawks": "Kansas"})
    @patch("pipeline.fetch_ncaam.requests.get")
    def test_handles_missing_box_score(self, mock_get, mock_sleep):
        """Games with missing/bad box score data should still appear in games_df."""
        scoreboard_resp = MagicMock()
        scoreboard_resp.json.return_value = {
            "events": [
                _make_espn_event("1", "Duke Blue Devils", "Kansas Jayhawks", 75, 70),
            ]
        }
        scoreboard_resp.raise_for_status = MagicMock()

        empty_scoreboard_resp = MagicMock()
        empty_scoreboard_resp.json.return_value = {"events": []}
        empty_scoreboard_resp.raise_for_status = MagicMock()

        summary_resp = MagicMock()
        summary_resp.json.return_value = {"boxscore": {"teams": []}}
        summary_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [scoreboard_resp, empty_scoreboard_resp, summary_resp]

        games_df, box_df = fetch_ncaam_games(season=2025, dates=["2026-02-15"])
        assert len(games_df) == 1
        assert len(box_df) == 0


# ---------------------------------------------------------------------------
# fetch_ncaam_schedule
# ---------------------------------------------------------------------------


class TestFetchNcaamSchedule:
    @patch("pipeline.fetch_ncaam._team_map", {"Duke Blue Devils": "Duke", "Kansas Jayhawks": "Kansas"})
    @patch("pipeline.fetch_ncaam.requests.get")
    def test_returns_upcoming_games(self, mock_get):
        # Single request for today's scoreboard
        game_resp = MagicMock()
        game_resp.json.return_value = {
            "events": [
                _make_espn_event("1", "Duke Blue Devils", "Kansas Jayhawks", 0, 0, completed=False, date="2026-02-19T00:00Z"),
            ]
        }
        game_resp.raise_for_status = MagicMock()
        mock_get.return_value = game_resp

        fixtures = fetch_ncaam_schedule()

        assert isinstance(fixtures, list)
        assert len(fixtures) == 1
        assert fixtures[0]["home_team"] == "Duke"
        assert fixtures[0]["away_team"] == "Kansas"
        # Date is the queried ET date (YYYY-MM-DD), not the raw UTC event timestamp
        import re
        assert re.match(r"\d{4}-\d{2}-\d{2}$", fixtures[0]["date"])

    @patch("pipeline.fetch_ncaam._team_map", {"Duke Blue Devils": "Duke", "Kansas Jayhawks": "Kansas", "UConn Huskies": "UConn"})
    @patch("pipeline.fetch_ncaam.requests.get")
    def test_includes_final_games_with_completed_flag(self, mock_get):
        mixed_resp = MagicMock()
        mixed_resp.json.return_value = {
            "events": [
                _make_espn_event("1", "Duke Blue Devils", "Kansas Jayhawks", 75, 70, completed=True),
                _make_espn_event("2", "Kansas Jayhawks", "UConn Huskies", 0, 0, completed=False, date="2026-02-20T00:00Z"),
            ]
        }
        mixed_resp.raise_for_status = MagicMock()
        mock_get.return_value = mixed_resp

        fixtures = fetch_ncaam_schedule()
        assert len(fixtures) == 2
        completed = next(f for f in fixtures if f["home_team"] == "Duke")
        pending = next(f for f in fixtures if f["home_team"] == "Kansas")
        assert completed["completed"] is True
        assert pending["completed"] is False

    @patch("pipeline.fetch_ncaam._team_map", {})
    @patch("pipeline.fetch_ncaam.requests.get")
    def test_fetches_only_today(self, mock_get):
        """Should make exactly 1 request (today's scoreboard only)."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"events": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        fetch_ncaam_schedule()
        assert mock_get.call_count == 1


# ---------------------------------------------------------------------------
# TestFetchNcaamGamesCache
# ---------------------------------------------------------------------------


class TestFetchNcaamGamesCache:
    @patch("pipeline.fetch_ncaam.time.sleep")
    @patch("pipeline.fetch_ncaam._team_map", {"Duke Blue Devils": "Duke", "North Carolina Tar Heels": "North Carolina"})
    @patch("pipeline.fetch_ncaam.requests.get")
    def test_skips_box_score_for_cached_game(self, mock_get, mock_sleep, tmp_path):
        """Game already in cache with box_scores → summary endpoint NOT called."""
        cache_path = str(tmp_path / "espn_cache.json")
        existing_cache = {
            "games": {
                "401825518": {
                    "date": "2026-02-15",
                    "home_team": "Duke",
                    "away_team": "North Carolina",
                    "home_goals": 75,
                    "away_goals": 70,
                    "box_scores": [
                        {
                            "team": "Duke",
                            "pts": 75, "fgm": 28, "fga": 58, "fg3m": 8, "fg3a": 20,
                            "ftm": 11, "fta": 14, "orb": 8, "drb": 27, "to": 10,
                            "possessions": 66.16,
                        },
                        {
                            "team": "North Carolina",
                            "pts": 70, "fgm": 25, "fga": 55, "fg3m": 6, "fg3a": 18,
                            "ftm": 14, "fta": 18, "orb": 9, "drb": 30, "to": 12,
                            "possessions": 63.92,
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
                _make_espn_event("401825518", "Duke Blue Devils", "North Carolina Tar Heels", 75, 70),
            ]
        }
        scoreboard_resp.raise_for_status = MagicMock()

        empty_scoreboard_resp = MagicMock()
        empty_scoreboard_resp.json.return_value = {"events": []}
        empty_scoreboard_resp.raise_for_status = MagicMock()

        # Two scoreboard requests (groups 50 and 100); no summary call.
        mock_get.side_effect = [scoreboard_resp, empty_scoreboard_resp]

        games_df, box_df = fetch_ncaam_games(
            season=2025, dates=["2026-02-15"], cache_path=cache_path
        )

        assert mock_get.call_count == 2
        assert len(games_df) == 1
        assert len(box_df) == 2

    @patch("pipeline.fetch_ncaam.time.sleep")
    @patch("pipeline.fetch_ncaam._team_map", {"Duke Blue Devils": "Duke", "North Carolina Tar Heels": "North Carolina"})
    @patch("pipeline.fetch_ncaam.requests.get")
    def test_cache_written_after_fetch(self, mock_get, mock_sleep, tmp_path):
        """Cache file is created after fetching a new game."""
        cache_path = str(tmp_path / "espn_cache.json")

        scoreboard_resp = MagicMock()
        scoreboard_resp.json.return_value = {
            "events": [
                _make_espn_event("401825518", "Duke Blue Devils", "North Carolina Tar Heels", 75, 70),
            ]
        }
        scoreboard_resp.raise_for_status = MagicMock()

        summary_resp = MagicMock()
        summary_resp.json.return_value = _make_summary_response(
            SAMPLE_TOTALS, SAMPLE_TOTALS,
            home_name="Duke Blue Devils", away_name="North Carolina Tar Heels",
        )
        summary_resp.raise_for_status = MagicMock()

        empty_scoreboard_resp = MagicMock()
        empty_scoreboard_resp.json.return_value = {"events": []}
        empty_scoreboard_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [scoreboard_resp, empty_scoreboard_resp, summary_resp]

        fetch_ncaam_games(season=2025, dates=["2026-02-15"], cache_path=cache_path)

        assert os.path.exists(cache_path)
        with open(cache_path) as f:
            cache = json.load(f)
        assert "401825518" in cache["games"]
        assert cache["games"]["401825518"]["home_team"] == "Duke"
        assert len(cache["games"]["401825518"]["box_scores"]) == 2

    @patch("pipeline.fetch_ncaam.time.sleep")
    @patch("pipeline.fetch_ncaam._team_map", {"Duke Blue Devils": "Duke", "North Carolina Tar Heels": "North Carolina"})
    @patch("pipeline.fetch_ncaam.requests.get")
    def test_only_recent_dates_fetched_when_cache_has_data(self, mock_get, mock_sleep, tmp_path):
        """When cache has max date 2026-02-18, only dates >= 2026-02-16 are fetched."""
        cache_path = str(tmp_path / "espn_cache.json")
        existing_cache = {
            "games": {
                "999": {
                    "date": "2026-02-18",
                    "home_team": "Duke",
                    "away_team": "North Carolina",
                    "home_goals": 80,
                    "away_goals": 75,
                    "box_scores": [],
                }
            }
        }
        with open(cache_path, "w") as f:
            json.dump(existing_cache, f)

        all_dates = [
            "2026-02-14", "2026-02-15", "2026-02-16", "2026-02-17",
            "2026-02-18", "2026-02-19", "2026-02-20",
        ]

        empty_resp = MagicMock()
        empty_resp.json.return_value = {"events": []}
        empty_resp.raise_for_status = MagicMock()

        # 5 dates from 2026-02-16 onward × 2 groups → 10 scoreboard requests
        mock_get.side_effect = [empty_resp] * 10

        fetch_ncaam_games(season=2025, dates=all_dates, cache_path=cache_path)

        # Only dates 2026-02-16 through 2026-02-20 are fetched, but with 2 groups each.
        assert mock_get.call_count == 10
