"""Tests for pipeline.fetch_ncaam — ESPN API client for NCAAM."""

from unittest.mock import patch, MagicMock, call
import pandas as pd
import pytest

from pipeline.fetch_ncaam import (
    normalize_ncaam_team_name,
    _parse_box_score_totals,
    fetch_ncaam_games,
    fetch_ncaam_schedule,
    _build_team_map,
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
                     status="final", date="2026-02-15T00:00Z"):
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
            "status": {"type": {"name": status}},
        }],
    }


def _make_summary_response(home_totals, away_totals):
    """Helper: build an ESPN summary response with box score totals."""
    return {
        "boxscore": {
            "teams": [
                {"statistics": [{"totals": home_totals}]},
                {"statistics": [{"totals": away_totals}]},
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

        summary_resp = MagicMock()
        summary_resp.json.return_value = _make_summary_response(SAMPLE_TOTALS, SAMPLE_TOTALS)
        summary_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [scoreboard_resp, summary_resp]

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
                _make_espn_event("1", "Duke Blue Devils", "Kansas Jayhawks", 75, 70, status="final"),
                _make_espn_event("2", "Duke Blue Devils", "Kansas Jayhawks", 0, 0, status="pre"),
            ]
        }
        scoreboard_resp.raise_for_status = MagicMock()

        summary_resp = MagicMock()
        summary_resp.json.return_value = _make_summary_response(SAMPLE_TOTALS, SAMPLE_TOTALS)
        summary_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [scoreboard_resp, summary_resp]

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

        resp2 = MagicMock()
        resp2.json.return_value = {
            "events": [_make_espn_event("2", "Kansas Jayhawks", "Duke Blue Devils", 90, 85, date="2026-02-16T00:00Z")]
        }
        resp2.raise_for_status = MagicMock()

        summary1 = MagicMock()
        summary1.json.return_value = _make_summary_response(SAMPLE_TOTALS, SAMPLE_TOTALS)
        summary1.raise_for_status = MagicMock()

        summary2 = MagicMock()
        summary2.json.return_value = _make_summary_response(SAMPLE_TOTALS, SAMPLE_TOTALS)
        summary2.raise_for_status = MagicMock()

        mock_get.side_effect = [resp1, summary1, resp2, summary2]

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

        summary_resp = MagicMock()
        summary_resp.json.return_value = {"boxscore": {"teams": []}}
        summary_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [scoreboard_resp, summary_resp]

        games_df, box_df = fetch_ncaam_games(season=2025, dates=["2026-02-15"])
        assert len(games_df) == 1
        assert len(box_df) == 0


# ---------------------------------------------------------------------------
# fetch_ncaam_schedule
# ---------------------------------------------------------------------------


class TestFetchNcaamSchedule:
    @patch("pipeline.fetch_ncaam.time.sleep")
    @patch("pipeline.fetch_ncaam._team_map", {"Duke Blue Devils": "Duke", "Kansas Jayhawks": "Kansas"})
    @patch("pipeline.fetch_ncaam.requests.get")
    def test_returns_upcoming_games(self, mock_get, mock_sleep):
        # First day has one game, remaining 7 days are empty
        game_resp = MagicMock()
        game_resp.json.return_value = {
            "events": [
                _make_espn_event("1", "Duke Blue Devils", "Kansas Jayhawks", 0, 0, status="pre", date="2026-02-19T00:00Z"),
            ]
        }
        game_resp.raise_for_status = MagicMock()

        empty_resp = MagicMock()
        empty_resp.json.return_value = {"events": []}
        empty_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [game_resp] + [empty_resp] * 7

        fixtures = fetch_ncaam_schedule()

        assert isinstance(fixtures, list)
        assert len(fixtures) == 1
        assert fixtures[0]["home_team"] == "Duke"
        assert fixtures[0]["away_team"] == "Kansas"
        assert fixtures[0]["date"] == "2026-02-19"

    @patch("pipeline.fetch_ncaam.time.sleep")
    @patch("pipeline.fetch_ncaam._team_map", {"Duke Blue Devils": "Duke", "Kansas Jayhawks": "Kansas", "UConn Huskies": "UConn"})
    @patch("pipeline.fetch_ncaam.requests.get")
    def test_excludes_final_games(self, mock_get, mock_sleep):
        mixed_resp = MagicMock()
        mixed_resp.json.return_value = {
            "events": [
                _make_espn_event("1", "Duke Blue Devils", "Kansas Jayhawks", 75, 70, status="final"),
                _make_espn_event("2", "Kansas Jayhawks", "UConn Huskies", 0, 0, status="pre", date="2026-02-20T00:00Z"),
            ]
        }
        mixed_resp.raise_for_status = MagicMock()

        empty_resp = MagicMock()
        empty_resp.json.return_value = {"events": []}
        empty_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [mixed_resp] + [empty_resp] * 7

        fixtures = fetch_ncaam_schedule()
        assert len(fixtures) == 1
        assert fixtures[0]["home_team"] == "Kansas"

    @patch("pipeline.fetch_ncaam.time.sleep")
    @patch("pipeline.fetch_ncaam._team_map", {})
    @patch("pipeline.fetch_ncaam.requests.get")
    def test_fetches_multiple_days(self, mock_get, mock_sleep):
        """Should make 8 requests (today + 7 days)."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"events": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        fetch_ncaam_schedule()
        assert mock_get.call_count == 8
