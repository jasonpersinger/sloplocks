"""Tests for pipeline.fetch_nhl."""

from unittest.mock import MagicMock, patch

from pipeline.fetch_nhl import fetch_nhl_games, fetch_nhl_schedule, normalize_nhl_team_name


def _make_nhl_event(event_id, home_name, away_name, home_goals, away_goals, completed=True):
    return {
        "id": event_id,
        "date": "2026-03-29T23:00Z",
        "competitions": [
            {
                "date": "2026-03-29T23:00Z",
                "neutralSite": False,
                "status": {"type": {"completed": completed}},
                "competitors": [
                    {
                        "homeAway": "home",
                        "team": {"displayName": home_name},
                        "score": str(home_goals),
                        "probables": [
                            {
                                "name": "probableStartingGoalie",
                                "playerId": 31,
                                "athlete": {"id": "31", "displayName": "Home Goalie"},
                                "status": {"type": "confirmed"},
                            }
                        ],
                        "statistics": [
                            {"name": "saves", "displayValue": "28"},
                            {"name": "savePct", "displayValue": ".933"},
                        ],
                    },
                    {
                        "homeAway": "away",
                        "team": {"displayName": away_name},
                        "score": str(away_goals),
                        "probables": [
                            {
                                "name": "probableStartingGoalie",
                                "playerId": 35,
                                "athlete": {"id": "35", "displayName": "Away Goalie"},
                                "status": {"type": "projected"},
                            }
                        ],
                        "statistics": [
                            {"name": "saves", "displayValue": "24"},
                            {"name": "savePct", "displayValue": ".889"},
                        ],
                    },
                ],
            }
        ],
    }


class TestNormalizeNhlTeamName:
    @patch("pipeline.fetch_nhl._team_map", {"Boston Bruins": "Bruins"})
    def test_maps_display_name_to_short_name(self):
        assert normalize_nhl_team_name("Boston Bruins") == "Bruins"
        assert normalize_nhl_team_name("Unknown Club") == "Unknown Club"

    @patch("pipeline.fetch_nhl._team_map", {})
    def test_handles_common_alias_and_accent_variants(self):
        assert normalize_nhl_team_name("Montréal Canadiens") == "Canadiens"
        assert normalize_nhl_team_name("Utah Hockey Club") == "Utah"


class TestFetchNhlGames:
    @patch("pipeline.fetch_nhl._team_map", {"Boston Bruins": "Bruins", "Toronto Maple Leafs": "Maple Leafs"})
    @patch("pipeline.fetch_nhl.requests.get")
    def test_fetches_completed_games(self, mock_get):
        scoreboard_resp = MagicMock()
        scoreboard_resp.raise_for_status = MagicMock()
        scoreboard_resp.json.return_value = {
            "events": [_make_nhl_event("1", "Boston Bruins", "Toronto Maple Leafs", 4, 2)]
        }
        summary_resp = MagicMock()
        summary_resp.raise_for_status = MagicMock()
        summary_resp.json.return_value = {
            "boxscore": {
                "teams": [
                    {
                        "team": {"displayName": "Boston Bruins"},
                        "statistics": [
                            {"name": "shotsTotal", "displayValue": "31"},
                            {"name": "faceoffPercent", "displayValue": "57.3"},
                            {"name": "powerPlayPct", "displayValue": "25.0"},
                            {"name": "takeaways", "displayValue": "8"},
                            {"name": "giveaways", "displayValue": "5"},
                            {"name": "penaltyMinutes", "displayValue": "6"},
                        ],
                    },
                    {
                        "team": {"displayName": "Toronto Maple Leafs"},
                        "statistics": [
                            {"name": "shotsTotal", "displayValue": "26"},
                            {"name": "faceoffPercent", "displayValue": "42.7"},
                            {"name": "powerPlayPct", "displayValue": "12.5"},
                            {"name": "takeaways", "displayValue": "5"},
                            {"name": "giveaways", "displayValue": "7"},
                            {"name": "penaltyMinutes", "displayValue": "10"},
                        ],
                    },
                ],
                "players": [
                    {
                        "team": {"displayName": "Boston Bruins"},
                        "statistics": [
                            {
                                "name": "goalies",
                                "labels": ["GA", "SA", "SOS", "SOSA", "SV", "SV%", "ESSV", "PPSV", "SHSV", "TOI"],
                                "athletes": [
                                    {
                                        "athlete": {"id": "31", "displayName": "Linus Ullmark"},
                                        "stats": ["2", "26", "0", "0", "24", ".923", "20", "4", "0", "60:00"],
                                    }
                                ],
                            }
                        ],
                    },
                    {
                        "team": {"displayName": "Toronto Maple Leafs"},
                        "statistics": [
                            {
                                "name": "goalies",
                                "labels": ["GA", "SA", "SOS", "SOSA", "SV", "SV%", "ESSV", "PPSV", "SHSV", "TOI"],
                                "athletes": [
                                    {
                                        "athlete": {"id": "35", "displayName": "Joseph Woll"},
                                        "stats": ["4", "31", "0", "0", "27", ".871", "23", "4", "0", "58:14"],
                                    }
                                ],
                            }
                        ],
                    },
                ],
            }
        }
        mock_get.side_effect = [scoreboard_resp, summary_resp]

        games_df, box_df = fetch_nhl_games(season=2025, dates=["2026-03-29"])

        assert box_df is None
        assert len(games_df) == 1
        assert games_df.iloc[0]["home_team"] == "Bruins"
        assert games_df.iloc[0]["away_team"] == "Maple Leafs"
        assert games_df.iloc[0]["home_shots"] == 31
        assert games_df.iloc[0]["away_save_pct"] == 0.889
        assert games_df.iloc[0]["home_faceoff_pct"] == 0.573
        assert games_df.iloc[0]["away_power_play_pct"] == 0.125
        assert games_df.iloc[0]["away_penalty_minutes"] == 10.0
        assert games_df.iloc[0]["home_goalie"] == "Linus Ullmark"
        assert games_df.iloc[0]["away_goalie_save_pct"] == 0.871


class TestFetchNhlSchedule:
    @patch("pipeline.fetch_nhl._team_map", {"Boston Bruins": "Bruins", "Toronto Maple Leafs": "Maple Leafs"})
    @patch("pipeline.fetch_nhl.requests.get")
    def test_fetches_schedule(self, mock_get):
        scoreboard_resp = MagicMock()
        scoreboard_resp.raise_for_status = MagicMock()
        scoreboard_resp.json.return_value = {
            "events": [_make_nhl_event("1", "Boston Bruins", "Toronto Maple Leafs", 0, 0, completed=False)]
        }
        mock_get.return_value = scoreboard_resp

        fixtures = fetch_nhl_schedule()

        assert len(fixtures) == 1
        assert fixtures[0]["home_team"] == "Bruins"
        assert fixtures[0]["away_team"] == "Maple Leafs"
        assert fixtures[0]["completed"] is False
        assert fixtures[0]["start_time"] == "2026-03-29T23:00Z"
        assert fixtures[0]["home_goalie"] == "Home Goalie"
        assert fixtures[0]["away_goalie_status"] == "projected"
