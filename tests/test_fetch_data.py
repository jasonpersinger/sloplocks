"""Tests for pipeline.fetch_data — API client functions."""

from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

from pipeline.fetch_data import (
    normalize_team_name,
    fetch_epl_matches,
    fetch_epl_fixtures,
    fetch_odds,
)


# ---------------------------------------------------------------------------
# normalize_team_name
# ---------------------------------------------------------------------------


class TestNormalizeTeamName:
    """Test the football-data.org name -> short name mapping."""

    def test_standard_names(self):
        assert normalize_team_name("Arsenal FC") == "Arsenal"
        assert normalize_team_name("Chelsea FC") == "Chelsea"
        assert normalize_team_name("Liverpool FC") == "Liverpool"
        assert normalize_team_name("Tottenham Hotspur FC") == "Tottenham"

    def test_manchester_abbreviations(self):
        assert normalize_team_name("Manchester City FC") == "Man City"
        assert normalize_team_name("Manchester United FC") == "Man United"

    def test_nottingham(self):
        assert normalize_team_name("Nottingham Forest FC") == "Nottingham Forest"

    def test_already_clean(self):
        """Names not in the map should pass through unchanged."""
        assert normalize_team_name("Arsenal") == "Arsenal"
        assert normalize_team_name("Wolves") == "Wolves"
        assert normalize_team_name("Some Unknown FC") == "Some Unknown FC"


# ---------------------------------------------------------------------------
# fetch_epl_matches
# ---------------------------------------------------------------------------


def _make_fd_match(home, away, home_goals, away_goals, status="FINISHED", date="2025-09-14T15:00:00Z"):
    """Helper: build a single football-data.org match object."""
    full_time = {"home": home_goals, "away": away_goals}
    if home_goals is None:
        full_time = {"home": None, "away": None}
    return {
        "utcDate": date,
        "status": status,
        "homeTeam": {"name": home},
        "awayTeam": {"name": away},
        "score": {"fullTime": full_time},
        "matchday": 4,
    }


class TestFetchEPLMatches:
    """fetch_epl_matches should return a DataFrame of finished results."""

    @patch("pipeline.fetch_data.requests.get")
    def test_returns_dataframe_with_required_columns(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "matches": [
                _make_fd_match("Arsenal FC", "Wolverhampton Wanderers FC", 2, 0),
                _make_fd_match("Liverpool FC", "Ipswich Town FC", 2, 0),
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        df = fetch_epl_matches()

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["date", "home_team", "away_team", "home_goals", "away_goals"]
        assert len(df) == 2
        assert df.iloc[0]["home_team"] == "Arsenal"
        assert df.iloc[0]["away_team"] == "Wolves"
        assert df.iloc[0]["home_goals"] == 2
        assert df.iloc[0]["away_goals"] == 0

    @patch("pipeline.fetch_data.requests.get")
    def test_skips_unfinished_matches(self, mock_get):
        """Matches with None goals (no final score) should be excluded."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "matches": [
                _make_fd_match("Arsenal FC", "Chelsea FC", 1, 0),
                _make_fd_match("Liverpool FC", "Everton FC", None, None),
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        df = fetch_epl_matches()

        assert len(df) == 1
        assert df.iloc[0]["home_team"] == "Arsenal"


# ---------------------------------------------------------------------------
# fetch_epl_fixtures
# ---------------------------------------------------------------------------


class TestFetchEPLFixtures:
    """fetch_epl_fixtures should return upcoming matches."""

    @patch("pipeline.fetch_data.requests.get")
    def test_returns_upcoming_matches(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "matches": [
                {
                    "utcDate": "2026-02-22T15:00:00Z",
                    "homeTeam": {"name": "Arsenal FC"},
                    "awayTeam": {"name": "Chelsea FC"},
                    "matchday": 26,
                },
                {
                    "utcDate": "2026-02-22T17:30:00Z",
                    "homeTeam": {"name": "Manchester City FC"},
                    "awayTeam": {"name": "Liverpool FC"},
                    "matchday": 26,
                },
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        fixtures = fetch_epl_fixtures()

        assert isinstance(fixtures, list)
        assert len(fixtures) == 2
        assert fixtures[0]["home_team"] == "Arsenal"
        assert fixtures[0]["away_team"] == "Chelsea"
        assert fixtures[0]["matchday"] == 26
        assert fixtures[1]["home_team"] == "Man City"
        assert fixtures[1]["away_team"] == "Liverpool"


# ---------------------------------------------------------------------------
# fetch_odds
# ---------------------------------------------------------------------------


class TestFetchOdds:
    """fetch_odds should return best odds per match."""

    @patch("pipeline.fetch_data.requests.get")
    def test_returns_best_odds_per_match(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "commence_time": "2026-02-22T15:00:00Z",
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Arsenal", "price": 1.65},
                                    {"name": "Draw", "price": 3.70},
                                    {"name": "Chelsea", "price": 4.40},
                                ],
                            }
                        ],
                    },
                    {
                        "key": "fanduel",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Arsenal", "price": 1.67},
                                    {"name": "Draw", "price": 3.80},
                                    {"name": "Chelsea", "price": 4.50},
                                ],
                            }
                        ],
                    },
                ],
            }
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        odds = fetch_odds()

        assert isinstance(odds, list)
        assert len(odds) == 1

        match_odds = odds[0]
        assert match_odds["home_team"] == "Arsenal"
        assert match_odds["away_team"] == "Chelsea"
        assert match_odds["commence_time"] == "2026-02-22T15:00:00Z"
        # Best across both bookmakers
        assert match_odds["home_odds"] == 1.67
        assert match_odds["draw_odds"] == 3.80
        assert match_odds["away_odds"] == 4.50
