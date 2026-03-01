"""Tests for pipeline.fetch_data — API client functions."""

from unittest.mock import patch, MagicMock

from pipeline.fetch_data import fetch_odds


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

        odds = fetch_odds(sport_key="basketball_nba")

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

    @patch("pipeline.fetch_data.requests.get")
    def test_nba_two_way_odds_no_draw(self, mock_get):
        """NBA h2h odds have no Draw outcome — draw_odds should be 0.0."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {
                "home_team": "Los Angeles Lakers",
                "away_team": "Boston Celtics",
                "commence_time": "2026-02-19T00:30:00Z",
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Los Angeles Lakers", "price": 2.10},
                                    {"name": "Boston Celtics", "price": 1.75},
                                ],
                            }
                        ],
                    },
                ],
            }
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        odds = fetch_odds(sport_key="basketball_nba")

        assert len(odds) == 1
        match_odds = odds[0]
        assert match_odds["home_odds"] == 2.10
        assert match_odds["draw_odds"] == 0.0
        assert match_odds["away_odds"] == 1.75

    @patch("pipeline.fetch_data.requests.get")
    def test_sport_key_passed_to_url(self, mock_get):
        """sport_key should appear in the API URL."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        fetch_odds(sport_key="basketball_nba")

        actual_url = mock_get.call_args[0][0]
        assert "basketball_nba" in actual_url
