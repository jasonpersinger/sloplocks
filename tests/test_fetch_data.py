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

    @patch("pipeline.fetch_data.requests.get")
    def test_include_totals_returns_consensus_total_line(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {
                "home_team": "Atlanta Braves",
                "away_team": "New York Mets",
                "commence_time": "2026-03-29T23:20:00Z",
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Atlanta Braves", "price": 1.85},
                                    {"name": "New York Mets", "price": 2.05},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "price": 1.95, "point": 8.5},
                                    {"name": "Under", "price": 1.87, "point": 8.5},
                                ],
                            },
                        ],
                    },
                    {
                        "key": "fanduel",
                        "markets": [
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "price": 2.00, "point": 8.5},
                                    {"name": "Under", "price": 1.91, "point": 8.5},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "price": 1.82, "point": 9.0},
                                    {"name": "Under", "price": 2.05, "point": 9.0},
                                ],
                            },
                        ],
                    },
                ],
            }
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        odds = fetch_odds(sport_key="baseball_mlb", include_totals=True)

        assert len(odds) == 1
        match_odds = odds[0]
        assert match_odds["total_line"] == 8.5
        assert match_odds["over_odds"] == 2.00
        assert match_odds["under_odds"] == 1.91

    @patch("pipeline.fetch_data.requests.get")
    def test_include_totals_uses_combined_markets_param(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        fetch_odds(sport_key="baseball_mlb", include_totals=True)

        params = mock_get.call_args.kwargs["params"]
        assert params["markets"] == "h2h,totals"
