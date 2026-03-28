"""Tests for MLB fetch helpers."""

import pipeline.fetch_mlb as fetch_mlb
from pipeline.fetch_mlb import _extract_mlb_starting_pitchers, _innings_to_float


class TestInningsToFloat:
    def test_converts_baseball_innings_notation(self):
        assert _innings_to_float("5.0") == 5.0
        assert _innings_to_float("5.1") == 5 + (1 / 3)
        assert _innings_to_float("6.2") == 6 + (2 / 3)


class TestExtractMlbStartingPitchers:
    def test_extracts_starter_names_and_stats(self):
        fetch_mlb._team_map = {
            "New York Yankees": "Yankees",
            "Boston Red Sox": "Red Sox",
        }
        summary = {
            "boxscore": {
                "players": [
                    {
                        "team": {"displayName": "New York Yankees"},
                        "statistics": [
                            {
                                "athletes": [
                                    {
                                        "athlete": {"displayName": "Leadoff Guy"},
                                        "starter": True,
                                        "position": {"abbreviation": "CF"},
                                        "stats": [],
                                    }
                                ]
                            },
                            {
                                "athletes": [
                                    {
                                        "athlete": {"displayName": "Gerrit Cole"},
                                        "starter": True,
                                        "position": {"abbreviation": "P"},
                                        "stats": ["6.1", "4", "2", "2", "1", "8", "1", "94---"],
                                    }
                                ]
                            },
                        ],
                    },
                    {
                        "team": {"displayName": "Boston Red Sox"},
                        "statistics": [
                            {
                                "athletes": [
                                    {
                                        "athlete": {"displayName": "Bat Guy"},
                                        "starter": True,
                                        "position": {"abbreviation": "LF"},
                                        "stats": [],
                                    }
                                ]
                            },
                            {
                                "athletes": [
                                    {
                                        "athlete": {"displayName": "Garrett Crochet"},
                                        "starter": True,
                                        "position": {"abbreviation": "P"},
                                        "stats": ["5.0", "6", "3", "3", "2", "7", "0", "88---"],
                                    }
                                ]
                            },
                        ],
                    },
                ]
            }
        }

        starters = _extract_mlb_starting_pitchers(summary)

        assert starters["Yankees"]["name"] == "Gerrit Cole"
        assert starters["Yankees"]["innings_pitched"] == 6 + (1 / 3)
        assert starters["Yankees"]["earned_runs"] == 2
        assert starters["Yankees"]["strikeouts"] == 8
        assert starters["Red Sox"]["name"] == "Garrett Crochet"
