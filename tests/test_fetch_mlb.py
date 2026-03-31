"""Tests for MLB fetch helpers."""

import pipeline.fetch_mlb as fetch_mlb
from pipeline.fetch_mlb import (
    _extract_confirmed_mlb_lineups,
    _extract_mlb_starting_pitchers,
    _extract_mlb_team_pitching,
    _fetch_ballpark_weather,
    _fetch_team_lineup_profile,
    _fetch_pitcher_profile,
    _innings_to_float,
)


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


class TestExtractMlbTeamPitching:
    def test_extracts_bullpen_aggregate_stats(self):
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
                                        "athlete": {"displayName": "Gerrit Cole"},
                                        "starter": True,
                                        "position": {"abbreviation": "P"},
                                        "stats": ["6.0", "4", "2", "2", "1", "8"],
                                    },
                                    {
                                        "athlete": {"displayName": "Reliever One"},
                                        "starter": False,
                                        "position": {"abbreviation": "P"},
                                        "stats": ["1.2", "1", "0", "0", "1", "2"],
                                    },
                                    {
                                        "athlete": {"displayName": "Reliever Two"},
                                        "starter": False,
                                        "position": {"abbreviation": "P"},
                                        "stats": ["1.1", "0", "1", "1", "0", "1"],
                                    },
                                ]
                            }
                        ],
                    }
                ]
            }
        }

        teams = _extract_mlb_team_pitching(summary)

        assert teams["Yankees"]["starter"]["name"] == "Gerrit Cole"
        assert teams["Yankees"]["bullpen"]["innings_pitched"] == 3.0
        assert teams["Yankees"]["bullpen"]["runs_allowed"] == 1
        assert teams["Yankees"]["bullpen"]["earned_runs"] == 1
        assert teams["Yankees"]["bullpen"]["walks"] == 1
        assert teams["Yankees"]["bullpen"]["strikeouts"] == 3


class TestPitcherProfile:
    def test_fetches_and_caches_pitcher_handedness(self, monkeypatch):
        calls = []

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "throws": {"abbreviation": "R"},
                    "bats": {"abbreviation": "L"},
                }

        def fake_get(url, timeout=30):
            calls.append(url)
            return FakeResponse()

        monkeypatch.setattr(fetch_mlb.requests, "get", fake_get)
        cache = {"pitchers": {}}

        first = _fetch_pitcher_profile("12345", cache)
        second = _fetch_pitcher_profile("12345", cache)

        assert first["throws"] == "R"
        assert first["bats"] == "L"
        assert second == first
        assert len(calls) == 1


class TestBallparkWeather:
    def test_fetches_weather_for_outdoor_park(self, monkeypatch):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "hourly": {
                        "time": ["2026-03-29T19:00", "2026-03-29T20:00"],
                        "temperature_2m": [20.0, 21.0],
                        "wind_speed_10m": [16.0, 18.0],
                        "precipitation_probability": [5, 10],
                    }
                }

        monkeypatch.setattr(fetch_mlb.requests, "get", lambda *args, **kwargs: FakeResponse())
        weather = _fetch_ballpark_weather("Braves", "2026-03-29T20:10:00Z", {"weather": {}})

        assert weather["weather_exposed"] is True
        assert weather["temperature_f"] == 69.8
        assert weather["wind_mph"] == 11.2
        assert weather["precipitation_probability"] == 10

    def test_returns_indoor_stub_for_closed_roof_parks(self):
        weather = _fetch_ballpark_weather("Astros", "2026-03-29T20:10:00Z", {"weather": {}})

        assert weather["weather_exposed"] is False
        assert weather["temperature_f"] is None


class TestTeamLineupProfile:
    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def test_fetches_and_caches_active_hitter_profile(self, monkeypatch):
        calls = []

        roster_payload = {
            "athletes": [
                {
                    "position": "Pitchers",
                    "items": [],
                },
                {
                    "position": "Outfielders",
                    "items": [
                        {
                            "id": "101",
                            "displayName": "Lefty Bat",
                            "status": {"type": "active"},
                            "injuries": [],
                        },
                        {
                            "id": "102",
                            "displayName": "Righty Bat",
                            "status": {"type": "active"},
                            "injuries": [],
                        },
                        {
                            "id": "103",
                            "displayName": "Switch Bat",
                            "status": {"type": "active"},
                            "injuries": [],
                        },
                        {
                            "id": "104",
                            "displayName": "Injured Bat",
                            "status": {"type": "active"},
                            "injuries": [{"status": "day-to-day"}],
                        },
                    ],
                },
            ]
        }
        athlete_payloads = {
            "101": {"bats": {"abbreviation": "L"}, "throws": {"abbreviation": "R"}},
            "102": {"bats": {"abbreviation": "R"}, "throws": {"abbreviation": "R"}},
            "103": {"bats": {"abbreviation": "S"}, "throws": {"abbreviation": "R"}},
        }

        def fake_get(url, timeout=30):
            calls.append(url)
            if "/teams/15/roster" in url:
                return self.FakeResponse(roster_payload)
            player_id = url.rsplit("/", 1)[-1]
            return self.FakeResponse(athlete_payloads[player_id])

        monkeypatch.setattr(fetch_mlb.requests, "get", fake_get)
        cache = {"rosters": {}, "players": {}, "pitchers": {}, "weather": {}, "games": {}}

        first = _fetch_team_lineup_profile("15", cache)
        second = _fetch_team_lineup_profile("15", cache)

        assert first["active_hitters"] == 4
        assert first["available_hitters"] == 3
        assert first["injured_hitters"] == 1
        assert first["key_bat_absence_score"] == 0.0
        assert first["leader_absence_burden"] == 0.0
        assert first["left_handed_batters"] == 1
        assert first["right_handed_batters"] == 1
        assert first["switch_hitters"] == 1
        assert first["lefty_share"] == 0.3333
        assert second == first
        assert len(calls) == 4

    def test_lineup_profile_penalizes_missing_live_batting_leaders(self, monkeypatch):
        calls = []
        roster_payload = {
            "athletes": [
                {
                    "position": "Infielders",
                    "items": [
                        {
                            "id": "101",
                            "displayName": "Slugger",
                            "status": {"type": "active"},
                            "injuries": [{"status": "out"}],
                        },
                        {
                            "id": "102",
                            "displayName": "Healthy Bat",
                            "status": {"type": "active"},
                            "injuries": [],
                        },
                    ],
                },
            ]
        }
        athlete_payloads = {
            "101": {"bats": {"abbreviation": "L"}, "throws": {"abbreviation": "R"}},
            "102": {"bats": {"abbreviation": "R"}, "throws": {"abbreviation": "R"}},
        }

        def fake_get(url, timeout=30):
            calls.append(url)
            if "/teams/15/roster" in url:
                return self.FakeResponse(roster_payload)
            player_id = url.rsplit("/", 1)[-1]
            return self.FakeResponse(athlete_payloads[player_id])

        monkeypatch.setattr(fetch_mlb.requests, "get", fake_get)
        profile = _fetch_team_lineup_profile(
            "15",
            cache={"rosters": {}, "players": {}, "pitchers": {}, "weather": {}, "games": {}},
            leader_weights={"101": 1.0},
        )

        assert profile["injured_hitters"] == 1
        assert profile["key_bat_absence_score"] == 1.0
        assert profile["leader_absence_burden"] == 1.0

    def test_lineup_profile_merges_confirmed_batting_order(self, monkeypatch):
        roster_payload = {
            "athletes": [
                {
                    "position": "Infielders",
                    "items": [
                        {"id": "101", "status": {"type": "active"}, "injuries": []},
                        {"id": "102", "status": {"type": "active"}, "injuries": []},
                        {"id": "103", "status": {"type": "active"}, "injuries": []},
                    ],
                },
            ]
        }
        athlete_payloads = {
            "101": {"bats": {"abbreviation": "L"}, "throws": {"abbreviation": "R"}},
            "102": {"bats": {"abbreviation": "R"}, "throws": {"abbreviation": "R"}},
            "103": {"bats": {"abbreviation": "S"}, "throws": {"abbreviation": "R"}},
        }

        def fake_get(url, timeout=30):
            if "/teams/15/roster" in url:
                return self.FakeResponse(roster_payload)
            player_id = url.rsplit("/", 1)[-1]
            return self.FakeResponse(athlete_payloads[player_id])

        monkeypatch.setattr(fetch_mlb.requests, "get", fake_get)
        profile = _fetch_team_lineup_profile(
            "15",
            cache={"rosters": {}, "players": {}, "pitchers": {}, "weather": {}, "games": {}},
            leader_weights={"101": 1.0, "102": 0.8, "103": 0.6},
            confirmed_lineup={
                "confirmed_lineup": True,
                "confirmed_hitters": 2,
                "confirmed_top_order_score": 0.62,
                "confirmed_lefty_share": 0.5,
                "confirmed_righty_share": 0.5,
                "confirmed_switch_share": 0.0,
                "player_ids": ["101", "102"],
            },
        )

        assert profile["confirmed_lineup"] is True
        assert profile["confirmed_hitters"] == 2
        assert profile["confirmed_top_order_score"] == 0.62
        assert profile["confirmed_leader_absence_burden"] == 0.6


class TestConfirmedMlbLineups:
    def test_extracts_confirmed_batting_order_from_summary(self, monkeypatch):
        fetch_mlb._team_map = {"New York Yankees": "Yankees"}

        class FakeResponse:
            def __init__(self, bats):
                self.bats = bats

            def raise_for_status(self):
                return None

            def json(self):
                return {"bats": {"abbreviation": self.bats}, "throws": {"abbreviation": "R"}}

        bats = {"1": "L", "2": "R", "3": "S"}

        def fake_get(url, timeout=30):
            player_id = url.rsplit("/", 1)[-1]
            return FakeResponse(bats[player_id])

        monkeypatch.setattr(fetch_mlb.requests, "get", fake_get)
        lineups = _extract_confirmed_mlb_lineups(
            {
                "rosters": [
                    {
                        "team": {"displayName": "New York Yankees"},
                        "roster": [
                            {"starter": True, "batOrder": 1, "position": {"abbreviation": "CF"}, "athlete": {"id": "1"}},
                            {"starter": True, "batOrder": 2, "position": {"abbreviation": "1B"}, "athlete": {"id": "2"}},
                            {"starter": True, "batOrder": 3, "position": {"abbreviation": "DH"}, "athlete": {"id": "3"}},
                            {"starter": True, "batOrder": None, "position": {"abbreviation": "P"}, "athlete": {"id": "99"}},
                        ],
                    }
                ]
            },
            cache={"players": {}},
        )

        assert lineups["Yankees"]["confirmed_lineup"] is False
        assert lineups["Yankees"]["confirmed_hitters"] == 3
        assert lineups["Yankees"]["confirmed_lefty_share"] == 0.3333
        assert lineups["Yankees"]["confirmed_switch_share"] == 0.3333
