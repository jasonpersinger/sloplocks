"""Tests for MLB fetch helpers."""

from datetime import datetime, timedelta, timezone

import pipeline.fetch_mlb as fetch_mlb
from pipeline.fetch_mlb import (
    _extract_confirmed_mlb_lineups,
    _extract_mlb_starting_pitchers,
    _extract_mlb_team_pitching,
    _fetch_ballpark_weather,
    _fetch_statsapi_probable_pitchers,
    _fetch_team_lineup_profile,
    _fetch_pitcher_profile,
    _innings_to_float,
)


class TestInningsToFloat:
    def test_converts_baseball_innings_notation(self):
        assert _innings_to_float("5.0") == 5.0
        assert _innings_to_float("5.1") == 5 + (1 / 3)
        assert _innings_to_float("6.2") == 6 + (2 / 3)


def test_placeholder_pitcher_detection_covers_common_variants():
    for value in ("TBD", "TBA", "Undecided", "To Be Announced", "Unknown", "", None):
        assert fetch_mlb._is_placeholder_pitcher_name(value)
    assert not fetch_mlb._is_placeholder_pitcher_name("Gerrit Cole")


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

        def fake_get(url, timeout=30, **_kwargs):
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


class TestStatsApiProbables:
    def test_fetches_and_caches_probable_pitchers_by_team(self, monkeypatch):
        calls = []
        fetch_mlb._team_map = {
            "New York Yankees": "Yankees",
            "Boston Red Sox": "Red Sox",
        }

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "dates": [
                        {
                            "games": [
                                {
                                    "teams": {
                                        "home": {
                                            "team": {"name": "New York Yankees"},
                                            "probablePitcher": {
                                                "id": 123,
                                                "fullName": "Gerrit Cole",
                                                "pitchHand": {"code": "R"},
                                            },
                                        },
                                        "away": {
                                            "team": {"name": "Boston Red Sox"},
                                            "probablePitcher": {
                                                "id": 456,
                                                "fullName": "Garrett Crochet",
                                                "pitchHand": {"code": "L"},
                                            },
                                        },
                                    }
                                }
                            ]
                        }
                    ]
                }

        def fake_get(url, timeout=30, **_kwargs):
            calls.append(url)
            return FakeResponse()

        monkeypatch.setattr(fetch_mlb.requests, "get", fake_get)
        cache = {"statsapi_probables": {}}

        first = _fetch_statsapi_probable_pitchers("2026-05-27", cache)
        second = _fetch_statsapi_probable_pitchers("2026-05-27", cache)

        assert first["Yankees"]["name"] == "Gerrit Cole"
        assert first["Yankees"]["throws"] == "R"
        assert first["Red Sox"]["name"] == "Garrett Crochet"
        assert second == first
        assert len(calls) == 1

    def test_expired_cache_entry_is_refetched(self, monkeypatch):
        calls = []
        fetch_mlb._team_map = {"New York Yankees": "Yankees"}

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "dates": [
                        {
                            "games": [
                                {
                                    "teams": {
                                        "home": {
                                            "team": {"name": "New York Yankees"},
                                            "probablePitcher": {
                                                "id": 123,
                                                "fullName": "Fresh Starter",
                                                "pitchHand": {"code": "R"},
                                            },
                                        }
                                    }
                                }
                            ]
                        }
                    ]
                }

        def fake_get(url, timeout=30, **_kwargs):
            calls.append(url)
            return FakeResponse()

        now = datetime(2026, 5, 27, 18, 30, tzinfo=timezone.utc)
        cache = {
            "statsapi_probables": {
                "2026-05-27": {
                    "checked_at": (now - timedelta(minutes=121)).isoformat(),
                    "probables": {
                        "Yankees": {
                            "id": "999",
                            "name": "Stale Starter",
                            "throws": "L",
                            "source": "mlb_stats_api",
                            "last_checked": (now - timedelta(minutes=121)).isoformat(),
                        }
                    },
                }
            }
        }
        monkeypatch.setattr(fetch_mlb.requests, "get", fake_get)

        probables = _fetch_statsapi_probable_pitchers(
            "2026-05-27",
            cache,
            now=now,
            ttl_minutes=120,
        )

        assert probables["Yankees"]["name"] == "Fresh Starter"
        assert probables["Yankees"]["source"] == "mlb_stats_api"
        assert probables["Yankees"]["last_checked"] == now.isoformat()
        assert cache["statsapi_probables"]["2026-05-27"]["checked_at"] == now.isoformat()
        assert len(calls) == 1

    def test_force_refresh_bypasses_fresh_cache(self, monkeypatch):
        calls = []
        fetch_mlb._team_map = {"New York Yankees": "Yankees"}

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "dates": [
                        {
                            "games": [
                                {
                                    "teams": {
                                        "home": {
                                            "team": {"name": "New York Yankees"},
                                            "probablePitcher": {
                                                "id": 123,
                                                "fullName": "Refreshed Starter",
                                                "pitchHand": {"code": "R"},
                                            },
                                        }
                                    }
                                }
                            ]
                        }
                    ]
                }

        now = datetime(2026, 5, 27, 18, 30, tzinfo=timezone.utc)
        cache = {
            "statsapi_probables": {
                "2026-05-27": {
                    "checked_at": (now - timedelta(minutes=5)).isoformat(),
                    "probables": {
                        "Yankees": {
                            "id": "999",
                            "name": "Cached Starter",
                            "throws": "L",
                            "source": "mlb_stats_api",
                            "last_checked": (now - timedelta(minutes=5)).isoformat(),
                        }
                    },
                }
            }
        }
        monkeypatch.setattr(fetch_mlb.requests, "get", lambda url, timeout=30: calls.append(url) or FakeResponse())

        probables = _fetch_statsapi_probable_pitchers(
            "2026-05-27",
            cache,
            now=now,
            ttl_minutes=120,
            force_refresh=True,
        )

        assert probables["Yankees"]["name"] == "Refreshed Starter"
        assert len(calls) == 1

    def test_doubleheader_probables_are_cached_by_game_identity(self, monkeypatch):
        fetch_mlb._team_map = {
            "New York Yankees": "Yankees",
            "Boston Red Sox": "Red Sox",
        }
        now = datetime(2026, 7, 4, 14, 0, tzinfo=timezone.utc)

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "dates": [{
                        "games": [
                            {
                                "gamePk": 1,
                                "gameDate": "2026-07-04T17:05:00Z",
                                "teams": {
                                    "home": {
                                        "team": {"name": "New York Yankees"},
                                        "probablePitcher": {"id": 11, "fullName": "Game One Home", "pitchHand": {"code": "R"}},
                                    },
                                    "away": {
                                        "team": {"name": "Boston Red Sox"},
                                        "probablePitcher": {"id": 12, "fullName": "Game One Away", "pitchHand": {"code": "L"}},
                                    },
                                },
                            },
                            {
                                "gamePk": 2,
                                "gameDate": "2026-07-04T23:05:00Z",
                                "teams": {
                                    "home": {
                                        "team": {"name": "New York Yankees"},
                                        "probablePitcher": {"id": 21, "fullName": "Game Two Home", "pitchHand": {"code": "L"}},
                                    },
                                    "away": {
                                        "team": {"name": "Boston Red Sox"},
                                        "probablePitcher": {"id": 22, "fullName": "Game Two Away", "pitchHand": {"code": "R"}},
                                    },
                                },
                            },
                        ]
                    }]
                }

        monkeypatch.setattr(fetch_mlb.requests, "get", lambda *args, **kwargs: FakeResponse())
        cache = {"statsapi_probables": {}}

        probables = _fetch_statsapi_probable_pitchers("2026-07-04", cache, now=now)

        assert len(probables["__games__"]) == 2
        assert probables["__games__"]["1"]["home"]["name"] == "Game One Home"
        assert probables["__games__"]["2"]["home"]["name"] == "Game Two Home"
        assert cache["statsapi_probables"]["2026-07-04"]["games"]["1"]["away"]["name"] == "Game One Away"
        assert cache["statsapi_probables"]["2026-07-04"]["games"]["2"]["away"]["name"] == "Game Two Away"


class TestMlbSchedulePitcherProvenance:
    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def _scoreboard_payload(self, *, start_time, home_probable=None, away_probable=None):
        home = {
            "homeAway": "home",
            "team": {"id": "10", "displayName": "New York Yankees"},
            "leaders": [],
        }
        away = {
            "homeAway": "away",
            "team": {"id": "2", "displayName": "Boston Red Sox"},
            "leaders": [],
        }
        if home_probable is not None:
            home["probables"] = [home_probable]
        if away_probable is not None:
            away["probables"] = [away_probable]
        return {
            "events": [
                {
                    "id": "401",
                    "date": start_time,
                    "competitions": [
                        {
                            "date": start_time,
                            "neutralSite": False,
                            "status": {"type": {"completed": False}},
                            "competitors": [home, away],
                        }
                    ],
                }
            ]
        }

    def _event(self, event_id, start_time, home_name, away_name, home_id="10", away_id="2", home_probable=None, away_probable=None):
        home = {
            "homeAway": "home",
            "team": {"id": home_id, "displayName": home_name},
            "leaders": [],
        }
        away = {
            "homeAway": "away",
            "team": {"id": away_id, "displayName": away_name},
            "leaders": [],
        }
        if home_probable is not None:
            home["probables"] = [home_probable]
        if away_probable is not None:
            away["probables"] = [away_probable]
        return {
            "id": event_id,
            "date": start_time,
            "competitions": [{
                "date": start_time,
                "neutralSite": False,
                "status": {"type": {"completed": False}},
                "competitors": [home, away],
            }],
        }

    def _install_common_mocks(self, monkeypatch, scoreboard_payload, statsapi_payload):
        fetch_mlb._team_map = {
            "New York Yankees": "Yankees",
            "Boston Red Sox": "Red Sox",
        }

        def fake_get(url, timeout=30, **_kwargs):
            if "scoreboard" in url:
                return self.FakeResponse(scoreboard_payload)
            if "statsapi.mlb.com" in url:
                return self.FakeResponse(statsapi_payload)
            if "/summary?event=" in url:
                return self.FakeResponse({"injuries": [], "rosters": []})
            if "/teams/" in url and "/roster" in url:
                return self.FakeResponse({"athletes": []})
            if "/athletes/" in url:
                return self.FakeResponse({"throws": {"abbreviation": "R"}, "bats": {"abbreviation": "R"}})
            return self.FakeResponse({"hourly": {"time": [], "temperature_2m": [], "wind_speed_10m": [], "precipitation_probability": []}})

        monkeypatch.setattr(fetch_mlb.requests, "get", fake_get)
        monkeypatch.setattr(fetch_mlb.time, "sleep", lambda *_args, **_kwargs: None)

    def test_espn_probable_pitcher_includes_source_and_last_checked(self, monkeypatch, tmp_path):
        start_time = "2026-05-27T23:05:00Z"
        self._install_common_mocks(
            monkeypatch,
            self._scoreboard_payload(
                start_time=start_time,
                home_probable={"playerId": "123", "athlete": {"id": "123", "displayName": "Gerrit Cole"}},
                away_probable={"playerId": "456", "athlete": {"id": "456", "displayName": "Garrett Crochet"}},
            ),
            {"dates": []},
        )

        fixtures = fetch_mlb.fetch_mlb_schedule(cache_path=str(tmp_path / "espn_cache.json"))

        assert fixtures[0]["home_pitcher"] == "Gerrit Cole"
        assert fixtures[0]["home_pitcher_source"] == "espn"
        assert fixtures[0]["away_pitcher_source"] == "espn"
        assert fixtures[0]["home_pitcher_last_checked"]
        assert fixtures[0]["away_pitcher_last_checked"]
        assert fixtures[0]["pitcher_warnings"] == []

    def test_statsapi_fallback_includes_source_and_last_checked(self, monkeypatch, tmp_path):
        start_time = "2026-05-27T23:05:00Z"
        self._install_common_mocks(
            monkeypatch,
            self._scoreboard_payload(start_time=start_time),
            {
                "dates": [
                    {
                        "games": [
                            {
                                "teams": {
                                    "home": {
                                        "team": {"name": "New York Yankees"},
                                        "probablePitcher": {
                                            "id": 123,
                                            "fullName": "Gerrit Cole",
                                            "pitchHand": {"code": "R"},
                                        },
                                    },
                                    "away": {
                                        "team": {"name": "Boston Red Sox"},
                                        "probablePitcher": {
                                            "id": 456,
                                            "fullName": "Garrett Crochet",
                                            "pitchHand": {"code": "L"},
                                        },
                                    },
                                }
                            }
                        ]
                    }
                ]
            },
        )

        fixtures = fetch_mlb.fetch_mlb_schedule(cache_path=str(tmp_path / "espn_cache.json"))

        assert fixtures[0]["home_pitcher"] == "Gerrit Cole"
        assert fixtures[0]["home_pitcher_source"] == "mlb_stats_api"
        assert fixtures[0]["away_pitcher"] == "Garrett Crochet"
        assert fixtures[0]["away_pitcher_source"] == "mlb_stats_api"
        assert fixtures[0]["home_pitcher_last_checked"]
        assert fixtures[0]["away_pitcher_last_checked"]

    def test_tbd_pitchers_near_game_time_emit_warning(self, monkeypatch, tmp_path):
        now = datetime.now(timezone.utc)
        start_time = (now + timedelta(minutes=45)).isoformat().replace("+00:00", "Z")
        self._install_common_mocks(
            monkeypatch,
            self._scoreboard_payload(start_time=start_time),
            {"dates": []},
        )
        monkeypatch.setitem(fetch_mlb.SPORTS["mlb"], "pitcher_tbd_warning_hours", 2)

        fixtures = fetch_mlb.fetch_mlb_schedule(cache_path=str(tmp_path / "espn_cache.json"))

        assert fixtures[0]["home_pitcher"] == "TBD"
        assert fixtures[0]["home_pitcher_source"] == "unavailable"
        assert fixtures[0]["away_pitcher_source"] == "unavailable"
        assert fixtures[0]["pitcher_warnings"] == [
            "home_pitcher_tbd_inside_pregame_window",
            "away_pitcher_tbd_inside_pregame_window",
        ]

    def test_same_opponent_doubleheader_uses_game_level_probables(self, monkeypatch, tmp_path):
        now = datetime(2026, 7, 4, 14, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(fetch_mlb, "_utcnow", lambda: now)
        self._install_common_mocks(
            monkeypatch,
            {
                "events": [
                    self._event("espn-1", "2026-07-04T17:05:00Z", "New York Yankees", "Boston Red Sox"),
                    self._event("espn-2", "2026-07-04T23:05:00Z", "New York Yankees", "Boston Red Sox"),
                ]
            },
            {
                "dates": [{
                    "games": [
                        {
                            "gamePk": 1,
                            "gameDate": "2026-07-04T17:05:00Z",
                            "teams": {
                                "home": {"team": {"name": "New York Yankees"}, "probablePitcher": {"id": 11, "fullName": "Yankees Early", "pitchHand": {"code": "R"}}},
                                "away": {"team": {"name": "Boston Red Sox"}, "probablePitcher": {"id": 12, "fullName": "Red Sox Early", "pitchHand": {"code": "L"}}},
                            },
                        },
                        {
                            "gamePk": 2,
                            "gameDate": "2026-07-04T23:05:00Z",
                            "teams": {
                                "home": {"team": {"name": "New York Yankees"}, "probablePitcher": {"id": 21, "fullName": "Yankees Late", "pitchHand": {"code": "L"}}},
                                "away": {"team": {"name": "Boston Red Sox"}, "probablePitcher": {"id": 22, "fullName": "Red Sox Late", "pitchHand": {"code": "R"}}},
                            },
                        },
                    ]
                }]
            },
        )

        fixtures = fetch_mlb.fetch_mlb_schedule(cache_path=str(tmp_path / "espn_cache.json"))

        assert [fixture["home_pitcher"] for fixture in fixtures] == ["Yankees Early", "Yankees Late"]
        assert [fixture["away_pitcher"] for fixture in fixtures] == ["Red Sox Early", "Red Sox Late"]
        assert all(fixture["home_pitcher_source"] == "mlb_stats_api" for fixture in fixtures)

    def test_split_doubleheader_uses_game_level_probables(self, monkeypatch, tmp_path):
        now = datetime(2026, 7, 4, 14, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(fetch_mlb, "_utcnow", lambda: now)
        self._install_common_mocks(
            monkeypatch,
            {
                "events": [
                    self._event("espn-1", "2026-07-04T17:05:00Z", "New York Yankees", "Boston Red Sox"),
                    self._event("espn-2", "2026-07-04T23:05:00Z", "Boston Red Sox", "New York Yankees", home_id="2", away_id="10"),
                ]
            },
            {
                "dates": [{
                    "games": [
                        {
                            "gamePk": 1,
                            "gameDate": "2026-07-04T17:05:00Z",
                            "teams": {
                                "home": {"team": {"name": "New York Yankees"}, "probablePitcher": {"id": 11, "fullName": "Yankees Home", "pitchHand": {"code": "R"}}},
                                "away": {"team": {"name": "Boston Red Sox"}, "probablePitcher": {"id": 12, "fullName": "Red Sox Away", "pitchHand": {"code": "L"}}},
                            },
                        },
                        {
                            "gamePk": 2,
                            "gameDate": "2026-07-04T23:05:00Z",
                            "teams": {
                                "home": {"team": {"name": "Boston Red Sox"}, "probablePitcher": {"id": 21, "fullName": "Red Sox Home", "pitchHand": {"code": "R"}}},
                                "away": {"team": {"name": "New York Yankees"}, "probablePitcher": {"id": 22, "fullName": "Yankees Away", "pitchHand": {"code": "L"}}},
                            },
                        },
                    ]
                }]
            },
        )

        fixtures = fetch_mlb.fetch_mlb_schedule(cache_path=str(tmp_path / "espn_cache.json"))

        assert fixtures[0]["home_pitcher"] == "Yankees Home"
        assert fixtures[0]["away_pitcher"] == "Red Sox Away"
        assert fixtures[1]["home_pitcher"] == "Red Sox Home"
        assert fixtures[1]["away_pitcher"] == "Yankees Away"

    def test_stale_statsapi_fallback_is_exposed_in_fixture_warnings(self, monkeypatch, tmp_path):
        now = datetime(2026, 7, 4, 14, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(fetch_mlb, "_utcnow", lambda: now)
        fetch_mlb._team_map = {"New York Yankees": "Yankees", "Boston Red Sox": "Red Sox"}
        cache_path = tmp_path / "espn_cache.json"
        stale_checked = (now - timedelta(hours=8)).isoformat()
        cache_path.write_text(
            __import__("json").dumps({
                "games": {},
                "pitchers": {},
                "players": {},
                "weather": {},
                "rosters": {},
                "statsapi_probables": {
                    "2026-07-04": {
                        "checked_at": stale_checked,
                        "games": {
                            "1": {
                                "game_id": "1",
                                "game_date": "2026-07-04T17:05:00Z",
                                "home_team": "Yankees",
                                "away_team": "Red Sox",
                                "home": {"name": "Cached Home", "source": "mlb_stats_api", "last_checked": stale_checked},
                                "away": {"name": "Cached Away", "source": "mlb_stats_api", "last_checked": stale_checked},
                            }
                        },
                        "probables": {},
                    }
                },
            })
        )

        def fake_get(url, timeout=30, **_kwargs):
            if "scoreboard" in url:
                return self.FakeResponse({"events": [self._event("espn-1", "2026-07-04T17:05:00Z", "New York Yankees", "Boston Red Sox")]})
            if "statsapi.mlb.com" in url:
                raise fetch_mlb.requests.RequestException("down")
            if "/summary?event=" in url:
                return self.FakeResponse({"injuries": [], "rosters": []})
            if "/teams/" in url and "/roster" in url:
                return self.FakeResponse({"athletes": []})
            return self.FakeResponse({"hourly": {"time": [], "temperature_2m": [], "wind_speed_10m": [], "precipitation_probability": []}})

        monkeypatch.setattr(fetch_mlb.requests, "get", fake_get)
        monkeypatch.setattr(fetch_mlb.time, "sleep", lambda *_args, **_kwargs: None)

        fixtures = fetch_mlb.fetch_mlb_schedule(cache_path=str(cache_path))

        assert fixtures[0]["home_pitcher"] == "Cached Home"
        assert fixtures[0]["home_pitcher_cache_stale"] is True
        assert fixtures[0]["away_pitcher_cache_stale"] is True
        assert "home_pitcher_from_stale_cache" in fixtures[0]["pitcher_warnings"]
        assert "away_pitcher_from_stale_cache" in fixtures[0]["pitcher_warnings"]

    def test_placeholder_espn_probable_triggers_statsapi_fallback(self, monkeypatch, tmp_path):
        now = datetime(2026, 7, 4, 14, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(fetch_mlb, "_utcnow", lambda: now)
        self._install_common_mocks(
            monkeypatch,
            self._scoreboard_payload(
                start_time="2026-07-04T17:05:00Z",
                home_probable={"playerId": "0", "athlete": {"id": "0", "displayName": "To Be Announced"}},
                away_probable={"playerId": "0", "athlete": {"id": "0", "displayName": "Unknown"}},
            ),
            {
                "dates": [{
                    "games": [{
                        "gamePk": 1,
                        "gameDate": "2026-07-04T17:05:00Z",
                        "teams": {
                            "home": {"team": {"name": "New York Yankees"}, "probablePitcher": {"id": 11, "fullName": "Fallback Home", "pitchHand": {"code": "R"}}},
                            "away": {"team": {"name": "Boston Red Sox"}, "probablePitcher": {"id": 12, "fullName": "Fallback Away", "pitchHand": {"code": "L"}}},
                        },
                    }]
                }]
            },
        )

        fixtures = fetch_mlb.fetch_mlb_schedule(cache_path=str(tmp_path / "espn_cache.json"))

        assert fixtures[0]["home_pitcher"] == "Fallback Home"
        assert fixtures[0]["away_pitcher"] == "Fallback Away"
        assert fixtures[0]["home_pitcher_source"] == "mlb_stats_api"
        assert fixtures[0]["away_pitcher_source"] == "mlb_stats_api"

    def test_schedule_date_uses_new_york_timezone_during_dst_boundary(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(fetch_mlb, "_utcnow", lambda: datetime(2026, 7, 1, 4, 30, tzinfo=timezone.utc))
        self._install_common_mocks(monkeypatch, {"events": []}, {"dates": []})

        def fake_get(url, timeout=30, **_kwargs):
            calls.append(url)
            if "scoreboard" in url:
                return self.FakeResponse({"events": []})
            if "statsapi.mlb.com" in url:
                return self.FakeResponse({"dates": []})
            return self.FakeResponse({"athletes": []})

        monkeypatch.setattr(fetch_mlb.requests, "get", fake_get)

        fetch_mlb.fetch_mlb_schedule(cache_path=str(tmp_path / "espn_cache.json"))

        assert any("dates=20260701" in url for url in calls)

    def test_legacy_cached_completed_game_pitchers_are_marked_unverified(self, monkeypatch, tmp_path):
        fetch_mlb._team_map = {
            "New York Yankees": "Yankees",
            "Boston Red Sox": "Red Sox",
        }
        cache_path = tmp_path / "espn_cache.json"
        cache_path.write_text(
            __import__("json").dumps({
                "games": {
                    "legacy-1": {
                        "date": "2026-07-04",
                        "home_team": "Yankees",
                        "away_team": "Red Sox",
                        "home_goals": 5,
                        "away_goals": 3,
                        "home_pitcher": "Legacy Home",
                        "away_pitcher": "Legacy Away",
                        "home_pitcher_stats": {},
                        "away_pitcher_stats": {},
                    }
                },
                "pitchers": {},
                "players": {},
                "weather": {},
                "rosters": {},
                "statsapi_probables": {},
            })
        )

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "events": [{
                        "id": "legacy-1",
                        "date": "2026-07-04T17:05:00Z",
                        "competitions": [{
                            "status": {"type": {"completed": True}},
                            "competitors": [
                                {"homeAway": "home", "team": {"displayName": "New York Yankees"}, "score": "5"},
                                {"homeAway": "away", "team": {"displayName": "Boston Red Sox"}, "score": "3"},
                            ],
                        }],
                    }]
                }

        monkeypatch.setattr(fetch_mlb.requests, "get", lambda *args, **kwargs: FakeResponse())
        monkeypatch.setattr(fetch_mlb.time, "sleep", lambda *_args, **_kwargs: None)

        games_df, _ = fetch_mlb.fetch_mlb_games(dates=["2026-07-04"], cache_path=str(cache_path))
        row = games_df.iloc[0]

        assert row["home_pitcher"] == "Legacy Home"
        assert row["home_pitcher_source"] == "legacy_cache_unverified"
        assert row["away_pitcher_source"] == "legacy_cache_unverified"

    def test_legacy_cached_completed_game_pitchers_are_repaired_from_summary(self, monkeypatch, tmp_path):
        fetch_mlb._team_map = {
            "New York Yankees": "Yankees",
            "Boston Red Sox": "Red Sox",
        }
        cache_path = tmp_path / "espn_cache.json"
        cache_path.write_text(
            __import__("json").dumps({
                "games": {
                    "legacy-1": {
                        "date": "2026-07-04",
                        "home_team": "Yankees",
                        "away_team": "Red Sox",
                        "home_goals": 5,
                        "away_goals": 3,
                        "home_pitcher": "Legacy Home",
                        "away_pitcher": "Legacy Away",
                        "home_pitcher_source": "legacy_cache_unverified",
                        "away_pitcher_source": "legacy_cache_unverified",
                        "home_pitcher_stats": {},
                        "away_pitcher_stats": {},
                    }
                },
                "pitchers": {},
                "players": {},
                "weather": {},
                "rosters": {},
                "statsapi_probables": {},
            })
        )

        scoreboard_payload = {
            "events": [{
                "id": "legacy-1",
                "date": "2026-07-04T17:05:00Z",
                "competitions": [{
                    "status": {"type": {"completed": True}},
                    "competitors": [
                        {"homeAway": "home", "team": {"displayName": "New York Yankees"}, "score": "5"},
                        {"homeAway": "away", "team": {"displayName": "Boston Red Sox"}, "score": "3"},
                    ],
                }],
            }]
        }
        summary_payload = {
            "boxscore": {
                "players": [
                    {
                        "team": {"displayName": "New York Yankees"},
                        "statistics": [{
                            "athletes": [{
                                "starter": True,
                                "position": {"abbreviation": "P"},
                                "athlete": {"id": "11", "displayName": "Verified Home"},
                                "stats": ["6.0", "4", "2", "2", "1", "7"],
                            }],
                        }],
                    },
                    {
                        "team": {"displayName": "Boston Red Sox"},
                        "statistics": [{
                            "athletes": [{
                                "starter": True,
                                "position": {"abbreviation": "P"},
                                "athlete": {"id": "12", "displayName": "Verified Away"},
                                "stats": ["5.0", "5", "3", "3", "2", "5"],
                            }],
                        }],
                    },
                ]
            }
        }

        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self.payload

        def fake_get(url, timeout=30, **_kwargs):
            if "/summary?event=" in url:
                return FakeResponse(summary_payload)
            if "/athletes/11" in url:
                return FakeResponse({"throws": {"abbreviation": "R"}, "bats": {"abbreviation": "R"}})
            if "/athletes/12" in url:
                return FakeResponse({"throws": {"abbreviation": "L"}, "bats": {"abbreviation": "L"}})
            return FakeResponse(scoreboard_payload)

        monkeypatch.setattr(fetch_mlb.requests, "get", fake_get)
        monkeypatch.setattr(fetch_mlb.time, "sleep", lambda *_args, **_kwargs: None)

        games_df, _ = fetch_mlb.fetch_mlb_games(dates=["2026-07-04"], cache_path=str(cache_path))
        row = games_df.iloc[0]

        assert row["home_pitcher"] == "Verified Home"
        assert row["away_pitcher"] == "Verified Away"
        assert row["home_pitcher_source"] == "espn_summary"
        assert row["away_pitcher_source"] == "espn_summary"


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
