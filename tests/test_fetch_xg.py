"""Tests for pipeline.fetch_xg — Understat xG scraping helpers."""

import pandas as pd
import pytest

from pipeline.fetch_xg import normalize_understat_name, parse_understat_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def played_matches():
    """Two played matches in raw Understat format."""
    return [
        {
            "isResult": True,
            "datetime": "2025-08-16 15:00:00",
            "h": {"title": "Manchester City"},
            "a": {"title": "Chelsea"},
            "xG": {"h": "1.52", "a": "1.18"},
        },
        {
            "isResult": True,
            "datetime": "2025-08-17 14:00:00",
            "h": {"title": "Wolverhampton Wanderers"},
            "a": {"title": "Arsenal"},
            "xG": {"h": "0.73", "a": "1.95"},
        },
    ]


@pytest.fixture
def unplayed_match():
    """An unplayed match that should be skipped."""
    return {
        "isResult": False,
        "datetime": "2026-05-25 16:00:00",
        "h": {"title": "Liverpool"},
        "a": {"title": "Newcastle United"},
        "xG": {"h": "0", "a": "0"},
    }


# ---------------------------------------------------------------------------
# TestNormalizeUnderstatName
# ---------------------------------------------------------------------------

class TestNormalizeUnderstatName:
    """Verify that Understat's full team names map to short canonical names."""

    def test_man_city(self):
        assert normalize_understat_name("Manchester City") == "Man City"

    def test_man_united(self):
        assert normalize_understat_name("Manchester United") == "Man United"

    def test_wolves(self):
        assert normalize_understat_name("Wolverhampton Wanderers") == "Wolves"

    def test_newcastle(self):
        assert normalize_understat_name("Newcastle United") == "Newcastle"

    def test_west_ham(self):
        assert normalize_understat_name("West Ham United") == "West Ham"

    def test_bournemouth(self):
        assert normalize_understat_name("AFC Bournemouth") == "Bournemouth"

    def test_brighton(self):
        assert normalize_understat_name("Brighton and Hove Albion") == "Brighton"

    def test_tottenham(self):
        assert normalize_understat_name("Tottenham Hotspur") == "Tottenham"

    def test_ipswich(self):
        assert normalize_understat_name("Ipswich Town") == "Ipswich"

    def test_leicester(self):
        assert normalize_understat_name("Leicester City") == "Leicester"

    def test_passthrough_arsenal(self):
        """Names not in the map should be returned unchanged."""
        assert normalize_understat_name("Arsenal") == "Arsenal"

    def test_passthrough_chelsea(self):
        assert normalize_understat_name("Chelsea") == "Chelsea"


# ---------------------------------------------------------------------------
# TestParseUnderstatData
# ---------------------------------------------------------------------------

class TestParseUnderstatData:
    """Verify parse_understat_data produces the expected DataFrame."""

    def test_parses_played_matches(self, played_matches):
        df = parse_understat_data(played_matches)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == [
            "date", "home_team", "away_team", "home_xg", "away_xg",
        ]

    def test_team_names_normalized(self, played_matches):
        df = parse_understat_data(played_matches)

        assert df.iloc[0]["home_team"] == "Man City"
        assert df.iloc[0]["away_team"] == "Chelsea"
        assert df.iloc[1]["home_team"] == "Wolves"
        assert df.iloc[1]["away_team"] == "Arsenal"

    def test_xg_values_correct(self, played_matches):
        df = parse_understat_data(played_matches)

        assert df.iloc[0]["home_xg"] == 1.52
        assert df.iloc[0]["away_xg"] == 1.18
        assert df.iloc[1]["home_xg"] == 0.73
        assert df.iloc[1]["away_xg"] == 1.95

    def test_dates_extracted(self, played_matches):
        df = parse_understat_data(played_matches)

        assert df.iloc[0]["date"] == "2025-08-16"
        assert df.iloc[1]["date"] == "2025-08-17"

    def test_skips_unplayed_matches(self, played_matches, unplayed_match):
        raw = played_matches + [unplayed_match]
        df = parse_understat_data(raw)

        assert len(df) == 2  # unplayed match excluded

    def test_all_unplayed_returns_empty(self, unplayed_match):
        df = parse_understat_data([unplayed_match])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_empty_input(self):
        df = parse_understat_data([])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
