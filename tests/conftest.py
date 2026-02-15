"""Shared test fixtures for SLOP LOCKS pipeline tests."""

import pytest
import pandas as pd
from datetime import datetime, timedelta


@pytest.fixture
def sample_matches():
    """Minimal set of EPL match results for testing."""
    base_date = datetime(2025, 8, 16)
    results = [
        ("Arsenal", "Wolves", 2, 0),
        ("Liverpool", "Ipswich", 2, 0),
        ("Man City", "Chelsea", 0, 2),
        ("Newcastle", "Southampton", 1, 0),
        ("Aston Villa", "West Ham", 2, 1),
        ("Wolves", "Arsenal", 0, 1),
        ("Chelsea", "Man City", 1, 1),
        ("Ipswich", "Liverpool", 0, 3),
        ("Southampton", "Newcastle", 1, 2),
        ("West Ham", "Aston Villa", 0, 0),
    ]
    matches = []
    for i, (home, away, hg, ag) in enumerate(results):
        matches.append({
            "date": (base_date + timedelta(weeks=i // 5, days=(i % 5))).isoformat(),
            "home_team": home,
            "away_team": away,
            "home_goals": hg,
            "away_goals": ag,
        })
    return pd.DataFrame(matches)


@pytest.fixture
def sample_xg():
    """Matching xG data for the sample matches."""
    base_date = datetime(2025, 8, 16)
    results = [
        ("Arsenal", "Wolves", 1.8, 0.4),
        ("Liverpool", "Ipswich", 2.5, 0.3),
        ("Man City", "Chelsea", 1.5, 1.2),
        ("Newcastle", "Southampton", 0.9, 0.6),
        ("Aston Villa", "West Ham", 1.4, 1.1),
        ("Wolves", "Arsenal", 0.5, 1.6),
        ("Chelsea", "Man City", 0.8, 1.8),
        ("Ipswich", "Liverpool", 0.2, 2.8),
        ("Southampton", "Newcastle", 1.0, 1.5),
        ("West Ham", "Aston Villa", 0.7, 0.7),
    ]
    data = []
    for i, (home, away, hxg, axg) in enumerate(results):
        data.append({
            "date": (base_date + timedelta(weeks=i // 5, days=(i % 5))).isoformat(),
            "home_team": home,
            "away_team": away,
            "home_xg": hxg,
            "away_xg": axg,
        })
    return pd.DataFrame(data)


@pytest.fixture
def sample_odds():
    """Bookmaker odds for upcoming matches."""
    return [
        {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "commence_time": "2026-02-22T15:00:00Z",
            "home_odds": 1.67,
            "draw_odds": 3.80,
            "away_odds": 4.50,
        }
    ]


@pytest.fixture
def teams():
    """List of EPL teams for the current season."""
    return [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
        "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich",
        "Leicester", "Liverpool", "Man City", "Man United", "Newcastle",
        "Nottingham Forest", "Southampton", "Tottenham", "West Ham", "Wolves",
    ]
