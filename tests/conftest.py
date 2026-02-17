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


@pytest.fixture
def ncaam_games():
    """Minimal set of NCAAM game results for testing."""
    base_date = datetime(2026, 1, 5)
    results = [
        ("Duke", "North Carolina", 82, 75),
        ("Kansas", "Kentucky", 70, 68),
        ("Gonzaga", "UCLA", 85, 78),
        ("North Carolina", "Kansas", 71, 65),
        ("Kentucky", "Duke", 80, 77),
        ("UCLA", "Gonzaga", 68, 72),
        ("Duke", "Kansas", 88, 82),
        ("North Carolina", "Kentucky", 76, 74),
        ("Gonzaga", "Duke", 79, 81),
        ("UCLA", "Kansas", 65, 70),
    ]
    games = []
    for i, (home, away, hg, ag) in enumerate(results):
        games.append({
            "game_id": i + 1,
            "date": (base_date + timedelta(days=i * 3)).isoformat(),
            "home_team": home,
            "away_team": away,
            "home_goals": hg,
            "away_goals": ag,
        })
    return pd.DataFrame(games)


def _make_box_score(game_id, date, team, pts, is_home):
    """Generate plausible box-score stats from a point total."""
    import random
    rng = random.Random(hash((game_id, team)))
    # Work backwards from points to shooting stats
    fg3m = rng.randint(4, 10)
    ftm = rng.randint(8, 18)
    fg2_pts = pts - 3 * fg3m - ftm
    if fg2_pts < 0:
        fg3m = max(0, fg3m - 3)
        fg2_pts = pts - 3 * fg3m - ftm
    if fg2_pts < 0:
        ftm = max(0, ftm - 5)
        fg2_pts = pts - 3 * fg3m - ftm
    fg2m = max(1, fg2_pts // 2)
    fgm = fg2m + fg3m
    fga = fgm + rng.randint(18, 28)
    fg3a = fg3m + rng.randint(5, 12)
    fta = ftm + rng.randint(3, 8)
    orb = rng.randint(6, 14)
    drb = rng.randint(18, 28)
    to = rng.randint(8, 16)
    possessions = fga - orb + to + int(0.44 * fta)
    return {
        "game_id": game_id,
        "team": team,
        "date": date,
        "pts": pts,
        "fgm": fgm,
        "fga": fga,
        "fg3m": fg3m,
        "fg3a": fg3a,
        "ftm": ftm,
        "fta": fta,
        "orb": orb,
        "drb": drb,
        "to": to,
        "possessions": possessions,
    }


@pytest.fixture
def ncaam_box_scores(ncaam_games):
    """Box scores matching the ncaam_games fixture."""
    rows = []
    for _, g in ncaam_games.iterrows():
        rows.append(_make_box_score(
            g["game_id"], g["date"], g["home_team"], g["home_goals"], True,
        ))
        rows.append(_make_box_score(
            g["game_id"], g["date"], g["away_team"], g["away_goals"], False,
        ))
    return pd.DataFrame(rows)
