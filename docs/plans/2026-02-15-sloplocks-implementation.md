# SLOP LOCKS Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a daily-updating EPL match prediction engine with a 3-model ensemble and automated betting edge detection, deployed as a free static site at sloplocks.lol.

**Architecture:** A GitHub Action runs daily, executing a Python pipeline that fetches EPL data (results, xG, odds), runs a Dixon-Coles + xG + Elo ensemble, computes edges vs bookmaker lines, and writes JSON files. A single-file static HTML frontend reads those JSON files and displays predictions with highlighted value bets. Deployed on Netlify.

**Tech Stack:** Python 3.11+ (scipy, pandas, numpy, requests, beautifulsoup4), HTML/CSS/JS, GitHub Actions, Netlify

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pipeline/requirements.txt`
- Create: `pipeline/__init__.py`
- Create: `pipeline/config.py`
- Create: `data/.gitkeep`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `.gitignore`

**Step 1: Create requirements.txt**

```
# pipeline/requirements.txt
scipy>=1.11.0
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
pytest>=7.4.0
pytest-mock>=3.11.0
```

**Step 2: Create config.py with API endpoints and constants**

```python
# pipeline/config.py
"""Central configuration for the SLOP LOCKS pipeline."""

import os

# API Keys (from environment / GitHub Secrets)
FOOTBALL_DATA_API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "")
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")

# football-data.org
FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"
EPL_COMPETITION_ID = "PL"  # Premier League

# Understat
UNDERSTAT_BASE = "https://understat.com"
EPL_UNDERSTAT_LEAGUE = "EPL"
CURRENT_SEASON = "2025"  # Understat uses start year of season

# The Odds API
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_SPORT = "soccer_epl"
ODDS_REGIONS = "us"  # American odds format
ODDS_MARKETS = "h2h"  # Moneyline (home/draw/away)

# Model parameters
TIME_DECAY_RATE = 0.005  # Higher = more weight on recent matches
FORM_WINDOW = 6  # Last N matches for form momentum
FORM_WEIGHT_MULTIPLIER = 2.0  # How much extra weight form matches get
CONGESTION_THRESHOLD_DAYS = 4  # Penalty if rest < this many days
CONGESTION_PENALTY = 0.05  # Reduce attack strength by this factor
ELO_K_FACTOR = 20  # Elo update magnitude
ELO_HOME_ADVANTAGE = 65  # Elo points added for home team
VALUE_EDGE_THRESHOLD = 0.05  # 5% edge = value bet
ENSEMBLE_ACCURACY_WINDOW = 10  # Rolling N matches for model weighting
MAX_GOALS = 6  # Max goals in scoreline matrix (0-6 x 0-6)

# Output paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "predictions.json")
HISTORY_PATH = os.path.join(DATA_DIR, "history.json")
ACCURACY_PATH = os.path.join(DATA_DIR, "model_accuracy.json")
```

**Step 3: Create .gitignore**

```
# .gitignore
__pycache__/
*.pyc
.env
.venv/
venv/
*.egg-info/
.pytest_cache/
node_modules/
.DS_Store
```

**Step 4: Create empty init files and data directory**

```bash
touch pipeline/__init__.py tests/__init__.py
mkdir -p data
touch data/.gitkeep
```

**Step 5: Create test conftest with shared fixtures**

```python
# tests/conftest.py
"""Shared test fixtures for SLOP LOCKS pipeline tests."""

import pytest
import pandas as pd
from datetime import datetime, timedelta


@pytest.fixture
def sample_matches():
    """Minimal set of EPL match results for testing."""
    base_date = datetime(2025, 8, 16)
    matches = []
    # 10 matches with plausible EPL scores
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
            "bookmakers": [
                {
                    "key": "fanduel",
                    "markets": [{
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Arsenal", "price": -150},
                            {"name": "Draw", "price": +280},
                            {"name": "Chelsea", "price": +350},
                        ]
                    }]
                }
            ]
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
```

**Step 6: Install dependencies and verify**

Run: `cd /home/jason/sloplocks && python -m venv venv && source venv/bin/activate && pip install -r pipeline/requirements.txt`
Expected: All packages install successfully

**Step 7: Run pytest to verify scaffold**

Run: `cd /home/jason/sloplocks && source venv/bin/activate && pytest tests/ -v`
Expected: "no tests ran" (0 collected), no errors

**Step 8: Commit**

```bash
git add -A
git commit -m "feat: project scaffolding with config, fixtures, and dependencies"
```

---

## Task 2: Data Fetching — football-data.org

**Files:**
- Create: `pipeline/fetch_data.py`
- Create: `tests/test_fetch_data.py`

**Step 1: Write tests for football-data.org client**

```python
# tests/test_fetch_data.py
"""Tests for data fetching from football-data.org and The Odds API."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pipeline.fetch_data import (
    fetch_epl_matches,
    fetch_epl_fixtures,
    fetch_odds,
    normalize_team_name,
)


class TestNormalizeTeamName:
    def test_standard_name(self):
        assert normalize_team_name("Arsenal FC") == "Arsenal"

    def test_manchester_abbreviation(self):
        assert normalize_team_name("Manchester United FC") == "Man United"
        assert normalize_team_name("Manchester City FC") == "Man City"

    def test_nottingham(self):
        assert normalize_team_name("Nottingham Forest FC") == "Nottingham Forest"

    def test_already_clean(self):
        assert normalize_team_name("Liverpool") == "Liverpool"


class TestFetchEPLMatches:
    @patch("pipeline.fetch_data.requests.get")
    def test_returns_dataframe_with_required_columns(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "matches": [
                    {
                        "utcDate": "2025-08-16T15:00:00Z",
                        "homeTeam": {"name": "Arsenal FC"},
                        "awayTeam": {"name": "Wolves"},
                        "score": {
                            "fullTime": {"home": 2, "away": 0}
                        },
                        "status": "FINISHED",
                    }
                ]
            },
        )
        df = fetch_epl_matches()
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) >= {"date", "home_team", "away_team", "home_goals", "away_goals"}
        assert df.iloc[0]["home_team"] == "Arsenal"
        assert df.iloc[0]["home_goals"] == 2

    @patch("pipeline.fetch_data.requests.get")
    def test_skips_unfinished_matches(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "matches": [
                    {
                        "utcDate": "2025-08-16T15:00:00Z",
                        "homeTeam": {"name": "Arsenal FC"},
                        "awayTeam": {"name": "Wolves"},
                        "score": {"fullTime": {"home": None, "away": None}},
                        "status": "SCHEDULED",
                    }
                ]
            },
        )
        df = fetch_epl_matches()
        assert len(df) == 0


class TestFetchEPLFixtures:
    @patch("pipeline.fetch_data.requests.get")
    def test_returns_upcoming_matches(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "matches": [
                    {
                        "utcDate": "2026-02-22T15:00:00Z",
                        "homeTeam": {"name": "Arsenal FC"},
                        "awayTeam": {"name": "Chelsea FC"},
                        "status": "SCHEDULED",
                        "matchday": 26,
                    }
                ]
            },
        )
        fixtures = fetch_epl_fixtures()
        assert len(fixtures) == 1
        assert fixtures[0]["home_team"] == "Arsenal"
        assert fixtures[0]["away_team"] == "Chelsea"


class TestFetchOdds:
    @patch("pipeline.fetch_data.requests.get")
    def test_returns_best_odds_per_match(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [
                {
                    "home_team": "Arsenal",
                    "away_team": "Chelsea",
                    "commence_time": "2026-02-22T15:00:00Z",
                    "bookmakers": [
                        {
                            "key": "fanduel",
                            "markets": [{
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Arsenal", "price": 1.67},
                                    {"name": "Draw", "price": 3.80},
                                    {"name": "Chelsea", "price": 4.50},
                                ]
                            }]
                        }
                    ],
                }
            ],
        )
        odds = fetch_odds()
        assert len(odds) == 1
        assert "home_odds" in odds[0]
        assert "draw_odds" in odds[0]
        assert "away_odds" in odds[0]
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/jason/sloplocks && source venv/bin/activate && pytest tests/test_fetch_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.fetch_data'`

**Step 3: Implement fetch_data.py**

```python
# pipeline/fetch_data.py
"""Fetch EPL match data from football-data.org, odds from The Odds API."""

import requests
import pandas as pd
from datetime import datetime
from pipeline.config import (
    FOOTBALL_DATA_API_KEY,
    FOOTBALL_DATA_BASE,
    EPL_COMPETITION_ID,
    ODDS_API_KEY,
    ODDS_API_BASE,
    ODDS_SPORT,
    ODDS_REGIONS,
    ODDS_MARKETS,
)

# football-data.org uses full names, we want short display names
TEAM_NAME_MAP = {
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "AFC Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton",
    "Chelsea FC": "Chelsea",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Ipswich Town FC": "Ipswich",
    "Leicester City FC": "Leicester",
    "Liverpool FC": "Liverpool",
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle",
    "Nottingham Forest FC": "Nottingham Forest",
    "Southampton FC": "Southampton",
    "Tottenham Hotspur FC": "Tottenham",
    "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",
}


def normalize_team_name(name):
    """Convert API team name to short display name."""
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]
    # Fallback: strip trailing "FC"
    return name.replace(" FC", "").strip()


def fetch_epl_matches():
    """Fetch all completed EPL matches for the current season.

    Returns a DataFrame with columns:
        date, home_team, away_team, home_goals, away_goals
    """
    url = f"{FOOTBALL_DATA_BASE}/competitions/{EPL_COMPETITION_ID}/matches"
    headers = {"X-Auth-Token": FOOTBALL_DATA_API_KEY}
    params = {"status": "FINISHED"}

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for m in data.get("matches", []):
        ft = m["score"]["fullTime"]
        if ft["home"] is None or ft["away"] is None:
            continue
        rows.append({
            "date": m["utcDate"][:10],
            "home_team": normalize_team_name(m["homeTeam"]["name"]),
            "away_team": normalize_team_name(m["awayTeam"]["name"]),
            "home_goals": ft["home"],
            "away_goals": ft["away"],
        })

    return pd.DataFrame(rows)


def fetch_epl_fixtures():
    """Fetch upcoming scheduled EPL matches.

    Returns a list of dicts with:
        home_team, away_team, date, matchday
    """
    url = f"{FOOTBALL_DATA_BASE}/competitions/{EPL_COMPETITION_ID}/matches"
    headers = {"X-Auth-Token": FOOTBALL_DATA_API_KEY}
    params = {"status": "SCHEDULED"}

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    fixtures = []
    for m in data.get("matches", []):
        fixtures.append({
            "home_team": normalize_team_name(m["homeTeam"]["name"]),
            "away_team": normalize_team_name(m["awayTeam"]["name"]),
            "date": m["utcDate"],
            "matchday": m.get("matchday"),
        })

    return fixtures


def fetch_odds():
    """Fetch current bookmaker odds for upcoming EPL matches.

    Returns a list of dicts with:
        home_team, away_team, commence_time, home_odds, draw_odds, away_odds
    The odds are the best available across all bookmakers (decimal format from API).
    """
    url = f"{ODDS_API_BASE}/sports/{ODDS_SPORT}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": ODDS_REGIONS,
        "markets": ODDS_MARKETS,
        "oddsFormat": "decimal",
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    events = resp.json()

    results = []
    for event in events:
        best = {"home": 0, "draw": 0, "away": 0}
        home_name = event["home_team"]
        away_name = event["away_team"]

        for bm in event.get("bookmakers", []):
            for market in bm.get("markets", []):
                if market["key"] != "h2h":
                    continue
                for outcome in market["outcomes"]:
                    price = outcome["price"]
                    if outcome["name"] == home_name:
                        best["home"] = max(best["home"], price)
                    elif outcome["name"] == away_name:
                        best["away"] = max(best["away"], price)
                    elif outcome["name"] == "Draw":
                        best["draw"] = max(best["draw"], price)

        if best["home"] > 0 and best["draw"] > 0 and best["away"] > 0:
            results.append({
                "home_team": home_name,
                "away_team": away_name,
                "commence_time": event["commence_time"],
                "home_odds": best["home"],
                "draw_odds": best["draw"],
                "away_odds": best["away"],
            })

    return results
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/jason/sloplocks && source venv/bin/activate && pytest tests/test_fetch_data.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add pipeline/fetch_data.py tests/test_fetch_data.py
git commit -m "feat: data fetching from football-data.org and The Odds API"
```

---

## Task 3: Data Fetching — Understat xG Scraper

**Files:**
- Create: `pipeline/fetch_xg.py`
- Create: `tests/test_fetch_xg.py`

**Step 1: Write tests for Understat scraper**

```python
# tests/test_fetch_xg.py
"""Tests for Understat xG scraping."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pipeline.fetch_xg import fetch_understat_xg, parse_understat_data


class TestParseUnderstatData:
    def test_parses_match_data(self):
        # Understat embeds JSON in script tags as: var datesData = JSON.parse('...')
        raw = [
            {
                "id": "1",
                "isResult": True,
                "datetime": "2025-08-16 15:00:00",
                "h": {"title": "Arsenal", "short_title": "ARS"},
                "a": {"title": "Wolverhampton Wanderers", "short_title": "WOL"},
                "xG": {"h": "1.82", "a": "0.41"},
                "goals": {"h": "2", "a": "0"},
            }
        ]
        df = parse_understat_data(raw)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["home_team"] == "Arsenal"
        assert df.iloc[0]["away_team"] == "Wolves"
        assert abs(df.iloc[0]["home_xg"] - 1.82) < 0.01

    def test_skips_unplayed_matches(self):
        raw = [
            {
                "id": "1",
                "isResult": False,
                "datetime": "2026-03-01 15:00:00",
                "h": {"title": "Arsenal", "short_title": "ARS"},
                "a": {"title": "Chelsea", "short_title": "CHE"},
                "xG": {"h": "0", "a": "0"},
                "goals": {"h": "0", "a": "0"},
            }
        ]
        df = parse_understat_data(raw)
        assert len(df) == 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fetch_xg.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement fetch_xg.py**

```python
# pipeline/fetch_xg.py
"""Scrape xG data from Understat for EPL matches."""

import json
import re
import requests
import pandas as pd
from pipeline.config import UNDERSTAT_BASE, CURRENT_SEASON

# Understat uses full names, map to our short names
UNDERSTAT_TEAM_MAP = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Wolverhampton Wanderers": "Wolves",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nottingham Forest",
    "West Ham": "West Ham",
    "Aston Villa": "Aston Villa",
    "Crystal Palace": "Crystal Palace",
    "Leicester": "Leicester",
    "Ipswich": "Ipswich",
    "Tottenham": "Tottenham",
    "Southampton": "Southampton",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Fulham": "Fulham",
    "Everton": "Everton",
    "Arsenal": "Arsenal",
    "Liverpool": "Liverpool",
    "Chelsea": "Chelsea",
}


def normalize_understat_name(name):
    """Convert Understat team name to our standard short name."""
    return UNDERSTAT_TEAM_MAP.get(name, name)


def parse_understat_data(raw_matches):
    """Parse raw Understat match data into a DataFrame.

    Args:
        raw_matches: List of match dicts from Understat's embedded JSON.

    Returns:
        DataFrame with: date, home_team, away_team, home_xg, away_xg
    """
    rows = []
    for m in raw_matches:
        if not m.get("isResult", False):
            continue
        rows.append({
            "date": m["datetime"][:10],
            "home_team": normalize_understat_name(m["h"]["title"]),
            "away_team": normalize_understat_name(m["a"]["title"]),
            "home_xg": float(m["xG"]["h"]),
            "away_xg": float(m["xG"]["a"]),
        })
    return pd.DataFrame(rows)


def fetch_understat_xg():
    """Fetch xG data for all EPL matches this season from Understat.

    Understat embeds match data as JSON inside a <script> tag.
    We extract it with regex — no JS rendering needed.

    Returns:
        DataFrame with: date, home_team, away_team, home_xg, away_xg
    """
    url = f"{UNDERSTAT_BASE}/league/{CURRENT_SEASON}"

    # Understat expects the league in the URL as "EPL"
    url = f"{UNDERSTAT_BASE}/league/EPL/{CURRENT_SEASON}"

    resp = requests.get(url, timeout=30, headers={
        "User-Agent": "Mozilla/5.0 (compatible; SlopLocks/1.0)"
    })
    resp.raise_for_status()

    # Understat stores data as: var datesData = JSON.parse('...')
    pattern = r"var\s+datesData\s*=\s*JSON\.parse\('(.+?)'\)"
    match = re.search(pattern, resp.text)
    if not match:
        raise ValueError("Could not find datesData in Understat response")

    # The JSON string has escaped characters
    raw_json = match.group(1).encode().decode("unicode_escape")
    matches = json.loads(raw_json)

    return parse_understat_data(matches)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fetch_xg.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add pipeline/fetch_xg.py tests/test_fetch_xg.py
git commit -m "feat: Understat xG scraper with team name normalization"
```

---

## Task 4: Dixon-Coles Model

This is the core prediction engine. The most complex single piece.

**Files:**
- Create: `pipeline/models.py`
- Create: `tests/test_models.py`

**Step 1: Write tests for Dixon-Coles**

```python
# tests/test_models.py
"""Tests for prediction models: Dixon-Coles, xG-adjusted, and Elo."""

import pytest
import numpy as np
import pandas as pd
from pipeline.models import (
    dixon_coles_predict,
    fit_dixon_coles,
    elo_predict,
    EloRatings,
    scoreline_to_probabilities,
)


class TestScorelineToProbabilities:
    def test_probabilities_sum_to_one(self, sample_matches):
        params = fit_dixon_coles(sample_matches)
        matrix = dixon_coles_predict("Arsenal", "Chelsea", params)
        probs = scoreline_to_probabilities(matrix)
        total = probs["home"] + probs["draw"] + probs["away"]
        assert abs(total - 1.0) < 0.001

    def test_returns_dict_with_three_keys(self, sample_matches):
        params = fit_dixon_coles(sample_matches)
        matrix = dixon_coles_predict("Arsenal", "Chelsea", params)
        probs = scoreline_to_probabilities(matrix)
        assert set(probs.keys()) == {"home", "draw", "away"}

    def test_all_probabilities_positive(self, sample_matches):
        params = fit_dixon_coles(sample_matches)
        matrix = dixon_coles_predict("Arsenal", "Chelsea", params)
        probs = scoreline_to_probabilities(matrix)
        assert all(v > 0 for v in probs.values())


class TestDixonColes:
    def test_fit_returns_params_for_all_teams(self, sample_matches):
        params = fit_dixon_coles(sample_matches)
        assert "attack" in params
        assert "defense" in params
        assert "home_advantage" in params
        assert "rho" in params
        # Should have params for every team in the data
        teams_in_data = set(sample_matches["home_team"]) | set(sample_matches["away_team"])
        assert set(params["attack"].keys()) == teams_in_data

    def test_strong_team_has_higher_attack(self, sample_matches):
        params = fit_dixon_coles(sample_matches)
        # Liverpool scored 5 goals in 2 matches, should have high attack
        assert params["attack"]["Liverpool"] > params["attack"]["Southampton"]

    def test_predict_returns_scoreline_matrix(self, sample_matches):
        params = fit_dixon_coles(sample_matches)
        matrix = dixon_coles_predict("Arsenal", "Wolves", params)
        assert matrix.shape == (7, 7)  # 0-6 goals each
        assert abs(matrix.sum() - 1.0) < 0.001


class TestElo:
    def test_initial_ratings(self, teams):
        elo = EloRatings(teams)
        # All teams start at 1500
        for team in teams:
            assert elo.get_rating(team) == 1500

    def test_winner_gains_rating(self, teams):
        elo = EloRatings(teams)
        before_home = elo.get_rating("Arsenal")
        before_away = elo.get_rating("Wolves")
        elo.update("Arsenal", "Wolves", 2, 0)
        assert elo.get_rating("Arsenal") > before_home
        assert elo.get_rating("Wolves") < before_away

    def test_draw_adjusts_toward_parity(self, teams):
        elo = EloRatings(teams)
        # Give Arsenal a big lead first
        for _ in range(5):
            elo.update("Arsenal", "Wolves", 3, 0)
        before_arsenal = elo.get_rating("Arsenal")
        before_wolves = elo.get_rating("Wolves")
        elo.update("Arsenal", "Wolves", 1, 1)
        # Arsenal should lose rating (they were expected to win)
        assert elo.get_rating("Arsenal") < before_arsenal

    def test_predict_returns_three_probs(self, teams):
        elo = EloRatings(teams)
        probs = elo_predict(elo, "Arsenal", "Wolves")
        assert set(probs.keys()) == {"home", "draw", "away"}
        assert abs(sum(probs.values()) - 1.0) < 0.001
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement models.py**

```python
# pipeline/models.py
"""Prediction models: Dixon-Coles, xG-adjusted Dixon-Coles, and Elo ratings."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from datetime import datetime
from pipeline.config import (
    TIME_DECAY_RATE,
    MAX_GOALS,
    ELO_K_FACTOR,
    ELO_HOME_ADVANTAGE,
    FORM_WINDOW,
    FORM_WEIGHT_MULTIPLIER,
    CONGESTION_THRESHOLD_DAYS,
    CONGESTION_PENALTY,
)


# ── Dixon-Coles Model ──────────────────────────────────────────────────────

def _tau(x, y, lambda_h, lambda_a, rho):
    """Dixon-Coles correction factor for low-scoring outcomes.

    Adjusts the probability of 0-0, 1-0, 0-1, 1-1 scorelines to account
    for correlation that independent Poisson misses.
    """
    if x == 0 and y == 0:
        return 1 - lambda_h * lambda_a * rho
    elif x == 0 and y == 1:
        return 1 + lambda_h * rho
    elif x == 1 and y == 0:
        return 1 + lambda_a * rho
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1.0


def _dc_log_likelihood(params, matches, teams, weights):
    """Negative log-likelihood for the Dixon-Coles model.

    Parameters are packed as:
        [attack_0, ..., attack_n, defense_0, ..., defense_n, home_adv, rho]
    """
    n_teams = len(teams)
    team_idx = {t: i for i, t in enumerate(teams)}

    attack = params[:n_teams]
    defense = params[n_teams:2 * n_teams]
    home_adv = params[2 * n_teams]
    rho = params[2 * n_teams + 1]

    log_lik = 0.0
    for i, row in matches.iterrows():
        hi = team_idx[row["home_team"]]
        ai = team_idx[row["away_team"]]

        lambda_h = np.exp(attack[hi] + defense[ai] + home_adv)
        lambda_a = np.exp(attack[ai] + defense[hi])

        # Clamp lambdas to avoid numerical issues
        lambda_h = max(lambda_h, 0.001)
        lambda_a = max(lambda_a, 0.001)

        hg = int(row["home_goals"])
        ag = int(row["away_goals"])
        w = weights[i] if i < len(weights) else 1.0

        tau = _tau(hg, ag, lambda_h, lambda_a, rho)
        if tau <= 0:
            tau = 0.001

        log_lik += w * (
            np.log(tau)
            + np.log(poisson.pmf(hg, lambda_h) + 1e-10)
            + np.log(poisson.pmf(ag, lambda_a) + 1e-10)
        )

    return -log_lik


def _compute_weights(matches, decay_rate=TIME_DECAY_RATE, form_window=FORM_WINDOW,
                     form_multiplier=FORM_WEIGHT_MULTIPLIER):
    """Compute time-decay weights with form momentum boost.

    More recent matches get higher weight (exponential decay).
    The last `form_window` matches get an additional multiplier.
    """
    dates = pd.to_datetime(matches["date"])
    max_date = dates.max()
    days_ago = (max_date - dates).dt.days.values

    weights = np.exp(-decay_rate * days_ago)

    # Form momentum: boost the most recent matches
    n = len(matches)
    if n > form_window:
        # Sort by date, mark last form_window as boosted
        sorted_idx = dates.argsort()
        form_indices = sorted_idx[-form_window:]
        for idx in form_indices:
            weights[idx] *= form_multiplier

    return weights


def fit_dixon_coles(matches, goals_col_home="home_goals", goals_col_away="away_goals"):
    """Fit the Dixon-Coles model to historical match data.

    Args:
        matches: DataFrame with date, home_team, away_team, home_goals, away_goals
        goals_col_home: Column name for home goals (or home_xg for xG variant)
        goals_col_away: Column name for away goals (or away_xg for xG variant)

    Returns:
        dict with keys: attack, defense, home_advantage, rho
        attack/defense are dicts mapping team_name -> float
    """
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    n_teams = len(teams)

    # Rename columns if using xG
    fit_data = matches.copy()
    fit_data["home_goals"] = fit_data[goals_col_home]
    fit_data["away_goals"] = fit_data[goals_col_away]

    weights = _compute_weights(fit_data)

    # Initial params: attack=0, defense=0, home_adv=0.25, rho=-0.05
    x0 = np.zeros(2 * n_teams + 2)
    x0[2 * n_teams] = 0.25  # home advantage
    x0[2 * n_teams + 1] = -0.05  # rho (low-score correlation)

    # Constraint: sum of attack params = 0 (identifiability)
    constraints = [{"type": "eq", "fun": lambda p: np.sum(p[:n_teams])}]

    result = minimize(
        _dc_log_likelihood,
        x0,
        args=(fit_data, teams, weights),
        method="SLSQP",
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-6},
    )

    attack = {teams[i]: result.x[i] for i in range(n_teams)}
    defense = {teams[i]: result.x[n_teams + i] for i in range(n_teams)}
    home_advantage = result.x[2 * n_teams]
    rho = result.x[2 * n_teams + 1]

    return {
        "attack": attack,
        "defense": defense,
        "home_advantage": home_advantage,
        "rho": rho,
    }


def dixon_coles_predict(home_team, away_team, params, congestion_home=False,
                        congestion_away=False):
    """Predict scoreline probabilities for a single match.

    Args:
        home_team: Name of home team
        away_team: Name of away team
        params: Output of fit_dixon_coles()
        congestion_home: If True, apply fatigue penalty to home attack
        congestion_away: If True, apply fatigue penalty to away attack

    Returns:
        numpy array of shape (MAX_GOALS+1, MAX_GOALS+1) — scoreline probability matrix
        matrix[i][j] = P(home scores i, away scores j)
    """
    attack = params["attack"]
    defense = params["defense"]
    home_adv = params["home_advantage"]
    rho = params["rho"]

    att_h = attack.get(home_team, 0)
    def_h = defense.get(home_team, 0)
    att_a = attack.get(away_team, 0)
    def_a = defense.get(away_team, 0)

    lambda_h = np.exp(att_h + def_a + home_adv)
    lambda_a = np.exp(att_a + def_h)

    # Apply congestion penalty
    if congestion_home:
        lambda_h *= (1 - CONGESTION_PENALTY)
    if congestion_away:
        lambda_a *= (1 - CONGESTION_PENALTY)

    n = MAX_GOALS + 1
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            base = poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a)
            tau = _tau(i, j, lambda_h, lambda_a, rho)
            matrix[i][j] = base * max(tau, 0.001)

    # Normalize to sum to 1
    matrix /= matrix.sum()
    return matrix


def scoreline_to_probabilities(matrix):
    """Convert a scoreline matrix to home/draw/away probabilities.

    Args:
        matrix: (n, n) array where matrix[i][j] = P(home=i, away=j)

    Returns:
        dict with keys: home, draw, away (float probabilities summing to 1)
    """
    n = matrix.shape[0]
    home = sum(matrix[i][j] for i in range(n) for j in range(n) if i > j)
    draw = sum(matrix[i][i] for i in range(n))
    away = sum(matrix[i][j] for i in range(n) for j in range(n) if i < j)

    total = home + draw + away
    return {
        "home": home / total,
        "draw": draw / total,
        "away": away / total,
    }


# ── Elo Rating System ──────────────────────────────────────────────────────

class EloRatings:
    """Dynamic Elo rating system for EPL teams."""

    def __init__(self, teams, initial_rating=1500):
        self.ratings = {team: initial_rating for team in teams}

    def get_rating(self, team):
        return self.ratings.get(team, 1500)

    def expected_score(self, rating_a, rating_b):
        """Expected score for team A against team B (0-1 scale)."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update(self, home_team, away_team, home_goals, away_goals):
        """Update ratings after a match result."""
        hr = self.get_rating(home_team) + ELO_HOME_ADVANTAGE
        ar = self.get_rating(away_team)

        # Actual result: 1 for win, 0.5 for draw, 0 for loss
        if home_goals > away_goals:
            actual_h, actual_a = 1.0, 0.0
        elif home_goals == away_goals:
            actual_h, actual_a = 0.5, 0.5
        else:
            actual_h, actual_a = 0.0, 1.0

        expected_h = self.expected_score(hr, ar)
        expected_a = 1 - expected_h

        # Goal difference multiplier (bigger wins = bigger shifts)
        gd = abs(home_goals - away_goals)
        gd_mult = np.log(max(gd, 1) + 1)

        k = ELO_K_FACTOR * gd_mult

        self.ratings[home_team] += k * (actual_h - expected_h)
        self.ratings[away_team] += k * (actual_a - expected_a)

    def process_season(self, matches):
        """Process all matches in chronological order to build ratings."""
        sorted_matches = matches.sort_values("date")
        for _, row in sorted_matches.iterrows():
            self.update(
                row["home_team"], row["away_team"],
                int(row["home_goals"]), int(row["away_goals"])
            )


def elo_predict(elo, home_team, away_team):
    """Convert Elo ratings to home/draw/away probabilities.

    Uses a logistic model with a draw margin parameter.
    """
    hr = elo.get_rating(home_team) + ELO_HOME_ADVANTAGE
    ar = elo.get_rating(away_team)
    diff = hr - ar

    # Convert Elo difference to probabilities using logistic function
    # Draw probability modeled as a function of how close the teams are
    home_win_raw = 1 / (1 + 10 ** (-diff / 400))
    away_win_raw = 1 - home_win_raw

    # EPL average draw rate ~25%. Scale draw probability by rating closeness.
    draw_base = 0.25
    closeness = 1 - abs(home_win_raw - 0.5) * 2  # 1 when equal, 0 when dominant
    draw_prob = draw_base * (0.5 + 0.5 * closeness)

    # Distribute remaining probability
    remaining = 1 - draw_prob
    home_prob = remaining * home_win_raw
    away_prob = remaining * away_win_raw

    return {
        "home": home_prob,
        "draw": draw_prob,
        "away": away_prob,
    }
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add pipeline/models.py tests/test_models.py
git commit -m "feat: Dixon-Coles model and Elo rating system"
```

---

## Task 5: Ensemble Blending + Edge Detection

**Files:**
- Create: `pipeline/ensemble.py`
- Create: `tests/test_ensemble.py`

**Step 1: Write tests for ensemble and edge detection**

```python
# tests/test_ensemble.py
"""Tests for ensemble blending and edge detection."""

import pytest
from pipeline.ensemble import (
    blend_predictions,
    compute_edges,
    decimal_to_american,
    implied_probability,
)


class TestOddsConversion:
    def test_decimal_to_american_favorite(self):
        # 1.67 decimal = -149 American (approx -150)
        american = decimal_to_american(1.67)
        assert american < 0
        assert abs(american - (-149)) < 2

    def test_decimal_to_american_underdog(self):
        # 3.00 decimal = +200 American
        american = decimal_to_american(3.0)
        assert american > 0
        assert abs(american - 200) < 1

    def test_decimal_to_american_even(self):
        # 2.00 decimal = +100 American
        american = decimal_to_american(2.0)
        assert american == 100

    def test_implied_probability(self):
        # 2.00 decimal → 50% implied
        assert abs(implied_probability(2.0) - 0.50) < 0.001
        # 1.50 decimal → 66.7% implied
        assert abs(implied_probability(1.5) - 0.667) < 0.01


class TestBlendPredictions:
    def test_equal_weights(self):
        preds = [
            {"home": 0.5, "draw": 0.3, "away": 0.2},
            {"home": 0.4, "draw": 0.3, "away": 0.3},
            {"home": 0.6, "draw": 0.2, "away": 0.2},
        ]
        weights = [1 / 3, 1 / 3, 1 / 3]
        blended = blend_predictions(preds, weights)
        assert abs(blended["home"] - 0.5) < 0.001
        assert abs(blended["draw"] - 0.267) < 0.01
        assert abs(sum(blended.values()) - 1.0) < 0.001

    def test_weighted_blend(self):
        preds = [
            {"home": 0.6, "draw": 0.2, "away": 0.2},
            {"home": 0.4, "draw": 0.3, "away": 0.3},
        ]
        weights = [0.75, 0.25]
        blended = blend_predictions(preds, weights)
        # 0.75*0.6 + 0.25*0.4 = 0.55
        assert abs(blended["home"] - 0.55) < 0.001


class TestComputeEdges:
    def test_positive_edge_flagged(self):
        model_probs = {"home": 0.50, "draw": 0.25, "away": 0.25}
        odds = {"home_odds": 2.50, "draw_odds": 3.50, "away_odds": 3.00}
        edges = compute_edges(model_probs, odds)
        # home implied = 1/2.50 = 40%, model = 50%, edge = +10%
        assert edges["home"]["edge"] > 0.05
        assert edges["home"]["is_value"] is True

    def test_negative_edge_not_flagged(self):
        model_probs = {"home": 0.30, "draw": 0.30, "away": 0.40}
        odds = {"home_odds": 2.00, "draw_odds": 3.00, "away_odds": 3.50}
        edges = compute_edges(model_probs, odds)
        # home implied = 50%, model = 30%, edge = -20%
        assert edges["home"]["is_value"] is False
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ensemble.py -v`
Expected: FAIL

**Step 3: Implement ensemble.py**

```python
# pipeline/ensemble.py
"""Ensemble blending, edge detection, and odds conversion."""

from pipeline.config import VALUE_EDGE_THRESHOLD


def decimal_to_american(decimal_odds):
    """Convert decimal odds to American format.

    Decimal 2.00 = American +100 (even money)
    Decimal < 2.00 = negative American (favorite)
    Decimal > 2.00 = positive American (underdog)
    """
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1) * 100)
    else:
        return round(-100 / (decimal_odds - 1))


def implied_probability(decimal_odds):
    """Convert decimal odds to implied probability (0-1)."""
    return 1 / decimal_odds


def blend_predictions(predictions, weights):
    """Blend multiple model predictions using weighted average.

    Args:
        predictions: List of dicts, each with {home, draw, away} probabilities
        weights: List of floats summing to ~1.0

    Returns:
        dict with blended {home, draw, away} probabilities
    """
    blended = {"home": 0, "draw": 0, "away": 0}
    total_weight = sum(weights)

    for pred, w in zip(predictions, weights):
        norm_w = w / total_weight
        blended["home"] += pred["home"] * norm_w
        blended["draw"] += pred["draw"] * norm_w
        blended["away"] += pred["away"] * norm_w

    return blended


def compute_edges(model_probs, odds):
    """Compare model probabilities vs bookmaker odds to find edges.

    Args:
        model_probs: dict with {home, draw, away} from ensemble
        odds: dict with {home_odds, draw_odds, away_odds} in decimal format

    Returns:
        dict with home/draw/away sub-dicts containing:
            model_prob, implied_prob, edge, decimal_odds, american_odds, is_value
    """
    results = {}
    mapping = [
        ("home", "home_odds"),
        ("draw", "draw_odds"),
        ("away", "away_odds"),
    ]

    for outcome, odds_key in mapping:
        dec = odds[odds_key]
        imp = implied_probability(dec)
        model_p = model_probs[outcome]
        edge = model_p - imp

        results[outcome] = {
            "model_prob": round(model_p, 4),
            "implied_prob": round(imp, 4),
            "edge": round(edge, 4),
            "decimal_odds": dec,
            "american_odds": decimal_to_american(dec),
            "is_value": edge >= VALUE_EDGE_THRESHOLD,
        }

    return results
```

**Step 4: Run tests**

Run: `pytest tests/test_ensemble.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add pipeline/ensemble.py tests/test_ensemble.py
git commit -m "feat: ensemble blending and edge detection with odds conversion"
```

---

## Task 6: Backtest & Accuracy Tracking

**Files:**
- Create: `pipeline/backtest.py`
- Create: `tests/test_backtest.py`

**Step 1: Write tests**

```python
# tests/test_backtest.py
"""Tests for backtesting and accuracy tracking."""

import pytest
from pipeline.backtest import (
    evaluate_prediction,
    compute_model_weights,
    compute_roi,
    update_accuracy_log,
)


class TestEvaluatePrediction:
    def test_correct_home_prediction(self):
        pred = {"home": 0.55, "draw": 0.25, "away": 0.20}
        result = evaluate_prediction(pred, home_goals=2, away_goals=0)
        assert result["predicted"] == "home"
        assert result["actual"] == "home"
        assert result["correct"] is True

    def test_correct_draw_prediction(self):
        pred = {"home": 0.30, "draw": 0.40, "away": 0.30}
        result = evaluate_prediction(pred, home_goals=1, away_goals=1)
        assert result["predicted"] == "draw"
        assert result["actual"] == "draw"
        assert result["correct"] is True

    def test_incorrect_prediction(self):
        pred = {"home": 0.55, "draw": 0.25, "away": 0.20}
        result = evaluate_prediction(pred, home_goals=0, away_goals=1)
        assert result["predicted"] == "home"
        assert result["actual"] == "away"
        assert result["correct"] is False


class TestComputeModelWeights:
    def test_better_model_gets_more_weight(self):
        accuracies = [0.60, 0.45, 0.50]  # Model 0 is best
        weights = compute_model_weights(accuracies)
        assert weights[0] > weights[1]
        assert weights[0] > weights[2]
        assert abs(sum(weights) - 1.0) < 0.001

    def test_equal_accuracies_equal_weights(self):
        accuracies = [0.50, 0.50, 0.50]
        weights = compute_model_weights(accuracies)
        assert abs(weights[0] - weights[1]) < 0.001


class TestComputeROI:
    def test_profitable_bets(self):
        bets = [
            {"stake": 1.0, "odds": 2.50, "won": True},
            {"stake": 1.0, "odds": 2.00, "won": False},
            {"stake": 1.0, "odds": 3.00, "won": True},
        ]
        roi = compute_roi(bets)
        # Won: 2.50 + 3.00 = 5.50 return, staked 3.00, profit 2.50
        # ROI = 2.50 / 3.00 = 83.3%
        assert roi > 0

    def test_no_bets_returns_zero(self):
        assert compute_roi([]) == 0.0
```

**Step 2: Run tests to verify fail**

Run: `pytest tests/test_backtest.py -v`
Expected: FAIL

**Step 3: Implement backtest.py**

```python
# pipeline/backtest.py
"""Backtesting, accuracy tracking, and model weight computation."""

import numpy as np
from pipeline.config import ENSEMBLE_ACCURACY_WINDOW


def evaluate_prediction(probs, home_goals, away_goals):
    """Evaluate a single prediction against actual result.

    Args:
        probs: dict with {home, draw, away} probabilities
        home_goals: actual home goals scored
        away_goals: actual away goals scored

    Returns:
        dict with predicted, actual, correct, log_loss
    """
    predicted = max(probs, key=probs.get)

    if home_goals > away_goals:
        actual = "home"
    elif home_goals == away_goals:
        actual = "draw"
    else:
        actual = "away"

    # Log loss for this prediction (lower = better)
    actual_prob = max(probs[actual], 0.001)
    log_loss = -np.log(actual_prob)

    return {
        "predicted": predicted,
        "actual": actual,
        "correct": predicted == actual,
        "log_loss": round(log_loss, 4),
        "actual_prob": round(probs[actual], 4),
    }


def compute_model_weights(accuracies):
    """Compute ensemble weights from model accuracies.

    Better-performing models get more weight. Uses softmax-like
    scaling so weights sum to 1 and differences are meaningful.

    Args:
        accuracies: list of float (0-1) accuracy scores per model

    Returns:
        list of float weights summing to 1.0
    """
    if not accuracies or all(a == 0 for a in accuracies):
        n = len(accuracies) if accuracies else 1
        return [1 / n] * n

    # Softmax with temperature=0.5 for moderate differentiation
    arr = np.array(accuracies)
    exp = np.exp(arr * 2)  # temperature scaling
    weights = exp / exp.sum()
    return weights.tolist()


def compute_roi(bets):
    """Compute return on investment from a list of bets.

    Args:
        bets: list of dicts with {stake, odds (decimal), won (bool)}

    Returns:
        float: ROI as a fraction (0.10 = 10% ROI)
    """
    if not bets:
        return 0.0

    total_staked = sum(b["stake"] for b in bets)
    total_return = sum(b["stake"] * b["odds"] for b in bets if b["won"])
    profit = total_return - total_staked

    return round(profit / total_staked, 4) if total_staked > 0 else 0.0


def update_accuracy_log(accuracy_log, model_name, prediction_result):
    """Append a prediction result to a model's accuracy log.

    Args:
        accuracy_log: dict mapping model_name -> list of {correct, log_loss}
        model_name: string key for the model
        prediction_result: output of evaluate_prediction()

    Returns:
        Updated accuracy_log (mutated in place and returned)
    """
    if model_name not in accuracy_log:
        accuracy_log[model_name] = []

    accuracy_log[model_name].append({
        "correct": prediction_result["correct"],
        "log_loss": prediction_result["log_loss"],
    })

    # Keep only the most recent window
    if len(accuracy_log[model_name]) > ENSEMBLE_ACCURACY_WINDOW:
        accuracy_log[model_name] = accuracy_log[model_name][-ENSEMBLE_ACCURACY_WINDOW:]

    return accuracy_log


def get_rolling_accuracy(accuracy_log, model_name):
    """Get rolling accuracy for a model from its log.

    Returns:
        float: accuracy (0-1) over the log window, or 0.5 if no data
    """
    entries = accuracy_log.get(model_name, [])
    if not entries:
        return 0.5  # Default: no data = average

    correct = sum(1 for e in entries if e["correct"])
    return correct / len(entries)
```

**Step 4: Run tests**

Run: `pytest tests/test_backtest.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add pipeline/backtest.py tests/test_backtest.py
git commit -m "feat: backtesting, accuracy tracking, and model weight computation"
```

---

## Task 7: Pipeline Orchestrator

**Files:**
- Create: `pipeline/run.py`
- Create: `tests/test_run.py`

**Step 1: Write integration test**

```python
# tests/test_run.py
"""Integration tests for the pipeline orchestrator."""

import json
import os
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from pipeline.run import run_pipeline


class TestRunPipeline:
    @patch("pipeline.run.fetch_odds")
    @patch("pipeline.run.fetch_understat_xg")
    @patch("pipeline.run.fetch_epl_fixtures")
    @patch("pipeline.run.fetch_epl_matches")
    def test_produces_valid_predictions_json(
        self, mock_matches, mock_fixtures, mock_xg, mock_odds,
        sample_matches, sample_xg, sample_odds, tmp_path
    ):
        mock_matches.return_value = sample_matches
        mock_fixtures.return_value = [
            {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "date": "2026-02-22T15:00:00Z",
                "matchday": 26,
            }
        ]
        mock_xg.return_value = sample_xg
        mock_odds.return_value = [
            {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "commence_time": "2026-02-22T15:00:00Z",
                "home_odds": 1.67,
                "draw_odds": 3.80,
                "away_odds": 4.50,
            }
        ]

        output_dir = str(tmp_path)
        run_pipeline(output_dir=output_dir)

        predictions_path = os.path.join(output_dir, "predictions.json")
        assert os.path.exists(predictions_path)

        with open(predictions_path) as f:
            data = json.load(f)

        assert "generated_at" in data
        assert "matches" in data
        assert len(data["matches"]) >= 1

        match = data["matches"][0]
        assert match["home_team"] == "Arsenal"
        assert match["away_team"] == "Chelsea"
        assert "model_probs" in match
        assert "edges" in match
        assert abs(sum(match["model_probs"].values()) - 1.0) < 0.01
```

**Step 2: Run test to verify fail**

Run: `pytest tests/test_run.py -v`
Expected: FAIL

**Step 3: Implement run.py**

```python
# pipeline/run.py
"""Pipeline orchestrator — fetches data, runs models, writes output."""

import json
import os
from datetime import datetime, timedelta
import pandas as pd

from pipeline.config import DATA_DIR, PREDICTIONS_PATH, HISTORY_PATH, ACCURACY_PATH
from pipeline.fetch_data import fetch_epl_matches, fetch_epl_fixtures, fetch_odds
from pipeline.fetch_xg import fetch_understat_xg
from pipeline.models import (
    fit_dixon_coles,
    dixon_coles_predict,
    scoreline_to_probabilities,
    EloRatings,
    elo_predict,
)
from pipeline.ensemble import blend_predictions, compute_edges, decimal_to_american
from pipeline.backtest import (
    evaluate_prediction,
    compute_model_weights,
    compute_roi,
    get_rolling_accuracy,
    update_accuracy_log,
)


def _check_congestion(team, fixtures, matches, threshold_days=4):
    """Check if a team has fixture congestion (played recently)."""
    if matches.empty:
        return False
    team_matches = matches[
        (matches["home_team"] == team) | (matches["away_team"] == team)
    ]
    if team_matches.empty:
        return False
    last_date = pd.to_datetime(team_matches["date"]).max()
    # This is a rough check — ideally we'd know the exact fixture date
    days_since = (datetime.now() - last_date).days
    return days_since < threshold_days


def _load_json(path, default=None):
    """Load JSON file, return default if not found."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default if default is not None else {}


def _save_json(path, data):
    """Save data as formatted JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def run_pipeline(output_dir=None):
    """Run the full SLOP LOCKS prediction pipeline.

    1. Fetch match results, xG data, fixtures, and odds
    2. Fit Dixon-Coles (goals), Dixon-Coles (xG), and Elo models
    3. Blend into ensemble predictions
    4. Compare vs bookmaker odds to find edges
    5. Write predictions.json, update history.json and model_accuracy.json
    """
    if output_dir is None:
        output_dir = DATA_DIR

    predictions_path = os.path.join(output_dir, "predictions.json")
    history_path = os.path.join(output_dir, "history.json")
    accuracy_path = os.path.join(output_dir, "model_accuracy.json")

    print("Fetching EPL match results...")
    matches = fetch_epl_matches()
    print(f"  → {len(matches)} completed matches")

    print("Fetching xG data from Understat...")
    try:
        xg_data = fetch_understat_xg()
        print(f"  → {len(xg_data)} matches with xG")
        has_xg = len(xg_data) > 0
    except Exception as e:
        print(f"  → xG fetch failed: {e}. Running without xG model.")
        xg_data = pd.DataFrame()
        has_xg = False

    print("Fetching upcoming fixtures...")
    fixtures = fetch_epl_fixtures()
    print(f"  → {len(fixtures)} upcoming matches")

    print("Fetching bookmaker odds...")
    try:
        odds_data = fetch_odds()
        print(f"  → Odds for {len(odds_data)} matches")
    except Exception as e:
        print(f"  → Odds fetch failed: {e}. Running without odds.")
        odds_data = []

    if matches.empty or not fixtures:
        print("Not enough data to generate predictions.")
        return

    # ── Fit Models ──
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))

    print("Fitting Dixon-Coles (goals)...")
    dc_params = fit_dixon_coles(matches)

    dc_xg_params = None
    if has_xg:
        print("Fitting Dixon-Coles (xG)...")
        # Merge xG into matches
        xg_matches = matches.merge(
            xg_data[["date", "home_team", "away_team", "home_xg", "away_xg"]],
            on=["date", "home_team", "away_team"],
            how="inner",
        )
        if len(xg_matches) >= 20:
            dc_xg_params = fit_dixon_coles(
                xg_matches, goals_col_home="home_xg", goals_col_away="away_xg"
            )
            print(f"  → Fitted on {len(xg_matches)} matches with xG")
        else:
            print(f"  → Only {len(xg_matches)} xG matches, skipping xG model")

    print("Building Elo ratings...")
    elo = EloRatings(teams)
    elo.process_season(matches)

    # ── Load accuracy log for model weighting ──
    accuracy_log = _load_json(accuracy_path, {"dixon_coles": [], "xg": [], "elo": []})

    dc_acc = get_rolling_accuracy(accuracy_log, "dixon_coles")
    xg_acc = get_rolling_accuracy(accuracy_log, "xg") if dc_xg_params else 0
    elo_acc = get_rolling_accuracy(accuracy_log, "elo")

    if dc_xg_params:
        model_weights = compute_model_weights([dc_acc, xg_acc, elo_acc])
    else:
        model_weights = compute_model_weights([dc_acc, elo_acc])

    print(f"Model weights: DC={model_weights[0]:.2f}" +
          (f", xG={model_weights[1]:.2f}" if dc_xg_params else "") +
          f", Elo={model_weights[-1]:.2f}")

    # ── Build odds lookup ──
    odds_lookup = {}
    for o in odds_data:
        key = (o["home_team"], o["away_team"])
        odds_lookup[key] = o

    # ── Generate Predictions ──
    print("Generating predictions...")
    predictions = []

    for fixture in fixtures:
        home = fixture["home_team"]
        away = fixture["away_team"]

        if home not in dc_params["attack"] or away not in dc_params["attack"]:
            continue

        # Congestion check
        cong_home = _check_congestion(home, fixtures, matches)
        cong_away = _check_congestion(away, fixtures, matches)

        # Model predictions
        dc_matrix = dixon_coles_predict(home, away, dc_params, cong_home, cong_away)
        dc_probs = scoreline_to_probabilities(dc_matrix)

        model_preds = [dc_probs]

        if dc_xg_params and home in dc_xg_params["attack"] and away in dc_xg_params["attack"]:
            xg_matrix = dixon_coles_predict(home, away, dc_xg_params, cong_home, cong_away)
            xg_probs = scoreline_to_probabilities(xg_matrix)
            model_preds.append(xg_probs)

        elo_probs = elo_predict(elo, home, away)
        model_preds.append(elo_probs)

        # Blend
        ensemble_probs = blend_predictions(model_preds, model_weights[:len(model_preds)])

        # Edge detection
        odds_key = (home, away)
        edges = None
        if odds_key in odds_lookup:
            edges = compute_edges(ensemble_probs, odds_lookup[odds_key])

        match_pred = {
            "home_team": home,
            "away_team": away,
            "date": fixture["date"],
            "matchday": fixture.get("matchday"),
            "model_probs": {k: round(v, 4) for k, v in ensemble_probs.items()},
            "individual_models": {
                "dixon_coles": {k: round(v, 4) for k, v in dc_probs.items()},
                "elo": {k: round(v, 4) for k, v in elo_probs.items()},
            },
        }

        if dc_xg_params and len(model_preds) == 3:
            match_pred["individual_models"]["xg"] = {k: round(v, 4) for k, v in xg_probs.items()}

        if edges:
            match_pred["edges"] = edges
            match_pred["best_odds"] = {
                "home": decimal_to_american(odds_lookup[odds_key]["home_odds"]),
                "draw": decimal_to_american(odds_lookup[odds_key]["draw_odds"]),
                "away": decimal_to_american(odds_lookup[odds_key]["away_odds"]),
            }

        predictions.append(match_pred)

    # ── Evaluate past predictions ──
    history = _load_json(history_path, {"predictions": [], "bets": []})

    # Check if any previous predictions now have results
    for past in history.get("predictions", []):
        if past.get("result") is not None:
            continue  # Already evaluated
        home = past["home_team"]
        away = past["away_team"]
        match_result = matches[
            (matches["home_team"] == home) & (matches["away_team"] == away) &
            (matches["date"] == past["date"][:10])
        ]
        if not match_result.empty:
            row = match_result.iloc[0]
            past["result"] = {
                "home_goals": int(row["home_goals"]),
                "away_goals": int(row["away_goals"]),
            }
            # Evaluate each model
            eval_result = evaluate_prediction(
                past["model_probs"], int(row["home_goals"]), int(row["away_goals"])
            )
            past["evaluation"] = eval_result
            accuracy_log = update_accuracy_log(accuracy_log, "dixon_coles", eval_result)

    # Add new predictions to history
    for pred in predictions:
        # Don't duplicate
        existing = [p for p in history["predictions"]
                    if p["home_team"] == pred["home_team"]
                    and p["away_team"] == pred["away_team"]
                    and p.get("date") == pred.get("date")]
        if not existing:
            history["predictions"].append(pred)

    # Keep history manageable (last 200 predictions)
    history["predictions"] = history["predictions"][-200:]

    # ── Compute season stats ──
    evaluated = [p for p in history["predictions"] if p.get("evaluation")]
    total_preds = len(evaluated)
    correct_preds = sum(1 for p in evaluated if p["evaluation"]["correct"])
    accuracy = correct_preds / total_preds if total_preds > 0 else 0

    # Value bet tracking
    value_bets = [p for p in evaluated if p.get("edges") and
                  any(e["is_value"] for e in p["edges"].values())]
    value_correct = sum(1 for p in value_bets if p["evaluation"]["correct"])

    stats = {
        "total_predictions": total_preds,
        "correct_predictions": correct_preds,
        "accuracy": round(accuracy, 4),
        "value_bets_placed": len(value_bets),
        "value_bets_correct": value_correct,
        "value_bet_accuracy": round(value_correct / len(value_bets), 4) if value_bets else 0,
    }

    # ── Write Output ──
    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "matches": predictions,
        "season_stats": stats,
        "model_weights": {
            "dixon_coles": round(model_weights[0], 4),
            "elo": round(model_weights[-1], 4),
        },
    }
    if dc_xg_params and len(model_weights) == 3:
        output["model_weights"]["xg"] = round(model_weights[1], 4)

    _save_json(predictions_path, output)
    _save_json(history_path, history)
    _save_json(accuracy_path, accuracy_log)

    print(f"\nPredictions written for {len(predictions)} matches")
    print(f"Season accuracy: {correct_preds}/{total_preds} ({accuracy:.1%})")
    if value_bets:
        print(f"Value bets: {value_correct}/{len(value_bets)} ({value_correct/len(value_bets):.1%})")
    print("Done.")


if __name__ == "__main__":
    run_pipeline()
```

**Step 4: Run tests**

Run: `pytest tests/test_run.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pipeline/run.py tests/test_run.py
git commit -m "feat: pipeline orchestrator — fetches data, runs ensemble, writes predictions"
```

---

## Task 8: GitHub Action Workflow

**Files:**
- Create: `.github/workflows/daily.yml`

**Step 1: Create the workflow file**

```yaml
# .github/workflows/daily.yml
name: Daily EPL Predictions

on:
  schedule:
    - cron: '0 6 * * *'  # 6am UTC daily
  workflow_dispatch:  # Manual trigger

permissions:
  contents: write

jobs:
  predict:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r pipeline/requirements.txt

      - name: Run prediction pipeline
        env:
          FOOTBALL_DATA_API_KEY: ${{ secrets.FOOTBALL_DATA_API_KEY }}
          ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}
        run: python -m pipeline.run

      - name: Commit and push predictions
        run: |
          git config user.name "sloplocks-bot"
          git config user.email "bot@sloplocks.lol"
          git add data/
          git diff --staged --quiet || git commit -m "data: daily predictions $(date -u +%Y-%m-%d)"
          git push
```

**Step 2: Commit**

```bash
git add .github/workflows/daily.yml
git commit -m "feat: daily GitHub Action for automated predictions"
```

---

## Task 9: Frontend — HTML/CSS/JS

**Files:**
- Create: `index.html`

This is the big one. Single file, dark theme, reads the JSON data and renders match cards with edge highlighting.

**Step 1: Create index.html**

The frontend should:
- Fetch `data/predictions.json` on load
- Render upcoming match cards with model probabilities
- Highlight value bets (edge > 5%) in neon green
- Show American odds
- Display season accuracy stats
- Show recent prediction results
- Be responsive for mobile
- Be a PWA (manifest + service worker)

The complete HTML is too large to inline here, but the key sections are:

**CSS:** Dark theme (#0a0a0a background), neon green (#00ff88) for value bets, white text, monospace for odds, Oswald for headings. Compact match cards with a 3-column probability bar (home/draw/away).

**JS:** On load, fetch `data/predictions.json`. Parse matches, render cards. For each match: show team names, date, model probabilities as a visual bar, American odds, edge percentages (green if value). Season stats section reads from the same JSON.

**Structure:**
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SLOP LOCKS — EPL Predictions</title>
  <link rel="manifest" href="/manifest.json">
  <link rel="icon" href="/icons/icon-192.png">
  <link href="https://fonts.googleapis.com/css2?family=Oswald:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>/* ... full CSS ... */</style>
</head>
<body>
  <header><!-- SLOP LOCKS branding --></header>
  <main>
    <section id="matches"><!-- Rendered by JS --></section>
    <section id="stats"><!-- Season accuracy --></section>
    <section id="recent"><!-- Past predictions --></section>
  </main>
  <script>/* ... fetch JSON, render cards ... */</script>
</body>
</html>
```

> **Note for implementer:** The full index.html will be ~800-1000 lines. Build it section by section: first the CSS variables and layout, then the match card rendering, then the stats section, then the recent predictions. Use the frontend-design skill for the visual execution.

**Step 2: Commit**

```bash
git add index.html
git commit -m "feat: frontend — match cards, edge highlighting, season stats"
```

---

## Task 10: PWA — Manifest + Service Worker

**Files:**
- Create: `manifest.json`
- Create: `sw.js`
- Create: `icons/icon-192.png` (placeholder)
- Create: `icons/icon-512.png` (placeholder)

**Step 1: Create manifest.json**

```json
{
  "name": "SLOP LOCKS",
  "short_name": "SLOP LOCKS",
  "description": "EPL match predictions and betting edge finder",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#0a0a0a",
  "theme_color": "#00ff88",
  "icons": [
    { "src": "/icons/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any" },
    { "src": "/icons/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable" }
  ]
}
```

**Step 2: Create sw.js**

```javascript
const CACHE_NAME = 'sloplocks-v1';
const ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/icons/icon-192.png',
  '/icons/icon-512.png',
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(ASSETS))
      .then(() => self.skipWaiting())
      .catch(err => {
        console.error('SW install failed:', err);
        self.skipWaiting();
      })
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', event => {
  // Always fetch data/ files from network (they update daily)
  if (event.request.url.includes('/data/')) {
    event.respondWith(
      fetch(event.request)
        .then(response => {
          const clone = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
          return response;
        })
        .catch(() => caches.match(event.request))
    );
    return;
  }

  // Cache-first for static assets
  event.respondWith(
    caches.match(event.request).then(cached => {
      if (cached) return cached;
      return fetch(event.request);
    }).catch(() => {
      if (event.request.mode === 'navigate') {
        return caches.match('/');
      }
      return new Response('', { status: 503, statusText: 'Offline' });
    })
  );
});
```

**Step 3: Commit**

```bash
mkdir -p icons
git add manifest.json sw.js icons/
git commit -m "feat: PWA manifest and service worker with network-first data strategy"
```

---

## Task 11: CLAUDE.md + README

**Files:**
- Create: `CLAUDE.md`
- Create: `README.md`

**Step 1: Create CLAUDE.md**

```markdown
# SLOP LOCKS — Project Brief

## What is this?
EPL match prediction engine with a 3-model ensemble and automated betting edge detection. Live at https://sloplocks.lol.

## Tech Stack
- Pipeline: Python 3.11+ (scipy, pandas, numpy, requests, beautifulsoup4)
- Frontend: Single-file HTML/CSS/JS
- Automation: GitHub Actions (daily cron)
- Hosting: Netlify (auto-deploy from GitHub)
- Domain: sloplocks.lol

## Architecture
- GitHub Action runs daily at 6am UTC
- Python pipeline fetches data, runs models, writes JSON to `data/`
- Static frontend reads JSON files, no backend

## The Ensemble
Three models blended by rolling accuracy:
1. **Dixon-Coles** — Attack/defense parameters from goals, time-decay, low-score correction
2. **xG-adjusted Dixon-Coles** — Same model using expected goals from Understat
3. **Elo ratings** — Dynamic power ratings updated after each match

## Data Sources
- football-data.org (results, fixtures) — API key in `FOOTBALL_DATA_API_KEY`
- Understat (xG) — scraped, no key needed
- The Odds API (bookmaker odds) — API key in `ODDS_API_KEY`

## Commands
- Run pipeline locally: `python -m pipeline.run`
- Run tests: `pytest tests/ -v`
- Install deps: `pip install -r pipeline/requirements.txt`

## File Structure
```
sloplocks/
├── index.html         ← Frontend (single file)
├── data/              ← Generated daily by pipeline
├── pipeline/          ← Python prediction pipeline
├── tests/             ← pytest tests
├── .github/workflows/ ← GitHub Action
├── manifest.json      ← PWA manifest
└── sw.js              ← Service worker
```

## Color Scheme
- Background: #0a0a0a
- Value/edge: #00ff88 (neon green)
- Text: #ffffff
- Secondary: #888888
- Danger/negative: #ff4444

## Deployment
Push to master → Netlify auto-deploys.
Pipeline runs daily via GitHub Action, commits updated predictions.
```

**Step 2: Create README.md**

```markdown
# SLOP LOCKS 🔒

EPL match predictions powered by a Dixon-Coles + xG + Elo ensemble.

**Live:** [sloplocks.lol](https://sloplocks.lol)

## How It Works

A GitHub Action runs daily, pulling EPL results, xG data, and bookmaker odds. Three prediction models are blended into an ensemble, and the output is compared against bookmaker lines to surface value bets.

## Setup

1. Register for free API keys:
   - [football-data.org](https://www.football-data.org/client/register)
   - [The Odds API](https://the-odds-api.com/#get-access)

2. Add keys as GitHub Secrets:
   - `FOOTBALL_DATA_API_KEY`
   - `ODDS_API_KEY`

3. Push to GitHub — the Action handles the rest.

## Local Development

```bash
python -m venv venv
source venv/bin/activate
pip install -r pipeline/requirements.txt
pytest tests/ -v
python -m pipeline.run
```
```

**Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: add CLAUDE.md and README with setup instructions"
```

---

## Task 12: API Key Registration + GitHub Repo Setup

**Step 1:** Register for a free API key at [football-data.org](https://www.football-data.org/client/register)

**Step 2:** Register for a free API key at [the-odds-api.com](https://the-odds-api.com/#get-access)

**Step 3:** Create GitHub repo

```bash
gh repo create sloplocks --public --source=/home/jason/sloplocks --push
```

**Step 4:** Add API keys as GitHub Secrets

```bash
gh secret set FOOTBALL_DATA_API_KEY
gh secret set ODDS_API_KEY
```

**Step 5:** Set up Netlify

- Connect the `sloplocks` repo to Netlify
- Point `sloplocks.lol` DNS to Netlify
- No build command needed (static site)
- Publish directory: `/` (root)

---

## Build Order Summary

| Task | What | Depends On |
|------|------|-----------|
| 1 | Project scaffolding | — |
| 2 | football-data.org client | 1 |
| 3 | Understat xG scraper | 1 |
| 4 | Dixon-Coles + Elo models | 1 |
| 5 | Ensemble + edge detection | 4 |
| 6 | Backtest + accuracy | 5 |
| 7 | Pipeline orchestrator | 2, 3, 4, 5, 6 |
| 8 | GitHub Action | 7 |
| 9 | Frontend | 7 |
| 10 | PWA | 9 |
| 11 | CLAUDE.md + README | — |
| 12 | API keys + deploy | 8, 10, 11 |
