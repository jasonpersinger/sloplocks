# SLOPLOCKS Revamp Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Revamp SLOPLOCKS to post ≤5 curated, line-shopped betting picks daily to Discord via a clean curator layer, adding NHL support and removing the website and pick evaluation machinery.

**Architecture:** `run.py` fetches data and fits models, writing raw candidate matches to per-sport `predictions.json`. `curator.py` reads all sports' candidates, applies strict guardrails (8% min edge, -130 to +350 odds, de-vigged line shopping across 6 books), ranks by edge, and returns ≤5 picks. `notify_discord.py` formats and posts the card.

**Tech Stack:** Python 3.11+, requests, pandas, numpy, scipy, pytest. ESPN NHL API (free). The Odds API (free tier, h2h only).

---

## Chunk 1: Foundation — Config, fetch_data, fetch_nhl

### Task 1: Update `pipeline/config.py`

**Files:**
- Modify: `pipeline/config.py`

- [ ] **Step 1: Open config.py and make these changes**

Remove `SLOP_LOCK_FALLBACK_MIN_ODDS`. Add `BOOKMAKERS`, `BOOK_ABBREV`, NHL ESPN base URL, and NHL sport config.

```python
# pipeline/config.py — full replacement

"""Central configuration for the SLOP LOCKS pipeline."""

import os

# API Keys (from environment / GitHub Secrets)
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
BALLDONTLIE_API_KEY = os.environ.get("BALLDONTLIE_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# The Odds API
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_REGIONS = "us"
ODDS_MARKETS = "h2h"

# balldontlie.io
BALLDONTLIE_BASE = "https://api.balldontlie.io/v1"

# ESPN (no API key needed)
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
NBA_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
NHL_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl"

# Model parameters
TIME_DECAY_RATE = 0.005
FORM_WINDOW = 6
FORM_WEIGHT_MULTIPLIER = 2.0
CONGESTION_THRESHOLD_DAYS = 4
CONGESTION_PENALTY = 0.05
ELO_K_FACTOR = 20
ELO_HOME_ADVANTAGE = 65
NBA_B2B_PENALTY = 30
VALUE_EDGE_THRESHOLD = 0.05
SLOP_LOCK_MIN_ODDS = -150
SLOP_LOCK_MAX_ODDS = 195
ENSEMBLE_ACCURACY_WINDOW = 10
MAX_GOALS = 6

# Curator guardrails
CURATOR_MIN_EDGE = 0.08
CURATOR_MIN_ODDS = -130   # American
CURATOR_MAX_ODDS = 350    # American
CURATOR_MAX_PICKS = 5
CURATOR_MIN_BOOKS = 2     # Minimum bookmakers required to trust implied prob

# Line shopping — bookmaker API keys and display abbreviations
BOOKMAKERS = [
    "betmgm",
    "williamhill_us",   # Caesars
    "draftkings",
    "fanduel",
    "bet365",
    "ballysports",      # Bally Bet
]

BOOK_ABBREV = {
    "betmgm": "MGM",
    "williamhill_us": "CZR",
    "draftkings": "DK",
    "fanduel": "FD",
    "bet365": "B365",
    "ballysports": "BALLY",
}

# Sport emoji for Discord
SPORT_EMOJI = {
    "nba": "🏀",
    "ncaam": "🎓",
    "nhl": "🏒",
}

# Output paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "predictions.json")
HISTORY_PATH = os.path.join(DATA_DIR, "history.json")
ACCURACY_PATH = os.path.join(DATA_DIR, "model_accuracy.json")

# Per-sport configuration
SPORTS = {
    "nba": {
        "name": "NBA",
        "display_name": "NBA",
        "odds_sport": "basketball_nba",
        "outcomes": ["home", "away"],
        "models": ["elo", "efficiency", "four_factors"],
        "elo_k_factor": 20,
        "elo_home_advantage": 65,
        "efficiency_home_bonus": 3.5,
        "data_dir": os.path.join(DATA_DIR, "nba"),
    },
    "ncaam": {
        "name": "NCAAM",
        "display_name": "NCAAM",
        "odds_sport": "basketball_ncaab",
        "outcomes": ["home", "away"],
        "models": ["elo", "efficiency", "four_factors"],
        "elo_k_factor": 32,
        "elo_home_advantage": 125,
        "efficiency_home_bonus": 3.5,
        "data_dir": os.path.join(DATA_DIR, "ncaam"),
    },
    "nhl": {
        "name": "NHL",
        "display_name": "NHL",
        "odds_sport": "icehockey_nhl",
        "outcomes": ["home", "away"],
        "models": ["elo"],
        "elo_k_factor": 6,
        "elo_home_advantage": 15,
        "data_dir": os.path.join(DATA_DIR, "nhl"),
    },
}
```

- [ ] **Step 2: Run existing tests to confirm nothing broke**

```bash
pytest tests/ -v -x
```
Expected: all existing tests pass (config changes are additive except removing `SLOP_LOCK_FALLBACK_MIN_ODDS`).

- [ ] **Step 3: Fix any import errors** (search for `SLOP_LOCK_FALLBACK_MIN_ODDS` in the codebase and remove those imports — they'll be cleaned up fully in Task 5)

```bash
grep -r "SLOP_LOCK_FALLBACK_MIN_ODDS" pipeline/ --include="*.py"
```

- [ ] **Step 4: Commit**

```bash
git add pipeline/config.py
git commit -m "config: add NHL, BOOKMAKERS, curator constants; remove fallback odds"
```

---

### Task 2: Update `pipeline/fetch_data.py` to return per-book odds

**Files:**
- Modify: `pipeline/fetch_data.py`
- Modify: `tests/test_fetch_data.py`

The current `fetch_data.py` returns only the best odds across all bookmakers. We need it to also return per-book odds filtered to `BOOKMAKERS`, so the curator can de-vig per book.

- [ ] **Step 1: Write the failing test first**

Add to `tests/test_fetch_data.py`:

```python
from unittest.mock import patch, MagicMock
from pipeline.fetch_data import fetch_odds
from pipeline.config import BOOKMAKERS

class TestFetchOddsPerBook:
    """fetch_odds returns per-book odds for our configured bookmakers."""

    def _make_mock_response(self):
        """Build a fake Odds API response with two bookmakers."""
        return [
            {
                "id": "abc123",
                "home_team": "Boston Bruins",
                "away_team": "New York Rangers",
                "commence_time": "2026-03-15T23:00:00Z",
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "markets": [{"key": "h2h", "outcomes": [
                            {"name": "Boston Bruins", "price": 2.10},
                            {"name": "New York Rangers", "price": 1.80},
                        ]}],
                    },
                    {
                        "key": "fanduel",
                        "markets": [{"key": "h2h", "outcomes": [
                            {"name": "Boston Bruins", "price": 2.05},
                            {"name": "New York Rangers", "price": 1.85},
                        ]}],
                    },
                    {
                        "key": "unknown_book",  # not in BOOKMAKERS — should be ignored
                        "markets": [{"key": "h2h", "outcomes": [
                            {"name": "Boston Bruins", "price": 2.20},
                            {"name": "New York Rangers", "price": 1.70},
                        ]}],
                    },
                ],
            }
        ]

    @patch("pipeline.fetch_data.requests.get")
    def test_returns_book_odds_dict(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._make_mock_response()
        mock_get.return_value = mock_resp

        results = fetch_odds("icehockey_nhl")

        assert len(results) == 1
        event = results[0]
        assert "book_odds" in event

    @patch("pipeline.fetch_data.requests.get")
    def test_book_odds_filtered_to_configured_books(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._make_mock_response()
        mock_get.return_value = mock_resp

        results = fetch_odds("icehockey_nhl")
        book_odds = results[0]["book_odds"]

        assert "draftkings" in book_odds
        assert "fanduel" in book_odds
        assert "unknown_book" not in book_odds

    @patch("pipeline.fetch_data.requests.get")
    def test_book_odds_home_away_structure(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._make_mock_response()
        mock_get.return_value = mock_resp

        results = fetch_odds("icehockey_nhl")
        dk = results[0]["book_odds"]["draftkings"]

        assert dk["home_odds"] == pytest.approx(2.10)
        assert dk["away_odds"] == pytest.approx(1.80)

    @patch("pipeline.fetch_data.requests.get")
    def test_legacy_best_odds_still_present(self, mock_get):
        """home_odds / away_odds (best across books) are still returned for backward compat."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._make_mock_response()
        mock_get.return_value = mock_resp

        results = fetch_odds("icehockey_nhl")
        event = results[0]

        # Best home odds across DK (2.10) and FD (2.05) = 2.10
        assert event["home_odds"] == pytest.approx(2.10)
        assert event["away_odds"] == pytest.approx(1.85)
```

- [ ] **Step 2: Run to confirm it fails**

```bash
pytest tests/test_fetch_data.py::TestFetchOddsPerBook -v
```
Expected: FAIL — `book_odds` key not present.

- [ ] **Step 3: Update `pipeline/fetch_data.py`**

```python
"""Fetch odds from The Odds API."""

import requests

from pipeline.config import (
    BOOKMAKERS,
    ODDS_API_KEY,
    ODDS_API_BASE,
    ODDS_REGIONS,
    ODDS_MARKETS,
)


def fetch_odds(sport_key: str) -> list[dict]:
    """Fetch h2h odds for upcoming matches, with per-book breakdown.

    Returns a list of dicts with keys:
        home_team, away_team, commence_time,
        home_odds, away_odds  (best across configured books — backward compat),
        book_odds  (dict: book_key → {home_odds, away_odds} for our books only)
    """
    url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": ODDS_REGIONS,
        "markets": ODDS_MARKETS,
        "oddsFormat": "decimal",
        "bookmakers": ",".join(BOOKMAKERS),
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    events = resp.json()

    results = []
    for event in events:
        home_name = event["home_team"]
        away_name = event["away_team"]

        best_home = 0.0
        best_draw = 0.0
        best_away = 0.0
        book_odds: dict[str, dict] = {}

        for bookmaker in event.get("bookmakers", []):
            book_key = bookmaker.get("key", "")
            if book_key not in BOOKMAKERS:
                continue

            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcome_map = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                h = outcome_map.get(home_name, 0.0)
                d = outcome_map.get("Draw", 0.0)
                a = outcome_map.get(away_name, 0.0)

                if h > 1.0 and a > 1.0:
                    book_odds[book_key] = {"home_odds": h, "away_odds": a}

                best_home = max(best_home, h)
                best_draw = max(best_draw, d)
                best_away = max(best_away, a)

        results.append({
            "home_team": home_name,
            "away_team": away_name,
            "commence_time": event["commence_time"],
            "home_odds": best_home,
            "draw_odds": best_draw,
            "away_odds": best_away,
            "book_odds": book_odds,
        })

    return results
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_fetch_data.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pipeline/fetch_data.py tests/test_fetch_data.py
git commit -m "feat: fetch per-book odds for line shopping"
```

---

### Task 3: Create `pipeline/fetch_nhl.py`

**Files:**
- Create: `pipeline/fetch_nhl.py`
- Create: `tests/test_fetch_nhl.py`

The ESPN NHL API uses the same scoreboard structure as NCAAM. We need historical game results (to fit Elo) and today's schedule (to predict).

- [ ] **Step 1: Write failing tests**

Create `tests/test_fetch_nhl.py`:

```python
"""Tests for pipeline.fetch_nhl — ESPN NHL data fetcher."""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from pipeline.fetch_nhl import fetch_nhl_games, fetch_nhl_schedule, normalize_nhl_team_name


def _make_scoreboard(events):
    return {"events": events}


def _make_event(home_name, away_name, home_score, away_score,
                date="2026-03-10T23:00Z", completed=True, event_id="1"):
    return {
        "id": event_id,
        "date": date,
        "competitions": [{
            "status": {"type": {"completed": completed}},
            "competitors": [
                {
                    "homeAway": "home",
                    "team": {"displayName": home_name, "abbreviation": home_name[:3].upper()},
                    "score": str(home_score),
                },
                {
                    "homeAway": "away",
                    "team": {"displayName": away_name, "abbreviation": away_name[:3].upper()},
                    "score": str(away_score),
                },
            ],
        }],
    }


class TestNormalizeNhlTeamName:
    def test_known_team(self):
        assert normalize_nhl_team_name("Boston Bruins") == "Bruins"

    def test_unknown_team_returns_original(self):
        assert normalize_nhl_team_name("Unknown Team") == "Unknown Team"


class TestFetchNhlGames:
    @patch("pipeline.fetch_nhl.requests.get")
    def test_returns_dataframe(self, mock_get):
        mock_get.return_value = MagicMock(
            json=lambda: _make_scoreboard([
                _make_event("Boston Bruins", "New York Rangers", 3, 2),
                _make_event("Toronto Maple Leafs", "Montreal Canadiens", 4, 1,
                            event_id="2"),
            ])
        )
        games, _ = fetch_nhl_games()
        assert isinstance(games, pd.DataFrame)
        assert len(games) == 2

    @patch("pipeline.fetch_nhl.requests.get")
    def test_columns_present(self, mock_get):
        mock_get.return_value = MagicMock(
            json=lambda: _make_scoreboard([
                _make_event("Boston Bruins", "New York Rangers", 3, 2),
            ])
        )
        games, _ = fetch_nhl_games()
        for col in ("date", "home_team", "away_team", "home_goals", "away_goals"):
            assert col in games.columns

    @patch("pipeline.fetch_nhl.requests.get")
    def test_skips_incomplete_games(self, mock_get):
        mock_get.return_value = MagicMock(
            json=lambda: _make_scoreboard([
                _make_event("Boston Bruins", "New York Rangers", 3, 2, completed=True),
                _make_event("Toronto Maple Leafs", "Montreal Canadiens", 0, 0,
                            completed=False, event_id="2"),
            ])
        )
        games, _ = fetch_nhl_games()
        assert len(games) == 1

    @patch("pipeline.fetch_nhl.requests.get")
    def test_team_names_normalized(self, mock_get):
        mock_get.return_value = MagicMock(
            json=lambda: _make_scoreboard([
                _make_event("Boston Bruins", "New York Rangers", 3, 2),
            ])
        )
        games, _ = fetch_nhl_games()
        assert games.iloc[0]["home_team"] == "Bruins"
        assert games.iloc[0]["away_team"] == "Rangers"


class TestFetchNhlSchedule:
    @patch("pipeline.fetch_nhl.requests.get")
    def test_returns_list(self, mock_get):
        mock_get.return_value = MagicMock(
            json=lambda: _make_scoreboard([
                _make_event("Boston Bruins", "New York Rangers", 0, 0,
                            completed=False, date="2026-03-15T23:00Z"),
            ])
        )
        schedule = fetch_nhl_schedule()
        assert isinstance(schedule, list)

    @patch("pipeline.fetch_nhl.requests.get")
    def test_schedule_item_structure(self, mock_get):
        mock_get.return_value = MagicMock(
            json=lambda: _make_scoreboard([
                _make_event("Boston Bruins", "New York Rangers", 0, 0,
                            completed=False, date="2026-03-15T23:00Z"),
            ])
        )
        schedule = fetch_nhl_schedule()
        assert len(schedule) == 1
        item = schedule[0]
        assert "home_team" in item
        assert "away_team" in item
        assert "date" in item
        assert item["home_team"] == "Bruins"
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
pytest tests/test_fetch_nhl.py -v
```
Expected: ModuleNotFoundError (file doesn't exist yet).

- [ ] **Step 3: Create `pipeline/fetch_nhl.py`**

```python
"""Fetch NHL game results and schedule from ESPN."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from pipeline.config import NHL_ESPN_BASE

_NHL_TEAM_NAME_MAP = {
    "Anaheim Ducks": "Ducks",
    "Arizona Coyotes": "Coyotes",
    "Boston Bruins": "Bruins",
    "Buffalo Sabres": "Sabres",
    "Calgary Flames": "Flames",
    "Carolina Hurricanes": "Hurricanes",
    "Chicago Blackhawks": "Blackhawks",
    "Colorado Avalanche": "Avalanche",
    "Columbus Blue Jackets": "Blue Jackets",
    "Dallas Stars": "Stars",
    "Detroit Red Wings": "Red Wings",
    "Edmonton Oilers": "Oilers",
    "Florida Panthers": "Panthers",
    "Los Angeles Kings": "Kings",
    "Minnesota Wild": "Wild",
    "Montreal Canadiens": "Canadiens",
    "Nashville Predators": "Predators",
    "New Jersey Devils": "Devils",
    "New York Islanders": "Islanders",
    "New York Rangers": "Rangers",
    "Ottawa Senators": "Senators",
    "Philadelphia Flyers": "Flyers",
    "Pittsburgh Penguins": "Penguins",
    "San Jose Sharks": "Sharks",
    "Seattle Kraken": "Kraken",
    "St. Louis Blues": "Blues",
    "Tampa Bay Lightning": "Lightning",
    "Toronto Maple Leafs": "Maple Leafs",
    "Utah Hockey Club": "Utah",
    "Vancouver Canucks": "Canucks",
    "Vegas Golden Knights": "Golden Knights",
    "Washington Capitals": "Capitals",
    "Winnipeg Jets": "Jets",
}


def normalize_nhl_team_name(name: str) -> str:
    """Map an ESPN full NHL team name to its short display name."""
    return _NHL_TEAM_NAME_MAP.get(name, name)


def _parse_scoreboard(data: dict, completed_only: bool) -> list[dict]:
    """Parse ESPN scoreboard JSON into a list of game dicts."""
    games = []
    for event in data.get("events", []):
        competition = event.get("competitions", [{}])[0]
        status = competition.get("status", {}).get("type", {})
        is_completed = status.get("completed", False)

        if completed_only and not is_completed:
            continue
        if not completed_only and is_completed:
            continue

        competitors = competition.get("competitors", [])
        if len(competitors) < 2:
            continue

        home = next((c for c in competitors if c["homeAway"] == "home"), None)
        away = next((c for c in competitors if c["homeAway"] == "away"), None)
        if not home or not away:
            continue

        date_str = event.get("date", "")[:10]
        home_name = normalize_nhl_team_name(home["team"]["displayName"])
        away_name = normalize_nhl_team_name(away["team"]["displayName"])

        game = {
            "game_id": event["id"],
            "date": date_str,
            "home_team": home_name,
            "away_team": away_name,
        }
        if completed_only:
            try:
                game["home_goals"] = int(home.get("score", 0))
                game["away_goals"] = int(away.get("score", 0))
            except (ValueError, TypeError):
                continue
        games.append(game)
    return games


def fetch_nhl_games(days_back: int = 90) -> tuple[pd.DataFrame, None]:
    """Fetch completed NHL games from the past `days_back` days via ESPN.

    Returns
    -------
    tuple[pd.DataFrame, None]
        games DataFrame with columns: game_id, date, home_team, away_team,
        home_goals, away_goals. Second element is None (no box scores for NHL).
    """
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=days_back)

    all_games: list[dict] = []
    current = start
    # ESPN supports date range via calendar param; fetch month by month
    seen_months: set[str] = set()
    while current <= today:
        month_key = current.strftime("%Y%m")
        if month_key not in seen_months:
            seen_months.add(month_key)
            url = f"{NHL_ESPN_BASE}/scoreboard"
            params = {"limit": 1000, "dates": current.strftime("%Y%m01")}
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                all_games.extend(_parse_scoreboard(resp.json(), completed_only=True))
            except Exception:
                pass
        current += timedelta(days=32)
        current = current.replace(day=1)

    if not all_games:
        return pd.DataFrame(columns=["game_id", "date", "home_team", "away_team",
                                     "home_goals", "away_goals"]), None

    df = pd.DataFrame(all_games).drop_duplicates(subset="game_id")
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce").fillna(0).astype(int)
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce").fillna(0).astype(int)
    return df, None


def fetch_nhl_schedule() -> list[dict]:
    """Fetch today's upcoming NHL games from ESPN.

    Returns a list of dicts with keys: home_team, away_team, date.
    """
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    url = f"{NHL_ESPN_BASE}/scoreboard"
    params = {"dates": today}
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        games = _parse_scoreboard(resp.json(), completed_only=False)
    except Exception:
        return []

    return [
        {"home_team": g["home_team"], "away_team": g["away_team"], "date": g["date"]}
        for g in games
    ]
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_fetch_nhl.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pipeline/fetch_nhl.py tests/test_fetch_nhl.py
git commit -m "feat: add NHL data fetcher via ESPN API"
```

---

## Chunk 2: Curator

### Task 4: Create `pipeline/curator.py`

**Files:**
- Create: `pipeline/curator.py`
- Create: `tests/test_curator.py`

This is the core new component. It reads all sports' `predictions.json`, applies guardrails, de-vigs odds, and returns ≤5 ranked picks.

- [ ] **Step 1: Write failing tests**

Create `tests/test_curator.py`:

```python
"""Tests for pipeline.curator — unified pick selection."""

import json
import pytest
from pathlib import Path
from pipeline.curator import (
    devig_fair_prob,
    best_line_for_outcome,
    curate_picks,
    american_to_decimal,
)


# ── Unit: de-vig ────────────────────────────────────────────────────────────


class TestDevigFairProb:
    def test_even_market(self):
        """50/50 market with no vig should return 0.5 each."""
        fair_h, fair_a = devig_fair_prob(2.0, 2.0)
        assert fair_h == pytest.approx(0.5)
        assert fair_a == pytest.approx(0.5)

    def test_viggy_market(self):
        """Books with vig: raw probs sum > 1, fair probs sum to exactly 1."""
        # -110 / -110 (typical spread market)
        dec = 100 / 110 + 1  # ≈ 1.909
        fair_h, fair_a = devig_fair_prob(dec, dec)
        assert fair_h + fair_a == pytest.approx(1.0)
        assert fair_h == pytest.approx(0.5)

    def test_favorite_market(self):
        """Favorite has higher raw implied prob → higher fair prob after de-vig."""
        fair_h, fair_a = devig_fair_prob(1.50, 2.75)
        assert fair_h > fair_a
        assert fair_h + fair_a == pytest.approx(1.0)


# ── Unit: best_line_for_outcome ─────────────────────────────────────────────


class TestBestLineForOutcome:
    def _book_odds(self):
        return {
            "draftkings": {"home_odds": 2.10, "away_odds": 1.80},
            "fanduel":    {"home_odds": 2.05, "away_odds": 1.85},
        }

    def test_returns_none_when_too_few_books(self):
        book_odds = {"draftkings": {"home_odds": 2.10, "away_odds": 1.80}}
        result = best_line_for_outcome(book_odds, "home", model_prob=0.60)
        assert result is None  # only 1 book, below MIN_BOOKS=2

    def test_edge_computed_correctly(self):
        book_odds = self._book_odds()
        result = best_line_for_outcome(book_odds, "home", model_prob=0.60)
        assert result is not None
        # Both books: fair home prob ~ 1/2.10 normalized and 1/2.05 normalized
        # Exact value isn't critical; edge = model_prob - avg_fair_prob
        assert "edge" in result
        assert result["edge"] == pytest.approx(result["model_prob"] - result["avg_fair_prob"])

    def test_best_american_is_highest_odds(self):
        """Best line = highest American odds (most favorable to bettor)."""
        book_odds = self._book_odds()
        # DK home: 2.10 → +110, FD home: 2.05 → +105 — DK is better
        result = best_line_for_outcome(book_odds, "home", model_prob=0.60)
        assert result["best_american"] == 110
        assert result["best_book"] == "DK"

    def test_away_outcome(self):
        book_odds = self._book_odds()
        result = best_line_for_outcome(book_odds, "away", model_prob=0.40)
        assert result is not None
        # FD away: 1.85 → -118, DK away: 1.80 → -125 — FD is better (+higher American)
        assert result["best_book"] == "FD"


# ── Unit: _passes_guardrails ────────────────────────────────────────────────


class TestPassesGuardrails:
    """_passes_guardrails edge cases for the odds window and edge threshold."""

    def _line(self, edge=0.10, american=150):
        return {"edge": edge, "best_american": american}

    def test_exactly_min_edge_passes(self):
        from pipeline.curator import _passes_guardrails
        assert _passes_guardrails(self._line(edge=0.08, american=150)) is True

    def test_below_min_edge_fails(self):
        from pipeline.curator import _passes_guardrails
        assert _passes_guardrails(self._line(edge=0.079, american=150)) is False

    def test_exactly_min_odds_passes(self):
        from pipeline.curator import _passes_guardrails
        assert _passes_guardrails(self._line(edge=0.10, american=-130)) is True

    def test_below_min_odds_fails(self):
        from pipeline.curator import _passes_guardrails
        assert _passes_guardrails(self._line(edge=0.10, american=-131)) is False

    def test_exactly_max_odds_passes(self):
        from pipeline.curator import _passes_guardrails
        assert _passes_guardrails(self._line(edge=0.10, american=350)) is True

    def test_above_max_odds_fails(self):
        from pipeline.curator import _passes_guardrails
        assert _passes_guardrails(self._line(edge=0.10, american=351)) is False


# ── Integration: curate_picks ───────────────────────────────────────────────


def _write_predictions(tmp_path: Path, sport: str, matches: list[dict]):
    sport_dir = tmp_path / sport
    sport_dir.mkdir(parents=True)
    (sport_dir / "predictions.json").write_text(
        json.dumps({"sport": sport, "matches": matches})
    )


def _make_match(home, away, home_prob, away_prob, book_odds=None, date="2026-03-15"):
    return {
        "home_team": home,
        "away_team": away,
        "date": date,
        "model_probs": {"home": home_prob, "away": away_prob},
        "book_odds": book_odds or {
            "draftkings": {"home_odds": 2.10, "away_odds": 1.80},
            "fanduel":    {"home_odds": 2.05, "away_odds": 1.85},
        },
    }


class TestCuratePicks:
    def test_returns_list(self, tmp_path):
        _write_predictions(tmp_path, "nba", [_make_match("Lakers", "Celtics", 0.70, 0.30)])
        picks = curate_picks(str(tmp_path))
        assert isinstance(picks, list)

    def test_high_edge_pick_included(self, tmp_path):
        """A pick with clearly positive edge clears the guardrails."""
        # model_prob=0.70, DK home 2.10 → fair ≈ 0.488 → edge ≈ 21% ✓
        match = _make_match("Lakers", "Celtics", 0.70, 0.30)
        _write_predictions(tmp_path, "nba", [match])
        picks = curate_picks(str(tmp_path))
        assert len(picks) == 1
        assert picks[0]["home_team"] == "Lakers"

    def test_low_edge_pick_excluded(self, tmp_path):
        """A pick with < 8% edge is filtered out."""
        # model_prob=0.52 → fair ≈ 0.488 → edge ≈ 3% ✗
        match = _make_match("Lakers", "Celtics", 0.52, 0.48)
        _write_predictions(tmp_path, "nba", [match])
        picks = curate_picks(str(tmp_path))
        assert len(picks) == 0

    def test_heavy_favorite_excluded(self, tmp_path):
        """A pick with American odds below -130 is excluded."""
        # home at 1.40 decimal → -250 American — below -130 floor
        match = _make_match(
            "Lakers", "Celtics", 0.85, 0.15,
            book_odds={
                "draftkings": {"home_odds": 1.40, "away_odds": 2.90},
                "fanduel":    {"home_odds": 1.38, "away_odds": 3.00},
            }
        )
        _write_predictions(tmp_path, "nba", [match])
        picks = curate_picks(str(tmp_path))
        assert len(picks) == 0

    def test_max_5_picks(self, tmp_path):
        """Returns at most 5 picks even when more qualify."""
        matches = [
            _make_match(f"Team{i}", f"Opp{i}", 0.75, 0.25)
            for i in range(10)
        ]
        _write_predictions(tmp_path, "nba", matches)
        picks = curate_picks(str(tmp_path))
        assert len(picks) <= 5

    def test_ranked_by_edge_descending(self, tmp_path):
        """Picks are returned highest edge first."""
        matches = [
            _make_match("Team1", "Opp1", 0.72, 0.28),  # moderate edge
            _make_match("Team2", "Opp2", 0.85, 0.15),  # high edge
        ]
        _write_predictions(tmp_path, "nba", matches)
        picks = curate_picks(str(tmp_path))
        if len(picks) >= 2:
            assert picks[0]["edge"] >= picks[1]["edge"]

    def test_one_pick_per_game(self, tmp_path):
        # Rule: per game, only the outcome with the highest edge is selected.
        # If both home and away pass all filters, only the higher-edge outcome appears.
        """Only one outcome per game is picked."""
        # Both home and away have high edge — only the better one should appear
        match = {
            "home_team": "Lakers",
            "away_team": "Celtics",
            "date": "2026-03-15",
            "model_probs": {"home": 0.70, "away": 0.70},  # pathological
            "book_odds": {
                "draftkings": {"home_odds": 2.10, "away_odds": 2.10},
                "fanduel":    {"home_odds": 2.05, "away_odds": 2.05},
            },
        }
        _write_predictions(tmp_path, "nba", [match])
        picks = curate_picks(str(tmp_path))
        games = [(p["home_team"], p["away_team"]) for p in picks]
        assert len(games) == len(set(games))

    def test_fewer_books_than_min_excluded(self, tmp_path):
        """Games with < 2 books providing lines are skipped."""
        match = _make_match(
            "Lakers", "Celtics", 0.70, 0.30,
            book_odds={"draftkings": {"home_odds": 2.10, "away_odds": 1.80}}  # 1 book only
        )
        _write_predictions(tmp_path, "nba", [match])
        picks = curate_picks(str(tmp_path))
        assert len(picks) == 0

    def test_multi_sport_combined(self, tmp_path):
        """Picks from different sports are combined and ranked globally."""
        _write_predictions(tmp_path, "nba", [_make_match("Lakers", "Celtics", 0.75, 0.25)])
        _write_predictions(tmp_path, "nhl", [_make_match("Bruins", "Rangers", 0.72, 0.28)])
        picks = curate_picks(str(tmp_path))
        sports = {p["sport"] for p in picks}
        assert len(sports) == 2

    def test_pick_has_required_fields(self, tmp_path):
        match = _make_match("Lakers", "Celtics", 0.70, 0.30)
        _write_predictions(tmp_path, "nba", [match])
        picks = curate_picks(str(tmp_path))
        assert len(picks) == 1
        pick = picks[0]
        for field in ("sport", "home_team", "away_team", "date", "pick",
                      "model_prob", "edge", "best_american", "best_book"):
            assert field in pick, f"Missing field: {field}"
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
pytest tests/test_curator.py -v
```
Expected: ModuleNotFoundError.

- [ ] **Step 3: Create `pipeline/curator.py`**

```python
"""Unified pick curation for SLOP LOCKS.

Reads raw candidate matches from all sports' predictions.json files,
applies guardrails, performs de-vigged line shopping across configured
bookmakers, ranks by edge, and returns at most CURATOR_MAX_PICKS picks.
"""
from __future__ import annotations

import json
from pathlib import Path

from pipeline.config import (
    BOOK_ABBREV,
    BOOKMAKERS,
    CURATOR_MAX_ODDS,
    CURATOR_MAX_PICKS,
    CURATOR_MIN_BOOKS,
    CURATOR_MIN_EDGE,
    CURATOR_MIN_ODDS,
    SPORTS,
)
from pipeline.ensemble import decimal_to_american


def american_to_decimal(american: int) -> float:
    """Convert American odds to decimal format."""
    if american >= 0:
        return american / 100 + 1
    return 100 / abs(american) + 1


def devig_fair_prob(home_dec: float, away_dec: float) -> tuple[float, float]:
    """Return de-vigged fair probabilities for a two-way market.

    Uses multiplicative normalization: divide each raw implied probability
    by the sum of both, removing the bookmaker's margin.
    """
    raw_h = 1.0 / home_dec
    raw_a = 1.0 / away_dec
    total = raw_h + raw_a
    return raw_h / total, raw_a / total


def best_line_for_outcome(
    book_odds: dict[str, dict],
    outcome: str,
    model_prob: float,
) -> dict | None:
    """Compute de-vigged edge and best available line for one outcome.

    Parameters
    ----------
    book_odds : dict
        Mapping of book_key → {home_odds: float, away_odds: float} (decimal).
    outcome : str
        "home" or "away".
    model_prob : float
        Ensemble's predicted probability for this outcome.

    Returns None if fewer than CURATOR_MIN_BOOKS have lines.
    """
    fair_probs: list[float] = []
    best_american: int | None = None
    best_book: str | None = None

    for book_key, odds in book_odds.items():
        h = odds.get("home_odds", 0.0)
        a = odds.get("away_odds", 0.0)
        if h <= 1.0 or a <= 1.0:
            continue

        fair_h, fair_a = devig_fair_prob(h, a)
        fair_prob = fair_h if outcome == "home" else fair_a
        fair_probs.append(fair_prob)

        dec = h if outcome == "home" else a
        american = decimal_to_american(dec)
        if best_american is None or american > best_american:
            best_american = american
            best_book = BOOK_ABBREV.get(book_key, book_key)

    if len(fair_probs) < CURATOR_MIN_BOOKS:
        return None

    avg_fair_prob = sum(fair_probs) / len(fair_probs)
    edge = model_prob - avg_fair_prob

    return {
        "model_prob": model_prob,
        "avg_fair_prob": round(avg_fair_prob, 4),
        "edge": round(edge, 4),
        "best_american": best_american,
        "best_book": best_book,
        "num_books": len(fair_probs),
    }


def _passes_guardrails(line: dict) -> bool:
    """Return True if a candidate line passes all hard filters."""
    if line["edge"] < CURATOR_MIN_EDGE:
        return False
    if not (CURATOR_MIN_ODDS <= line["best_american"] <= CURATOR_MAX_ODDS):
        return False
    return True


def curate_picks(data_dir: str) -> list[dict]:
    """Read all sports' predictions.json and return ≤CURATOR_MAX_PICKS picks.

    Applies guardrails, de-vigs per-book odds, ranks by edge descending.
    Returns at most one pick per game.

    Parameters
    ----------
    data_dir : str
        Root data directory containing per-sport subdirectories.
    """
    root = Path(data_dir)
    candidates: list[dict] = []

    for sport_key, sport_cfg in SPORTS.items():
        pred_path = root / sport_key / "predictions.json"
        if not pred_path.exists():
            continue

        try:
            data = json.loads(pred_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        for match in data.get("matches", []):
            model_probs = match.get("model_probs", {})
            book_odds = match.get("book_odds", {})
            outcomes = sport_cfg["outcomes"]

            # One pick per game: collect all qualifying outcomes, keep the best
            game_candidates: list[dict] = []
            for outcome in outcomes:
                mp = model_probs.get(outcome)
                if mp is None:
                    continue

                line = best_line_for_outcome(book_odds, outcome, mp)
                if line is None:
                    continue
                if not _passes_guardrails(line):
                    continue

                game_candidates.append({
                    "sport": sport_key,
                    "home_team": match["home_team"],
                    "away_team": match["away_team"],
                    "date": match.get("date", ""),
                    "pick": outcome,
                    **line,
                })

            if not game_candidates:
                continue
            # Best outcome for this game = highest edge
            best = max(game_candidates, key=lambda x: x["edge"])
            candidates.append(best)

    # Global rank by edge descending, return top N
    candidates.sort(key=lambda x: x["edge"], reverse=True)
    return candidates[:CURATOR_MAX_PICKS]
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_curator.py -v
```
Expected: all pass.

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add pipeline/curator.py tests/test_curator.py
git commit -m "feat: add curator — unified pick selection with guardrails and line shopping"
```

---

## Chunk 3: Pipeline Wiring — run.py, notify_discord.py, refresh_picks.py, cleanup

### Task 5: Update `pipeline/run.py`

**Files:**
- Modify: `pipeline/run.py`
- Modify: `tests/test_run.py`

Remove: `_compute_slop_locks`, `_compute_longslop`, `_generate_blurbs`, pick history evaluation loop.
Add: NHL sport branch, `book_odds` in prediction records.
Change: predictions.json schema drops `slop_locks`/`longslop` keys.

- [ ] **Step 1: Update the imports at the top of run.py**

Remove:
```python
from pipeline.config import (
    ANTHROPIC_API_KEY,
    DATA_DIR,
    NBA_B2B_PENALTY,
    SLOP_LOCK_MIN_ODDS,
    SLOP_LOCK_MAX_ODDS,
    SLOP_LOCK_FALLBACK_MIN_ODDS,
    SPORTS,
)
```

Replace with:
```python
from pipeline.config import (
    DATA_DIR,
    NBA_B2B_PENALTY,
    SPORTS,
)
```

Also add the NHL import:
```python
from pipeline.fetch_nhl import fetch_nhl_games, fetch_nhl_schedule, normalize_nhl_team_name
```

Remove from backtest imports:
```python
    compute_roi,
    evaluate_prediction,
    update_accuracy_log,
```

- [ ] **Step 2: Remove dead functions from run.py**

Delete these functions entirely:
- `_generate_blurbs()`
- `_compute_slop_locks()`
- `_compute_longslop()`
- `_compute_pick_stats()`

- [ ] **Step 3: Add NHL branch in `run_sport_pipeline()`**

In the `# 1. Fetch data` section, add after the `elif sport_key == "ncaam":` block:

```python
    elif sport_key == "nhl":
        games_df, box_scores_df = fetch_nhl_games()
        fixtures = fetch_nhl_schedule()
        matches = games_df
```

And update the normalizer section:

```python
    if sport_key == "nba":
        normalizer = normalize_nba_team_name
    elif sport_key == "ncaam":
        normalizer = normalize_ncaam_team_name
    elif sport_key == "nhl":
        normalizer = normalize_nhl_team_name
    else:
        normalizer = lambda x: x
```

Remove the bare `else: raise ValueError(...)` line.

- [ ] **Step 4: Add `book_odds` to prediction records and remove slop_locks from output**

> **book_odds schema:** `{book_key: {"home_odds": float, "away_odds": float}}` — decimal odds, only books from `BOOKMAKERS` that returned a line. Example: `{"draftkings": {"home_odds": 2.10, "away_odds": 1.80}, "fanduel": {"home_odds": 2.05, "away_odds": 1.85}}`. This is the same structure stored by `fetch_data.py` and consumed by `curator.py`.

In the fixture loop, update the `record` dict construction to include `book_odds`:

```python
        record = {
            "home_team": home,
            "away_team": away,
            "date": fix["date"],
            "matchday": fix.get("matchday"),
            "model_probs": {k: round(v, 4) for k, v in blended.items()},
            "individual_models": {
                name: {k: round(v, 4) for k, v in probs.items()}
                for name, probs in individual_models.items()
            },
            "edges": edges,
            "best_odds": best_odds,
            "book_odds": match_odds.get("book_odds", {}) if match_odds else {},
        }
```

In the output section (step 5b), remove the `slop_locks` and `longslop` computation and blurb calls. Replace with nothing — the curator handles selection.

Update the `_save_json` call for predictions to use the new schema:

```python
    _save_json(predictions_path, {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sport": sport_key,
        "sport_name": sport["name"],
        "outcomes": outcomes,
        "matches": prediction_records,
    })
```

- [ ] **Step 5: Remove the pick history evaluation loop (steps 6 and 6b)**

Delete the entire sections:
- `# 6. Evaluate past predictions` block
- `# 6b. Track and evaluate picks` block
- All `_save_json` calls for `pick_history_path` and `accuracy_path`

Keep the `accuracy_log` load (step 3) and model weights computation — those are still used.

- [ ] **Step 6: Update `run_pipeline()` to include NHL**

```python
def run_pipeline():
    """Run the full SLOP LOCKS pipeline for all configured sports."""
    for sport_key in SPORTS:
        try:
            run_sport_pipeline(sport_key)
        except Exception as exc:
            print(f"[{sport_key}] pipeline error: {exc}")
```

- [ ] **Step 7: Run tests, fix any failures**

```bash
pytest tests/test_run.py -v
```

The main things to fix in `test_run.py`:
- Remove any assertions on `slop_locks` or `longslop` keys in predictions output
- Remove mock patches for `_generate_blurbs`
- Assert predictions output has `matches` key with a list

- [ ] **Step 8: Run full suite**

```bash
pytest tests/ -v
```

- [ ] **Step 9: Commit**

```bash
git add pipeline/run.py pipeline/fetch_nhl.py tests/test_run.py
git commit -m "feat: add NHL to pipeline; remove slop_lock selection from run.py; add book_odds to match records"
```

---

### Task 6: Rewrite `pipeline/notify_discord.py`

**Files:**
- Modify: `pipeline/notify_discord.py`
- Create: `tests/test_notify_discord.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_notify_discord.py`:

```python
"""Tests for pipeline.notify_discord."""

import pytest
from unittest.mock import patch, MagicMock
from pipeline.notify_discord import build_payload, _fmt_pick_line, _fmt_odds


def _make_pick(sport="nba", home="Lakers", away="Celtics", pick="home",
               model_prob=0.70, edge=0.12, best_american=115, best_book="DK",
               date="2026-03-15"):
    return {
        "sport": sport,
        "home_team": home,
        "away_team": away,
        "date": date,
        "pick": pick,
        "model_prob": model_prob,
        "edge": edge,
        "best_american": best_american,
        "best_book": best_book,
    }


class TestFmtOdds:
    def test_positive_odds(self):
        assert _fmt_odds(115) == "+115"

    def test_negative_odds(self):
        assert _fmt_odds(-130) == "-130"

    def test_even_odds(self):
        assert _fmt_odds(100) == "+100"


class TestFmtPickLine:
    def test_home_pick_shows_home_team(self):
        pick = _make_pick(home="Lakers", away="Celtics", pick="home")
        line = _fmt_pick_line(pick)
        assert "LAKERS" in line

    def test_away_pick_shows_away_team(self):
        pick = _make_pick(home="Lakers", away="Celtics", pick="away")
        line = _fmt_pick_line(pick)
        assert "CELTICS" in line

    def test_contains_odds_and_book(self):
        pick = _make_pick(best_american=115, best_book="DK")
        line = _fmt_pick_line(pick)
        assert "+115" in line
        assert "DK" in line

    def test_contains_model_prob(self):
        pick = _make_pick(model_prob=0.70)
        line = _fmt_pick_line(pick)
        assert "70%" in line

    def test_contains_edge(self):
        pick = _make_pick(edge=0.12)
        line = _fmt_pick_line(pick)
        assert "12.0%" in line

    def test_contains_sport_emoji_nba(self):
        pick = _make_pick(sport="nba")
        line = _fmt_pick_line(pick)
        assert "🏀" in line

    def test_contains_sport_emoji_nhl(self):
        pick = _make_pick(sport="nhl")
        line = _fmt_pick_line(pick)
        assert "🏒" in line


class TestBuildPayload:
    def test_no_picks_returns_no_locks_message(self):
        payload = build_payload([])
        assert "no locks" in payload["content"].lower() or \
               any("no locks" in str(e).lower() for e in payload.get("embeds", []))

    def test_picks_appear_in_content(self):
        picks = [_make_pick()]
        payload = build_payload(picks)
        # Content or embed fields should reference the team
        full_text = str(payload)
        assert "LAKERS" in full_text

    def test_pick_count_in_footer(self):
        picks = [_make_pick(), _make_pick(home="Warriors", away="Heat", sport="nhl")]
        payload = build_payload(picks)
        full_text = str(payload)
        assert "2" in full_text

    def test_payload_has_username(self):
        payload = build_payload([_make_pick()])
        assert payload.get("username") == "BIG SLIME"
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
pytest tests/test_notify_discord.py -v
```

- [ ] **Step 3: Rewrite `pipeline/notify_discord.py`**

```python
"""Post daily SLOP LOCKS to a Discord webhook.

Reads curated picks from pipeline.curator and sends one Discord message.

Usage:
    python -m pipeline.notify_discord
"""

import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

from pipeline.config import DATA_DIR, SPORT_EMOJI
from pipeline.curator import curate_picks

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
COLOR_SLIME = 0x39FF14


def _fmt_odds(american: int) -> str:
    return f"+{american}" if american >= 0 else str(american)


def _pick_team(pick: dict) -> str:
    if pick["pick"] == "home":
        return pick["home_team"].upper()
    return pick["away_team"].upper()


def _fmt_pick_line(pick: dict) -> str:
    emoji = SPORT_EMOJI.get(pick["sport"], "🎯")
    team = _pick_team(pick)
    odds_str = _fmt_odds(pick["best_american"])
    conf = f"{pick['model_prob'] * 100:.0f}%"
    edge = f"{pick['edge'] * 100:.1f}%"
    return f"{emoji} {team} · {odds_str} ({pick['best_book']}) · {conf} model · +{edge} edge"


def build_payload(picks: list[dict]) -> dict:
    et_now = datetime.now(ZoneInfo("America/New_York"))
    date_str = et_now.strftime("%b %d")

    if not picks:
        return {
            "username": "BIG SLIME",
            "content": f"🔒 **SLOP LOCKS · {date_str}**\nNo locks today — nothing cleared the bar.",
            "embeds": [],
        }

    lines = [_fmt_pick_line(p) for p in picks]
    count_line = f"\n{len(picks)} lock{'s' if len(picks) != 1 else ''} today"
    body = "\n".join(lines) + count_line

    return {
        "username": "BIG SLIME",
        "content": f"🔒 **SLOP LOCKS · {date_str}**",
        "embeds": [
            {
                "description": body,
                "color": COLOR_SLIME,
            }
        ],
    }


def main() -> None:
    # Assumes predictions.json is already fresh — always called after run.py in the workflow.
    if not DISCORD_WEBHOOK_URL:
        print("DISCORD_WEBHOOK_URL not set — skipping Discord notification")
        sys.exit(0)

    picks = curate_picks(DATA_DIR)
    payload = build_payload(picks)

    resp = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=15)
    if resp.status_code == 204:
        print(f"Discord notification sent — {len(picks)} pick(s)")
    else:
        print(f"Discord webhook error: {resp.status_code} — {resp.text}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_notify_discord.py -v
```

- [ ] **Step 5: Run full suite**

```bash
pytest tests/ -v
```

- [ ] **Step 6: Commit**

```bash
git add pipeline/notify_discord.py tests/test_notify_discord.py
git commit -m "feat: rewrite notify_discord — clean pick card format with sport emoji and book"
```

---

### Task 7: Update `pipeline/refresh_picks.py`

**Files:**
- Modify: `pipeline/refresh_picks.py`

The refresh workflow re-fetches odds without retraining models. It should now: fetch fresh odds for each sport, patch `book_odds` into each sport's `predictions.json`, run the curator, and post to Discord.

- [ ] **Step 1: Rewrite `pipeline/refresh_picks.py`**

```python
"""Quick odds refresh and Discord re-post without model retraining.

Fetches fresh h2h odds from The Odds API, patches book_odds into the
existing predictions.json for each sport, then runs the curator and
posts a fresh Discord card.

Usage:
    python -m pipeline.refresh_picks            # all sports
    python -m pipeline.refresh_picks nba nhl    # specific sports
"""

import json
import sys
from pathlib import Path

from pipeline.config import DATA_DIR, SPORTS
from pipeline.curator import curate_picks
from pipeline.fetch_data import fetch_odds
from pipeline.notify_discord import build_payload, DISCORD_WEBHOOK_URL

import requests


def _normalizer_for(sport_key: str):
    if sport_key == "nba":
        from pipeline.fetch_nba import normalize_nba_team_name
        return normalize_nba_team_name
    if sport_key == "ncaam":
        from pipeline.fetch_ncaam import normalize_ncaam_team_name
        return normalize_ncaam_team_name
    if sport_key == "nhl":
        from pipeline.fetch_nhl import normalize_nhl_team_name
        return normalize_nhl_team_name
    return lambda x: x


def refresh_sport(sport_key: str) -> None:
    """Fetch fresh odds and patch book_odds into predictions.json for one sport."""
    sport = SPORTS[sport_key]
    pred_path = Path(DATA_DIR) / sport_key / "predictions.json"

    if not pred_path.exists():
        print(f"[{sport_key}] no predictions.json — skipping")
        return

    data = json.loads(pred_path.read_text())
    matches = data.get("matches", [])
    if not matches:
        return

    try:
        odds_list = fetch_odds(sport["odds_sport"])
    except Exception as exc:
        print(f"[{sport_key}] odds fetch failed: {exc}")
        return

    normalizer = _normalizer_for(sport_key)
    for o in odds_list:
        o["home_team"] = normalizer(o["home_team"])
        o["away_team"] = normalizer(o["away_team"])

    odds_lookup = {(o["home_team"], o["away_team"]): o for o in odds_list}

    for match in matches:
        key = (match["home_team"], match["away_team"])
        if key in odds_lookup:
            match["book_odds"] = odds_lookup[key].get("book_odds", {})

    pred_path.write_text(json.dumps(data, indent=2))
    print(f"[{sport_key}] patched {len(matches)} match(es) with fresh odds")


def main(sports: list[str] | None = None) -> None:
    targets = sports or list(SPORTS.keys())
    for sport_key in targets:
        if sport_key not in SPORTS:
            print(f"Unknown sport: {sport_key}")
            continue
        refresh_sport(sport_key)

    picks = curate_picks(DATA_DIR)
    payload = build_payload(picks)

    if not DISCORD_WEBHOOK_URL:
        print("DISCORD_WEBHOOK_URL not set — skipping Discord notification")
        print(f"Would post {len(picks)} pick(s)")
        return

    resp = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=15)
    if resp.status_code == 204:
        print(f"Discord refresh sent — {len(picks)} pick(s)")
    else:
        print(f"Discord webhook error: {resp.status_code} — {resp.text}")
        sys.exit(1)


if __name__ == "__main__":
    sports_arg = sys.argv[1:] or None
    main(sports_arg)
```

- [ ] **Step 2: Run full test suite**

```bash
pytest tests/ -v
```

- [ ] **Step 3: Commit**

```bash
git add pipeline/refresh_picks.py
git commit -m "feat: update refresh_picks to patch book_odds and invoke curator"
```

---

### Task 8: Delete website files and final cleanup

**Files:**
- Delete: `index.html`, `sw.js`, `manifest.json`, `netlify.toml`, `icons/`
- Delete: `data/sotd.json`

- [ ] **Step 1: Delete website artifacts**

```bash
git rm index.html sw.js manifest.json netlify.toml
git rm -r icons/
git rm -f data/sotd.json
```

- [ ] **Step 2: Run full test suite one final time**

```bash
pytest tests/ -v
```
Expected: all pass.

- [ ] **Step 3: Final commit**

```bash
git commit -m "chore: remove website files (index.html, sw.js, manifest, netlify, icons)"
```

- [ ] **Step 4: Push**

```bash
git push origin master
```

- [ ] **Step 5: Trigger a manual workflow run to verify end-to-end**

In the GitHub Actions UI, trigger `Daily Predictions` via `workflow_dispatch`. Confirm:
- Pipeline runs without errors
- All 3 sports (NBA, NCAAM, NHL) produce `predictions.json` with `matches` and `book_odds`
- Discord card posts correctly
