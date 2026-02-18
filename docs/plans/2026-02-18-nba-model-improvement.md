# NBA Model Improvement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the single-model NBA Elo pipeline with a 3-model ensemble (Elo + AdjustedEfficiency + FourFactors) using ESPN box scores, a recalibrated home advantage, and a back-to-back rest adjustment.

**Architecture:** Switch NBA data fetching from balldontlie.io to the ESPN API (same infrastructure as NCAAM). Reuse existing `AdjustedEfficiency` and `FourFactorsModel` classes from `models.py` — no new model code needed. Add rest adjustment as an ephemeral Elo rating modifier applied per-prediction.

**Tech Stack:** Python, ESPN public API (no key), existing `pipeline/models.py` classes, pandas, pytest

---

### Task 1: Config + `elo_predict` rest adjustment parameter

**Files:**
- Modify: `pipeline/config.py`
- Modify: `pipeline/models.py` (lines ~409–455, `elo_predict` function)
- Test: `tests/test_models.py`

**Step 1: Write the failing test**

Add to `tests/test_models.py` inside `TestElo`:

```python
def test_rest_adjustment_reduces_home_win_prob(self):
    """B2B penalty on home team should lower their win probability."""
    teams = ["A", "B"]
    elo = EloRatings(teams, k_factor=20, home_advantage=65)

    prob_fresh = elo_predict(elo, "A", "B", outcomes=["home", "away"])
    prob_b2b   = elo_predict(elo, "A", "B", outcomes=["home", "away"],
                             home_rest_adj=-30.0)

    assert prob_b2b["home"] < prob_fresh["home"]

def test_rest_adjustment_zero_is_unchanged(self):
    prob_default = elo_predict(elo_ratings_fixture, "A", "B", outcomes=["home", "away"])
    prob_zero    = elo_predict(elo_ratings_fixture, "A", "B", outcomes=["home", "away"],
                               home_rest_adj=0.0, away_rest_adj=0.0)
    assert prob_default == prob_zero
```

(Note: `elo_ratings_fixture` — add a module-level fixture to `TestElo`:
```python
@pytest.fixture
def elo_ratings_fixture(self):
    elo = EloRatings(["A", "B"], k_factor=20, home_advantage=65)
    return elo
```
)

**Step 2: Run to verify it fails**

```bash
pytest tests/test_models.py::TestElo::test_rest_adjustment_reduces_home_win_prob -v
```

Expected: `TypeError: elo_predict() got an unexpected keyword argument 'home_rest_adj'`

**Step 3: Update `elo_predict` in `pipeline/models.py`**

Change the signature and first two rating lines:

```python
def elo_predict(elo, home_team, away_team, outcomes=None,
                home_rest_adj=0.0, away_rest_adj=0.0):
    if outcomes is None:
        outcomes = ["home", "draw", "away"]

    r_home = elo.get_rating(home_team) + elo.home_advantage + home_rest_adj
    r_away = elo.get_rating(away_team) + away_rest_adj

    diff = r_home - r_away
    # ... rest unchanged
```

**Step 4: Update `pipeline/config.py`**

```python
# NBA model parameters
NBA_B2B_PENALTY = 30  # Elo points subtracted for back-to-back game

# ESPN base URLs
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
NBA_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
```

In the `SPORTS` dict, update the `"nba"` entry:

```python
"nba": {
    "name": "NBA",
    "display_name": "NBA",
    "odds_sport": "basketball_nba",
    "outcomes": ["home", "away"],
    "models": ["elo", "efficiency", "four_factors"],   # was: ["elo"]
    "elo_k_factor": 20,
    "elo_home_advantage": 65,                          # was: 100
    "efficiency_home_bonus": 3.5,
    "data_dir": os.path.join(DATA_DIR, "nba"),
},
```

**Step 5: Run tests**

```bash
pytest tests/test_models.py -v
```

Expected: all pass.

**Step 6: Commit**

```bash
git add pipeline/config.py pipeline/models.py tests/test_models.py
git commit -m "feat(nba): recalibrate home advantage to 65, add rest adj param to elo_predict"
```

---

### Task 2: NBA ESPN data fetcher

**Files:**
- Modify: `pipeline/fetch_nba.py`
- Test: `tests/test_fetch_nba.py`

**Step 1: Write the failing tests**

Add to `tests/test_fetch_nba.py`:

```python
from pipeline.fetch_nba import (
    normalize_nba_team_name,
    fetch_nba_games,
    fetch_nba_schedule,
    fetch_nba_espn_games,
    fetch_nba_espn_schedule,
)

# ---- helpers ----------------------------------------------------------------

NBA_SAMPLE_TOTALS = [
    "", "110", "44-95", "17-42", "5-7",
    "49", "25", "10", "9", "6", "15", "34", "17", "",
]

def _make_nba_espn_event(event_id, home_name, away_name,
                         home_score, away_score,
                         completed=True, date="2026-01-15T00:00Z"):
    return {
        "id": event_id,
        "date": date,
        "competitions": [{
            "competitors": [
                {
                    "homeAway": "home",
                    "team": {"displayName": home_name},
                    "score": str(home_score),
                },
                {
                    "homeAway": "away",
                    "team": {"displayName": away_name},
                    "score": str(away_score),
                },
            ],
            "status": {"type": {"completed": completed}},
        }],
    }

def _make_nba_summary(home_name, away_name, home_totals, away_totals):
    return {
        "boxscore": {
            "players": [
                {
                    "team": {"displayName": home_name},
                    "statistics": [{"totals": home_totals}],
                },
                {
                    "team": {"displayName": away_name},
                    "statistics": [{"totals": away_totals}],
                },
            ]
        }
    }


# ---- fetch_nba_espn_games ---------------------------------------------------

class TestFetchNbaEspnGames:
    @patch("pipeline.fetch_nba.requests.get")
    def test_returns_games_and_box_scores(self, mock_get):
        scoreboard_resp = MagicMock()
        scoreboard_resp.raise_for_status = MagicMock()
        scoreboard_resp.json.return_value = {
            "events": [
                _make_nba_espn_event("1", "Los Angeles Lakers", "Boston Celtics",
                                     112, 108),
            ]
        }
        summary_resp = MagicMock()
        summary_resp.raise_for_status = MagicMock()
        summary_resp.json.return_value = _make_nba_summary(
            "Los Angeles Lakers", "Boston Celtics",
            NBA_SAMPLE_TOTALS, NBA_SAMPLE_TOTALS,
        )
        mock_get.side_effect = [scoreboard_resp, summary_resp]

        games_df, box_df = fetch_nba_espn_games(season=2025, dates=["2026-01-15"])

        assert len(games_df) == 1
        assert list(games_df.columns) == [
            "game_id", "date", "home_team", "away_team", "home_goals", "away_goals"
        ]
        assert games_df.iloc[0]["home_team"] == "Lakers"
        assert games_df.iloc[0]["away_team"] == "Celtics"
        assert games_df.iloc[0]["home_goals"] == 112
        assert "game_id" in games_df.columns
        assert len(box_df) == 2
        assert box_df.iloc[0]["pts"] == 110

    @patch("pipeline.fetch_nba.requests.get")
    def test_skips_non_completed_games(self, mock_get):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "events": [
                _make_nba_espn_event("1", "Lakers", "Celtics", 0, 0, completed=False),
            ]
        }
        mock_get.return_value = resp

        games_df, box_df = fetch_nba_espn_games(season=2025, dates=["2026-01-15"])
        assert len(games_df) == 0
        assert len(box_df) == 0

    @patch("pipeline.fetch_nba.requests.get")
    def test_handles_missing_box_score(self, mock_get):
        scoreboard_resp = MagicMock()
        scoreboard_resp.raise_for_status = MagicMock()
        scoreboard_resp.json.return_value = {
            "events": [
                _make_nba_espn_event("1", "Los Angeles Lakers", "Boston Celtics", 112, 108),
            ]
        }
        summary_resp = MagicMock()
        summary_resp.raise_for_status.side_effect = requests.RequestException("timeout")
        mock_get.side_effect = [scoreboard_resp, summary_resp]

        games_df, box_df = fetch_nba_espn_games(season=2025, dates=["2026-01-15"])
        assert len(games_df) == 1   # game still recorded
        assert len(box_df) == 0     # box score silently skipped


# ---- fetch_nba_espn_schedule ------------------------------------------------

class TestFetchNbaEspnSchedule:
    @patch("pipeline.fetch_nba.requests.get")
    def test_returns_upcoming_games(self, mock_get):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "events": [
                _make_nba_espn_event("1", "Los Angeles Lakers", "Boston Celtics",
                                     0, 0, completed=False, date="2026-02-19T00:00Z"),
            ]
        }
        mock_get.return_value = resp

        fixtures = fetch_nba_espn_schedule()
        assert len(fixtures) >= 1
        assert fixtures[0]["home_team"] == "Lakers"
        assert fixtures[0]["away_team"] == "Celtics"
        assert fixtures[0]["date"] == "2026-02-19"

    @patch("pipeline.fetch_nba.requests.get")
    def test_excludes_completed_games(self, mock_get):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "events": [
                _make_nba_espn_event("1", "Lakers", "Celtics", 112, 108,
                                     completed=True),
                _make_nba_espn_event("2", "Heat", "Bulls", 0, 0,
                                     completed=False, date="2026-02-20T00:00Z"),
            ]
        }
        mock_get.return_value = resp

        fixtures = fetch_nba_espn_schedule()
        assert all(f["home_team"] != "Lakers" for f in fixtures)
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_fetch_nba.py::TestFetchNbaEspnGames -v
pytest tests/test_fetch_nba.py::TestFetchNbaEspnSchedule -v
```

Expected: `ImportError: cannot import name 'fetch_nba_espn_games'`

**Step 3: Implement ESPN fetchers in `pipeline/fetch_nba.py`**

Add imports at the top of `fetch_nba.py`:

```python
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from pipeline.config import BALLDONTLIE_API_KEY, BALLDONTLIE_BASE, NBA_ESPN_BASE
from pipeline.fetch_ncaam import _parse_box_score_totals  # reuse ESPN totals parser

_ESPN_REQUEST_DELAY = 0.5
```

Add these functions after the existing `fetch_nba_schedule()`:

```python
# ---------------------------------------------------------------------------
# NBA season date range
# ---------------------------------------------------------------------------

def _nba_season_date_range(season: int) -> list[str]:
    """Generate YYYY-MM-DD dates for an NBA regular season.

    Season starts Oct 1 of `season`, ends Apr 20 of `season+1` or today.
    """
    start = datetime(season, 10, 1)
    end = min(
        datetime(season + 1, 4, 20),
        datetime.now(timezone.utc).replace(tzinfo=None),
    )
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


# ---------------------------------------------------------------------------
# ESPN event parsing
# ---------------------------------------------------------------------------

def _parse_nba_espn_event(event: dict) -> dict | None:
    """Parse an ESPN NBA scoreboard event, returning None if not final."""
    comp = event["competitions"][0]
    status_type = comp.get("status", {}).get("type", {})
    if not status_type.get("completed", False):
        return None

    home = away = None
    for competitor in comp["competitors"]:
        if competitor["homeAway"] == "home":
            home = competitor
        else:
            away = competitor

    if home is None or away is None:
        return None

    return {
        "event_id": event["id"],
        "date": event["date"][:10],
        "home_name": home["team"]["displayName"],
        "away_name": away["team"]["displayName"],
        "home_score": int(home["score"]),
        "away_score": int(away["score"]),
    }


# ---------------------------------------------------------------------------
# ESPN game + box score fetch
# ---------------------------------------------------------------------------

def fetch_nba_espn_games(
    season: int | None = None,
    dates: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch finished NBA games and box scores via ESPN API.

    Parameters
    ----------
    season : int or None
        Season start year (e.g. 2025 for 2025-26). Defaults to current season.
    dates : list[str] or None
        Explicit YYYY-MM-DD dates. Defaults to full season range.

    Returns
    -------
    (games_df, box_scores_df)
        games_df columns : game_id, date, home_team, away_team, home_goals, away_goals
        box_scores_df columns : game_id, team, date, pts, fgm, fga, fg3m, fg3a,
                                ftm, fta, orb, drb, to, possessions
    """
    if season is None:
        season = _current_nba_season()
    if dates is None:
        dates = _nba_season_date_range(season)

    game_rows = []
    box_rows = []

    for date_str in dates:
        espn_date = date_str.replace("-", "")
        url = f"{NBA_ESPN_BASE}/scoreboard?dates={espn_date}&limit=50&seasontype=2"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        final_events = []
        for event in data.get("events", []):
            parsed = _parse_nba_espn_event(event)
            if parsed is not None:
                final_events.append(parsed)
                game_rows.append({
                    "game_id": parsed["event_id"],
                    "date": parsed["date"],
                    "home_team": normalize_nba_team_name(parsed["home_name"]),
                    "away_team": normalize_nba_team_name(parsed["away_name"]),
                    "home_goals": parsed["home_score"],
                    "away_goals": parsed["away_score"],
                })

        for parsed in final_events:
            time.sleep(_ESPN_REQUEST_DELAY)
            summary_url = f"{NBA_ESPN_BASE}/summary?event={parsed['event_id']}"
            try:
                s_resp = requests.get(summary_url, timeout=30)
                s_resp.raise_for_status()
                s_data = s_resp.json()
                player_groups = s_data.get("boxscore", {}).get("players", [])
                if len(player_groups) < 2:
                    continue
                for player_group in player_groups:
                    try:
                        totals = player_group["statistics"][0]["totals"]
                        stats = _parse_box_score_totals(totals)
                        team_name = player_group["team"]["displayName"]
                        box_rows.append({
                            "game_id": parsed["event_id"],
                            "team": normalize_nba_team_name(team_name),
                            "date": parsed["date"],
                            **stats,
                        })
                    except (KeyError, IndexError):
                        continue
            except requests.RequestException:
                continue

        time.sleep(_ESPN_REQUEST_DELAY)

    games_df = pd.DataFrame(
        game_rows,
        columns=["game_id", "date", "home_team", "away_team", "home_goals", "away_goals"],
    )
    box_cols = [
        "game_id", "team", "date", "pts", "fgm", "fga", "fg3m", "fg3a",
        "ftm", "fta", "orb", "drb", "to", "possessions",
    ]
    box_scores_df = pd.DataFrame(box_rows, columns=box_cols)
    return games_df, box_scores_df


def fetch_nba_espn_schedule() -> list[dict]:
    """Fetch upcoming NBA games (today + next 7 days) via ESPN API."""
    today = datetime.now(timezone.utc).date()
    fixtures = []

    for day_offset in range(8):
        date = today + timedelta(days=day_offset)
        espn_date = date.strftime("%Y%m%d")
        url = f"{NBA_ESPN_BASE}/scoreboard?dates={espn_date}&limit=50"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for event in data.get("events", []):
            comp = event["competitions"][0]
            status_type = comp.get("status", {}).get("type", {})
            if status_type.get("completed", False):
                continue

            home = away = None
            for competitor in comp["competitors"]:
                if competitor["homeAway"] == "home":
                    home = competitor
                else:
                    away = competitor

            if home is None or away is None:
                continue

            fixtures.append({
                "home_team": normalize_nba_team_name(home["team"]["displayName"]),
                "away_team": normalize_nba_team_name(away["team"]["displayName"]),
                "date": event["date"][:10],
            })

        time.sleep(_ESPN_REQUEST_DELAY)

    return fixtures
```

**Step 4: Run tests**

```bash
pytest tests/test_fetch_nba.py -v
```

Expected: all pass (including existing balldontlie tests).

**Step 5: Commit**

```bash
git add pipeline/fetch_nba.py pipeline/config.py tests/test_fetch_nba.py
git commit -m "feat(nba): add ESPN-based game, box score, and schedule fetchers"
```

---

### Task 3: Rest adjustment helper in run.py

**Files:**
- Modify: `pipeline/run.py`
- Test: inline test in `tests/test_run.py` (or add to conftest — check if `test_run.py` exists; if not, create it)

**Step 1: Write the failing test**

Check if `tests/test_run.py` exists. If not, create it. Add:

```python
"""Tests for pipeline.run helpers."""
import pandas as pd
import pytest
from pipeline.run import _days_since_last_game


class TestDaysSinceLastGame:
    def _make_matches(self, rows):
        return pd.DataFrame(rows, columns=["date", "home_team", "away_team",
                                           "home_goals", "away_goals"])

    def test_returns_correct_days_for_known_team(self):
        matches = self._make_matches([
            {"date": "2026-02-10", "home_team": "Lakers", "away_team": "Celtics",
             "home_goals": 110, "away_goals": 105},
        ])
        result = _days_since_last_game("Lakers", "2026-02-11", matches)
        assert result == 1

    def test_returns_none_for_unknown_team(self):
        matches = self._make_matches([
            {"date": "2026-02-10", "home_team": "Lakers", "away_team": "Celtics",
             "home_goals": 110, "away_goals": 105},
        ])
        result = _days_since_last_game("Thunder", "2026-02-11", matches)
        assert result is None

    def test_ignores_future_games(self):
        matches = self._make_matches([
            {"date": "2026-02-10", "home_team": "Lakers", "away_team": "Celtics",
             "home_goals": 110, "away_goals": 105},
            {"date": "2026-02-12", "home_team": "Lakers", "away_team": "Nuggets",
             "home_goals": 100, "away_goals": 98},
        ])
        # Asking "as of Feb 11", the Feb 12 game is in the future
        result = _days_since_last_game("Lakers", "2026-02-11", matches)
        assert result == 1  # only Feb 10 counts

    def test_handles_empty_matches(self):
        matches = pd.DataFrame(
            columns=["date", "home_team", "away_team", "home_goals", "away_goals"]
        )
        result = _days_since_last_game("Lakers", "2026-02-11", matches)
        assert result is None
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_run.py::TestDaysSinceLastGame -v
```

Expected: `ImportError: cannot import name '_days_since_last_game' from 'pipeline.run'`

**Step 3: Add `_days_since_last_game` to `pipeline/run.py`**

Add after the `_check_congestion` function (around line 106):

```python
def _days_since_last_game(team: str, before_date: str, matches: pd.DataFrame) -> int | None:
    """Return days since team's most recent game strictly before before_date.

    Parameters
    ----------
    team : str
        Normalised team name.
    before_date : str
        ISO date string (YYYY-MM-DD) of the upcoming fixture.
    matches : pd.DataFrame
        Historical game results with a ``date`` column.

    Returns
    -------
    int or None
        Days since last game, or None if the team has no recorded games.
    """
    cutoff = pd.to_datetime(before_date)
    team_mask = (matches["home_team"] == team) | (matches["away_team"] == team)
    team_games = matches[team_mask].copy()
    team_games["_dt"] = pd.to_datetime(team_games["date"])
    past_games = team_games[team_games["_dt"] < cutoff]

    if past_games.empty:
        return None

    last_game = past_games["_dt"].max()
    return (cutoff - last_game).days
```

Also add `NBA_B2B_PENALTY` to the imports from `pipeline.config` at the top of `run.py`:

```python
from pipeline.config import (
    ANTHROPIC_API_KEY,
    CONGESTION_THRESHOLD_DAYS,
    DATA_DIR,
    NBA_B2B_PENALTY,
    SPORTS,
)
```

**Step 4: Run tests**

```bash
pytest tests/test_run.py -v
```

Expected: all `TestDaysSinceLastGame` tests pass.

**Step 5: Commit**

```bash
git add pipeline/run.py tests/test_run.py
git commit -m "feat(nba): add _days_since_last_game helper for B2B rest adjustment"
```

---

### Task 4: Wire up NBA branch in run.py

**Files:**
- Modify: `pipeline/run.py`

No new tests — existing integration is validated by a local dry-run in Step 4.

**Step 1: Update imports in `pipeline/run.py`**

Change the NBA fetch import line:

```python
from pipeline.fetch_nba import (
    fetch_nba_games,
    fetch_nba_schedule,
    normalize_nba_team_name,
    fetch_nba_espn_games,
    fetch_nba_espn_schedule,
)
```

**Step 2: Update NBA data-fetch branch in `run_sport_pipeline()`**

Find the `elif sport_key == "nba":` block (around line 346) and replace:

```python
elif sport_key == "nba":
    games_df, box_scores_df = fetch_nba_espn_games()
    fixtures = fetch_nba_espn_schedule()
    matches = games_df
    xg_data = None
```

**Step 3: Add rest adjustment in the prediction loop**

Find the Elo prediction block inside the `for fix in fixtures:` loop (around line 496). After the existing Elo prediction block:

```python
# Elo (all sports)
if elo is not None and home in elo.ratings and away in elo.ratings:
    elo_probs = elo_predict(elo, home, away, outcomes=outcomes)
```

Replace with:

```python
# Elo (all sports) — with optional B2B rest adjustment for NBA
if elo is not None and home in elo.ratings and away in elo.ratings:
    home_rest_adj = 0.0
    away_rest_adj = 0.0
    if sport_key == "nba":
        home_rest = _days_since_last_game(home, fix["date"], matches)
        away_rest = _days_since_last_game(away, fix["date"], matches)
        if home_rest == 1:
            home_rest_adj = -NBA_B2B_PENALTY
        if away_rest == 1:
            away_rest_adj = -NBA_B2B_PENALTY
    elo_probs = elo_predict(elo, home, away, outcomes=outcomes,
                            home_rest_adj=home_rest_adj,
                            away_rest_adj=away_rest_adj)
    individual_preds.append(elo_probs)
    blend_weights.append(model_weight_dict["elo"])
    individual_models["elo"] = elo_probs
```

(Remove the old `individual_preds.append` / `blend_weights.append` / `individual_models` lines that follow the original Elo block — they're now inside the replacement above.)

**Step 4: Dry-run the NBA pipeline locally with limited dates**

```bash
python3 -c "
from pipeline.run import run_sport_pipeline
import json, tempfile, os

with tempfile.TemporaryDirectory() as d:
    result = run_sport_pipeline('nba', output_dir=d)
    print('Matches predicted:', len(result['matches']))
    print('Model weights:', result['model_weights'])
    print('Slop locks:', len(result['slop_locks']))
    if result['matches']:
        m = result['matches'][0]
        print('Sample models:', list(m['individual_models'].keys()))
"
```

Expected output:
```
Matches predicted: <N > 0>
Model weights: {'elo': ..., 'efficiency': ..., 'four_factors': ...}
Slop locks: <some number>
Sample models: ['elo', 'efficiency', 'four_factors']
```

**Step 5: Commit**

```bash
git add pipeline/run.py
git commit -m "feat(nba): wire up 3-model ensemble with ESPN data and B2B rest adjustment"
```

---

### Task 5: Full test suite + push

**Step 1: Run all tests**

```bash
pytest tests/ -v
```

Expected: all tests pass.

**Step 2: Run full pipeline (optional but recommended)**

```bash
python3 -m pipeline.run
```

Check `data/nba/predictions.json`:
- `model_weights` should have three keys: `elo`, `efficiency`, `four_factors`
- `slop_locks` should not be exclusively home picks
- `season_stats.total_matches` should be > 0

**Step 3: Commit and push**

```bash
git add data/
git diff --staged --quiet || git commit -m "data: NBA model improvement baseline predictions"
git push origin master
```
