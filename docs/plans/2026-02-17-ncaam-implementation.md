# NCAAM Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add NCAA Men's Basketball as the third sport on SLOP LOCKS with a 3-model ensemble (Elo + Adjusted Efficiency + Four Factors logistic regression).

**Architecture:** ESPN hidden API for scores + box scores, The Odds API for bookmaker odds. Three independent models produce win probabilities, blended via softmax-weighted ensemble. Pipeline integration follows existing sport-branching pattern in `run.py`.

**Tech Stack:** Python 3.11+, scipy, pandas, numpy, scikit-learn (new dep), requests

---

### Task 1: Add scikit-learn dependency and NCAAM config

**Files:**
- Modify: `pipeline/requirements.txt`
- Modify: `pipeline/config.py`

**Step 1: Add scikit-learn to requirements.txt**

Add after the `numpy` line in `pipeline/requirements.txt`:

```
scikit-learn>=1.3.0
```

**Step 2: Add NCAAM sport config to config.py**

Add ESPN base URL constant after the balldontlie constants (~line 26):

```python
# ESPN (no API key needed)
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
```

Add NCAAM entry to the `SPORTS` dict after the NBA entry (~line 67):

```python
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
```

**Step 3: Install deps**

Run: `pip install -r pipeline/requirements.txt`

**Step 4: Commit**

```bash
git add pipeline/requirements.txt pipeline/config.py
git commit -m "feat(ncaam): add scikit-learn dep and NCAAM sport config"
```

---

### Task 2: ESPN API client — team normalization and helpers

**Files:**
- Create: `pipeline/fetch_ncaam.py`
- Create: `tests/test_fetch_ncaam.py`

**Step 1: Write tests for team normalization**

Create `tests/test_fetch_ncaam.py`:

```python
"""Tests for pipeline.fetch_ncaam — ESPN API client for NCAAM."""

from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

from pipeline.fetch_ncaam import normalize_ncaam_team_name, _parse_box_score_totals


class TestNormalizeNcaamTeamName:
    def test_full_name_to_short(self):
        assert normalize_ncaam_team_name("Duke Blue Devils") == "Duke"
        assert normalize_ncaam_team_name("Gonzaga Bulldogs") == "Gonzaga"

    def test_already_short(self):
        assert normalize_ncaam_team_name("Duke") == "Duke"

    def test_unknown_passes_through(self):
        assert normalize_ncaam_team_name("Unknown Team") == "Unknown Team"


class TestParseBoxScoreTotals:
    def test_parses_standard_totals(self):
        # Totals array from ESPN: [MIN, PTS, FG, 3PT, FT, REB, AST, TO, STL, BLK, OREB, DREB, PF]
        totals = ["", "75", "28-58", "8-20", "11-14", "35", "15", "10", "5", "3", "8", "27", "12"]
        result = _parse_box_score_totals(totals)
        assert result["pts"] == 75
        assert result["fgm"] == 28
        assert result["fga"] == 58
        assert result["fg3m"] == 8
        assert result["fg3a"] == 20
        assert result["ftm"] == 11
        assert result["fta"] == 14
        assert result["orb"] == 8
        assert result["drb"] == 27
        assert result["to"] == 10

    def test_computes_possessions(self):
        totals = ["", "75", "28-58", "8-20", "11-14", "35", "15", "10", "5", "3", "8", "27", "12"]
        result = _parse_box_score_totals(totals)
        # possessions = FGA - ORB + TO + 0.44 * FTA = 58 - 8 + 10 + 0.44 * 14 = 66.16
        assert abs(result["possessions"] - 66.16) < 0.01
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fetch_ncaam.py -v`
Expected: FAIL (module not found)

**Step 3: Write the fetch_ncaam.py module with normalization and helpers**

Create `pipeline/fetch_ncaam.py`:

```python
"""Fetch NCAAM game results, box scores, and schedule from ESPN API."""

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from pipeline.config import ESPN_BASE

_REQUEST_DELAY = 0.5  # polite delay between ESPN requests


def normalize_ncaam_team_name(name: str) -> str:
    """Normalize ESPN display name to short team name.

    ESPN uses 'Duke Blue Devils' — we want 'Duke'.
    Uses the team's 'location' field from ESPN when available,
    falls back to splitting off the last word(s) as mascot.
    """
    return _NCAAM_TEAM_NAME_MAP.get(name, name)


# This map is populated dynamically by _build_team_map() on first use,
# but we seed it with common names for testing.
_NCAAM_TEAM_NAME_MAP: dict[str, str] = {}
_TEAM_MAP_BUILT = False


def _build_team_map():
    """Fetch all D1 teams from ESPN and build the name map."""
    global _NCAAM_TEAM_NAME_MAP, _TEAM_MAP_BUILT
    if _TEAM_MAP_BUILT:
        return

    try:
        # ESPN teams endpoint returns groups of teams
        url = f"{ESPN_BASE}/teams"
        params = {"limit": 400}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for group in data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
            team = group.get("team", {})
            display_name = team.get("displayName", "")
            location = team.get("location", "")
            if display_name and location:
                _NCAAM_TEAM_NAME_MAP[display_name] = location
    except Exception:
        pass  # fall back to passthrough normalization

    _TEAM_MAP_BUILT = True


def _parse_box_score_totals(totals: list[str]) -> dict:
    """Parse ESPN box score totals array into a stats dict.

    ESPN totals format (indices):
        0: MIN (empty), 1: PTS, 2: FG (m-a), 3: 3PT (m-a), 4: FT (m-a),
        5: REB, 6: AST, 7: TO, 8: STL, 9: BLK, 10: OREB, 11: DREB, 12: PF
    """
    def _split_made_att(s: str) -> tuple[int, int]:
        parts = s.split("-")
        return int(parts[0]), int(parts[1])

    pts = int(totals[1])
    fgm, fga = _split_made_att(totals[2])
    fg3m, fg3a = _split_made_att(totals[3])
    ftm, fta = _split_made_att(totals[4])
    orb = int(totals[10])
    drb = int(totals[11])
    to = int(totals[7])

    # Possessions estimate (KenPom formula)
    possessions = fga - orb + to + 0.44 * fta

    return {
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fetch_ncaam.py::TestNormalizeNcaamTeamName tests/test_fetch_ncaam.py::TestParseBoxScoreTotals -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pipeline/fetch_ncaam.py tests/test_fetch_ncaam.py
git commit -m "feat(ncaam): add ESPN team normalization and box score parser"
```

---

### Task 3: ESPN API client — fetch games and box scores

**Files:**
- Modify: `pipeline/fetch_ncaam.py`
- Modify: `tests/test_fetch_ncaam.py`

**Step 1: Write tests for fetch_ncaam_games**

Add to `tests/test_fetch_ncaam.py`:

```python
def _make_espn_scoreboard_event(event_id, home_name, away_name, home_score, away_score,
                                 date="2026-01-15T00:00Z", status_type="final"):
    """Helper: build an ESPN scoreboard event object."""
    return {
        "id": str(event_id),
        "date": date,
        "competitions": [{
            "competitors": [
                {
                    "homeAway": "home",
                    "score": str(home_score),
                    "team": {"displayName": home_name},
                },
                {
                    "homeAway": "away",
                    "score": str(away_score),
                    "team": {"displayName": away_name},
                },
            ],
            "status": {"type": {"name": status_type}},
        }],
    }


def _make_espn_summary(home_totals, away_totals, home_name="Duke Blue Devils",
                        away_name="North Carolina Tar Heels"):
    """Helper: build an ESPN game summary response with box score."""
    def _make_team(totals, name, home_away):
        return {
            "team": {"displayName": name},
            "homeAway": home_away,
            "statistics": [{
                "names": ["MIN", "PTS", "FG", "3PT", "FT", "REB", "AST", "TO",
                          "STL", "BLK", "OREB", "DREB", "PF"],
                "totals": totals,
            }],
        }

    return {
        "boxscore": {
            "teams": [
                _make_team(home_totals, home_name, "home"),
                _make_team(away_totals, away_name, "away"),
            ],
        },
    }


class TestFetchNcaamGames:
    @patch("pipeline.fetch_ncaam._build_team_map")
    @patch("pipeline.fetch_ncaam.requests.get")
    def test_returns_scores_and_box_scores(self, mock_get, mock_build):
        # Scoreboard response
        scoreboard_resp = MagicMock()
        scoreboard_resp.json.return_value = {
            "events": [
                _make_espn_scoreboard_event(
                    "100", "Duke Blue Devils", "North Carolina Tar Heels",
                    75, 70, status_type="final"
                ),
            ],
        }
        scoreboard_resp.raise_for_status = MagicMock()

        # Summary response (box score)
        summary_resp = MagicMock()
        summary_resp.json.return_value = _make_espn_summary(
            home_totals=["", "75", "28-58", "8-20", "11-14", "35", "15", "10", "5", "3", "8", "27", "12"],
            away_totals=["", "70", "25-55", "6-18", "14-18", "30", "12", "13", "3", "2", "6", "24", "15"],
        )
        summary_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [scoreboard_resp, summary_resp]

        from pipeline.fetch_ncaam import fetch_ncaam_games
        games_df, box_df = fetch_ncaam_games(season=2025, dates=["20260115"])

        # Scores DataFrame
        assert isinstance(games_df, pd.DataFrame)
        assert list(games_df.columns) == ["date", "home_team", "away_team", "home_goals", "away_goals"]
        assert len(games_df) == 1
        assert games_df.iloc[0]["home_goals"] == 75

        # Box scores DataFrame
        assert isinstance(box_df, pd.DataFrame)
        assert len(box_df) == 2  # one row per team per game
        assert "possessions" in box_df.columns
        assert "fgm" in box_df.columns

    @patch("pipeline.fetch_ncaam._build_team_map")
    @patch("pipeline.fetch_ncaam.requests.get")
    def test_skips_non_final_games(self, mock_get, mock_build):
        scoreboard_resp = MagicMock()
        scoreboard_resp.json.return_value = {
            "events": [
                _make_espn_scoreboard_event(
                    "100", "Duke Blue Devils", "UNC Tar Heels",
                    75, 70, status_type="final"
                ),
                _make_espn_scoreboard_event(
                    "101", "Kentucky Wildcats", "Kansas Jayhawks",
                    0, 0, status_type="scheduled"
                ),
            ],
        }
        scoreboard_resp.raise_for_status = MagicMock()

        summary_resp = MagicMock()
        summary_resp.json.return_value = _make_espn_summary(
            home_totals=["", "75", "28-58", "8-20", "11-14", "35", "15", "10", "5", "3", "8", "27", "12"],
            away_totals=["", "70", "25-55", "6-18", "14-18", "30", "12", "13", "3", "2", "6", "24", "15"],
        )
        summary_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [scoreboard_resp, summary_resp]

        from pipeline.fetch_ncaam import fetch_ncaam_games
        games_df, box_df = fetch_ncaam_games(season=2025, dates=["20260115"])
        assert len(games_df) == 1
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fetch_ncaam.py::TestFetchNcaamGames -v`
Expected: FAIL (fetch_ncaam_games not defined)

**Step 3: Implement fetch_ncaam_games**

Add to `pipeline/fetch_ncaam.py`:

```python
def _generate_date_range(season: int) -> list[str]:
    """Generate list of date strings (YYYYMMDD) for an NCAAM season.

    Season starts in early November, ends in early April.
    """
    start = datetime(season, 11, 1)
    today = datetime.now(timezone.utc).date()
    end = min(datetime(season + 1, 4, 10).date(), today)

    dates = []
    current = start.date()
    while current <= end:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return dates


def fetch_ncaam_games(season: int | None = None,
                      dates: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch completed NCAAM games and box scores from ESPN.

    Parameters
    ----------
    season : int or None
        Season start year (e.g. 2025 for 2025-26). Defaults to current.
    dates : list[str] or None
        Explicit list of date strings (YYYYMMDD) to fetch. Overrides season.

    Returns
    -------
    (games_df, box_scores_df)
        games_df: date, home_team, away_team, home_goals, away_goals
        box_scores_df: game_id, team, fgm, fga, fg3m, fg3a, ftm, fta, orb, drb, to, pts, possessions
    """
    _build_team_map()

    if season is None:
        now = datetime.now(timezone.utc)
        season = now.year if now.month >= 10 else now.year - 1

    if dates is None:
        dates = _generate_date_range(season)

    game_rows = []
    box_rows = []
    seen_ids = set()

    for date_str in dates:
        try:
            url = f"{ESPN_BASE}/scoreboard"
            resp = requests.get(url, params={"dates": date_str, "limit": 200}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            continue

        for event in data.get("events", []):
            event_id = event.get("id")
            if event_id in seen_ids:
                continue

            comp = event.get("competitions", [{}])[0]
            status = comp.get("status", {}).get("type", {}).get("name", "")
            if status.lower() != "final":
                continue

            seen_ids.add(event_id)

            competitors = comp.get("competitors", [])
            home_comp = next((c for c in competitors if c["homeAway"] == "home"), None)
            away_comp = next((c for c in competitors if c["homeAway"] == "away"), None)
            if not home_comp or not away_comp:
                continue

            home_name = normalize_ncaam_team_name(home_comp["team"]["displayName"])
            away_name = normalize_ncaam_team_name(away_comp["team"]["displayName"])
            home_score = int(home_comp.get("score", 0))
            away_score = int(away_comp.get("score", 0))
            game_date = event.get("date", "")[:10]

            game_rows.append({
                "date": game_date,
                "home_team": home_name,
                "away_team": away_name,
                "home_goals": home_score,
                "away_goals": away_score,
            })

            # Fetch box score
            try:
                summary_url = f"{ESPN_BASE}/summary"
                summary_resp = requests.get(summary_url, params={"event": event_id}, timeout=30)
                summary_resp.raise_for_status()
                summary = summary_resp.json()

                for team_data in summary.get("boxscore", {}).get("teams", []):
                    team_name = normalize_ncaam_team_name(
                        team_data.get("team", {}).get("displayName", "")
                    )
                    stats_list = team_data.get("statistics", [])
                    if not stats_list:
                        continue
                    totals = stats_list[0].get("totals", [])
                    if len(totals) < 13:
                        continue

                    parsed = _parse_box_score_totals(totals)
                    parsed["game_id"] = event_id
                    parsed["team"] = team_name
                    parsed["date"] = game_date
                    box_rows.append(parsed)

                time.sleep(_REQUEST_DELAY)
            except Exception:
                continue

    games_df = pd.DataFrame(
        game_rows,
        columns=["date", "home_team", "away_team", "home_goals", "away_goals"],
    )
    box_df = pd.DataFrame(box_rows)

    return games_df, box_df
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fetch_ncaam.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pipeline/fetch_ncaam.py tests/test_fetch_ncaam.py
git commit -m "feat(ncaam): add ESPN API client for games and box scores"
```

---

### Task 4: ESPN API client — fetch schedule

**Files:**
- Modify: `pipeline/fetch_ncaam.py`
- Modify: `tests/test_fetch_ncaam.py`

**Step 1: Write test for fetch_ncaam_schedule**

Add to `tests/test_fetch_ncaam.py`:

```python
class TestFetchNcaamSchedule:
    @patch("pipeline.fetch_ncaam._build_team_map")
    @patch("pipeline.fetch_ncaam.requests.get")
    def test_returns_upcoming_games(self, mock_get, mock_build):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "events": [
                _make_espn_scoreboard_event(
                    "200", "Duke Blue Devils", "North Carolina Tar Heels",
                    0, 0, date="2026-02-20T19:00Z", status_type="scheduled"
                ),
            ],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from pipeline.fetch_ncaam import fetch_ncaam_schedule
        fixtures = fetch_ncaam_schedule()

        assert isinstance(fixtures, list)
        assert len(fixtures) == 1
        assert fixtures[0]["home_team"] == "Duke"
        assert fixtures[0]["away_team"] == "North Carolina"
        assert fixtures[0]["date"] == "2026-02-20"

    @patch("pipeline.fetch_ncaam._build_team_map")
    @patch("pipeline.fetch_ncaam.requests.get")
    def test_excludes_final_games(self, mock_get, mock_build):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "events": [
                _make_espn_scoreboard_event(
                    "200", "Duke Blue Devils", "North Carolina Tar Heels",
                    75, 70, status_type="final"
                ),
                _make_espn_scoreboard_event(
                    "201", "Kentucky Wildcats", "Kansas Jayhawks",
                    0, 0, date="2026-02-21T19:00Z", status_type="scheduled"
                ),
            ],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from pipeline.fetch_ncaam import fetch_ncaam_schedule
        fixtures = fetch_ncaam_schedule()
        assert len(fixtures) == 1
        assert fixtures[0]["home_team"] == "Kentucky"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fetch_ncaam.py::TestFetchNcaamSchedule -v`
Expected: FAIL

**Step 3: Implement fetch_ncaam_schedule**

Add to `pipeline/fetch_ncaam.py`:

```python
def fetch_ncaam_schedule() -> list[dict]:
    """Fetch upcoming NCAAM games (today + 7 days).

    Returns
    -------
    list[dict]
        Each dict has keys: home_team, away_team, date.
    """
    _build_team_map()

    today = datetime.now(timezone.utc).date()
    fixtures = []

    for day_offset in range(8):
        date = today + timedelta(days=day_offset)
        date_str = date.strftime("%Y%m%d")

        try:
            url = f"{ESPN_BASE}/scoreboard"
            resp = requests.get(url, params={"dates": date_str, "limit": 200}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            continue

        for event in data.get("events", []):
            comp = event.get("competitions", [{}])[0]
            status = comp.get("status", {}).get("type", {}).get("name", "")
            if status.lower() == "final":
                continue

            competitors = comp.get("competitors", [])
            home_comp = next((c for c in competitors if c["homeAway"] == "home"), None)
            away_comp = next((c for c in competitors if c["homeAway"] == "away"), None)
            if not home_comp or not away_comp:
                continue

            fixtures.append({
                "home_team": normalize_ncaam_team_name(home_comp["team"]["displayName"]),
                "away_team": normalize_ncaam_team_name(away_comp["team"]["displayName"]),
                "date": event.get("date", "")[:10],
            })

    return fixtures
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fetch_ncaam.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add pipeline/fetch_ncaam.py tests/test_fetch_ncaam.py
git commit -m "feat(ncaam): add ESPN schedule fetcher"
```

---

### Task 5: Adjusted Efficiency model

**Files:**
- Modify: `pipeline/models.py`
- Modify: `tests/test_models.py`
- Create: `tests/conftest_ncaam.py` (shared NCAAM fixtures)

**Step 1: Create NCAAM test fixtures**

Create `tests/conftest_ncaam.py` with shared box score fixtures, then add these fixtures to `tests/conftest.py`:

Add to `tests/conftest.py`:

```python
@pytest.fixture
def ncaam_games():
    """Minimal set of NCAAM game results for testing."""
    base_date = datetime(2025, 11, 10)
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
    matches = []
    for i, (home, away, hg, ag) in enumerate(results):
        matches.append({
            "date": (base_date + timedelta(days=i * 3)).isoformat(),
            "home_team": home,
            "away_team": away,
            "home_goals": hg,
            "away_goals": ag,
        })
    return pd.DataFrame(matches)


@pytest.fixture
def ncaam_box_scores():
    """Box score stats matching ncaam_games fixture."""
    base_date = datetime(2025, 11, 10)
    games = [
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
    rows = []
    for i, (home, away, hpts, apts) in enumerate(games):
        date = (base_date + timedelta(days=i * 3)).isoformat()
        game_id = str(1000 + i)
        # Generate plausible box scores from points
        for team, pts in [(home, hpts), (away, apts)]:
            fga = int(pts / 1.1)
            fgm = int(pts * 0.42)
            fg3a = int(fga * 0.35)
            fg3m = int(fg3a * 0.34)
            fta = int(pts * 0.22)
            ftm = int(fta * 0.73)
            orb = int(fga * 0.15)
            drb = int(fga * 0.45)
            to = int(fga * 0.18)
            poss = fga - orb + to + 0.44 * fta
            rows.append({
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
                "possessions": poss,
            })
    return pd.DataFrame(rows)
```

**Step 2: Write tests for AdjustedEfficiency**

Add to `tests/test_models.py`:

```python
from pipeline.models import AdjustedEfficiency, efficiency_predict


class TestAdjustedEfficiency:
    def test_all_teams_have_ratings(self, ncaam_games, ncaam_box_scores):
        model = AdjustedEfficiency(ncaam_box_scores, ncaam_games)
        teams = set(ncaam_games["home_team"]) | set(ncaam_games["away_team"])
        for team in teams:
            assert team in model.off_efficiency
            assert team in model.def_efficiency
            assert team in model.tempo

    def test_efficiencies_are_positive(self, ncaam_games, ncaam_box_scores):
        model = AdjustedEfficiency(ncaam_box_scores, ncaam_games)
        for team in model.off_efficiency:
            assert model.off_efficiency[team] > 0
            assert model.def_efficiency[team] > 0
            assert model.tempo[team] > 0

    def test_predict_returns_two_way_probs(self, ncaam_games, ncaam_box_scores):
        model = AdjustedEfficiency(ncaam_box_scores, ncaam_games)
        probs = efficiency_predict(model, "Duke", "North Carolina")
        assert set(probs.keys()) == {"home", "away"}
        assert math.isclose(probs["home"] + probs["away"], 1.0, abs_tol=1e-9)

    def test_home_team_gets_bonus(self, ncaam_games, ncaam_box_scores):
        """With equal-strength teams, home should have >50% due to home bonus."""
        model = AdjustedEfficiency(ncaam_box_scores, ncaam_games)
        # Force equal ratings
        for team in model.off_efficiency:
            model.off_efficiency[team] = 100.0
            model.def_efficiency[team] = 100.0
        probs = efficiency_predict(model, "Duke", "North Carolina")
        assert probs["home"] > 0.5
```

**Step 3: Run tests to verify they fail**

Run: `pytest tests/test_models.py::TestAdjustedEfficiency -v`
Expected: FAIL

**Step 4: Implement AdjustedEfficiency**

Add to `pipeline/models.py` (after the Elo section):

```python
# ---------------------------------------------------------------------------
# Adjusted Efficiency model (KenPom-style)
# ---------------------------------------------------------------------------

class AdjustedEfficiency:
    """Tempo-adjusted, opponent-adjusted efficiency ratings.

    Parameters
    ----------
    box_scores : pd.DataFrame
        Per-team-per-game box scores with columns:
        game_id, team, pts, fga, possessions, etc.
    games : pd.DataFrame
        Game results with columns: date, home_team, away_team, home_goals, away_goals.
    iterations : int
        Number of opponent-adjustment iterations.
    """

    def __init__(self, box_scores, games, iterations=10):
        self.off_efficiency = {}
        self.def_efficiency = {}
        self.tempo = {}
        self._fit(box_scores, games, iterations)

    def _fit(self, box_scores, games, iterations):
        teams = sorted(set(box_scores["team"].unique()))

        # Step 1: Raw efficiency per team
        raw_off = {}  # points per 100 possessions
        raw_def = {}  # points allowed per 100 possessions
        raw_tempo = {}  # avg possessions per game
        opponents = {t: [] for t in teams}  # track who each team played

        # Build game lookup: game_id -> (home_team, away_team)
        game_teams = {}
        for _, row in games.iterrows():
            # Match by date and teams
            for _, brow in box_scores.iterrows():
                if brow["game_id"] not in game_teams:
                    game_id = brow["game_id"]
                    game_box = box_scores[box_scores["game_id"] == game_id]
                    if len(game_box) == 2:
                        t1, t2 = game_box["team"].values
                        game_teams[game_id] = (t1, t2)

        # Compute raw stats
        for team in teams:
            team_box = box_scores[box_scores["team"] == team]
            if len(team_box) == 0:
                raw_off[team] = 100.0
                raw_def[team] = 100.0
                raw_tempo[team] = 68.0
                continue

            total_pts = team_box["pts"].sum()
            total_poss = team_box["possessions"].sum()

            if total_poss > 0:
                raw_off[team] = (total_pts / total_poss) * 100
            else:
                raw_off[team] = 100.0

            raw_tempo[team] = team_box["possessions"].mean()

            # Defensive: points allowed
            pts_allowed = 0.0
            poss_against = 0.0
            for _, trow in team_box.iterrows():
                gid = trow["game_id"]
                opp_box = box_scores[
                    (box_scores["game_id"] == gid) & (box_scores["team"] != team)
                ]
                if len(opp_box) == 1:
                    pts_allowed += opp_box.iloc[0]["pts"]
                    poss_against += opp_box.iloc[0]["possessions"]
                    opponents[team].append(opp_box.iloc[0]["team"])

            if poss_against > 0:
                raw_def[team] = (pts_allowed / poss_against) * 100
            else:
                raw_def[team] = 100.0

        # League averages
        league_off = sum(raw_off.values()) / len(teams) if teams else 100.0

        # Step 2: Iterative opponent adjustment
        adj_off = dict(raw_off)
        adj_def = dict(raw_def)

        for _ in range(iterations):
            new_off = {}
            new_def = {}
            avg_def = sum(adj_def.values()) / len(teams)
            avg_off = sum(adj_off.values()) / len(teams)

            for team in teams:
                opps = opponents.get(team, [])
                if not opps:
                    new_off[team] = adj_off[team]
                    new_def[team] = adj_def[team]
                    continue

                # Adjust offense by opponent defensive strength
                opp_def_avg = sum(adj_def.get(o, avg_def) for o in opps) / len(opps)
                if opp_def_avg > 0:
                    new_off[team] = raw_off[team] * (avg_def / opp_def_avg)
                else:
                    new_off[team] = raw_off[team]

                # Adjust defense by opponent offensive strength
                opp_off_avg = sum(adj_off.get(o, avg_off) for o in opps) / len(opps)
                if opp_off_avg > 0:
                    new_def[team] = raw_def[team] * (avg_off / opp_off_avg)
                else:
                    new_def[team] = raw_def[team]

            adj_off = new_off
            adj_def = new_def

        self.off_efficiency = adj_off
        self.def_efficiency = adj_def
        self.tempo = raw_tempo


def efficiency_predict(model, home_team, away_team, home_bonus=3.5):
    """Predict win probabilities from adjusted efficiency ratings.

    Parameters
    ----------
    model : AdjustedEfficiency
        Fitted efficiency model.
    home_team, away_team : str
        Team names.
    home_bonus : float
        Points added to home team's expected margin.

    Returns
    -------
    dict
        {"home": float, "away": float} probabilities summing to 1.
    """
    league_avg = sum(model.off_efficiency.values()) / len(model.off_efficiency)

    # Expected tempo for this matchup
    home_tempo = model.tempo.get(home_team, 68.0)
    away_tempo = model.tempo.get(away_team, 68.0)
    expected_tempo = (home_tempo + away_tempo) / 2

    # Expected points
    home_off = model.off_efficiency.get(home_team, league_avg)
    away_def = model.def_efficiency.get(away_team, league_avg)
    away_off = model.off_efficiency.get(away_team, league_avg)
    home_def = model.def_efficiency.get(home_team, league_avg)

    home_pts = (home_off * away_def / league_avg) * (expected_tempo / 100)
    away_pts = (away_off * home_def / league_avg) * (expected_tempo / 100)

    # Point spread + home bonus -> win probability via logistic
    spread = (home_pts - away_pts) + home_bonus
    # Sigma calibrated for college basketball (~11 points SD)
    sigma = 11.0
    home_prob = 1.0 / (1.0 + math.exp(-spread / sigma))

    return {"home": home_prob, "away": 1.0 - home_prob}
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_models.py::TestAdjustedEfficiency -v`
Expected: PASS

**Step 6: Commit**

```bash
git add pipeline/models.py tests/conftest.py tests/test_models.py
git commit -m "feat(ncaam): add Adjusted Efficiency model (KenPom-style)"
```

---

### Task 6: Four Factors logistic regression model

**Files:**
- Modify: `pipeline/models.py`
- Modify: `tests/test_models.py`

**Step 1: Write tests for FourFactorsModel**

Add to `tests/test_models.py`:

```python
from pipeline.models import FourFactorsModel, four_factors_predict


class TestFourFactorsModel:
    def test_all_teams_have_stats(self, ncaam_games, ncaam_box_scores):
        model = FourFactorsModel(ncaam_box_scores, ncaam_games)
        teams = set(ncaam_games["home_team"]) | set(ncaam_games["away_team"])
        for team in teams:
            assert team in model.team_stats

    def test_team_stats_have_expected_keys(self, ncaam_games, ncaam_box_scores):
        model = FourFactorsModel(ncaam_box_scores, ncaam_games)
        for team, stats in model.team_stats.items():
            for key in ["off_efg", "off_to_rate", "off_orb_pct", "off_ft_rate",
                        "def_efg", "def_to_rate", "def_orb_pct", "def_ft_rate"]:
                assert key in stats, f"{team} missing {key}"

    def test_predict_returns_two_way_probs(self, ncaam_games, ncaam_box_scores):
        model = FourFactorsModel(ncaam_box_scores, ncaam_games)
        probs = four_factors_predict(model, "Duke", "North Carolina")
        assert set(probs.keys()) == {"home", "away"}
        assert math.isclose(probs["home"] + probs["away"], 1.0, abs_tol=1e-9)

    def test_probabilities_are_reasonable(self, ncaam_games, ncaam_box_scores):
        model = FourFactorsModel(ncaam_box_scores, ncaam_games)
        probs = four_factors_predict(model, "Duke", "North Carolina")
        assert 0.1 < probs["home"] < 0.9
        assert 0.1 < probs["away"] < 0.9
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models.py::TestFourFactorsModel -v`
Expected: FAIL

**Step 3: Implement FourFactorsModel**

Add to `pipeline/models.py`:

```python
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Four Factors logistic regression model
# ---------------------------------------------------------------------------

class FourFactorsModel:
    """Logistic regression model using Dean Oliver's Four Factors.

    Computes per-team offensive and defensive four factors from box scores,
    then trains a logistic regression on historical matchup features.

    Parameters
    ----------
    box_scores : pd.DataFrame
        Per-team-per-game box scores.
    games : pd.DataFrame
        Game results (home_team, away_team, home_goals, away_goals).
    """

    def __init__(self, box_scores, games):
        self.team_stats = {}
        self.model = None
        self._fit(box_scores, games)

    def _compute_team_four_factors(self, box_scores):
        """Compute season-average four factors per team."""
        teams = sorted(box_scores["team"].unique())

        for team in teams:
            team_box = box_scores[box_scores["team"] == team]
            if len(team_box) == 0:
                continue

            # Offensive four factors
            fgm = team_box["fgm"].sum()
            fga = team_box["fga"].sum()
            fg3m = team_box["fg3m"].sum()
            fta = team_box["fta"].sum()
            to = team_box["to"].sum()
            orb = team_box["orb"].sum()
            poss = team_box["possessions"].sum()

            off_efg = (fgm + 0.5 * fg3m) / max(fga, 1)
            off_to_rate = to / max(poss, 1)
            off_ft_rate = fta / max(fga, 1)

            # ORB% needs opponent DRB — compute from game-level
            opp_drb_total = 0.0
            for _, trow in team_box.iterrows():
                gid = trow["game_id"]
                opp_box = box_scores[
                    (box_scores["game_id"] == gid) & (box_scores["team"] != team)
                ]
                if len(opp_box) == 1:
                    opp_drb_total += opp_box.iloc[0]["drb"]

            off_orb_pct = orb / max(orb + opp_drb_total, 1)

            # Defensive four factors (what opponents do against this team)
            opp_fgm = 0.0
            opp_fga = 0.0
            opp_fg3m = 0.0
            opp_fta = 0.0
            opp_to = 0.0
            opp_orb = 0.0
            opp_poss = 0.0
            team_drb_total = team_box["drb"].sum()

            for _, trow in team_box.iterrows():
                gid = trow["game_id"]
                opp_box = box_scores[
                    (box_scores["game_id"] == gid) & (box_scores["team"] != team)
                ]
                if len(opp_box) == 1:
                    opp = opp_box.iloc[0]
                    opp_fgm += opp["fgm"]
                    opp_fga += opp["fga"]
                    opp_fg3m += opp["fg3m"]
                    opp_fta += opp["fta"]
                    opp_to += opp["to"]
                    opp_orb += opp["orb"]
                    opp_poss += opp["possessions"]

            def_efg = (opp_fgm + 0.5 * opp_fg3m) / max(opp_fga, 1)
            def_to_rate = opp_to / max(opp_poss, 1)
            def_ft_rate = opp_fta / max(opp_fga, 1)
            def_orb_pct = opp_orb / max(opp_orb + team_drb_total, 1)

            self.team_stats[team] = {
                "off_efg": off_efg,
                "off_to_rate": off_to_rate,
                "off_orb_pct": off_orb_pct,
                "off_ft_rate": off_ft_rate,
                "def_efg": def_efg,
                "def_to_rate": def_to_rate,
                "def_orb_pct": def_orb_pct,
                "def_ft_rate": def_ft_rate,
            }

    def _build_feature_vector(self, home_team, away_team):
        """Build 16-feature vector for a matchup."""
        hs = self.team_stats.get(home_team, {})
        aws = self.team_stats.get(away_team, {})
        keys = ["off_efg", "off_to_rate", "off_orb_pct", "off_ft_rate",
                "def_efg", "def_to_rate", "def_orb_pct", "def_ft_rate"]
        return [hs.get(k, 0.5) for k in keys] + [aws.get(k, 0.5) for k in keys]

    def _fit(self, box_scores, games):
        self._compute_team_four_factors(box_scores)

        if len(self.team_stats) < 2:
            return

        # Build training data from historical games
        X = []
        y = []
        for _, row in games.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            if home not in self.team_stats or away not in self.team_stats:
                continue
            X.append(self._build_feature_vector(home, away))
            y.append(1 if row["home_goals"] > row["away_goals"] else 0)

        if len(set(y)) < 2 or len(X) < 5:
            return

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y)


def four_factors_predict(model, home_team, away_team):
    """Predict win probabilities from Four Factors model.

    Parameters
    ----------
    model : FourFactorsModel
        Fitted model.
    home_team, away_team : str
        Team names.

    Returns
    -------
    dict
        {"home": float, "away": float} probabilities summing to 1.
    """
    if model.model is None:
        return {"home": 0.5, "away": 0.5}

    if home_team not in model.team_stats or away_team not in model.team_stats:
        return {"home": 0.5, "away": 0.5}

    features = [model._build_feature_vector(home_team, away_team)]
    proba = model.model.predict_proba(features)[0]

    # proba[1] = P(home wins), proba[0] = P(away wins)
    return {"home": float(proba[1]), "away": float(proba[0])}
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_models.py::TestFourFactorsModel -v`
Expected: PASS

**Step 6: Commit**

```bash
git add pipeline/models.py tests/test_models.py
git commit -m "feat(ncaam): add Four Factors logistic regression model"
```

---

### Task 7: Pipeline integration — run_sport_pipeline for NCAAM

**Files:**
- Modify: `pipeline/run.py`
- Create: `tests/test_run_ncaam.py`

**Step 1: Write integration test**

Create `tests/test_run_ncaam.py`:

```python
"""Tests for NCAAM pipeline integration."""

import json
import os
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from pipeline.run import run_sport_pipeline


@pytest.fixture
def ncaam_pipeline_mocks(ncaam_games, ncaam_box_scores, tmp_path):
    """Mock all external calls for NCAAM pipeline."""
    fixtures = [
        {"home_team": "Duke", "away_team": "North Carolina", "date": "2026-02-20"},
        {"home_team": "Kansas", "away_team": "Kentucky", "date": "2026-02-21"},
    ]
    odds = [
        {
            "home_team": "Duke",
            "away_team": "North Carolina",
            "home_odds": 1.65,
            "away_odds": 2.25,
        },
    ]

    return {
        "games": ncaam_games,
        "box_scores": ncaam_box_scores,
        "fixtures": fixtures,
        "odds": odds,
        "output_dir": str(tmp_path),
    }


class TestNcaamPipeline:
    @patch("pipeline.run.fetch_odds")
    @patch("pipeline.run.fetch_ncaam_schedule")
    @patch("pipeline.run.fetch_ncaam_games")
    def test_produces_predictions_json(self, mock_games, mock_sched, mock_odds,
                                        ncaam_pipeline_mocks):
        mock_games.return_value = (
            ncaam_pipeline_mocks["games"],
            ncaam_pipeline_mocks["box_scores"],
        )
        mock_sched.return_value = ncaam_pipeline_mocks["fixtures"]
        mock_odds.return_value = ncaam_pipeline_mocks["odds"]

        output_dir = ncaam_pipeline_mocks["output_dir"]
        result = run_sport_pipeline("ncaam", output_dir=output_dir)

        assert result is not None
        assert result["sport"] == "ncaam"
        assert "matches" in result
        assert "slop_locks" in result
        assert "model_weights" in result

        # Check model weights include all three models
        weights = result["model_weights"]
        assert "elo" in weights
        assert "efficiency" in weights
        assert "four_factors" in weights

    @patch("pipeline.run.fetch_odds")
    @patch("pipeline.run.fetch_ncaam_schedule")
    @patch("pipeline.run.fetch_ncaam_games")
    def test_predictions_have_two_way_probs(self, mock_games, mock_sched, mock_odds,
                                             ncaam_pipeline_mocks):
        mock_games.return_value = (
            ncaam_pipeline_mocks["games"],
            ncaam_pipeline_mocks["box_scores"],
        )
        mock_sched.return_value = ncaam_pipeline_mocks["fixtures"]
        mock_odds.return_value = ncaam_pipeline_mocks["odds"]

        result = run_sport_pipeline("ncaam", output_dir=ncaam_pipeline_mocks["output_dir"])

        for match in result.get("matches", []):
            probs = match["model_probs"]
            assert "home" in probs
            assert "away" in probs
            assert "draw" not in probs
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_run_ncaam.py -v`
Expected: FAIL

**Step 3: Integrate NCAAM into run.py**

Add import at the top of `pipeline/run.py` (~line 21):

```python
from pipeline.fetch_ncaam import fetch_ncaam_games, fetch_ncaam_schedule, normalize_ncaam_team_name
from pipeline.models import (
    AdjustedEfficiency,
    EloRatings,
    FourFactorsModel,
    dixon_coles_predict,
    efficiency_predict,
    elo_predict,
    fit_dixon_coles,
    four_factors_predict,
    scoreline_to_probabilities,
)
```

Add NCAAM branch in `run_sport_pipeline()` after the NBA branch (~line 339):

```python
    elif sport_key == "ncaam":
        games_df, box_scores_df = fetch_ncaam_games()
        fixtures = fetch_ncaam_schedule()
        matches = games_df
        xg_data = None
```

Add efficiency model fitting after the Elo block (~line 380):

```python
    # Adjusted Efficiency model (NCAAM)
    efficiency_model = None
    if "efficiency" in sport["models"] and 'box_scores_df' in dir():
        efficiency_model = AdjustedEfficiency(box_scores_df, matches)

    # Four Factors model (NCAAM)
    four_factors_model = None
    if "four_factors" in sport["models"] and 'box_scores_df' in dir():
        four_factors_model = FourFactorsModel(box_scores_df, matches)
```

Update model_names list building (~line 390):

```python
    if efficiency_model is not None:
        model_names.append("efficiency")
    if four_factors_model is not None:
        model_names.append("four_factors")
```

Add NCAAM normalizer (~line 405):

```python
    if sport_key == "ncaam":
        normalizer = normalize_ncaam_team_name
```

Add prediction blocks inside the fixture loop (~line 462):

```python
        # Adjusted Efficiency (NCAAM)
        if efficiency_model is not None and home in efficiency_model.off_efficiency and away in efficiency_model.off_efficiency:
            eff_probs = efficiency_predict(
                efficiency_model, home, away,
                home_bonus=sport.get("efficiency_home_bonus", 3.5),
            )
            individual_preds.append(eff_probs)
            blend_weights.append(model_weight_dict["efficiency"])
            individual_models["efficiency"] = eff_probs

        # Four Factors (NCAAM)
        if four_factors_model is not None and four_factors_model.model is not None:
            if home in four_factors_model.team_stats and away in four_factors_model.team_stats:
                ff_probs = four_factors_predict(four_factors_model, home, away)
                individual_preds.append(ff_probs)
                blend_weights.append(model_weight_dict["four_factors"])
                individual_models["four_factors"] = ff_probs
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_run_ncaam.py -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: ALL PASS (no regressions)

**Step 6: Commit**

```bash
git add pipeline/run.py tests/test_run_ncaam.py
git commit -m "feat(ncaam): integrate NCAAM 3-model pipeline into run.py"
```

---

### Task 8: Frontend — add NCAAM sport pill

**Files:**
- Modify: `index.html`

**Step 1: Find and update the sport toggle**

Search for the sport pill/toggle section in `index.html` and add an "NCAAM" pill after the NBA pill. The frontend already reads `predictions.json` per sport and renders based on the `outcomes` array, so no JS changes are needed beyond adding the pill.

Add the NCAAM pill with the same markup pattern as the existing EPL and NBA pills.

**Step 2: Verify locally**

Open `index.html` in a browser and confirm the NCAAM pill appears and is clickable.

**Step 3: Commit**

```bash
git add index.html
git commit -m "feat(ncaam): add NCAAM sport pill to frontend"
```

---

### Task 9: GitHub Actions — add NCAAM to daily pipeline

**Files:**
- Modify: `.github/workflows/daily.yml`

**Step 1: Verify daily.yml already runs all sports**

Check if `run_pipeline()` is called (which iterates all `SPORTS` keys) — if so, no changes needed. NCAAM was added to the `SPORTS` dict in Task 1, so it will be picked up automatically.

If the workflow calls individual sports, add `ncaam` to the list.

**Step 2: Ensure scikit-learn is installed**

Verify the workflow installs from `requirements.txt` — scikit-learn was added in Task 1.

**Step 3: Commit (if changes needed)**

```bash
git add .github/workflows/daily.yml
git commit -m "ci: add NCAAM to daily pipeline run"
```

---

### Task 10: End-to-end verification

**Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: ALL PASS

**Step 2: Dry-run the pipeline locally (if API access available)**

Run: `python -c "from pipeline.run import run_sport_pipeline; run_sport_pipeline('ncaam')"`

Verify:
- `data/ncaam/predictions.json` is created
- Contains `matches` array with model predictions
- Contains `model_weights` with elo, efficiency, four_factors
- Contains `slop_locks` array

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat(ncaam): complete NCAAM support with 3-model ensemble"
```
