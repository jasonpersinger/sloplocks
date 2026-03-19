# Pipeline Speed Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Cut the daily GitHub Actions pipeline from 25-30 minutes to ~3-5 minutes by adding pip caching, incremental ESPN data fetching, and box score caching.

**Architecture:** Each sport's ESPN fetcher gains an optional `cache_path` parameter pointing to `data/{sport}/espn_cache.json`. On each run the cache is loaded, only dates since `(max_cached_date - 2 days)` are fetched from ESPN, and box score API calls are skipped for game IDs already in the cache. The cache is saved back to disk and committed to git alongside `predictions.json`, so the next run picks up where the last one left off.

**Tech Stack:** Python 3.11+, `json` (stdlib), `pytest` + `unittest.mock` for tests, GitHub Actions `setup-python@v5` pip cache.

---

## Pre-flight: Fix broken schedule tests

Our earlier change to preserve full timestamps in `fetch_nba_espn_schedule` and `fetch_ncaam_schedule` broke two existing tests that assert `date == "2026-02-19"` (date-only). Fix these before adding new code.

### Task 0: Fix failing schedule tests

**Files:**
- Modify: `tests/test_fetch_nba.py:302` (`TestFetchNbaEspnSchedule.test_returns_upcoming_games`)
- Modify: `tests/test_fetch_ncaam.py:337` (`TestFetchNcaamSchedule.test_returns_upcoming_games`)

**Step 1: Confirm tests fail**

```bash
pytest tests/test_fetch_nba.py::TestFetchNbaEspnSchedule::test_returns_upcoming_games \
       tests/test_fetch_ncaam.py::TestFetchNcaamSchedule::test_returns_upcoming_games -v
```

Expected: both FAIL — `"2026-02-19T00:00Z" != "2026-02-19"`

**Step 2: Update assertions to expect full timestamp**

In `tests/test_fetch_nba.py`, change:
```python
assert fixtures[0]["date"] == "2026-02-19"
```
to:
```python
assert fixtures[0]["date"] == "2026-02-19T00:00Z"
```

In `tests/test_fetch_ncaam.py`, change:
```python
assert fixtures[0]["date"] == "2026-02-19"
```
to:
```python
assert fixtures[0]["date"] == "2026-02-19T00:00Z"
```

**Step 3: Run tests**

```bash
pytest tests/test_fetch_nba.py::TestFetchNbaEspnSchedule \
       tests/test_fetch_ncaam.py::TestFetchNcaamSchedule -v
```

Expected: PASS

**Step 4: Commit**

```bash
git add tests/test_fetch_nba.py tests/test_fetch_ncaam.py
git commit -m "fix(tests): update schedule date assertions to expect full ISO timestamps"
```

---

## Task 1: pip caching in daily.yml

**Files:**
- Modify: `.github/workflows/daily.yml`

**Step 1: Add cache to setup-python step**

In `.github/workflows/daily.yml`, change:
```yaml
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
```
to:
```yaml
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
```

**Step 2: Commit**

```bash
git add .github/workflows/daily.yml
git commit -m "ci: add pip dependency caching to daily workflow"
```

---

## Task 2: Incremental fetch + box score cache for NCAAM

**Files:**
- Modify: `pipeline/fetch_ncaam.py`
- Modify: `tests/test_fetch_ncaam.py`

### Cache format

`data/ncaam/espn_cache.json`:
```json
{
  "games": {
    "401825518": {
      "date": "2026-02-15",
      "home_team": "Duke",
      "away_team": "North Carolina",
      "home_goals": 75,
      "away_goals": 70,
      "box_scores": [
        {"team": "Duke", "pts": 75, "fgm": 28, "fga": 58, "fg3m": 8, "fg3a": 20,
         "ftm": 11, "fta": 14, "orb": 8, "drb": 27, "to": 10, "possessions": 66.16},
        {"team": "North Carolina", "pts": 70, "fgm": 25, "fga": 55, "fg3m": 6, "fg3a": 18,
         "ftm": 14, "fta": 18, "orb": 9, "drb": 30, "to": 12, "possessions": 63.92}
      ]
    }
  }
}
```

### Step 1: Write failing tests

Add to `tests/test_fetch_ncaam.py`:

```python
import json
import os

class TestFetchNcaamGamesCache:
    @patch("pipeline.fetch_ncaam.time.sleep")
    @patch("pipeline.fetch_ncaam._team_map", {"Duke Blue Devils": "Duke", "Kansas Jayhawks": "Kansas"})
    @patch("pipeline.fetch_ncaam.requests.get")
    def test_skips_box_score_for_cached_game(self, mock_get, mock_sleep, tmp_path):
        """Game already in cache → summary endpoint not called."""
        cache = {
            "games": {
                "101": {
                    "date": "2026-02-15",
                    "home_team": "Duke",
                    "away_team": "Kansas",
                    "home_goals": 75,
                    "away_goals": 70,
                    "box_scores": [
                        {"team": "Duke", "pts": 75, "fgm": 28, "fga": 58, "fg3m": 8,
                         "fg3a": 20, "ftm": 11, "fta": 14, "orb": 8, "drb": 27,
                         "to": 10, "possessions": 66.16},
                        {"team": "Kansas", "pts": 70, "fgm": 25, "fga": 55, "fg3m": 6,
                         "fg3a": 18, "ftm": 14, "fta": 18, "orb": 9, "drb": 30,
                         "to": 12, "possessions": 63.92},
                    ],
                }
            }
        }
        cache_path = tmp_path / "espn_cache.json"
        cache_path.write_text(json.dumps(cache))

        scoreboard_resp = MagicMock()
        scoreboard_resp.json.return_value = {
            "events": [_make_espn_event("101", "Duke Blue Devils", "Kansas Jayhawks", 75, 70)]
        }
        scoreboard_resp.raise_for_status = MagicMock()
        mock_get.return_value = scoreboard_resp

        games_df, box_df = fetch_ncaam_games(
            season=2025, dates=["2026-02-15"], cache_path=str(cache_path)
        )

        # Only the scoreboard call — no summary call for cached game
        assert mock_get.call_count == 1
        assert "summary" not in str(mock_get.call_args_list[0])
        assert len(games_df) == 1
        assert len(box_df) == 2

    @patch("pipeline.fetch_ncaam.time.sleep")
    @patch("pipeline.fetch_ncaam._team_map", {"Duke Blue Devils": "Duke", "Kansas Jayhawks": "Kansas"})
    @patch("pipeline.fetch_ncaam.requests.get")
    def test_cache_written_after_fetch(self, mock_get, mock_sleep, tmp_path):
        """Cache file is created/updated after fetching new games."""
        cache_path = tmp_path / "espn_cache.json"

        scoreboard_resp = MagicMock()
        scoreboard_resp.json.return_value = {
            "events": [_make_espn_event("201", "Duke Blue Devils", "Kansas Jawhawks", 80, 72)]
        }
        scoreboard_resp.raise_for_status = MagicMock()

        summary_resp = MagicMock()
        summary_resp.json.return_value = _make_summary_response(SAMPLE_TOTALS, SAMPLE_TOTALS)
        summary_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [scoreboard_resp, summary_resp]

        fetch_ncaam_games(season=2025, dates=["2026-02-15"], cache_path=str(cache_path))

        assert cache_path.exists()
        saved = json.loads(cache_path.read_text())
        assert "201" in saved["games"]
        assert saved["games"]["201"]["home_team"] == "Duke"
        assert len(saved["games"]["201"]["box_scores"]) == 2

    @patch("pipeline.fetch_ncaam.time.sleep")
    @patch("pipeline.fetch_ncaam._team_map", {"Duke Blue Devils": "Duke", "Kansas Jayhawks": "Kansas"})
    @patch("pipeline.fetch_ncaam.requests.get")
    def test_only_recent_dates_fetched_when_cache_has_data(self, mock_get, mock_sleep, tmp_path):
        """Cache with max_date 2026-02-18 → only dates >= 2026-02-16 are fetched."""
        cache = {
            "games": {
                "101": {
                    "date": "2026-02-18",
                    "home_team": "Duke",
                    "away_team": "Kansas",
                    "home_goals": 75,
                    "away_goals": 70,
                    "box_scores": [],
                }
            }
        }
        cache_path = tmp_path / "espn_cache.json"
        cache_path.write_text(json.dumps(cache))

        empty_resp = MagicMock()
        empty_resp.json.return_value = {"events": []}
        empty_resp.raise_for_status = MagicMock()
        mock_get.return_value = empty_resp

        # Full season range has 5 dates; only 3 should be fetched (Feb 16, 17, 18)
        all_dates = ["2026-02-14", "2026-02-15", "2026-02-16", "2026-02-17", "2026-02-18"]
        fetch_ncaam_games(season=2025, dates=all_dates, cache_path=str(cache_path))

        # Should only fetch scoreboard for dates >= max_date - 2 days
        assert mock_get.call_count == 3  # Feb 16, 17, 18
```

**Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_fetch_ncaam.py::TestFetchNcaamGamesCache -v
```

Expected: FAIL — `fetch_ncaam_games() got unexpected keyword argument 'cache_path'`

**Step 3: Implement caching in `fetch_ncaam_games`**

Add a `_load_cache` / `_save_cache` helper and update `fetch_ncaam_games` in `pipeline/fetch_ncaam.py`:

```python
import json as _json   # add to top-level imports

def _load_espn_cache(cache_path: str | None) -> dict:
    """Load ESPN cache from disk, returning empty cache if missing."""
    if cache_path is None or not os.path.exists(cache_path):
        return {"games": {}}
    with open(cache_path) as f:
        return _json.load(f)


def _save_espn_cache(cache_path: str | None, cache: dict) -> None:
    """Write ESPN cache to disk."""
    if cache_path is None:
        return
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        _json.dump(cache, f)


def _incremental_dates(cache: dict, all_dates: list[str], lookback_days: int = 2) -> list[str]:
    """Return only dates that need fetching based on cache contents.

    If cache is empty, returns all_dates (full season).
    Otherwise returns dates >= (max_cached_date - lookback_days).
    """
    games = cache.get("games", {})
    if not games:
        return all_dates
    max_cached = max(v["date"] for v in games.values())
    cutoff = (datetime.strptime(max_cached, "%Y-%m-%d") - timedelta(days=lookback_days)).date()
    return [d for d in all_dates if datetime.strptime(d, "%Y-%m-%d").date() >= cutoff]
```

Then update the `fetch_ncaam_games` signature and body:

```python
def fetch_ncaam_games(
    season: int | None = None,
    dates: list[str] | None = None,
    cache_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ...
    if season is None:
        now = datetime.now(timezone.utc)
        season = now.year if now.month >= 10 else now.year - 1

    if dates is None:
        dates = _season_date_range(season)

    cache = _load_espn_cache(cache_path)
    fetch_dates = _incremental_dates(cache, dates)

    game_rows = []
    box_rows = []

    for date_str in fetch_dates:
        espn_date = date_str.replace("-", "")
        url = f"{ESPN_BASE}/scoreboard?dates={espn_date}&limit=200"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        final_events = []
        for event in data.get("events", []):
            parsed = _parse_event(event)
            if parsed is None:
                continue
            final_events.append(parsed)
            game_id = parsed["event_id"]
            # Always update game result in cache (score may have been corrected)
            if game_id not in cache["games"]:
                cache["games"][game_id] = {
                    "date": parsed["date"],
                    "home_team": normalize_ncaam_team_name(parsed["home_name"]),
                    "away_team": normalize_ncaam_team_name(parsed["away_name"]),
                    "home_goals": parsed["home_score"],
                    "away_goals": parsed["away_score"],
                    "box_scores": [],
                }
            else:
                cache["games"][game_id]["home_goals"] = parsed["home_score"]
                cache["games"][game_id]["away_goals"] = parsed["away_score"]

        for parsed in final_events:
            game_id = parsed["event_id"]
            # Skip box score fetch if already cached
            if cache["games"][game_id].get("box_scores"):
                continue
            time.sleep(_REQUEST_DELAY)
            summary_url = f"{ESPN_BASE}/summary?event={game_id}"
            try:
                s_resp = requests.get(summary_url, timeout=30)
                s_resp.raise_for_status()
                s_data = s_resp.json()
                player_groups = s_data.get("boxscore", {}).get("players", [])
                if len(player_groups) < 2:
                    continue
                box_entries = []
                for player_group in player_groups:
                    try:
                        totals = player_group["statistics"][0]["totals"]
                        stats = _parse_box_score_totals(totals)
                        team_name = normalize_ncaam_team_name(player_group["team"]["displayName"])
                        box_entries.append({"team": team_name, **stats})
                    except (KeyError, IndexError):
                        continue
                cache["games"][game_id]["box_scores"] = box_entries
            except requests.RequestException:
                continue

        time.sleep(_REQUEST_DELAY)

    _save_espn_cache(cache_path, cache)

    # Build DataFrames from full cache (not just newly fetched dates)
    for game_id, entry in cache["games"].items():
        game_rows.append({
            "game_id": game_id,
            "date": entry["date"],
            "home_team": entry["home_team"],
            "away_team": entry["away_team"],
            "home_goals": entry["home_goals"],
            "away_goals": entry["away_goals"],
        })
        for bs in entry.get("box_scores", []):
            box_rows.append({
                "game_id": game_id,
                "team": bs["team"],
                "date": entry["date"],
                "pts": bs["pts"],
                "fgm": bs["fgm"],
                "fga": bs["fga"],
                "fg3m": bs["fg3m"],
                "fg3a": bs["fg3a"],
                "ftm": bs["ftm"],
                "fta": bs["fta"],
                "orb": bs["orb"],
                "drb": bs["drb"],
                "to": bs["to"],
                "possessions": bs["possessions"],
            })

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
```

Also add `import os` and `import json as _json` to the top of `fetch_ncaam.py` if not already present.

**Step 4: Run tests**

```bash
pytest tests/test_fetch_ncaam.py -v
```

Expected: all PASS

**Step 5: Commit**

```bash
git add pipeline/fetch_ncaam.py tests/test_fetch_ncaam.py
git commit -m "feat(ncaam): incremental ESPN fetch + box score cache"
```

---

## Task 3: Incremental fetch + box score cache for NBA

**Files:**
- Modify: `pipeline/fetch_nba.py`
- Modify: `tests/test_fetch_nba.py`

Same pattern as Task 2. The NBA functions live in `fetch_nba.py` and share the same ESPN structure.

**Step 1: Write failing tests**

Add to `tests/test_fetch_nba.py`:

```python
import json
import os

class TestFetchNbaEspnGamesCache:
    @patch("pipeline.fetch_nba.time.sleep")
    @patch("pipeline.fetch_nba.requests.get")
    def test_skips_box_score_for_cached_game(self, mock_get, mock_sleep, tmp_path):
        """Game already in cache → summary endpoint not called."""
        cache = {
            "games": {
                "1": {
                    "date": "2026-01-15",
                    "home_team": "Lakers",
                    "away_team": "Celtics",
                    "home_goals": 112,
                    "away_goals": 108,
                    "box_scores": [
                        {"team": "Lakers", "pts": 112, "fgm": 44, "fga": 95, "fg3m": 17,
                         "fg3a": 42, "ftm": 5, "fta": 7, "orb": 15, "drb": 34,
                         "to": 10, "possessions": 91.08},
                        {"team": "Celtics", "pts": 108, "fgm": 40, "fga": 90, "fg3m": 14,
                         "fg3a": 38, "ftm": 8, "fta": 10, "orb": 12, "drb": 32,
                         "to": 12, "possessions": 89.4},
                    ],
                }
            }
        }
        cache_path = tmp_path / "espn_cache.json"
        cache_path.write_text(json.dumps(cache))

        scoreboard_resp = MagicMock()
        scoreboard_resp.raise_for_status = MagicMock()
        scoreboard_resp.json.return_value = {
            "events": [_make_nba_espn_event("1", "Los Angeles Lakers", "Boston Celtics", 112, 108)]
        }
        mock_get.return_value = scoreboard_resp

        games_df, box_df = fetch_nba_espn_games(
            season=2025, dates=["2026-01-15"], cache_path=str(cache_path)
        )

        assert mock_get.call_count == 1  # scoreboard only, no summary
        assert "summary" not in str(mock_get.call_args_list[0])
        assert len(games_df) == 1
        assert len(box_df) == 2

    @patch("pipeline.fetch_nba.time.sleep")
    @patch("pipeline.fetch_nba.requests.get")
    def test_cache_written_after_fetch(self, mock_get, mock_sleep, tmp_path):
        """Cache file is created/updated after fetching new games."""
        cache_path = tmp_path / "espn_cache.json"

        scoreboard_resp = MagicMock()
        scoreboard_resp.raise_for_status = MagicMock()
        scoreboard_resp.json.return_value = {
            "events": [_make_nba_espn_event("99", "Los Angeles Lakers", "Boston Celtics", 112, 108)]
        }
        summary_resp = MagicMock()
        summary_resp.raise_for_status = MagicMock()
        summary_resp.json.return_value = _make_nba_summary(
            "Los Angeles Lakers", "Boston Celtics", NBA_SAMPLE_TOTALS, NBA_SAMPLE_TOTALS
        )
        mock_get.side_effect = [scoreboard_resp, summary_resp]

        fetch_nba_espn_games(season=2025, dates=["2026-01-15"], cache_path=str(cache_path))

        assert cache_path.exists()
        saved = json.loads(cache_path.read_text())
        assert "99" in saved["games"]
        assert saved["games"]["99"]["home_team"] == "Lakers"
        assert len(saved["games"]["99"]["box_scores"]) == 2

    @patch("pipeline.fetch_nba.time.sleep")
    @patch("pipeline.fetch_nba.requests.get")
    def test_only_recent_dates_fetched_when_cache_has_data(self, mock_get, mock_sleep, tmp_path):
        """Cache with max_date 2026-01-18 → only dates >= 2026-01-16 are fetched."""
        cache = {
            "games": {
                "55": {
                    "date": "2026-01-18",
                    "home_team": "Lakers",
                    "away_team": "Celtics",
                    "home_goals": 112,
                    "away_goals": 108,
                    "box_scores": [],
                }
            }
        }
        cache_path = tmp_path / "espn_cache.json"
        cache_path.write_text(json.dumps(cache))

        empty_resp = MagicMock()
        empty_resp.raise_for_status = MagicMock()
        empty_resp.json.return_value = {"events": []}
        mock_get.return_value = empty_resp

        all_dates = ["2026-01-14", "2026-01-15", "2026-01-16", "2026-01-17", "2026-01-18"]
        fetch_nba_espn_games(season=2025, dates=all_dates, cache_path=str(cache_path))

        assert mock_get.call_count == 3  # Jan 16, 17, 18
```

**Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_fetch_nba.py::TestFetchNbaEspnGamesCache -v
```

Expected: FAIL — `fetch_nba_espn_games() got unexpected keyword argument 'cache_path'`

**Step 3: Implement caching in `fetch_nba_espn_games`**

Copy `_load_espn_cache`, `_save_espn_cache`, and `_incremental_dates` from `fetch_ncaam.py` into `fetch_nba.py` (they are identical). Then update `fetch_nba_espn_games`:

```python
def fetch_nba_espn_games(
    season: int | None = None,
    dates: list[str] | None = None,
    cache_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ...
    if season is None:
        season = _current_nba_season()
    if dates is None:
        dates = _nba_season_date_range(season)

    cache = _load_espn_cache(cache_path)
    fetch_dates = _incremental_dates(cache, dates)

    game_rows = []
    box_rows = []

    for date_str in fetch_dates:
        espn_date = date_str.replace("-", "")
        url = f"{NBA_ESPN_BASE}/scoreboard?dates={espn_date}&limit=50&seasontype=2"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        final_events = []
        for event in data.get("events", []):
            parsed = _parse_nba_espn_event(event)
            if parsed is None:
                continue
            final_events.append(parsed)
            game_id = parsed["event_id"]
            if game_id not in cache["games"]:
                cache["games"][game_id] = {
                    "date": parsed["date"],
                    "home_team": normalize_nba_team_name(parsed["home_name"]),
                    "away_team": normalize_nba_team_name(parsed["away_name"]),
                    "home_goals": parsed["home_score"],
                    "away_goals": parsed["away_score"],
                    "box_scores": [],
                }
            else:
                cache["games"][game_id]["home_goals"] = parsed["home_score"]
                cache["games"][game_id]["away_goals"] = parsed["away_score"]

        for parsed in final_events:
            game_id = parsed["event_id"]
            if cache["games"][game_id].get("box_scores"):
                continue
            time.sleep(_ESPN_REQUEST_DELAY)
            summary_url = f"{NBA_ESPN_BASE}/summary?event={game_id}"
            try:
                s_resp = requests.get(summary_url, timeout=30)
                s_resp.raise_for_status()
                s_data = s_resp.json()
                player_groups = s_data.get("boxscore", {}).get("players", [])
                if len(player_groups) < 2:
                    continue
                box_entries = []
                for player_group in player_groups:
                    try:
                        totals = player_group["statistics"][0]["totals"]
                        stats = _parse_box_score_totals(totals)
                        team_name = normalize_nba_team_name(player_group["team"]["displayName"])
                        box_entries.append({"team": team_name, **stats})
                    except (KeyError, IndexError):
                        continue
                cache["games"][game_id]["box_scores"] = box_entries
            except requests.RequestException:
                continue

        time.sleep(_ESPN_REQUEST_DELAY)

    _save_espn_cache(cache_path, cache)

    # Build DataFrames from full cache
    for game_id, entry in cache["games"].items():
        game_rows.append({
            "game_id": game_id,
            "date": entry["date"],
            "home_team": entry["home_team"],
            "away_team": entry["away_team"],
            "home_goals": entry["home_goals"],
            "away_goals": entry["away_goals"],
        })
        for bs in entry.get("box_scores", []):
            box_rows.append({
                "game_id": game_id,
                "team": bs["team"],
                "date": entry["date"],
                "pts": bs["pts"],
                "fgm": bs["fgm"],
                "fga": bs["fga"],
                "fg3m": bs["fg3m"],
                "fg3a": bs["fg3a"],
                "ftm": bs["ftm"],
                "fta": bs["fta"],
                "orb": bs["orb"],
                "drb": bs["drb"],
                "to": bs["to"],
                "possessions": bs["possessions"],
            })

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
```

Also add `import os` and `import json as _json` if not already present.

**Step 4: Run tests**

```bash
pytest tests/test_fetch_nba.py -v
```

Expected: all PASS

**Step 5: Commit**

```bash
git add pipeline/fetch_nba.py tests/test_fetch_nba.py
git commit -m "feat(nba): incremental ESPN fetch + box score cache"
```

---

## Task 4: Wire cache_path into run.py

**Files:**
- Modify: `pipeline/run.py`

**Step 1: Pass cache_path to both fetch calls**

In `run_sport_pipeline`, the `sport_dir` variable holds the per-sport data directory. Pass a `cache_path` derived from it to both fetch calls.

Find the NBA branch (around line 479):
```python
    elif sport_key == "nba":
        games_df, box_scores_df = fetch_nba_espn_games()
```
Change to:
```python
    elif sport_key == "nba":
        games_df, box_scores_df = fetch_nba_espn_games(
            cache_path=os.path.join(sport_dir, "espn_cache.json")
        )
```

Find the NCAAM branch (around line 483):
```python
    elif sport_key == "ncaam":
        games_df, box_scores_df = fetch_ncaam_games()
```
Change to:
```python
    elif sport_key == "ncaam":
        games_df, box_scores_df = fetch_ncaam_games(
            cache_path=os.path.join(sport_dir, "espn_cache.json")
        )
```

**Step 2: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all PASS

**Step 3: Commit**

```bash
git add pipeline/run.py
git commit -m "feat(pipeline): wire espn_cache.json into nba and ncaam fetch"
```

---

## Task 5: Bootstrap caches from current season data and push

The first pipeline run after this change will still be slow — there's no cache yet so it fetches the full season. Bootstrap the caches locally so the next GitHub Actions run is fast immediately.

**Step 1: Run the pipeline locally (NBA + NCAAM only)**

```bash
ODDS_API_KEY=<key> FOOTBALL_DATA_API_KEY=<key> ANTHROPIC_API_KEY=<key> \
  python -c "
from pipeline.run import run_sport_pipeline
run_sport_pipeline('nba')
run_sport_pipeline('ncaam')
"
```

This will be slow (full season), but only needs to happen once. It creates `data/nba/espn_cache.json` and `data/ncaam/espn_cache.json`.

**Step 2: Verify cache files were created**

```bash
python3 -c "
import json
for sport in ['nba', 'ncaam']:
    with open(f'data/{sport}/espn_cache.json') as f:
        c = json.load(f)
    print(f'{sport}: {len(c[\"games\"])} games cached')
"
```

Expected output: something like:
```
nba: 847 games cached
ncaam: 4312 games cached
```

**Step 3: Commit and push cache files**

```bash
git add data/nba/espn_cache.json data/ncaam/espn_cache.json data/nba/predictions.json data/ncaam/predictions.json
git commit -m "data: bootstrap espn game caches for nba and ncaam"
git push
```

After this push, subsequent GitHub Actions runs will only fetch the last 2 days of scoreboard data instead of the full season.

---

## Verification

After the next scheduled or manual `daily.yml` run completes, check the Actions log. You should see:
- Python dependencies restored from cache (not re-downloaded)
- NBA scoreboard fetches: ~3 dates instead of ~140+
- NCAAM scoreboard fetches: ~3 dates instead of ~110+
- Total runtime: 3-5 minutes instead of 25-30

If runtime is still high, check whether the cache files are being committed and available at the start of the next run.
