"""Fetch NBA game results and schedule from balldontlie.io."""

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from pipeline.config import BALLDONTLIE_API_KEY, BALLDONTLIE_BASE

_RATE_LIMIT_SLEEP = 12  # seconds between paginated requests (free tier: 5/min)

# ---- team-name normalisation ------------------------------------------------

_NBA_TEAM_NAME_MAP = {
    "Atlanta Hawks": "Hawks",
    "Boston Celtics": "Celtics",
    "Brooklyn Nets": "Nets",
    "Charlotte Hornets": "Hornets",
    "Chicago Bulls": "Bulls",
    "Cleveland Cavaliers": "Cavaliers",
    "Dallas Mavericks": "Mavericks",
    "Denver Nuggets": "Nuggets",
    "Detroit Pistons": "Pistons",
    "Golden State Warriors": "Warriors",
    "Houston Rockets": "Rockets",
    "Indiana Pacers": "Pacers",
    "Los Angeles Clippers": "Clippers",
    "LA Clippers": "Clippers",
    "Los Angeles Lakers": "Lakers",
    "Memphis Grizzlies": "Grizzlies",
    "Miami Heat": "Heat",
    "Milwaukee Bucks": "Bucks",
    "Minnesota Timberwolves": "Timberwolves",
    "New Orleans Pelicans": "Pelicans",
    "New York Knicks": "Knicks",
    "Oklahoma City Thunder": "Thunder",
    "Orlando Magic": "Magic",
    "Philadelphia 76ers": "76ers",
    "Phoenix Suns": "Suns",
    "Portland Trail Blazers": "Trail Blazers",
    "Sacramento Kings": "Kings",
    "San Antonio Spurs": "Spurs",
    "Toronto Raptors": "Raptors",
    "Utah Jazz": "Jazz",
    "Washington Wizards": "Wizards",
}


def normalize_nba_team_name(name: str) -> str:
    """Map a balldontlie.io full team name to its short display name."""
    return _NBA_TEAM_NAME_MAP.get(name, name)


# ---- rate-limit retry -------------------------------------------------------


def _request_with_retry(url, headers, params, max_retries=3):
    """Make a GET request, retrying on 429 using the Retry-After header."""
    for attempt in range(max_retries + 1):
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code != 429 or attempt == max_retries:
            resp.raise_for_status()
            return resp
        wait = int(resp.headers.get("Retry-After", _RATE_LIMIT_SLEEP)) + 1
        time.sleep(wait)
    return resp  # unreachable, but keeps linters happy


# ---- balldontlie.io ---------------------------------------------------------


def _current_nba_season() -> int:
    """Return the start year of the current NBA season.

    NBA seasons start in October, so Oct-Dec → current year, Jan-Sep → previous year.
    """
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 10 else now.year - 1


def fetch_nba_games(season: int | None = None) -> pd.DataFrame:
    """Fetch finished NBA games for a season.

    Parameters
    ----------
    season : int or None
        The season start year (e.g. 2025 for the 2025-26 season).
        Defaults to the current season.

    Returns
    -------
    pd.DataFrame
        Columns: date, home_team, away_team, home_goals, away_goals
        (goals = points, keeping schema consistent with EPL).
    """
    if season is None:
        season = _current_nba_season()
    url = f"{BALLDONTLIE_BASE}/games"
    headers = {"Authorization": BALLDONTLIE_API_KEY}

    rows = []
    cursor = None
    page = 0

    while True:
        params = {
            "seasons[]": season,
            "per_page": 100,
        }
        if cursor is not None:
            params["cursor"] = cursor

        resp = _request_with_retry(url, headers, params)
        data = resp.json()

        for game in data.get("data", []):
            if game.get("status") != "Final":
                continue

            home_score = game.get("home_team_score")
            visitor_score = game.get("visitor_team_score")
            if home_score is None or visitor_score is None:
                continue

            rows.append({
                "date": game["date"][:10],
                "home_team": normalize_nba_team_name(game["home_team"]["full_name"]),
                "away_team": normalize_nba_team_name(game["visitor_team"]["full_name"]),
                "home_goals": int(home_score),
                "away_goals": int(visitor_score),
            })

        meta = data.get("meta", {})
        cursor = meta.get("next_cursor")
        if cursor is None:
            break

        page += 1
        if page > 0:
            time.sleep(_RATE_LIMIT_SLEEP)

    return pd.DataFrame(
        rows,
        columns=["date", "home_team", "away_team", "home_goals", "away_goals"],
    )


def fetch_nba_schedule() -> list[dict]:
    """Fetch upcoming NBA games (today + next 7 days).

    Returns
    -------
    list[dict]
        Each dict has keys: home_team, away_team, date.
    """
    url = f"{BALLDONTLIE_BASE}/games"
    headers = {"Authorization": BALLDONTLIE_API_KEY}

    today = datetime.now(timezone.utc).date()
    end_date = today + timedelta(days=7)

    fixtures = []
    cursor = None

    while True:
        params = {
            "start_date": today.isoformat(),
            "end_date": end_date.isoformat(),
            "per_page": 100,
        }
        if cursor is not None:
            params["cursor"] = cursor

        resp = _request_with_retry(url, headers, params)
        data = resp.json()

        for game in data.get("data", []):
            if game.get("status") == "Final":
                continue

            fixtures.append({
                "home_team": normalize_nba_team_name(game["home_team"]["full_name"]),
                "away_team": normalize_nba_team_name(game["visitor_team"]["full_name"]),
                "date": game["date"][:10],
            })

        meta = data.get("meta", {})
        cursor = meta.get("next_cursor")
        if cursor is None:
            break

    return fixtures
