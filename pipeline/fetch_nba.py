"""Fetch NBA game results and schedule from balldontlie.io."""

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from pipeline.config import BALLDONTLIE_API_KEY, BALLDONTLIE_BASE, NBA_ESPN_BASE
from pipeline.fetch_ncaam import _parse_box_score_totals

_ESPN_REQUEST_DELAY = 0.5

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
