"""Fetch NFL game results and schedule from ESPN."""

import json as _json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

from pipeline.config import NFL_ESPN_BASE

_REQUEST_DELAY = 0.25

# ---- team-name normalisation ------------------------------------------------

_team_map: Optional[dict[str, str]] = None

# The Odds API publishes full "City Nickname" names, which usually match ESPN's
# displayName. These are the ones that do not.
_ODDS_API_FALLBACK: dict[str, str] = {
    "Washington Football Team": "Commanders",
    "Washington Redskins": "Commanders",
    "Oakland Raiders": "Raiders",
    "San Diego Chargers": "Chargers",
    "St. Louis Rams": "Rams",
}


def _build_team_map() -> dict[str, str]:
    """Fetch the ESPN teams endpoint and build displayName -> nickname map."""
    url = f"{NFL_ESPN_BASE}/teams?limit=50"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    mapping = {}
    try:
        teams = data["sports"][0]["leagues"][0]["teams"]
        for entry in teams:
            team = entry["team"]
            # ``name`` is the nickname ("Chiefs"); keep it short and stable so
            # odds names, ESPN names and stored picks all collapse to one key.
            short = team.get("name") or team.get("shortDisplayName") or team["displayName"]
            mapping[team["displayName"]] = short
            mapping[short] = short
    except (KeyError, IndexError):
        pass
    return mapping


def normalize_nfl_team_name(name: str) -> str:
    """Map an ESPN or Odds API NFL team name to its short nickname."""
    global _team_map
    if _team_map is None:
        _team_map = _build_team_map()
    mapped = _team_map.get(name)
    if mapped:
        return mapped
    return _ODDS_API_FALLBACK.get(name, name)


# ---- season handling ---------------------------------------------------------


def _current_nfl_season() -> int:
    """Return the start year of the current NFL season.

    An NFL season is named for the year it starts in, but runs into February.
    Anything before August belongs to the previous season's playoffs.
    """
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 8 else now.year - 1


def _season_date_range(season: int, history_seasons: int = 1) -> list[str]:
    """Generate dates covering one or more NFL seasons.

    ``history_seasons`` counts back from ``season`` inclusive, so 3 spans the
    current season plus the two before it. Prior seasons are what give Elo a
    real prior in September; without them every team sits at 1500 until
    roughly week 5 of a 17-game season.
    """
    start = datetime(season - max(0, history_seasons - 1), 8, 1)
    end = min(datetime(season + 1, 2, 20), datetime.now(timezone.utc).replace(tzinfo=None))
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


# ---- ESPN cache helpers -------------------------------------------------------


def _load_espn_cache(cache_path: Optional[str]) -> dict:
    """Load cached ESPN game data."""
    if cache_path is None or not os.path.exists(cache_path):
        return {"games": {}}
    with open(cache_path) as f:
        cache = _json.load(f)
    if not isinstance(cache, dict):
        return {"games": {}}
    cache.setdefault("games", {})
    return cache


def _save_espn_cache(cache_path: Optional[str], cache: dict) -> None:
    """Write the ESPN game cache."""
    if cache_path is None:
        return
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        _json.dump(cache, f)


def _incremental_dates(cache: dict, all_dates: list[str], lookback_days: int = 3) -> list[str]:
    """Return only the dates that still need fetching."""
    games = cache.get("games", {})
    if not games:
        return all_dates
    cached_dates = [v["date"] for v in games.values()]
    max_cached = max(cached_dates)
    min_cached = min(cached_dates)
    cutoff = (datetime.strptime(max_cached, "%Y-%m-%d") - timedelta(days=lookback_days)).date()
    earliest = datetime.strptime(min_cached, "%Y-%m-%d").date()
    # Fetch forward from the cache head, and also backfill anything older than
    # the cache covers, so raising ``history_seasons`` actually pulls the
    # earlier seasons in instead of silently doing nothing.
    return [
        d
        for d in all_dates
        if datetime.strptime(d, "%Y-%m-%d").date() >= cutoff
        or datetime.strptime(d, "%Y-%m-%d").date() < earliest
    ]


# ---- scoreboard parsing -------------------------------------------------------


def _week_metadata(event: dict, comp: dict) -> dict:
    """Extract NFL scheduling context (week number and season type).

    Season type 1 = preseason, 2 = regular season, 3 = postseason. Preseason
    results are excluded from the modelling set because rosters bear little
    relation to the teams that play in September.
    """
    season = event.get("season") or {}
    week = event.get("week") or {}
    return {
        "season_year": season.get("year"),
        "season_type": season.get("type"),
        "week": week.get("number"),
        "neutral": bool(comp.get("neutralSite", False)),
    }


def _parse_final_event(event: dict) -> Optional[dict]:
    """Parse an ESPN scoreboard event into a finished-game dict.

    Returns ``None`` for games that are not complete or are preseason. Final
    scores already include overtime, so no separate OT handling is needed; a
    tie (legal in the NFL regular season) is preserved as an equal scoreline
    and handled downstream as a non-home, non-away outcome.
    """
    competitions = event.get("competitions") or []
    if not competitions:
        return None
    comp = competitions[0]
    status_type = comp.get("status", {}).get("type", {})
    if not status_type.get("completed", False):
        return None

    meta = _week_metadata(event, comp)
    if meta["season_type"] == 1:
        return None

    home = away = None
    for competitor in comp.get("competitors", []):
        if competitor.get("homeAway") == "home":
            home = competitor
        elif competitor.get("homeAway") == "away":
            away = competitor
    if home is None or away is None:
        return None

    try:
        home_score = int(home["score"])
        away_score = int(away["score"])
    except (KeyError, TypeError, ValueError):
        return None

    return {
        "date": event["date"][:10],
        "home_team": normalize_nfl_team_name(home["team"]["displayName"]),
        "away_team": normalize_nfl_team_name(away["team"]["displayName"]),
        "home_goals": home_score,
        "away_goals": away_score,
        "overtime": bool(comp.get("status", {}).get("period", 0) > 4),
        **meta,
    }


def fetch_nfl_games(
    season: Optional[int] = None,
    dates: Optional[list[str]] = None,
    cache_path: Optional[str] = None,
    history_seasons: int = 1,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Fetch finished NFL games via the ESPN scoreboard.

    Returns
    -------
    (games_df, None)
        ``games_df`` columns: game_id, date, home_team, away_team, home_goals,
        away_goals, neutral, season_year, season_type, week, overtime.
        The second element is ``None``: the NFL pipeline runs no box-score
        model, matching the pipeline's convention for sports without one.
    """
    if season is None:
        season = _current_nfl_season()
    if dates is None:
        dates = _season_date_range(season, history_seasons=history_seasons)

    cache = _load_espn_cache(cache_path)
    fetch_dates = _incremental_dates(cache, dates)

    # The NFL plays on a handful of days per week, so request whole weeks in one
    # call rather than issuing a request per empty Tuesday.
    for chunk_start in range(0, len(fetch_dates), 7):
        chunk = fetch_dates[chunk_start:chunk_start + 7]
        if not chunk:
            continue
        span = f"{chunk[0].replace('-', '')}-{chunk[-1].replace('-', '')}"
        url = f"{NFL_ESPN_BASE}/scoreboard?dates={span}&limit=100"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError):
            continue

        for event in data.get("events", []):
            parsed = _parse_final_event(event)
            if parsed is None:
                continue
            cache["games"][event["id"]] = parsed

        time.sleep(_REQUEST_DELAY)

    _save_espn_cache(cache_path, cache)

    rows = [{"game_id": game_id, **entry} for game_id, entry in cache["games"].items()]
    columns = [
        "game_id", "date", "home_team", "away_team", "home_goals", "away_goals",
        "neutral", "season_year", "season_type", "week", "overtime",
    ]
    games_df = pd.DataFrame(rows, columns=columns)
    if not games_df.empty:
        games_df = games_df.sort_values("date").reset_index(drop=True)
    return games_df, None


def fetch_nfl_schedule(cache_path: Optional[str] = None) -> list[dict]:
    """Fetch upcoming NFL fixtures from ESPN.

    Unlike the daily-cadence sports, the NFL plays on scattered days, so this
    requests a date span rather than a single day. The span matches the window
    ``run_sport_pipeline`` keeps (yesterday through two days out), so a
    Thursday, Saturday, Sunday or Monday slate is picked up on the day it runs.

    ``cache_path`` is accepted for signature parity with the other schedule
    fetchers; the schedule is always fetched live.
    """
    et_offset = timedelta(hours=5)
    today_et = (datetime.now(timezone.utc) - et_offset).date()
    start = today_et - timedelta(days=1)
    end = today_et + timedelta(days=2)
    span = f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"

    url = f"{NFL_ESPN_BASE}/scoreboard?dates={span}&limit=100"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    fixtures = []
    for event in data.get("events", []):
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        comp = competitions[0]
        status_type = comp.get("status", {}).get("type", {})

        season_type = (event.get("season") or {}).get("type")
        if season_type == 1:
            continue

        home = away = None
        for competitor in comp.get("competitors", []):
            if competitor.get("homeAway") == "home":
                home = competitor
            elif competitor.get("homeAway") == "away":
                away = competitor
        if home is None or away is None:
            continue

        summary_injuries = []
        try:
            time.sleep(_REQUEST_DELAY)
            summary_resp = requests.get(
                f"{NFL_ESPN_BASE}/summary?event={event.get('id')}", timeout=30
            )
            if summary_resp.status_code == 200:
                summary_injuries = summary_resp.json().get("injuries", [])
        except (requests.RequestException, ValueError):
            summary_injuries = []

        start_time = comp.get("date", event.get("date"))
        fixtures.append({
            "home_team": normalize_nfl_team_name(home["team"]["displayName"]),
            "away_team": normalize_nfl_team_name(away["team"]["displayName"]),
            # Use the kickoff's Eastern date so late Sunday/Monday night games
            # are not filed under the following UTC day.
            "date": _eastern_date(start_time, fallback=today_et),
            "start_time": start_time,
            "completed": status_type.get("completed", False),
            # International and Super Bowl games carry no home-field edge.
            "neutral": bool(comp.get("neutralSite", False)),
            "week": (event.get("week") or {}).get("number"),
            "season_type": season_type,
            "summary_injuries": summary_injuries,
        })

    return fixtures


def _eastern_date(start_time: Optional[str], fallback) -> str:
    """Return the US-Eastern calendar date for a UTC kickoff timestamp."""
    if not start_time:
        return fallback.strftime("%Y-%m-%d")
    try:
        parsed = datetime.fromisoformat(str(start_time).replace("Z", "+00:00"))
    except ValueError:
        return fallback.strftime("%Y-%m-%d")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (parsed.astimezone(timezone.utc) - timedelta(hours=5)).strftime("%Y-%m-%d")
