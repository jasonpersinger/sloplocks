"""Fetch NHL results and schedule from ESPN."""

import json as _json
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from pipeline.config import NHL_ESPN_BASE

_REQUEST_DELAY = 0.0

_team_map: dict[str, str] | None = None


def _build_team_map() -> dict[str, str]:
    """Fetch ESPN teams endpoint and build displayName -> shortDisplayName map."""
    url = f"{NHL_ESPN_BASE}/teams?limit=40"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    mapping = {}
    try:
        teams = data["sports"][0]["leagues"][0]["teams"]
        for entry in teams:
            team = entry["team"]
            mapping[team["displayName"]] = team["shortDisplayName"]
    except (KeyError, IndexError):
        pass
    return mapping


def normalize_nhl_team_name(name: str) -> str:
    """Map an ESPN full team name to its short display name."""
    global _team_map
    if _team_map is None:
        _team_map = _build_team_map()
    return _team_map.get(name, name)


def _current_nhl_season() -> int:
    """Return the start year of the current NHL season."""
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 9 else now.year - 1


def _season_date_range(season: int) -> list[str]:
    """Generate dates for one NHL season."""
    start = datetime(season, 10, 1)
    end = min(datetime(season + 1, 6, 30), datetime.now(timezone.utc).replace(tzinfo=None))
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


def _load_espn_cache(cache_path: str | None) -> dict:
    """Load cached ESPN game data."""
    if cache_path is None or not os.path.exists(cache_path):
        return {"games": {}}
    with open(cache_path) as f:
        cache = _json.load(f)
    if not isinstance(cache, dict):
        return {"games": {}}
    cache.setdefault("games", {})
    return cache


def _save_espn_cache(cache_path: str | None, cache: dict) -> None:
    """Write ESPN game cache."""
    if cache_path is None:
        return
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        _json.dump(cache, f)


def _incremental_dates(cache: dict, all_dates: list[str], lookback_days: int = 3) -> list[str]:
    """Return only dates that need fetching based on cached games."""
    games = cache.get("games", {})
    if not games:
        return all_dates
    max_cached = max(v["date"] for v in games.values())
    cutoff = (datetime.strptime(max_cached, "%Y-%m-%d") - timedelta(days=lookback_days)).date()
    return [d for d in all_dates if datetime.strptime(d, "%Y-%m-%d").date() >= cutoff]


def _stat_map(competitor: dict) -> dict[str, str]:
    """Convert competitor statistics list into a simple map."""
    return {
        item.get("name"): item.get("displayValue")
        for item in competitor.get("statistics", [])
        if item.get("name")
    }


def _parse_save_pct(value) -> float:
    """Parse ESPN save percentage strings like '.929' into floats."""
    if value in (None, ""):
        return 0.91
    text = str(value).strip()
    if text.startswith("."):
        text = "0" + text
    try:
        return float(text)
    except ValueError:
        return 0.91


def _parse_final_event(event: dict) -> dict | None:
    """Parse one final NHL scoreboard event into a cached game row."""
    comp = event.get("competitions", [{}])[0]
    status_type = comp.get("status", {}).get("type", {})
    if not status_type.get("completed", False):
        return None

    home = away = None
    for competitor in comp.get("competitors", []):
        if competitor.get("homeAway") == "home":
            home = competitor
        elif competitor.get("homeAway") == "away":
            away = competitor
    if home is None or away is None:
        return None

    home_stats = _stat_map(home)
    away_stats = _stat_map(away)
    home_goals = int(home.get("score", 0))
    away_goals = int(away.get("score", 0))
    home_saves = int(float(home_stats.get("saves", 0) or 0))
    away_saves = int(float(away_stats.get("saves", 0) or 0))

    return {
        "date": event["date"][:10],
        "home_team": normalize_nhl_team_name(home["team"]["displayName"]),
        "away_team": normalize_nhl_team_name(away["team"]["displayName"]),
        "home_goals": home_goals,
        "away_goals": away_goals,
        "home_saves": home_saves,
        "away_saves": away_saves,
        "home_save_pct": _parse_save_pct(home_stats.get("savePct")),
        "away_save_pct": _parse_save_pct(away_stats.get("savePct")),
        "home_shots": home_goals + away_saves,
        "away_shots": away_goals + home_saves,
    }


def fetch_nhl_games(
    season: int | None = None,
    dates: list[str] | None = None,
    cache_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch finished NHL games via ESPN scoreboard."""
    if season is None:
        season = _current_nhl_season()
    if dates is None:
        dates = _season_date_range(season)

    cache = _load_espn_cache(cache_path)
    fetch_dates = _incremental_dates(cache, dates)

    for date_str in fetch_dates:
        espn_date = date_str.replace("-", "")
        url = f"{NHL_ESPN_BASE}/scoreboard?dates={espn_date}&limit=50"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for event in data.get("events", []):
            parsed = _parse_final_event(event)
            if parsed is None:
                continue
            cache["games"][event["id"]] = parsed

    _save_espn_cache(cache_path, cache)

    rows = []
    for game_id, entry in cache["games"].items():
        rows.append({
            "game_id": game_id,
            "date": entry["date"],
            "home_team": entry["home_team"],
            "away_team": entry["away_team"],
            "home_goals": entry["home_goals"],
            "away_goals": entry["away_goals"],
            "home_saves": entry.get("home_saves", 0),
            "away_saves": entry.get("away_saves", 0),
            "home_save_pct": entry.get("home_save_pct", 0.91),
            "away_save_pct": entry.get("away_save_pct", 0.91),
            "home_shots": entry.get("home_shots", 0),
            "away_shots": entry.get("away_shots", 0),
        })

    games_df = pd.DataFrame(
        rows,
        columns=[
            "game_id", "date", "home_team", "away_team", "home_goals", "away_goals",
            "home_saves", "away_saves", "home_save_pct", "away_save_pct", "home_shots", "away_shots",
        ],
    )
    return games_df, None


def fetch_nhl_schedule(cache_path: str | None = None) -> list[dict]:
    """Fetch today's NHL games from ESPN."""
    et_offset = timedelta(hours=5)
    today_et = (datetime.now(timezone.utc) - et_offset).date()
    game_date_str = today_et.strftime("%Y-%m-%d")
    espn_date = today_et.strftime("%Y%m%d")

    url = f"{NHL_ESPN_BASE}/scoreboard?dates={espn_date}&limit=50"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    fixtures = []
    for event in data.get("events", []):
        comp = event.get("competitions", [{}])[0]
        status_type = comp.get("status", {}).get("type", {})
        is_completed = status_type.get("completed", False)

        home = away = None
        for competitor in comp.get("competitors", []):
            if competitor.get("homeAway") == "home":
                home = competitor
            elif competitor.get("homeAway") == "away":
                away = competitor
        if home is None or away is None:
            continue

        fixtures.append({
            "home_team": normalize_nhl_team_name(home["team"]["displayName"]),
            "away_team": normalize_nhl_team_name(away["team"]["displayName"]),
            "date": game_date_str,
            "start_time": comp.get("date", event.get("date")),
            "completed": is_completed,
            "neutral": comp.get("neutralSite", False),
        })

    return fixtures
