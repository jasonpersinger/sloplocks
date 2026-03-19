"""Fetch MLB game results, box scores, and schedule from ESPN."""

import json as _json
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from pipeline.config import MLB_ESPN_BASE

_REQUEST_DELAY = 0.5

# ---- team-name normalisation ------------------------------------------------

_team_map: dict[str, str] | None = None


def _build_team_map() -> dict[str, str]:
    """Fetch ESPN teams endpoint and build displayName -> shortName map."""
    url = f"{MLB_ESPN_BASE}/teams?limit=40"
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


def normalize_mlb_team_name(name: str) -> str:
    """Map an ESPN full team name to its short display name."""
    global _team_map
    if _team_map is None:
        _team_map = _build_team_map()
    return _team_map.get(name, name)


# ---- season date range -------------------------------------------------------


def _season_date_range(season: int) -> list[str]:
    """Generate list of YYYY-MM-DD date strings for a MLB season.

    Season starts March 20, ends Nov 5.
    """
    start = datetime(season, 3, 20)
    end = min(datetime(season, 11, 5), datetime.now(timezone.utc).replace(tzinfo=None))
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


# ---- ESPN cache helpers -------------------------------------------------------


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


def _incremental_dates(cache: dict, all_dates: list[str], lookback_days: int = 3) -> list[str]:
    """Return only dates that need fetching based on cache contents."""
    games = cache.get("games", {})
    if not games:
        return all_dates
    max_cached = max(v["date"] for v in games.values())
    cutoff = (datetime.strptime(max_cached, "%Y-%m-%d") - timedelta(days=lookback_days)).date()
    return [d for d in all_dates if datetime.strptime(d, "%Y-%m-%d").date() >= cutoff]


# ---- ESPN scoreboard / summary -----------------------------------------------


def _parse_event(event: dict) -> dict | None:
    """Parse an ESPN scoreboard event into a game dict, or None if not final."""
    if "competitions" not in event or not event["competitions"]:
        return None
    comp = event["competitions"][0]
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

    date_str = event["date"][:10]

    return {
        "event_id": event["id"],
        "date": date_str,
        "home_name": home["team"]["displayName"],
        "away_name": away["team"]["displayName"],
        "home_score": int(home["score"]),
        "away_score": int(away["score"]),
    }


def fetch_mlb_games(
    season: int | None = None,
    dates: list[str] | None = None,
    cache_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch finished MLB games for a season."""
    if season is None:
        now = datetime.now(timezone.utc)
        season = now.year

    if dates is None:
        dates = _season_date_range(season)

    cache = _load_espn_cache(cache_path)
    fetch_dates = _incremental_dates(cache, dates)

    for date_str in fetch_dates:
        espn_date = date_str.replace("-", "")
        url = f"{MLB_ESPN_BASE}/scoreboard?dates={espn_date}&limit=100"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            continue

        for event in data.get("events", []):
            parsed = _parse_event(event)
            if parsed is not None:
                game_id = parsed["event_id"]
                cache["games"][game_id] = {
                    "date": parsed["date"],
                    "home_team": normalize_mlb_team_name(parsed["home_name"]),
                    "away_team": normalize_mlb_team_name(parsed["away_name"]),
                    "home_goals": parsed["home_score"],
                    "away_goals": parsed["away_score"],
                }
        time.sleep(_REQUEST_DELAY)

    _save_espn_cache(cache_path, cache)

    game_rows = []
    for game_id, entry in cache["games"].items():
        game_rows.append({
            "game_id": game_id,
            "date": entry["date"],
            "home_team": entry["home_team"],
            "away_team": entry["away_team"],
            "home_goals": entry["home_goals"],
            "away_goals": entry["away_goals"],
        })

    games_df = pd.DataFrame(
        game_rows,
        columns=["game_id", "date", "home_team", "away_team", "home_goals", "away_goals"],
    )
    return games_df, None # Box scores not yet implemented for MLB


def fetch_mlb_schedule() -> list[dict]:
    """Fetch today's MLB games including probable pitchers."""
    et_offset = timedelta(hours=5)
    today_et = (datetime.now(timezone.utc) - et_offset).date()
    game_date_str = today_et.strftime("%Y-%m-%d")
    espn_date = today_et.strftime("%Y%m%d")

    url = f"{MLB_ESPN_BASE}/scoreboard?dates={espn_date}&limit=100"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    fixtures = []
    for event in data.get("events", []):
        if "competitions" not in event or not event["competitions"]:
            continue
        comp = event["competitions"][0]
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

        # MLB specific: probable pitchers
        home_pitcher = "TBD"
        away_pitcher = "TBD"
        for competitor in comp.get("competitors", []):
            prob = competitor.get("probables")
            if prob:
                p_name = prob[0].get("athlete", {}).get("displayName", "TBD")
                if competitor.get("homeAway") == "home":
                    home_pitcher = p_name
                elif competitor.get("homeAway") == "away":
                    away_pitcher = p_name

        fixtures.append({
            "home_team": normalize_mlb_team_name(home["team"]["displayName"]),
            "away_team": normalize_mlb_team_name(away["team"]["displayName"]),
            "date": game_date_str,
            "completed": is_completed,
            "neutral": comp.get("neutralSite", False),
            "home_pitcher": home_pitcher,
            "away_pitcher": away_pitcher,
        })

    return fixtures
