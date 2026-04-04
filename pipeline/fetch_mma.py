"""Fetch MMA (UFC) results and schedule from ESPN."""

import json as _json
import os
import re
import time
import unicodedata
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from pipeline.config import MMA_ESPN_BASE

_REQUEST_DELAY = 0.5

_MMA_NAME_ALIASES = {
    "abdul rakhman yakhyaev": "Abdulrakhman Yakhyaev",
    "abdulrakhman yakhyaev": "Abdulrakhman Yakhyaev",
    "charles radtke": "Charlie Radtke",
    "charlie radtke": "Charlie Radtke",
    "jiri prochazka": "Jiri Prochazka",
    "jose henrique": "Jose Henrique",
    "jose delano": "Jose Delano",
    "kai kamaka": "Kai Kamaka III",
    "kai kamaka iii": "Kai Kamaka III",
    "lando vannata": "Landon Vannata",
    "landon vannata": "Landon Vannata",
    "loopy godinez": "Lupita Godinez",
    "lupita godinez": "Lupita Godinez",
    "paulo costa": "Paulo Henrique Costa",
    "paulo henrique costa": "Paulo Henrique Costa",
    "robert ruchaa": "Robert Ruchala",
    "robert ruchala": "Robert Ruchala",
}

# ---- name normalisation -----------------------------------------------------

def _ascii_mma_name(name: str) -> str:
    """Return an ASCII-cleaned fighter name while preserving readable casing."""
    normalized = unicodedata.normalize("NFKD", name or "")
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_name = ascii_name.replace("\u2019", "'").replace("\u2018", "'")
    return " ".join(ascii_name.strip().split())


def _mma_name_key(name: str) -> str:
    """Return a canonical comparison key for fighter-name matching."""
    cleaned = _ascii_mma_name(name).lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    return " ".join(cleaned.split())


def normalize_mma_name(name: str) -> str:
    """Standardize fighter names."""
    display_name = _ascii_mma_name(name)
    if not display_name:
        return ""
    return _MMA_NAME_ALIASES.get(_mma_name_key(display_name), display_name)


def _competitor_display_name(competitor: dict) -> str:
    """Return the best available display name for a UFC competitor."""
    athlete_name = competitor.get("athlete", {}).get("displayName")
    if athlete_name:
        return normalize_mma_name(athlete_name)
    team_name = competitor.get("team", {}).get("displayName")
    if team_name:
        return normalize_mma_name(team_name)
    return ""


def _ordered_competitors(comp: dict) -> list[dict]:
    """Return the two competitors in a stable fight order.

    ESPN UFC payloads typically expose fighters as ``athlete`` entries with an
    ``order`` field rather than team-style ``homeAway`` markers.
    """
    competitors = [c for c in comp.get("competitors", []) if _competitor_display_name(c)]
    if any(c.get("homeAway") for c in competitors):
        competitors.sort(key=lambda c: 0 if c.get("homeAway") == "home" else 1)
    else:
        competitors.sort(key=lambda c: c.get("order", 99))
    return competitors[:2]


# ---- season date range -------------------------------------------------------


def _season_date_range(season: int) -> list[str]:
    """Generate list of YYYY-MM-DD date strings for a full year.
    MMA doesn't have a standard season, so we fetch the whole year.
    """
    start = datetime(season, 1, 1)
    # Don't fetch future dates beyond today
    end = min(datetime(season, 12, 31), datetime.now(timezone.utc).replace(tzinfo=None))
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=7) # MMA events are typically weekly, scoreboard often covers week
    return dates


# ---- ESPN cache helpers -------------------------------------------------------


def _load_espn_cache(cache_path: str | None) -> dict:
    """Load ESPN cache from disk."""
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


# ---- ESPN scoreboard / summary -----------------------------------------------


def _parse_event(event: dict) -> list[dict]:
    """MMA events contain multiple 'competitions' (fights) per event (card)."""
    fights = []
    if "competitions" not in event:
        return []

    for comp in event["competitions"]:
        status_type = comp.get("status", {}).get("type", {})
        if not status_type.get("completed", False):
            continue

        competitors = _ordered_competitors(comp)
        if len(competitors) != 2:
            continue
        home, away = competitors
        date_str = comp.get("date", event["date"])[:10]

        fights.append({
            "event_id": comp["id"],
            "date": date_str,
            "home_name": _competitor_display_name(home),
            "away_name": _competitor_display_name(away),
            "home_score": 1 if home.get("winner") else 0,
            "away_score": 1 if away.get("winner") else 0,
        })
    return fights


def fetch_mma_games(
    season: int | None = None,
    dates: list[str] | None = None,
    cache_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch finished UFC fights for a season."""
    if season is None:
        season = datetime.now(timezone.utc).year

    if dates is None:
        dates = _season_date_range(season)

    cache = _load_espn_cache(cache_path)
    
    # Simple logic for MMA: fetch every date in range once
    for date_str in dates:
        espn_date = date_str.replace("-", "")
        url = f"{MMA_ESPN_BASE}/scoreboard?dates={espn_date}&limit=100"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            continue

        for event in data.get("events", []):
            fights = _parse_event(event)
            for f in fights:
                game_id = f["event_id"]
                cache["games"][game_id] = {
                    "date": f["date"],
                    "home_team": normalize_mma_name(f["home_name"]),
                    "away_team": normalize_mma_name(f["away_name"]),
                    "home_goals": f["home_score"],
                    "away_goals": f["away_score"],
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
    return games_df, None


def fetch_mma_schedule() -> list[dict]:
    """Fetch upcoming UFC fights."""
    et_offset = timedelta(hours=5)
    today_et = (datetime.now(timezone.utc) - et_offset).date()
    start_date = (today_et - timedelta(days=1)).strftime("%Y%m%d")
    end_date = (today_et + timedelta(days=14)).strftime("%Y%m%d")
    url = f"{MMA_ESPN_BASE}/scoreboard?dates={start_date}-{end_date}&limit=100"
    
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    fixtures = []
    for event in data.get("events", []):
        if "competitions" not in event:
            continue

        for comp in event["competitions"]:
            status_type = comp.get("status", {}).get("type", {})
            is_completed = status_type.get("completed", False)
            competitors = _ordered_competitors(comp)
            if len(competitors) != 2:
                continue
            home, away = competitors
            date_str = comp.get("date", event["date"])[:10]

            fixtures.append({
                "home_team": _competitor_display_name(home),
                "away_team": _competitor_display_name(away),
                "date": date_str,
                "start_time": comp.get("date", event.get("date")),
                "completed": is_completed,
                "neutral": True, # Fights are always neutral site effectively
            })

    return fixtures
