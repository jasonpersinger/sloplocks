"""Fetch NHL results and schedule from ESPN."""

import json as _json
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from pipeline.config import NHL_ESPN_BASE

_REQUEST_DELAY = 0.25

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


def _parse_float(value, default=0.0) -> float:
    """Best-effort float parsing for ESPN stat payloads."""
    if value in (None, ""):
        return float(default)
    text = str(value).strip()
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return float(default)


def _extract_summary_team_stats(summary: dict) -> dict[str, dict]:
    """Extract team-level NHL summary stats keyed by normalized team name."""
    stats_by_team = {}
    for team_entry in summary.get("boxscore", {}).get("teams", []):
        team = team_entry.get("team", {})
        team_name = normalize_nhl_team_name(team.get("displayName", ""))
        if not team_name:
            continue
        stat_map = _stat_map(team_entry)
        stats_by_team[team_name] = {
            "shots": int(_parse_float(stat_map.get("shotsTotal"), 0.0)),
            "blocked_shots": int(_parse_float(stat_map.get("blockedShots"), 0.0)),
            "hits": int(_parse_float(stat_map.get("hits"), 0.0)),
            "takeaways": int(_parse_float(stat_map.get("takeaways"), 0.0)),
            "power_play_goals": int(_parse_float(stat_map.get("powerPlayGoals"), 0.0)),
            "power_play_opportunities": int(_parse_float(stat_map.get("powerPlayOpportunities"), 0.0)),
            "power_play_pct": _parse_float(stat_map.get("powerPlayPct"), 0.0) / 100.0,
            "faceoffs_won": int(_parse_float(stat_map.get("faceoffsWon"), 0.0)),
            "faceoff_pct": _parse_float(stat_map.get("faceoffPercent"), 50.0) / 100.0,
            "giveaways": int(_parse_float(stat_map.get("giveaways"), 0.0)),
            "penalties": int(_parse_float(stat_map.get("penalties"), 0.0)),
            "penalty_minutes": _parse_float(stat_map.get("penaltyMinutes"), 0.0),
        }
    return stats_by_team


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
        "home_blocked_shots": 0,
        "away_blocked_shots": 0,
        "home_hits": 0,
        "away_hits": 0,
        "home_takeaways": 0,
        "away_takeaways": 0,
        "home_power_play_goals": 0,
        "away_power_play_goals": 0,
        "home_power_play_opportunities": 0,
        "away_power_play_opportunities": 0,
        "home_power_play_pct": 0.0,
        "away_power_play_pct": 0.0,
        "home_faceoffs_won": 0,
        "away_faceoffs_won": 0,
        "home_faceoff_pct": 0.5,
        "away_faceoff_pct": 0.5,
        "home_giveaways": 0,
        "away_giveaways": 0,
        "home_penalties": 0,
        "away_penalties": 0,
        "home_penalty_minutes": 0.0,
        "away_penalty_minutes": 0.0,
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
            existing = cache["games"].get(event["id"], {})
            cache["games"][event["id"]] = {
                **parsed,
                "home_blocked_shots": existing.get("home_blocked_shots", parsed["home_blocked_shots"]),
                "away_blocked_shots": existing.get("away_blocked_shots", parsed["away_blocked_shots"]),
                "home_hits": existing.get("home_hits", parsed["home_hits"]),
                "away_hits": existing.get("away_hits", parsed["away_hits"]),
                "home_takeaways": existing.get("home_takeaways", parsed["home_takeaways"]),
                "away_takeaways": existing.get("away_takeaways", parsed["away_takeaways"]),
                "home_power_play_goals": existing.get("home_power_play_goals", parsed["home_power_play_goals"]),
                "away_power_play_goals": existing.get("away_power_play_goals", parsed["away_power_play_goals"]),
                "home_power_play_opportunities": existing.get(
                    "home_power_play_opportunities", parsed["home_power_play_opportunities"]
                ),
                "away_power_play_opportunities": existing.get(
                    "away_power_play_opportunities", parsed["away_power_play_opportunities"]
                ),
                "home_power_play_pct": existing.get("home_power_play_pct", parsed["home_power_play_pct"]),
                "away_power_play_pct": existing.get("away_power_play_pct", parsed["away_power_play_pct"]),
                "home_faceoffs_won": existing.get("home_faceoffs_won", parsed["home_faceoffs_won"]),
                "away_faceoffs_won": existing.get("away_faceoffs_won", parsed["away_faceoffs_won"]),
                "home_faceoff_pct": existing.get("home_faceoff_pct", parsed["home_faceoff_pct"]),
                "away_faceoff_pct": existing.get("away_faceoff_pct", parsed["away_faceoff_pct"]),
                "home_giveaways": existing.get("home_giveaways", parsed["home_giveaways"]),
                "away_giveaways": existing.get("away_giveaways", parsed["away_giveaways"]),
                "home_penalties": existing.get("home_penalties", parsed["home_penalties"]),
                "away_penalties": existing.get("away_penalties", parsed["away_penalties"]),
                "home_penalty_minutes": existing.get("home_penalty_minutes", parsed["home_penalty_minutes"]),
                "away_penalty_minutes": existing.get("away_penalty_minutes", parsed["away_penalty_minutes"]),
            }

        for event in data.get("events", []):
            game_id = event.get("id")
            entry = cache["games"].get(game_id)
            if not entry:
                continue
            if entry.get("home_faceoffs_won") or entry.get("away_faceoffs_won"):
                continue

            time.sleep(_REQUEST_DELAY)
            summary_url = f"{NHL_ESPN_BASE}/summary?event={game_id}"
            try:
                summary_resp = requests.get(summary_url, timeout=30)
                summary_resp.raise_for_status()
                summary = summary_resp.json()
            except requests.RequestException:
                continue

            team_stats = _extract_summary_team_stats(summary)
            home_stats = team_stats.get(entry["home_team"], {})
            away_stats = team_stats.get(entry["away_team"], {})
            for prefix, stats in (("home", home_stats), ("away", away_stats)):
                entry[f"{prefix}_shots"] = int(stats.get("shots", entry.get(f"{prefix}_shots", 0)))
                entry[f"{prefix}_blocked_shots"] = int(stats.get("blocked_shots", 0))
                entry[f"{prefix}_hits"] = int(stats.get("hits", 0))
                entry[f"{prefix}_takeaways"] = int(stats.get("takeaways", 0))
                entry[f"{prefix}_power_play_goals"] = int(stats.get("power_play_goals", 0))
                entry[f"{prefix}_power_play_opportunities"] = int(stats.get("power_play_opportunities", 0))
                entry[f"{prefix}_power_play_pct"] = float(stats.get("power_play_pct", 0.0))
                entry[f"{prefix}_faceoffs_won"] = int(stats.get("faceoffs_won", 0))
                entry[f"{prefix}_faceoff_pct"] = float(stats.get("faceoff_pct", 0.5))
                entry[f"{prefix}_giveaways"] = int(stats.get("giveaways", 0))
                entry[f"{prefix}_penalties"] = int(stats.get("penalties", 0))
                entry[f"{prefix}_penalty_minutes"] = float(stats.get("penalty_minutes", 0.0))

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
            "home_blocked_shots": entry.get("home_blocked_shots", 0),
            "away_blocked_shots": entry.get("away_blocked_shots", 0),
            "home_hits": entry.get("home_hits", 0),
            "away_hits": entry.get("away_hits", 0),
            "home_takeaways": entry.get("home_takeaways", 0),
            "away_takeaways": entry.get("away_takeaways", 0),
            "home_power_play_goals": entry.get("home_power_play_goals", 0),
            "away_power_play_goals": entry.get("away_power_play_goals", 0),
            "home_power_play_opportunities": entry.get("home_power_play_opportunities", 0),
            "away_power_play_opportunities": entry.get("away_power_play_opportunities", 0),
            "home_power_play_pct": entry.get("home_power_play_pct", 0.0),
            "away_power_play_pct": entry.get("away_power_play_pct", 0.0),
            "home_faceoffs_won": entry.get("home_faceoffs_won", 0),
            "away_faceoffs_won": entry.get("away_faceoffs_won", 0),
            "home_faceoff_pct": entry.get("home_faceoff_pct", 0.5),
            "away_faceoff_pct": entry.get("away_faceoff_pct", 0.5),
            "home_giveaways": entry.get("home_giveaways", 0),
            "away_giveaways": entry.get("away_giveaways", 0),
            "home_penalties": entry.get("home_penalties", 0),
            "away_penalties": entry.get("away_penalties", 0),
            "home_penalty_minutes": entry.get("home_penalty_minutes", 0.0),
            "away_penalty_minutes": entry.get("away_penalty_minutes", 0.0),
        })

    games_df = pd.DataFrame(
        rows,
        columns=[
            "game_id", "date", "home_team", "away_team", "home_goals", "away_goals",
            "home_saves", "away_saves", "home_save_pct", "away_save_pct", "home_shots", "away_shots",
            "home_blocked_shots", "away_blocked_shots", "home_hits", "away_hits",
            "home_takeaways", "away_takeaways", "home_power_play_goals", "away_power_play_goals",
            "home_power_play_opportunities", "away_power_play_opportunities",
            "home_power_play_pct", "away_power_play_pct",
            "home_faceoffs_won", "away_faceoffs_won", "home_faceoff_pct", "away_faceoff_pct",
            "home_giveaways", "away_giveaways", "home_penalties", "away_penalties",
            "home_penalty_minutes", "away_penalty_minutes",
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
