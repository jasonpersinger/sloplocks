from typing import Optional, Union
"""Fetch NHL results and schedule from ESPN."""

import json as _json
import os
import re
import time
import unicodedata
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from pipeline.config import NHL_ESPN_BASE

_REQUEST_DELAY = 0.25

_team_map: dict[str,Optional[ str] ] = None
_NHL_ALIAS_MAP = {
    "montreal canadiens": "Canadiens",
    "montréal canadiens": "Canadiens",
    "st louis blues": "Blues",
    "st. louis blues": "Blues",
    "new jersey devils": "Devils",
    "utah hockey club": "Utah",
    "utah mammoth": "Utah",
}


def _nhl_leader_weight(stat_name: Optional[str]) -> float:
    """Assign extra importance to key NHL leader categories."""
    mapping = {
        "goals": 1.0,
        "points": 1.0,
        "assists": 0.85,
        "shots": 0.6,
        "savePct": 1.0,
    }
    return mapping.get(str(stat_name or "").strip(), 0.55)


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
    mapped = _team_map.get(name)
    if mapped:
        return mapped
    folded = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode("ascii")
    simplified = re.sub(r"[^a-z0-9]+", " ", folded.lower()).strip()
    return _NHL_ALIAS_MAP.get(simplified, name)


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


def _parse_toi_minutes(value) -> float:
    """Parse ESPN hockey TOI strings like '59:43' into decimal minutes."""
    if value in (None, ""):
        return 0.0
    text = str(value).strip()
    parts = text.split(":")
    if len(parts) != 2:
        return _parse_float(text, 0.0)
    minutes = _parse_float(parts[0], 0.0)
    seconds = _parse_float(parts[1], 0.0)
    return float(minutes + (seconds / 60.0))


def _extract_summary_goalies(summary: dict) -> dict[str, dict]:
    """Extract the lead goalie row for each team from an ESPN summary payload."""
    goalies_by_team = {}
    for team_group in summary.get("boxscore", {}).get("players", []):
        team_name = normalize_nhl_team_name(team_group.get("team", {}).get("displayName", ""))
        if not team_name:
            continue

        goalie_group = None
        for stat_group in team_group.get("statistics", []):
            if stat_group.get("name") == "goalies":
                goalie_group = stat_group
                break
        if goalie_group is None:
            continue

        labels = goalie_group.get("labels", [])
        goalie_rows = []
        for athlete_row in goalie_group.get("athletes", []):
            athlete = athlete_row.get("athlete", {})
            raw_stats = athlete_row.get("stats", [])
            stat_map = {
                label: raw_stats[idx]
                for idx, label in enumerate(labels)
                if idx < len(raw_stats)
            }
            goalie_rows.append({
                "goalie": athlete.get("displayName"),
                "goalie_id": athlete.get("id"),
                "goals_allowed": int(_parse_float(stat_map.get("GA"), 0.0)),
                "shots_against": int(_parse_float(stat_map.get("SA"), 0.0)),
                "saves": int(_parse_float(stat_map.get("SV"), 0.0)),
                "save_pct": _parse_save_pct(stat_map.get("SV%")),
                "time_on_ice_minutes": _parse_toi_minutes(stat_map.get("TOI")),
            })

        if not goalie_rows:
            continue
        goalie_rows.sort(
            key=lambda row: (
                row.get("time_on_ice_minutes", 0.0),
                row.get("saves", 0),
                row.get("shots_against", 0),
            ),
            reverse=True,
        )
        goalies_by_team[team_name] = goalie_rows[0]
    return goalies_by_team


def _extract_probable_goalie(competitor: dict) -> dict:
    """Extract one probable-starting-goalie descriptor from an ESPN competitor row."""
    for probable in competitor.get("probables", []) or []:
        if probable.get("name") != "probableStartingGoalie":
            continue
        athlete = probable.get("athlete", {}) or {}
        status = probable.get("status", {}) or {}
        return {
            "goalie": athlete.get("displayName") or probable.get("displayValue"),
            "goalie_id": athlete.get("id") or probable.get("playerId"),
            "goalie_status": status.get("type") or status.get("name"),
        }
    return {
        "goalie": None,
        "goalie_id": None,
        "goalie_status": None,
    }


def _extract_nhl_event_injury_profile(summary_data: dict, team_id: str | Optional[int], leader_weights: dict[str,Optional[ float] ] = None) -> dict:
    """Extract event-specific NHL skater injury burden from an ESPN summary payload."""
    default = {
        "injury_burden": 0.0,
        "uncertainty_burden": 0.0,
        "key_absence_score": 0.0,
        "leader_absence_burden": 0.0,
        "leader_uncertainty_burden": 0.0,
    }
    if not team_id:
        return default

    weight_map = {
        str(player_id): float(weight)
        for player_id, weight in (leader_weights or {}).items()
        if player_id
    }
    summary_team_id = str(team_id)
    profile = dict(default)
    for injury_block in summary_data.get("injuries", []) or []:
        if str((injury_block.get("team") or {}).get("id")) != summary_team_id:
            continue
        for injury in injury_block.get("injuries", []) or []:
            status = str(injury.get("status") or "").strip().lower()
            if status in {"out", "injured reserve", "injured reserve/out", "doubtful"}:
                weight = 1.0 if status in {"out", "injured reserve", "injured reserve/out"} else 0.7
            elif status in {"questionable", "day-to-day"}:
                weight = 0.35
            else:
                continue
            profile["injury_burden"] += weight
            athlete_id = str((injury.get("athlete") or {}).get("id") or "")
            leader_weight_value = float(weight_map.get(athlete_id, 0.0) or 0.0)
            if leader_weight_value > 0:
                profile["key_absence_score"] += weight
                profile["leader_absence_burden"] += weight * leader_weight_value
            if status in {"questionable", "day-to-day", "doubtful"}:
                profile["uncertainty_burden"] += weight
                if leader_weight_value > 0:
                    profile["leader_uncertainty_burden"] += weight * leader_weight_value
    return {key: round(value, 3) for key, value in profile.items()}


def _parse_final_event(event: dict) -> Optional[dict]:
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
        "home_goalie": None,
        "away_goalie": None,
        "home_goalie_id": None,
        "away_goalie_id": None,
        "home_goalie_goals_allowed": home_goals,
        "away_goalie_goals_allowed": away_goals,
        "home_goalie_shots_against": home_goals + away_saves,
        "away_goalie_shots_against": away_goals + home_saves,
        "home_goalie_saves": home_saves,
        "away_goalie_saves": away_saves,
        "home_goalie_save_pct": _parse_save_pct(home_stats.get("savePct")),
        "away_goalie_save_pct": _parse_save_pct(away_stats.get("savePct")),
        "home_goalie_toi_minutes": 60.0,
        "away_goalie_toi_minutes": 60.0,
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
    season: Optional[int] = None,
    dates:Optional[ list[str] ] = None,
    cache_path: Optional[str] = None,
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
                "home_goalie": existing.get("home_goalie", parsed["home_goalie"]),
                "away_goalie": existing.get("away_goalie", parsed["away_goalie"]),
                "home_goalie_id": existing.get("home_goalie_id", parsed["home_goalie_id"]),
                "away_goalie_id": existing.get("away_goalie_id", parsed["away_goalie_id"]),
                "home_goalie_goals_allowed": existing.get(
                    "home_goalie_goals_allowed", parsed["home_goalie_goals_allowed"]
                ),
                "away_goalie_goals_allowed": existing.get(
                    "away_goalie_goals_allowed", parsed["away_goalie_goals_allowed"]
                ),
                "home_goalie_shots_against": existing.get(
                    "home_goalie_shots_against", parsed["home_goalie_shots_against"]
                ),
                "away_goalie_shots_against": existing.get(
                    "away_goalie_shots_against", parsed["away_goalie_shots_against"]
                ),
                "home_goalie_saves": existing.get("home_goalie_saves", parsed["home_goalie_saves"]),
                "away_goalie_saves": existing.get("away_goalie_saves", parsed["away_goalie_saves"]),
                "home_goalie_save_pct": existing.get("home_goalie_save_pct", parsed["home_goalie_save_pct"]),
                "away_goalie_save_pct": existing.get("away_goalie_save_pct", parsed["away_goalie_save_pct"]),
                "home_goalie_toi_minutes": existing.get("home_goalie_toi_minutes", parsed["home_goalie_toi_minutes"]),
                "away_goalie_toi_minutes": existing.get("away_goalie_toi_minutes", parsed["away_goalie_toi_minutes"]),
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
            goalie_stats = _extract_summary_goalies(summary)
            home_stats = team_stats.get(entry["home_team"], {})
            away_stats = team_stats.get(entry["away_team"], {})
            home_goalie = goalie_stats.get(entry["home_team"], {})
            away_goalie = goalie_stats.get(entry["away_team"], {})
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
            for prefix, stats in (("home", home_goalie), ("away", away_goalie)):
                if not stats:
                    continue
                entry[f"{prefix}_goalie"] = stats.get("goalie")
                entry[f"{prefix}_goalie_id"] = stats.get("goalie_id")
                entry[f"{prefix}_goalie_goals_allowed"] = int(stats.get("goals_allowed", 0))
                entry[f"{prefix}_goalie_shots_against"] = int(stats.get("shots_against", 0))
                entry[f"{prefix}_goalie_saves"] = int(stats.get("saves", 0))
                entry[f"{prefix}_goalie_save_pct"] = float(stats.get("save_pct", 0.91))
                entry[f"{prefix}_goalie_toi_minutes"] = float(stats.get("time_on_ice_minutes", 0.0))

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
            "home_goalie": entry.get("home_goalie"),
            "away_goalie": entry.get("away_goalie"),
            "home_goalie_id": entry.get("home_goalie_id"),
            "away_goalie_id": entry.get("away_goalie_id"),
            "home_goalie_goals_allowed": entry.get("home_goalie_goals_allowed", entry.get("home_goals", 0)),
            "away_goalie_goals_allowed": entry.get("away_goalie_goals_allowed", entry.get("away_goals", 0)),
            "home_goalie_shots_against": entry.get("home_goalie_shots_against", entry.get("home_shots", 0)),
            "away_goalie_shots_against": entry.get("away_goalie_shots_against", entry.get("away_shots", 0)),
            "home_goalie_saves": entry.get("home_goalie_saves", entry.get("home_saves", 0)),
            "away_goalie_saves": entry.get("away_goalie_saves", entry.get("away_saves", 0)),
            "home_goalie_save_pct": entry.get("home_goalie_save_pct", entry.get("home_save_pct", 0.91)),
            "away_goalie_save_pct": entry.get("away_goalie_save_pct", entry.get("away_save_pct", 0.91)),
            "home_goalie_toi_minutes": entry.get("home_goalie_toi_minutes", 60.0),
            "away_goalie_toi_minutes": entry.get("away_goalie_toi_minutes", 60.0),
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
            "home_goalie", "away_goalie", "home_goalie_id", "away_goalie_id",
            "home_goalie_goals_allowed", "away_goalie_goals_allowed",
            "home_goalie_shots_against", "away_goalie_shots_against",
            "home_goalie_saves", "away_goalie_saves",
            "home_goalie_save_pct", "away_goalie_save_pct",
            "home_goalie_toi_minutes", "away_goalie_toi_minutes",
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


def fetch_nhl_schedule(cache_path: Optional[str] = None) -> list[dict]:
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

        home_leader_weights = {}
        away_leader_weights = {}
        for leader_group in home.get("leaders", []):
            leaders = leader_group.get("leaders", [])
            if leaders:
                athlete_id = leaders[0].get("athlete", {}).get("id")
                if athlete_id:
                    home_leader_weights[str(athlete_id)] = (
                        home_leader_weights.get(str(athlete_id), 0.0)
                        + _nhl_leader_weight(leader_group.get("name"))
                    )
        for leader_group in away.get("leaders", []):
            leaders = leader_group.get("leaders", [])
            if leaders:
                athlete_id = leaders[0].get("athlete", {}).get("id")
                if athlete_id:
                    away_leader_weights[str(athlete_id)] = (
                        away_leader_weights.get(str(athlete_id), 0.0)
                        + _nhl_leader_weight(leader_group.get("name"))
                    )

        home_probable = _extract_probable_goalie(home)
        away_probable = _extract_probable_goalie(away)
        summary_data = {}
        try:
            time.sleep(_REQUEST_DELAY)
            summary_resp = requests.get(f"{NHL_ESPN_BASE}/summary?event={event['id']}", timeout=30)
            summary_resp.raise_for_status()
            summary_data = summary_resp.json()
        except requests.RequestException:
            summary_data = {}

        fixtures.append({
            "home_team": normalize_nhl_team_name(home["team"]["displayName"]),
            "away_team": normalize_nhl_team_name(away["team"]["displayName"]),
            "date": game_date_str,
            "start_time": comp.get("date", event.get("date")),
            "completed": is_completed,
            "neutral": comp.get("neutralSite", False),
            "home_goalie": home_probable.get("goalie"),
            "away_goalie": away_probable.get("goalie"),
            "home_goalie_id": home_probable.get("goalie_id"),
            "away_goalie_id": away_probable.get("goalie_id"),
            "home_goalie_status": home_probable.get("goalie_status"),
            "away_goalie_status": away_probable.get("goalie_status"),
            "home_injury_profile": _extract_nhl_event_injury_profile(
                summary_data,
                home["team"].get("id"),
                leader_weights=home_leader_weights,
            ),
            "away_injury_profile": _extract_nhl_event_injury_profile(
                summary_data,
                away["team"].get("id"),
                leader_weights=away_leader_weights,
            ),
            "summary_injuries": summary_data.get("injuries", []),
        })

    return fixtures
