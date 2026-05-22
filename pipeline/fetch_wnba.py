"""Fetch WNBA game results and schedule from ESPN."""

import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Union

import pandas as pd
import requests

from pipeline.config import WNBA_ESPN_BASE
from pipeline.fetch_nba import (
    _extract_nba_event_injury_profile,
    _incremental_dates,
    _injury_weight,
    _leader_weight,
    _load_espn_cache,
    _save_espn_cache,
)

_ESPN_REQUEST_DELAY = 0.5

_WNBA_TEAM_NAME_MAP = {
    "Atlanta Dream": "Dream",
    "Chicago Sky": "Sky",
    "Connecticut Sun": "Sun",
    "Dallas Wings": "Wings",
    "Golden State Valkyries": "Valkyries",
    "Indiana Fever": "Fever",
    "Las Vegas Aces": "Aces",
    "Los Angeles Sparks": "Sparks",
    "Minnesota Lynx": "Lynx",
    "New York Liberty": "Liberty",
    "Phoenix Mercury": "Mercury",
    "Portland Fire": "Fire",
    "Seattle Storm": "Storm",
    "Toronto Tempo": "Tempo",
    "Washington Mystics": "Mystics",
}


def normalize_wnba_team_name(name: str) -> str:
    """Map an ESPN or odds-provider WNBA full team name to its short name."""
    return _WNBA_TEAM_NAME_MAP.get(name, name)


def _current_wnba_season() -> int:
    """Return the current WNBA season year."""
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 4 else now.year - 1


def _wnba_season_date_range(season: int) -> list[str]:
    """Return YYYY-MM-DD strings covering the WNBA season to date."""
    start = date(season, 4, 1)
    end = min(date(season, 10, 20), datetime.now(timezone.utc).date())
    if end < start:
        return []

    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


def _default_availability_profile() -> dict:
    return {
        "active_players": 0,
        "injured_players": 0,
        "questionable_players": 0,
        "doubtful_players": 0,
        "injury_burden": 0.0,
        "uncertainty_burden": 0.0,
        "key_absence_score": 0.0,
        "leader_absence_burden": 0.0,
        "leader_uncertainty_burden": 0.0,
        "available_core_players": 0,
    }


def _iter_roster_athletes(roster_data: dict) -> list[dict]:
    athletes = roster_data.get("athletes", [])
    rows = []
    for item in athletes if isinstance(athletes, list) else []:
        if isinstance(item, dict) and isinstance(item.get("items"), list):
            rows.extend(athlete for athlete in item["items"] if isinstance(athlete, dict))
        elif isinstance(item, dict):
            rows.append(item)
    return rows


def _fetch_wnba_team_availability_profile(
    team_id: Union[str, int, None],
    leader_weights: Optional[dict[str, float]] = None,
    cache: Optional[dict] = None,
) -> dict:
    """Fetch and cache a coarse WNBA roster availability profile."""
    if not team_id:
        return _default_availability_profile()

    weight_map = {
        str(player_id): float(weight)
        for player_id, weight in (leader_weights or {}).items()
        if player_id
    }

    cache_store = None
    cache_key = str(team_id)
    roster_rows = None
    if cache is not None:
        cache_store = cache.setdefault("rosters", {})
        cached = cache_store.get(cache_key)
        if isinstance(cached, dict) and cached.get("player_rows") is not None:
            roster_rows = cached.get("player_rows")

    if roster_rows is None:
        try:
            resp = requests.get(f"{WNBA_ESPN_BASE}/teams/{team_id}/roster", timeout=30)
            resp.raise_for_status()
            roster_data = resp.json()
        except requests.RequestException:
            return _default_availability_profile()

        roster_rows = []
        for athlete in _iter_roster_athletes(roster_data):
            injuries = athlete.get("injuries") or []
            player_penalty = 0.0
            uncertain_penalty = 0.0
            for injury in injuries:
                status = injury.get("status")
                weight = _injury_weight(status)
                player_penalty = max(player_penalty, weight)
                if str(status or "").strip().lower() in {"questionable", "day-to-day", "doubtful"}:
                    uncertain_penalty = max(uncertain_penalty, weight)

            status_type = str((athlete.get("status") or {}).get("type") or "").lower()
            roster_rows.append({
                "id": str(athlete.get("id") or ""),
                "active": status_type == "active" or (status_type == "" and player_penalty == 0.0),
                "penalty": round(player_penalty, 3),
                "uncertain_penalty": round(uncertain_penalty, 3),
            })
        if cache_store is not None:
            cache_store[cache_key] = {"player_rows": roster_rows}

    profile = _default_availability_profile()
    for athlete in roster_rows:
        is_active = bool(athlete.get("active"))
        if is_active:
            profile["active_players"] += 1

        player_penalty = float(athlete.get("penalty", 0.0) or 0.0)
        uncertain_penalty = float(athlete.get("uncertain_penalty", 0.0) or 0.0)
        leader_weight_value = float(weight_map.get(str(athlete.get("id")), 0.0) or 0.0)
        if player_penalty > 0:
            profile["injured_players"] += 1
            profile["injury_burden"] += player_penalty
            if leader_weight_value > 0:
                profile["key_absence_score"] += player_penalty
                profile["leader_absence_burden"] += player_penalty * leader_weight_value
            if uncertain_penalty > 0:
                profile["uncertainty_burden"] += uncertain_penalty
                if uncertain_penalty >= 0.75:
                    profile["doubtful_players"] += 1
                else:
                    profile["questionable_players"] += 1
                if leader_weight_value > 0:
                    profile["leader_uncertainty_burden"] += uncertain_penalty * leader_weight_value
        elif is_active:
            profile["available_core_players"] += 1

    for key in (
        "injury_burden",
        "uncertainty_burden",
        "key_absence_score",
        "leader_absence_burden",
        "leader_uncertainty_burden",
    ):
        profile[key] = round(profile[key], 3)
    return profile


def _parse_wnba_espn_event(event: dict) -> Optional[dict]:
    """Parse an ESPN WNBA scoreboard event, returning None if not final."""
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
        "home_team_id": str(home["team"].get("id") or ""),
        "away_team_id": str(away["team"].get("id") or ""),
        "home_name": home["team"]["displayName"],
        "away_name": away["team"]["displayName"],
        "home_score": int(home["score"]),
        "away_score": int(away["score"]),
    }


def _safe_int(value, default: int = 0) -> int:
    text = str(value or "").strip()
    try:
        return int(float(text))
    except ValueError:
        return default


def _split_made_attempted(stats: dict, key: str) -> tuple[int, int]:
    value = str(stats.get(key, "0-0") or "0-0")
    parts = value.split("-")
    made = _safe_int(parts[0] if parts else 0)
    attempted = _safe_int(parts[1] if len(parts) > 1 else 0)
    return made, attempted


def _parse_wnba_box_score(summary_data: dict, event_id: str, game_date: str, parsed: dict) -> list[dict]:
    """Extract team-level box score rows from an ESPN WNBA summary payload."""
    names_by_id = {
        parsed["home_team_id"]: normalize_wnba_team_name(parsed["home_name"]),
        parsed["away_team_id"]: normalize_wnba_team_name(parsed["away_name"]),
    }
    score_by_id = {
        parsed["home_team_id"]: parsed["home_score"],
        parsed["away_team_id"]: parsed["away_score"],
    }

    rows = []
    for team_block in summary_data.get("boxscore", {}).get("teams", []):
        team = team_block.get("team") or {}
        team_id = str(team.get("id") or "")
        team_name = names_by_id.get(team_id) or normalize_wnba_team_name(team.get("displayName", ""))
        if not team_name:
            continue

        stats = {
            stat.get("name"): stat.get("displayValue", "")
            for stat in team_block.get("statistics", [])
            if isinstance(stat, dict)
        }
        fgm, fga = _split_made_attempted(stats, "fieldGoalsMade-fieldGoalsAttempted")
        fg3m, fg3a = _split_made_attempted(stats, "threePointFieldGoalsMade-threePointFieldGoalsAttempted")
        ftm, fta = _split_made_attempted(stats, "freeThrowsMade-freeThrowsAttempted")
        orb = _safe_int(stats.get("offensiveRebounds"))
        drb = _safe_int(stats.get("defensiveRebounds"))
        turnovers = _safe_int(stats.get("turnovers"))
        possessions = fga - orb + turnovers + 0.44 * fta
        points = _safe_int(stats.get("points"), score_by_id.get(team_id, 0))
        if points == 0 and (fgm or fg3m or ftm):
            points = (fgm * 2) + fg3m + ftm

        rows.append({
            "game_id": event_id,
            "team": team_name,
            "date": game_date,
            "pts": points,
            "fgm": fgm,
            "fga": fga,
            "fg3m": fg3m,
            "fg3a": fg3a,
            "ftm": ftm,
            "fta": fta,
            "orb": orb,
            "drb": drb,
            "to": turnovers,
            "possessions": round(possessions, 2),
        })
    return rows


def fetch_wnba_espn_games(
    season: Optional[int] = None,
    dates: Optional[list[str]] = None,
    cache_path: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch finished WNBA games and team box scores via ESPN API."""
    if season is None:
        season = _current_wnba_season()
    if dates is None:
        dates = _wnba_season_date_range(season)

    cache = _load_espn_cache(cache_path)
    fetch_dates = _incremental_dates(cache, dates)

    if fetch_dates:
        print(f"[*] Fetching WNBA box scores for {len(fetch_dates)} dates...")

    for date_str in fetch_dates:
        print(f"  - {date_str}...", end=" ", flush=True)
        resp = requests.get(
            f"{WNBA_ESPN_BASE}/scoreboard?dates={date_str.replace('-', '')}&limit=50",
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        final_events = []
        for event in data.get("events", []):
            parsed = _parse_wnba_espn_event(event)
            if parsed is None:
                continue
            final_events.append(parsed)
            game_id = parsed["event_id"]
            existing = cache["games"].get(game_id, {})
            cache["games"][game_id] = {
                "date": parsed["date"],
                "home_team": normalize_wnba_team_name(parsed["home_name"]),
                "away_team": normalize_wnba_team_name(parsed["away_name"]),
                "home_goals": parsed["home_score"],
                "away_goals": parsed["away_score"],
                "box_scores": existing.get("box_scores", []),
            }

        for parsed in final_events:
            game_id = parsed["event_id"]
            if cache["games"][game_id].get("box_scores"):
                continue
            try:
                time.sleep(_ESPN_REQUEST_DELAY)
                summary_resp = requests.get(f"{WNBA_ESPN_BASE}/summary?event={game_id}", timeout=30)
                summary_resp.raise_for_status()
                summary_data = summary_resp.json()
            except requests.RequestException:
                continue
            cache["games"][game_id]["box_scores"] = _parse_wnba_box_score(
                summary_data,
                game_id,
                parsed["date"],
                parsed,
            )

        print("DONE")
        time.sleep(_ESPN_REQUEST_DELAY)

    _save_espn_cache(cache_path, cache)

    game_rows = []
    box_rows = []
    for game_id, entry in cache.get("games", {}).items():
        game_rows.append({
            "game_id": game_id,
            "date": entry["date"],
            "home_team": entry["home_team"],
            "away_team": entry["away_team"],
            "home_goals": entry["home_goals"],
            "away_goals": entry["away_goals"],
        })
        for box_score in entry.get("box_scores", []):
            box_rows.append({
                "game_id": game_id,
                "team": box_score["team"],
                "date": entry["date"],
                "pts": box_score["pts"],
                "fgm": box_score["fgm"],
                "fga": box_score["fga"],
                "fg3m": box_score["fg3m"],
                "fg3a": box_score["fg3a"],
                "ftm": box_score["ftm"],
                "fta": box_score["fta"],
                "orb": box_score["orb"],
                "drb": box_score["drb"],
                "to": box_score["to"],
                "possessions": box_score["possessions"],
            })

    games_df = pd.DataFrame(
        game_rows,
        columns=["game_id", "date", "home_team", "away_team", "home_goals", "away_goals"],
    )
    box_df = pd.DataFrame(
        box_rows,
        columns=[
            "game_id", "team", "date", "pts", "fgm", "fga", "fg3m", "fg3a",
            "ftm", "fta", "orb", "drb", "to", "possessions",
        ],
    )
    return games_df, box_df


def _leader_weights(competitor: dict) -> dict[str, float]:
    weights = {}
    for leader_group in competitor.get("leaders", []):
        leaders = leader_group.get("leaders", [])
        if not leaders:
            continue
        athlete_id = leaders[0].get("athlete", {}).get("id")
        if athlete_id:
            weights[str(athlete_id)] = weights.get(str(athlete_id), 0.0) + _leader_weight(leader_group.get("name"))
    return weights


def fetch_wnba_espn_schedule(cache_path: Optional[str] = None) -> list[dict]:
    """Fetch today's WNBA games via ESPN API."""
    et_offset = timedelta(hours=5)
    today_et = (datetime.now(timezone.utc) - et_offset).date()
    game_date_str = today_et.strftime("%Y-%m-%d")

    cache = _load_espn_cache(cache_path)
    resp = requests.get(
        f"{WNBA_ESPN_BASE}/scoreboard?dates={today_et.strftime('%Y%m%d')}&limit=50",
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    fixtures = []
    events = data.get("events", [])
    if events:
        print(f"[*] Fetching WNBA summaries for {len(events)} matchups...")

    for event in events:
        comp = event["competitions"][0]
        status_type = comp.get("status", {}).get("type", {})
        is_completed = status_type.get("completed", False)

        home = away = None
        for competitor in comp["competitors"]:
            if competitor["homeAway"] == "home":
                home = competitor
            else:
                away = competitor
        if home is None or away is None:
            continue

        print(f"  - {away['team']['displayName']} @ {home['team']['displayName']}...", end=" ", flush=True)
        home_leader_weights = _leader_weights(home)
        away_leader_weights = _leader_weights(away)

        summary_data = {}
        try:
            time.sleep(_ESPN_REQUEST_DELAY)
            summary_resp = requests.get(f"{WNBA_ESPN_BASE}/summary?event={event['id']}", timeout=30)
            summary_resp.raise_for_status()
            summary_data = summary_resp.json()
        except requests.RequestException:
            summary_data = {}

        home_profile = _fetch_wnba_team_availability_profile(
            home["team"].get("id"),
            leader_weights=home_leader_weights,
            cache=cache,
        )
        away_profile = _fetch_wnba_team_availability_profile(
            away["team"].get("id"),
            leader_weights=away_leader_weights,
            cache=cache,
        )
        home_profile.update(
            _extract_nba_event_injury_profile(summary_data, home["team"].get("id"), leader_weights=home_leader_weights)
        )
        away_profile.update(
            _extract_nba_event_injury_profile(summary_data, away["team"].get("id"), leader_weights=away_leader_weights)
        )

        fixtures.append({
            "home_team": normalize_wnba_team_name(home["team"]["displayName"]),
            "away_team": normalize_wnba_team_name(away["team"]["displayName"]),
            "date": game_date_str,
            "start_time": comp.get("date", event.get("date")),
            "completed": is_completed,
            "neutral": comp.get("neutralSite", False),
            "home_availability_profile": home_profile,
            "away_availability_profile": away_profile,
            "summary_injuries": summary_data.get("injuries", []),
        })
        print("DONE")

    _save_espn_cache(cache_path, cache)
    return fixtures
