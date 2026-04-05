from typing import Optional, Union
"""Fetch MLB game results, box scores, and schedule from ESPN."""

import json as _json
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from pipeline.config import MLB_BALLPARKS, MLB_CORE_API_BASE, MLB_ESPN_BASE, OPEN_METEO_BASE, SPORTS

_REQUEST_DELAY = 0.5

# ---- team-name normalisation ------------------------------------------------

_team_map: dict[str, str] | None = None


def _mlb_leader_weight(stat_name: Optional[str]) -> float:
    """Assign extra importance to live batting leader categories."""
    mapping = {
        "battingAverage": 0.75,
        "hits": 0.75,
        "onBasePercentage": 0.85,
        "sluggingPercentage": 0.9,
        "ops": 0.95,
        "homeRuns": 1.0,
        "rbi": 0.9,
        "runsBattedIn": 0.9,
    }
    return mapping.get(str(stat_name or "").strip(), 0.6)


def _lineup_slot_weight(slot: int) -> float:
    """Weight earlier batting-order spots more heavily than the bottom third."""
    weights = {
        1: 1.00,
        2: 0.98,
        3: 1.00,
        4: 0.96,
        5: 0.88,
        6: 0.78,
        7: 0.68,
        8: 0.6,
        9: 0.54,
    }
    return weights.get(int(slot), 0.5)


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

    Start date is configurable via SPORTS["mlb"] config (defaults to Feb 20
    to capture spring training as Elo warmup). Ends Nov 5.
    """
    mlb_cfg = SPORTS.get("mlb", {})
    start_month = mlb_cfg.get("season_start_month", 2)
    start_day = mlb_cfg.get("season_start_day", 20)
    start = datetime(season, start_month, start_day)
    end = min(datetime(season, 11, 5), datetime.now(timezone.utc).replace(tzinfo=None))
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


# ---- ESPN cache helpers -------------------------------------------------------


def _load_espn_cache(cache_path: Optional[str]) -> dict:
    """Load ESPN cache from disk, returning empty cache if missing."""
    if cache_path is None or not os.path.exists(cache_path):
        return {"games": {}, "pitchers": {}, "players": {}, "weather": {}, "rosters": {}}
    with open(cache_path) as f:
        cache = _json.load(f)
    if not isinstance(cache, dict):
        return {"games": {}, "pitchers": {}, "players": {}, "weather": {}, "rosters": {}}
    cache.setdefault("games", {})
    cache.setdefault("pitchers", {})
    cache.setdefault("players", {})
    cache.setdefault("weather", {})
    cache.setdefault("rosters", {})
    return cache


def _save_espn_cache(cache_path: Optional[str], cache: dict) -> None:
    """Write ESPN cache to disk."""
    if cache_path is None:
        return
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        _json.dump(cache, f)


def _fetch_player_profile(player_id: str | Optional[int], cache: Optional[dict] = None) -> dict:
    """Fetch and cache handedness metadata for an MLB player."""
    default = {"throws": None, "bats": None}
    if not player_id:
        return default

    cache_store = None
    if cache is not None:
        cache_store = cache.setdefault("players", {})
        cached = cache_store.get(str(player_id))
        if cached is not None:
            return cached

    url = f"{MLB_CORE_API_BASE}/athletes/{player_id}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return default

    profile = {
        "throws": data.get("throws", {}).get("abbreviation"),
        "bats": data.get("bats", {}).get("abbreviation"),
    }
    if cache_store is not None:
        cache_store[str(player_id)] = profile
    return profile


def _fetch_pitcher_profile(player_id: str | Optional[int], cache: Optional[dict] = None) -> dict:
    """Fetch and cache MLB pitcher handedness metadata from ESPN core API."""
    profile = _fetch_player_profile(player_id, cache)
    if cache is not None and player_id:
        cache.setdefault("pitchers", {})[str(player_id)] = profile
    return profile


def _fetch_team_lineup_profile(
    team_id: str | Optional[int],
    cache: Optional[dict] = None,
    leader_weights: dict[str, float] | None = None,
    confirmed_lineup: Optional[dict] = None,
) -> dict:
    """Fetch and cache a coarse current-roster lineup profile for one MLB team."""
    default = {
        "active_hitters": 0,
        "available_hitters": 0,
        "injured_hitters": 0,
        "key_bat_absence_score": 0.0,
        "leader_absence_burden": 0.0,
        "left_handed_batters": 0,
        "right_handed_batters": 0,
        "switch_hitters": 0,
        "lefty_share": 0.0,
        "righty_share": 0.0,
        "switch_share": 0.0,
        "confirmed_lineup": False,
        "confirmed_hitters": 0,
        "confirmed_top_order_score": 0.0,
        "confirmed_leader_absence_burden": 0.0,
        "confirmed_lefty_share": 0.0,
        "confirmed_righty_share": 0.0,
        "confirmed_switch_share": 0.0,
    }
    if not team_id:
        return default

    cache_store = None
    cache_key = str(team_id)
    player_rows = None
    if cache is not None:
        cache_store = cache.setdefault("rosters", {})
        cached = cache_store.get(cache_key)
        if isinstance(cached, dict) and cached.get("player_rows") is not None:
            player_rows = cached.get("player_rows")

    if player_rows is None:
        url = f"{MLB_ESPN_BASE}/teams/{team_id}/roster"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException:
            return default

        player_rows = []
        for group in data.get("athletes", []):
            if not isinstance(group, dict) or group.get("position") == "Pitchers":
                continue
            for athlete in group.get("items", []):
                if not isinstance(athlete, dict):
                    continue
                status_type = athlete.get("status", {}).get("type")
                injuries = athlete.get("injuries") or []
                is_active = status_type == "active"
                penalty = 0.0
                for injury in injuries:
                    penalty = max(penalty, 1.0 if str(injury.get("status", "")).strip().lower() in {"out", "doubtful"} else 0.35)
                bats = None
                if is_active and not injuries:
                    bats = _fetch_player_profile(athlete.get("id"), cache).get("bats")
                player_rows.append({
                    "id": str(athlete.get("id")),
                    "active": is_active,
                    "injured": bool(injuries),
                    "penalty": round(penalty, 3),
                    "bats": bats,
                })
        if cache_store is not None:
            cache_store[cache_key] = {"player_rows": player_rows}

    weight_map = {
        str(player_id): float(weight)
        for player_id, weight in (leader_weights or {}).items()
        if player_id
    }
    active_hitters = 0
    injured_hitters = 0
    available_hitters = 0
    key_bat_absence_score = 0.0
    leader_absence_burden = 0.0
    left_handed_batters = 0
    right_handed_batters = 0
    switch_hitters = 0

    for athlete in player_rows:
        is_active = bool(athlete.get("active"))
        if is_active:
            active_hitters += 1
        if athlete.get("injured"):
            injured_hitters += 1

        penalty = float(athlete.get("penalty", 0.0) or 0.0)
        leader_weight_value = float(weight_map.get(str(athlete.get("id")), 0.0) or 0.0)
        if penalty > 0:
            if leader_weight_value > 0:
                key_bat_absence_score += penalty
                leader_absence_burden += penalty * leader_weight_value
            continue
        if not is_active:
            continue

        available_hitters += 1
        bats = athlete.get("bats")
        if bats == "L":
            left_handed_batters += 1
        elif bats == "S":
            switch_hitters += 1
        else:
            right_handed_batters += 1

    denominator = max(1, available_hitters)
    profile = {
        "active_hitters": active_hitters,
        "available_hitters": available_hitters,
        "injured_hitters": injured_hitters,
        "key_bat_absence_score": round(key_bat_absence_score, 3),
        "leader_absence_burden": round(leader_absence_burden, 3),
        "left_handed_batters": left_handed_batters,
        "right_handed_batters": right_handed_batters,
        "switch_hitters": switch_hitters,
        "lefty_share": round(left_handed_batters / denominator, 4),
        "righty_share": round(right_handed_batters / denominator, 4),
        "switch_share": round(switch_hitters / denominator, 4),
    }

    if confirmed_lineup:
        confirmed_ids = {
            str(player_id)
            for player_id in confirmed_lineup.get("player_ids", [])
            if player_id
        }
        profile.update({
            "confirmed_lineup": bool(confirmed_lineup.get("confirmed_lineup")),
            "confirmed_hitters": int(confirmed_lineup.get("confirmed_hitters", 0) or 0),
            "confirmed_top_order_score": round(float(confirmed_lineup.get("confirmed_top_order_score", 0.0) or 0.0), 3),
            "confirmed_lefty_share": round(float(confirmed_lineup.get("confirmed_lefty_share", 0.0) or 0.0), 4),
            "confirmed_righty_share": round(float(confirmed_lineup.get("confirmed_righty_share", 0.0) or 0.0), 4),
            "confirmed_switch_share": round(float(confirmed_lineup.get("confirmed_switch_share", 0.0) or 0.0), 4),
            "confirmed_leader_absence_burden": round(
                sum(weight for player_id, weight in weight_map.items() if player_id not in confirmed_ids),
                3,
            ),
        })
    return profile


def _extract_confirmed_mlb_lineups(summary_data: dict, cache: Optional[dict] = None) -> dict[str, dict]:
    """Extract confirmed same-day batting-order profiles from an ESPN summary payload."""
    lineups = {}
    full_order_weight = sum(_lineup_slot_weight(slot) for slot in range(1, 10))

    for roster in summary_data.get("rosters", []):
        team_name = normalize_mlb_team_name(roster.get("team", {}).get("displayName", ""))
        if not team_name:
            continue

        confirmed_rows = []
        for player in roster.get("roster", []) or []:
            if not player.get("starter"):
                continue
            if str((player.get("position") or {}).get("abbreviation", "")).upper() == "P":
                continue
            bat_order = player.get("batOrder")
            if bat_order in (None, "", 0):
                continue
            try:
                slot = int(bat_order)
            except (TypeError, ValueError):
                continue
            athlete = player.get("athlete", {}) or {}
            player_id = athlete.get("id")
            bats = _fetch_player_profile(player_id, cache).get("bats")
            confirmed_rows.append({
                "slot": slot,
                "player_id": str(player_id) if player_id else None,
                "bats": bats,
            })

        if not confirmed_rows:
            continue

        confirmed_rows.sort(key=lambda row: row["slot"])
        confirmed_hitters = len(confirmed_rows)
        lefties = sum(1 for row in confirmed_rows if row.get("bats") == "L")
        righties = sum(1 for row in confirmed_rows if row.get("bats") == "R")
        switch_hitters = sum(1 for row in confirmed_rows if row.get("bats") == "S")
        slot_weight_total = sum(_lineup_slot_weight(row["slot"]) for row in confirmed_rows[:9])
        denom = max(1, confirmed_hitters)
        lineups[team_name] = {
            "confirmed_lineup": confirmed_hitters >= 8,
            "confirmed_hitters": confirmed_hitters,
            "confirmed_top_order_score": round(slot_weight_total / full_order_weight, 4),
            "confirmed_lefty_share": round(lefties / denom, 4),
            "confirmed_righty_share": round(righties / denom, 4),
            "confirmed_switch_share": round(switch_hitters / denom, 4),
            "player_ids": [row["player_id"] for row in confirmed_rows if row.get("player_id")],
        }
    return lineups


def _weather_cache_key(home_team: str, start_time: Optional[str]) -> str:
    """Build a stable weather-cache key for one MLB fixture."""
    return f"{home_team}|{str(start_time or '')[:13]}"


def _fetch_ballpark_weather(home_team: str, start_time: Optional[str], cache: Optional[dict] = None) -> dict:
    """Fetch hourly Open-Meteo weather for an MLB ballpark at first pitch."""
    park = MLB_BALLPARKS.get(home_team)
    if not park:
        return {}

    weather_exposed = bool(park.get("weather_exposed", True))
    if not weather_exposed:
        return {
            "weather_exposed": False,
            "temperature_f": None,
            "wind_mph": None,
            "precipitation_probability": None,
        }

    key = _weather_cache_key(home_team, start_time)
    cache_store = None
    if cache is not None:
        cache_store = cache.setdefault("weather", {})
        cached = cache_store.get(key)
        if cached is not None:
            return cached

    try:
        event_dt = datetime.fromisoformat(str(start_time).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return {"weather_exposed": True}

    try:
        resp = requests.get(
            OPEN_METEO_BASE,
            params={
                "latitude": park["latitude"],
                "longitude": park["longitude"],
                "hourly": "temperature_2m,wind_speed_10m,precipitation_probability",
                "forecast_days": 2,
                "timezone": "UTC",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return {"weather_exposed": True}

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return {"weather_exposed": True}

    target_hour = event_dt.replace(minute=0, second=0, microsecond=0)
    best_index = min(
        range(len(times)),
        key=lambda idx: abs(
            datetime.fromisoformat(times[idx]).replace(tzinfo=timezone.utc) - target_hour
        ),
    )

    weather = {
        "weather_exposed": True,
        "temperature_f": round((float(hourly.get("temperature_2m", [0])[best_index]) * 9.0 / 5.0) + 32.0, 1),
        "wind_mph": round(float(hourly.get("wind_speed_10m", [0])[best_index]) * 0.621371, 1),
        "precipitation_probability": int(round(float(hourly.get("precipitation_probability", [0])[best_index]))),
    }
    if cache_store is not None:
        cache_store[key] = weather
    return weather


def _incremental_dates(cache: dict, all_dates: list[str], lookback_days: int = 3) -> list[str]:
    """Return only dates that need fetching based on cache contents."""
    games = cache.get("games", {})
    if not games:
        return all_dates
    max_cached = max(v["date"] for v in games.values())
    cutoff = (datetime.strptime(max_cached, "%Y-%m-%d") - timedelta(days=lookback_days)).date()
    return [d for d in all_dates if datetime.strptime(d, "%Y-%m-%d").date() >= cutoff]


# ---- ESPN scoreboard / summary -----------------------------------------------


def _parse_event(event: dict) -> Optional[dict]:
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


def _innings_to_float(value) -> float:
    """Convert baseball innings notation (e.g. 5.1, 6.2) to decimal innings."""
    if value in (None, "", "-"):
        return 0.0
    text = str(value)
    if "." not in text:
        try:
            return float(text)
        except (TypeError, ValueError):
            return 0.0
    whole, frac = text.split(".", 1)
    try:
        innings = float(whole)
    except (TypeError, ValueError):
        return 0.0
    if frac == "1":
        return innings + (1.0 / 3.0)
    if frac == "2":
        return innings + (2.0 / 3.0)
    try:
        return float(text)
    except (TypeError, ValueError):
        return innings


def _safe_stat_int(value) -> int:
    """Parse ESPN stat values into ints, defaulting invalid cells to zero."""
    if value in (None, "", "-"):
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _extract_mlb_team_pitching(summary_data: dict) -> dict[str, dict]:
    """Extract starter and bullpen pitching context for each team."""
    teams: dict[str, dict] = {}
    for team_group in summary_data.get("boxscore", {}).get("players", []):
        team_name = normalize_mlb_team_name(team_group.get("team", {}).get("displayName", ""))
        pitching_group = None
        for stat_group in team_group.get("statistics", []):
            athletes = stat_group.get("athletes", [])
            if not athletes:
                continue
            pos = athletes[0].get("position", {})
            if pos.get("abbreviation") == "P":
                pitching_group = stat_group
                break
        if pitching_group is None:
            continue

        starter = None
        bullpen_athletes = []
        for athlete in pitching_group.get("athletes", []):
            if athlete.get("starter"):
                starter = athlete
            else:
                bullpen_athletes.append(athlete)

        starter_stats = {
            "name": "TBD",
            "innings_pitched": 0.0,
            "hits_allowed": 0,
            "runs_allowed": 0,
            "earned_runs": 0,
            "walks": 0,
            "strikeouts": 0,
        }
        if starter is not None:
            stats = starter.get("stats", [])
            starter_stats = {
                "id": starter.get("athlete", {}).get("id"),
                "name": starter.get("athlete", {}).get("displayName", "TBD"),
                "innings_pitched": _innings_to_float(stats[0] if len(stats) > 0 else 0.0),
                "hits_allowed": _safe_stat_int(stats[1] if len(stats) > 1 else 0),
                "runs_allowed": _safe_stat_int(stats[2] if len(stats) > 2 else 0),
                "earned_runs": _safe_stat_int(stats[3] if len(stats) > 3 else 0),
                "walks": _safe_stat_int(stats[4] if len(stats) > 4 else 0),
                "strikeouts": _safe_stat_int(stats[5] if len(stats) > 5 else 0),
            }

        bullpen_stats = {
            "innings_pitched": 0.0,
            "hits_allowed": 0,
            "runs_allowed": 0,
            "earned_runs": 0,
            "walks": 0,
            "strikeouts": 0,
        }
        for reliever in bullpen_athletes:
            stats = reliever.get("stats", [])
            bullpen_stats["innings_pitched"] += _innings_to_float(stats[0] if len(stats) > 0 else 0.0)
            bullpen_stats["hits_allowed"] += _safe_stat_int(stats[1] if len(stats) > 1 else 0)
            bullpen_stats["runs_allowed"] += _safe_stat_int(stats[2] if len(stats) > 2 else 0)
            bullpen_stats["earned_runs"] += _safe_stat_int(stats[3] if len(stats) > 3 else 0)
            bullpen_stats["walks"] += _safe_stat_int(stats[4] if len(stats) > 4 else 0)
            bullpen_stats["strikeouts"] += _safe_stat_int(stats[5] if len(stats) > 5 else 0)

        teams[team_name] = {
            "starter": starter_stats,
            "bullpen": bullpen_stats,
        }
    return teams


def _extract_mlb_starting_pitchers(summary_data: dict) -> dict[str, dict]:
    """Extract per-team starter names and basic pitching stats from ESPN summary."""
    return {
        team_name: sections["starter"]
        for team_name, sections in _extract_mlb_team_pitching(summary_data).items()
    }


def fetch_mlb_games(
    season: Optional[int] = None,
    dates: list[str] | None = None,
    cache_path: Optional[str] = None,
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

        final_events = []
        for event in data.get("events", []):
            parsed = _parse_event(event)
            if parsed is not None:
                game_id = parsed["event_id"]
                final_events.append(parsed)
                existing = cache["games"].get(game_id, {})
                cache["games"][game_id] = {
                    "date": parsed["date"],
                    "home_team": normalize_mlb_team_name(parsed["home_name"]),
                    "away_team": normalize_mlb_team_name(parsed["away_name"]),
                    "home_goals": parsed["home_score"],
                    "away_goals": parsed["away_score"],
                    "home_pitcher": existing.get("home_pitcher", "TBD"),
                    "away_pitcher": existing.get("away_pitcher", "TBD"),
                    "home_pitcher_stats": existing.get("home_pitcher_stats", {}),
                    "away_pitcher_stats": existing.get("away_pitcher_stats", {}),
                    "home_pitcher_hand": existing.get("home_pitcher_hand"),
                    "away_pitcher_hand": existing.get("away_pitcher_hand"),
                    "home_bullpen_stats": existing.get("home_bullpen_stats", {}),
                    "away_bullpen_stats": existing.get("away_bullpen_stats", {}),
                }

        for parsed in final_events:
            game_id = parsed["event_id"]
            entry = cache["games"].get(game_id, {})
            if (
                entry.get("home_pitcher")
                and entry.get("away_pitcher")
                and entry.get("home_pitcher") != "TBD"
                and entry.get("away_pitcher") != "TBD"
            ):
                continue

            time.sleep(_REQUEST_DELAY)
            summary_url = f"{MLB_ESPN_BASE}/summary?event={game_id}"
            try:
                s_resp = requests.get(summary_url, timeout=30)
                s_resp.raise_for_status()
                team_pitching = _extract_mlb_team_pitching(s_resp.json())
            except requests.RequestException:
                continue

            home_team = entry.get("home_team")
            away_team = entry.get("away_team")
            home_pitching = team_pitching.get(home_team, {})
            away_pitching = team_pitching.get(away_team, {})
            home_starter = home_pitching.get("starter", {"name": entry.get("home_pitcher", "TBD")})
            away_starter = away_pitching.get("starter", {"name": entry.get("away_pitcher", "TBD")})
            home_profile = _fetch_pitcher_profile(home_starter.get("id"), cache)
            away_profile = _fetch_pitcher_profile(away_starter.get("id"), cache)
            entry["home_pitcher"] = home_starter.get("name", "TBD")
            entry["away_pitcher"] = away_starter.get("name", "TBD")
            entry["home_pitcher_stats"] = {k: v for k, v in home_starter.items() if k not in {"name", "id"}}
            entry["away_pitcher_stats"] = {k: v for k, v in away_starter.items() if k not in {"name", "id"}}
            entry["home_pitcher_hand"] = home_profile.get("throws")
            entry["away_pitcher_hand"] = away_profile.get("throws")
            entry["home_bullpen_stats"] = home_pitching.get("bullpen", {})
            entry["away_bullpen_stats"] = away_pitching.get("bullpen", {})
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
            "home_pitcher": entry.get("home_pitcher", "TBD"),
            "away_pitcher": entry.get("away_pitcher", "TBD"),
            "home_pitcher_hand": entry.get("home_pitcher_hand"),
            "away_pitcher_hand": entry.get("away_pitcher_hand"),
            "home_pitcher_ip": entry.get("home_pitcher_stats", {}).get("innings_pitched", 0.0),
            "home_pitcher_runs_allowed": entry.get("home_pitcher_stats", {}).get("runs_allowed", 0),
            "home_pitcher_earned_runs": entry.get("home_pitcher_stats", {}).get("earned_runs", 0),
            "home_pitcher_walks": entry.get("home_pitcher_stats", {}).get("walks", 0),
            "home_pitcher_strikeouts": entry.get("home_pitcher_stats", {}).get("strikeouts", 0),
            "home_bullpen_ip": entry.get("home_bullpen_stats", {}).get("innings_pitched", 0.0),
            "home_bullpen_runs_allowed": entry.get("home_bullpen_stats", {}).get("runs_allowed", 0),
            "home_bullpen_earned_runs": entry.get("home_bullpen_stats", {}).get("earned_runs", 0),
            "home_bullpen_walks": entry.get("home_bullpen_stats", {}).get("walks", 0),
            "home_bullpen_strikeouts": entry.get("home_bullpen_stats", {}).get("strikeouts", 0),
            "away_pitcher_ip": entry.get("away_pitcher_stats", {}).get("innings_pitched", 0.0),
            "away_pitcher_runs_allowed": entry.get("away_pitcher_stats", {}).get("runs_allowed", 0),
            "away_pitcher_earned_runs": entry.get("away_pitcher_stats", {}).get("earned_runs", 0),
            "away_pitcher_walks": entry.get("away_pitcher_stats", {}).get("walks", 0),
            "away_pitcher_strikeouts": entry.get("away_pitcher_stats", {}).get("strikeouts", 0),
            "away_bullpen_ip": entry.get("away_bullpen_stats", {}).get("innings_pitched", 0.0),
            "away_bullpen_runs_allowed": entry.get("away_bullpen_stats", {}).get("runs_allowed", 0),
            "away_bullpen_earned_runs": entry.get("away_bullpen_stats", {}).get("earned_runs", 0),
            "away_bullpen_walks": entry.get("away_bullpen_stats", {}).get("walks", 0),
            "away_bullpen_strikeouts": entry.get("away_bullpen_stats", {}).get("strikeouts", 0),
        })

    games_df = pd.DataFrame(
        game_rows,
        columns=[
            "game_id", "date", "home_team", "away_team", "home_goals", "away_goals",
            "home_pitcher", "away_pitcher",
            "home_pitcher_hand", "away_pitcher_hand",
            "home_pitcher_ip", "home_pitcher_runs_allowed", "home_pitcher_earned_runs",
            "home_pitcher_walks", "home_pitcher_strikeouts",
            "home_bullpen_ip", "home_bullpen_runs_allowed", "home_bullpen_earned_runs",
            "home_bullpen_walks", "home_bullpen_strikeouts",
            "away_pitcher_ip", "away_pitcher_runs_allowed", "away_pitcher_earned_runs",
            "away_pitcher_walks", "away_pitcher_strikeouts",
            "away_bullpen_ip", "away_bullpen_runs_allowed", "away_bullpen_earned_runs",
            "away_bullpen_walks", "away_bullpen_strikeouts",
        ],
    )
    return games_df, None # Box scores not yet implemented for MLB


def fetch_mlb_schedule(cache_path: Optional[str] = None) -> list[dict]:
    """Fetch today's MLB games including probable pitchers."""
    cache = _load_espn_cache(cache_path)
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

        home_leader_weights = {}
        away_leader_weights = {}
        for leader_group in home.get("leaders", []):
            leaders = leader_group.get("leaders", [])
            if leaders:
                athlete_id = leaders[0].get("athlete", {}).get("id")
                if athlete_id:
                    home_leader_weights[str(athlete_id)] = (
                        home_leader_weights.get(str(athlete_id), 0.0)
                        + _mlb_leader_weight(leader_group.get("name"))
                    )
        for leader_group in away.get("leaders", []):
            leaders = leader_group.get("leaders", [])
            if leaders:
                athlete_id = leaders[0].get("athlete", {}).get("id")
                if athlete_id:
                    away_leader_weights[str(athlete_id)] = (
                        away_leader_weights.get(str(athlete_id), 0.0)
                        + _mlb_leader_weight(leader_group.get("name"))
                    )

        # MLB specific: probable pitchers
        home_pitcher = "TBD"
        away_pitcher = "TBD"
        home_pitcher_hand = None
        away_pitcher_hand = None
        for competitor in comp.get("competitors", []):
            prob = competitor.get("probables")
            if prob:
                player_id = prob[0].get("playerId") or prob[0].get("athlete", {}).get("id")
                p_name = prob[0].get("athlete", {}).get("displayName", "TBD")
                profile = _fetch_pitcher_profile(player_id, cache)
                if competitor.get("homeAway") == "home":
                    home_pitcher = p_name
                    home_pitcher_hand = profile.get("throws")
                elif competitor.get("homeAway") == "away":
                    away_pitcher = p_name
                    away_pitcher_hand = profile.get("throws")

        start_time = comp.get("date", event.get("date"))
        home_team_id = home["team"].get("id")
        away_team_id = away["team"].get("id")
        confirmed_lineups = {}
        summary_url = f"{MLB_ESPN_BASE}/summary?event={event.get('id')}"
        summary_injuries = []
        try:
            time.sleep(_REQUEST_DELAY)
            summary_resp = requests.get(summary_url, timeout=30)
            summary_resp.raise_for_status()
            summary_json = summary_resp.json()
            confirmed_lineups = _extract_confirmed_mlb_lineups(summary_json, cache)
            summary_injuries = summary_json.get("injuries", [])
        except requests.RequestException:
            confirmed_lineups = {}

        weather = _fetch_ballpark_weather(

            normalize_mlb_team_name(home["team"]["displayName"]),
            start_time,
            cache,
        )
        home_lineup_profile = _fetch_team_lineup_profile(
            home_team_id,
            cache,
            leader_weights=home_leader_weights,
            confirmed_lineup=confirmed_lineups.get(normalize_mlb_team_name(home["team"]["displayName"])),
        )
        away_lineup_profile = _fetch_team_lineup_profile(
            away_team_id,
            cache,
            leader_weights=away_leader_weights,
            confirmed_lineup=confirmed_lineups.get(normalize_mlb_team_name(away["team"]["displayName"])),
        )

        fixtures.append({
            "home_team": normalize_mlb_team_name(home["team"]["displayName"]),
            "away_team": normalize_mlb_team_name(away["team"]["displayName"]),
            "date": game_date_str,
            "start_time": start_time,
            "completed": is_completed,
            "neutral": comp.get("neutralSite", False),
            "home_pitcher": home_pitcher,
            "home_pitcher_hand": home_pitcher_hand,
            "away_pitcher": away_pitcher,
            "away_pitcher_hand": away_pitcher_hand,
            "home_lineup_profile": home_lineup_profile,
            "away_lineup_profile": away_lineup_profile,
            "weather": weather,
            "summary_injuries": summary_injuries,
        })

    _save_espn_cache(cache_path, cache)
    return fixtures
