"""Fetch MLB game results, box scores, and schedule from ESPN."""

import json as _json
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from pipeline.config import MLB_ESPN_BASE, SPORTS

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


def _extract_mlb_starting_pitchers(summary_data: dict) -> dict[str, dict]:
    """Extract per-team starter names and basic pitching stats from ESPN summary."""
    starters: dict[str, dict] = {}
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
        for athlete in pitching_group.get("athletes", []):
            if athlete.get("starter"):
                starter = athlete
                break
        if starter is None:
            continue

        stats = starter.get("stats", [])
        starters[team_name] = {
            "name": starter.get("athlete", {}).get("displayName", "TBD"),
            "innings_pitched": _innings_to_float(stats[0] if len(stats) > 0 else 0.0),
            "hits_allowed": int(float(stats[1])) if len(stats) > 1 and str(stats[1]).replace(".", "", 1).isdigit() else 0,
            "runs_allowed": int(float(stats[2])) if len(stats) > 2 and str(stats[2]).replace(".", "", 1).isdigit() else 0,
            "earned_runs": int(float(stats[3])) if len(stats) > 3 and str(stats[3]).replace(".", "", 1).isdigit() else 0,
            "walks": int(float(stats[4])) if len(stats) > 4 and str(stats[4]).replace(".", "", 1).isdigit() else 0,
            "strikeouts": int(float(stats[5])) if len(stats) > 5 and str(stats[5]).replace(".", "", 1).isdigit() else 0,
        }
    return starters


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
                starters = _extract_mlb_starting_pitchers(s_resp.json())
            except requests.RequestException:
                continue

            home_team = entry.get("home_team")
            away_team = entry.get("away_team")
            home_starter = starters.get(home_team, {"name": entry.get("home_pitcher", "TBD")})
            away_starter = starters.get(away_team, {"name": entry.get("away_pitcher", "TBD")})
            entry["home_pitcher"] = home_starter.get("name", "TBD")
            entry["away_pitcher"] = away_starter.get("name", "TBD")
            entry["home_pitcher_stats"] = {k: v for k, v in home_starter.items() if k != "name"}
            entry["away_pitcher_stats"] = {k: v for k, v in away_starter.items() if k != "name"}
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
            "home_pitcher_ip": entry.get("home_pitcher_stats", {}).get("innings_pitched", 0.0),
            "home_pitcher_runs_allowed": entry.get("home_pitcher_stats", {}).get("runs_allowed", 0),
            "home_pitcher_earned_runs": entry.get("home_pitcher_stats", {}).get("earned_runs", 0),
            "home_pitcher_walks": entry.get("home_pitcher_stats", {}).get("walks", 0),
            "home_pitcher_strikeouts": entry.get("home_pitcher_stats", {}).get("strikeouts", 0),
            "away_pitcher_ip": entry.get("away_pitcher_stats", {}).get("innings_pitched", 0.0),
            "away_pitcher_runs_allowed": entry.get("away_pitcher_stats", {}).get("runs_allowed", 0),
            "away_pitcher_earned_runs": entry.get("away_pitcher_stats", {}).get("earned_runs", 0),
            "away_pitcher_walks": entry.get("away_pitcher_stats", {}).get("walks", 0),
            "away_pitcher_strikeouts": entry.get("away_pitcher_stats", {}).get("strikeouts", 0),
        })

    games_df = pd.DataFrame(
        game_rows,
        columns=[
            "game_id", "date", "home_team", "away_team", "home_goals", "away_goals",
            "home_pitcher", "away_pitcher",
            "home_pitcher_ip", "home_pitcher_runs_allowed", "home_pitcher_earned_runs",
            "home_pitcher_walks", "home_pitcher_strikeouts",
            "away_pitcher_ip", "away_pitcher_runs_allowed", "away_pitcher_earned_runs",
            "away_pitcher_walks", "away_pitcher_strikeouts",
        ],
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
            "start_time": comp.get("date", event.get("date")),
            "completed": is_completed,
            "neutral": comp.get("neutralSite", False),
            "home_pitcher": home_pitcher,
            "away_pitcher": away_pitcher,
        })

    return fixtures
