"""Fetch NBA game results and schedule from balldontlie.io."""

import json as _json
import os
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
# ESPN cache helpers
# ---------------------------------------------------------------------------


def _load_espn_cache(cache_path: str | None) -> dict:
    """Load ESPN cache from disk, returning empty cache if missing."""
    if cache_path is None or not os.path.exists(cache_path):
        return {"games": {}, "rosters": {}}
    with open(cache_path) as f:
        cache = _json.load(f)
    if not isinstance(cache, dict):
        return {"games": {}, "rosters": {}}
    cache.setdefault("games", {})
    cache.setdefault("rosters", {})
    return cache


def _save_espn_cache(cache_path: str | None, cache: dict) -> None:
    """Write ESPN cache to disk."""
    if cache_path is None:
        return
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        _json.dump(cache, f)


def _injury_weight(status: str | None) -> float:
    """Convert ESPN injury text into a coarse availability penalty."""
    normalized = str(status or "").strip().lower()
    if normalized in {"out", "suspended"}:
        return 1.0
    if normalized in {"doubtful"}:
        return 0.75
    if normalized in {"questionable", "day-to-day"}:
        return 0.35
    return 0.0


def _leader_weight(stat_name: str | None) -> float:
    """Assign extra importance to key leader categories."""
    mapping = {
        "pointsPerGame": 1.0,
        "assistsPerGame": 0.9,
        "reboundsPerGame": 0.65,
        "stealsPerGame": 0.4,
        "blocksPerGame": 0.4,
    }
    return mapping.get(str(stat_name or ""), 0.5)


def _fetch_nba_team_availability_profile(
    team_id: str | int | None,
    leader_ids=None,
    leader_weights: dict[str, float] | None = None,
    cache: dict | None = None,
) -> dict:
    """Fetch and cache a coarse NBA roster availability profile."""
    default = {
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
    if not team_id:
        return default

    weight_map = {
        str(player_id): float(weight)
        for player_id, weight in (leader_weights or {}).items()
        if player_id
    }
    if not weight_map:
        weight_map = {str(player_id): 1.0 for player_id in (leader_ids or []) if player_id}

    cache_store = None
    cache_key = str(team_id)
    roster_rows = None
    if cache is not None:
        cache_store = cache.setdefault("rosters", {})
        cached = cache_store.get(cache_key)
        if isinstance(cached, dict) and cached.get("player_rows") is not None:
            roster_rows = cached.get("player_rows")

    if roster_rows is None:
        url = f"{NBA_ESPN_BASE}/teams/{team_id}/roster"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException:
            return default

        roster_rows = []
        for athlete in data.get("athletes", []):
            if not isinstance(athlete, dict):
                continue
            status_type = athlete.get("status", {}).get("type")
            injuries = athlete.get("injuries") or []
            player_penalty = 0.0
            uncertain_penalty = 0.0
            for injury in injuries:
                injury_status = injury.get("status")
                weight = _injury_weight(injury_status)
                player_penalty = max(player_penalty, weight)
                normalized = str(injury_status or "").strip().lower()
                if normalized in {"questionable", "day-to-day", "doubtful"}:
                    uncertain_penalty = max(uncertain_penalty, weight)
            roster_rows.append({
                "id": str(athlete.get("id")),
                "active": status_type == "active",
                "penalty": round(player_penalty, 3),
                "uncertain_penalty": round(uncertain_penalty, 3),
            })
        if cache_store is not None:
            cache_store[cache_key] = {"player_rows": roster_rows}

    active_players = 0
    injured_players = 0
    questionable_players = 0
    doubtful_players = 0
    injury_burden = 0.0
    uncertainty_burden = 0.0
    key_absence_score = 0.0
    leader_absence_burden = 0.0
    leader_uncertainty_burden = 0.0
    available_core_players = 0

    for athlete in roster_rows:
        is_active = bool(athlete.get("active"))
        if is_active:
            active_players += 1

        player_penalty = float(athlete.get("penalty", 0.0) or 0.0)
        uncertain_penalty = float(athlete.get("uncertain_penalty", 0.0) or 0.0)
        if player_penalty > 0:
            injured_players += 1
            injury_burden += player_penalty
            leader_weight_value = float(weight_map.get(str(athlete.get("id")), 0.0) or 0.0)
            if leader_weight_value > 0:
                key_absence_score += player_penalty
                leader_absence_burden += player_penalty * leader_weight_value
            if uncertain_penalty > 0:
                uncertainty_burden += uncertain_penalty
                if uncertain_penalty >= 0.75:
                    doubtful_players += 1
                else:
                    questionable_players += 1
                if leader_weight_value > 0:
                    leader_uncertainty_burden += uncertain_penalty * leader_weight_value
        elif is_active:
            available_core_players += 1

    profile = {
        "active_players": active_players,
        "injured_players": injured_players,
        "questionable_players": questionable_players,
        "doubtful_players": doubtful_players,
        "injury_burden": round(injury_burden, 3),
        "uncertainty_burden": round(uncertainty_burden, 3),
        "key_absence_score": round(key_absence_score, 3),
        "leader_absence_burden": round(leader_absence_burden, 3),
        "leader_uncertainty_burden": round(leader_uncertainty_burden, 3),
        "available_core_players": available_core_players,
    }
    return profile


def _incremental_dates(cache: dict, all_dates: list[str], lookback_days: int = 2) -> list[str]:
    """Return only dates that need fetching based on cache contents.

    If cache is empty, returns all_dates (full season).
    Otherwise returns dates >= (max_cached_date - lookback_days).
    """
    games = cache.get("games", {})
    if not games:
        return all_dates
    max_cached = max(v["date"] for v in games.values())
    cutoff = (datetime.strptime(max_cached, "%Y-%m-%d") - timedelta(days=lookback_days)).date()
    return [d for d in all_dates if datetime.strptime(d, "%Y-%m-%d").date() >= cutoff]


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
    cache_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch finished NBA games and box scores via ESPN API.

    Parameters
    ----------
    season : int or None
        Season start year (e.g. 2025 for 2025-26). Defaults to current season.
    dates : list[str] or None
        Explicit YYYY-MM-DD dates. Defaults to full season range.
    cache_path : str or None
        Path to the ESPN cache JSON file. If provided, previously fetched
        games and box scores are loaded from disk and only missing/recent
        dates are re-fetched.

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

    # Load cache and restrict fetching to incremental dates
    cache = _load_espn_cache(cache_path)
    fetch_dates = _incremental_dates(cache, dates)

    for date_str in fetch_dates:
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
                game_id = parsed["event_id"]
                # Add/update game entry in cache (without touching existing box_scores)
                existing = cache["games"].get(game_id, {})
                cache["games"][game_id] = {
                    "date": parsed["date"],
                    "home_team": normalize_nba_team_name(parsed["home_name"]),
                    "away_team": normalize_nba_team_name(parsed["away_name"]),
                    "home_goals": parsed["home_score"],
                    "away_goals": parsed["away_score"],
                    "box_scores": existing.get("box_scores", []),
                }

        # Fetch box scores for games that don't already have them in cache
        for parsed in final_events:
            game_id = parsed["event_id"]
            if cache["games"][game_id].get("box_scores"):
                # Already have box scores for this game — skip the summary call
                continue

            time.sleep(_ESPN_REQUEST_DELAY)
            summary_url = f"{NBA_ESPN_BASE}/summary?event={game_id}"
            try:
                s_resp = requests.get(summary_url, timeout=30)
                s_resp.raise_for_status()
                s_data = s_resp.json()
                player_groups = s_data.get("boxscore", {}).get("players", [])
                if len(player_groups) < 2:
                    continue
                box_scores = []
                for player_group in player_groups:
                    try:
                        totals = player_group["statistics"][0]["totals"]
                        stats = _parse_box_score_totals(totals)
                        team_name = player_group["team"]["displayName"]
                        box_scores.append({
                            "team": normalize_nba_team_name(team_name),
                            **stats,
                        })
                    except (KeyError, IndexError):
                        continue
                cache["games"][game_id]["box_scores"] = box_scores
            except requests.RequestException:
                continue

        time.sleep(_ESPN_REQUEST_DELAY)

    _save_espn_cache(cache_path, cache)

    # Build DataFrames from the full cache (includes cached + newly fetched data)
    game_rows = []
    box_rows = []
    for game_id, entry in cache["games"].items():
        game_rows.append({
            "game_id": game_id,
            "date": entry["date"],
            "home_team": entry["home_team"],
            "away_team": entry["away_team"],
            "home_goals": entry["home_goals"],
            "away_goals": entry["away_goals"],
        })
        for bs in entry.get("box_scores", []):
            box_rows.append({
                "game_id": game_id,
                "team": bs["team"],
                "date": entry["date"],
                "pts": bs["pts"],
                "fgm": bs["fgm"],
                "fga": bs["fga"],
                "fg3m": bs["fg3m"],
                "fg3a": bs["fg3a"],
                "ftm": bs["ftm"],
                "fta": bs["fta"],
                "orb": bs["orb"],
                "drb": bs["drb"],
                "to": bs["to"],
                "possessions": bs["possessions"],
            })

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


def fetch_nba_espn_schedule(cache_path: str | None = None) -> list[dict]:
    """Fetch today's NBA games via ESPN API.

    Uses the ESPN scoreboard for today's date in US Eastern Time (UTC-5/UTC-4).
    Stores the game date as the local scoreboard date, not the raw UTC event
    timestamp (which shifts late-night games to the following UTC day).
    """
    # Derive today's date in US Eastern Time so late-night games (e.g. 9pm ET =
    # 2am UTC next day) are attributed to the correct local game date.
    et_offset = timedelta(hours=5)  # UTC-5; close enough for schedule purposes
    today_et = (datetime.now(timezone.utc) - et_offset).date()
    game_date_str = today_et.strftime("%Y-%m-%d")
    espn_date = today_et.strftime("%Y%m%d")

    cache = _load_espn_cache(cache_path)
    url = f"{NBA_ESPN_BASE}/scoreboard?dates={espn_date}&limit=50"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    fixtures = []
    for event in data.get("events", []):
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

        home_leader_weights = {}
        away_leader_weights = {}
        for leader_group in home.get("leaders", []):
            leaders = leader_group.get("leaders", [])
            if leaders:
                athlete_id = leaders[0].get("athlete", {}).get("id")
                if athlete_id:
                    home_leader_weights[str(athlete_id)] = (
                        home_leader_weights.get(str(athlete_id), 0.0) +
                        _leader_weight(leader_group.get("name"))
                    )
        for leader_group in away.get("leaders", []):
            leaders = leader_group.get("leaders", [])
            if leaders:
                athlete_id = leaders[0].get("athlete", {}).get("id")
                if athlete_id:
                    away_leader_weights[str(athlete_id)] = (
                        away_leader_weights.get(str(athlete_id), 0.0) +
                        _leader_weight(leader_group.get("name"))
                    )

        fixtures.append({
            "home_team": normalize_nba_team_name(home["team"]["displayName"]),
            "away_team": normalize_nba_team_name(away["team"]["displayName"]),
            # Use the local scoreboard date, not event["date"] (UTC timestamp)
            "date": game_date_str,
            "start_time": comp.get("date", event.get("date")),
            "completed": is_completed,
            "neutral": comp.get("neutralSite", False),
            "home_availability_profile": _fetch_nba_team_availability_profile(
                home["team"].get("id"),
                leader_weights=home_leader_weights,
                cache=cache,
            ),
            "away_availability_profile": _fetch_nba_team_availability_profile(
                away["team"].get("id"),
                leader_weights=away_leader_weights,
                cache=cache,
            ),
        })

    _save_espn_cache(cache_path, cache)
    return fixtures
