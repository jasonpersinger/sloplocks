"""Fetch NCAAM game results, box scores, and schedule from ESPN."""

import json as _json
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from pipeline.config import NCAAM_ESPN_BASE

_REQUEST_DELAY = 0.5

# ---- team-name normalisation ------------------------------------------------

_team_map: dict[str, str] | None = None

# Fallback map for Odds API team names that don't appear in ESPN's displayName list
_ODDS_API_FALLBACK: dict[str, str] = {
    "Michigan St Spartans": "Michigan State",
    "Mississippi St Bulldogs": "Mississippi State",
    "Kansas St Wildcats": "Kansas State",
    "Iowa St Cyclones": "Iowa State",
    "Ohio St Buckeyes": "Ohio State",
    "Penn St Nittany Lions": "Penn State",
    "Florida St Seminoles": "Florida State",
    "Arizona St Sun Devils": "Arizona State",
    "Colorado St Rams": "Colorado State",
    "Fresno St Bulldogs": "Fresno State",
    "San Diego St Aztecs": "San Diego State",
    "Boise St Broncos": "Boise State",
    "Utah St Aggies": "Utah State",
    "Weber St Wildcats": "Weber State",
    "Sacramento St Hornets": "Sacramento State",
    "Portland St Vikings": "Portland State",
    "Indiana St Sycamores": "Indiana State",
    "Wichita St Shockers": "Wichita State",
    "Kennesaw St Owls": "Kennesaw State",
    "Jacksonville St Gamecocks": "Jacksonville State",
    "McNeese St Cowboys": "McNeese State",
    "Nicholls St Colonels": "Nicholls State",
    "Grambling St Tigers": "Grambling State",
    "Alcorn St Braves": "Alcorn State",
    "Delaware St Hornets": "Delaware State",
    "Morgan St Bears": "Morgan State",
    "Norfolk St Spartans": "Norfolk State",
    "Coppin St Eagles": "Coppin State",
    "Savannah St Tigers": "Savannah State",
    "Tennessee St Tigers": "Tennessee State",
    "NC State Wolfpack": "NC State",
    "UConn Huskies": "UConn",
    "UNC Tar Heels": "North Carolina",
    "UCF Knights": "UCF",
    "UNLV Rebels": "UNLV",
    "USC Trojans": "USC",
    "LSU Tigers": "LSU",
    "SMU Mustangs": "SMU",
    "TCU Horned Frogs": "TCU",
    "VCU Rams": "VCU",
    "BYU Cougars": "BYU",
    "Ole Miss Rebels": "Ole Miss",
    "Mississippi Rebels": "Ole Miss",
    "Miami (OH) Redhawks": "Miami (OH)",
    "Miami Hurricanes": "Miami",
    "Sam Houston St Bearkats": "Sam Houston State",
    "Middle Tennessee Blue Raiders": "Middle Tennessee",
    "UT Arlington Mavericks": "UT Arlington",
    "UTSA Roadrunners": "UTSA",
    "UTEP Miners": "UTEP",
    "New Mexico Lobos": "New Mexico",
    "Saint Joseph's Hawks": "Saint Joseph's",
    "St. Joseph's (PA) Hawks": "Saint Joseph's",
    "St. Joseph's Hawks": "Saint Joseph's",
    "Saint Joseph's Hawks": "Saint Joseph's",
    "Tulsa Golden Hurricane": "Tulsa",
    "Wichita St. Shockers": "Wichita State",
    "Connecticut Huskies": "UConn",
    "Connecticut": "UConn",
    "Southern California Trojans": "USC",
    "Southern California": "USC",
    "Central Florida Knights": "UCF",
    "Central Florida": "UCF",
    "Mississippi": "Ole Miss",
    "Texas Christian": "TCU",
    "Texas Christian Horned Frogs": "TCU",
    "Brigham Young": "BYU",
    "Brigham Young Cougars": "BYU",
    "Virginia Commonwealth": "VCU",
    "Virginia Commonwealth Rams": "VCU",
    "Nevada-Las Vegas": "UNLV",
    "Nevada-Las Vegas Rebels": "UNLV",
    "Louisiana State": "LSU",
    "Louisiana State Tigers": "LSU",
    "Southern Methodist": "SMU",
    "Southern Methodist Mustangs": "SMU",
}


def _build_team_map() -> dict[str, str]:
    """Fetch ESPN teams endpoint and build displayName -> location map."""
    url = f"{NCAAM_ESPN_BASE}/teams?limit=400"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    mapping = {}
    try:
        teams = data["sports"][0]["leagues"][0]["teams"]
        for entry in teams:
            team = entry["team"]
            mapping[team["displayName"]] = team["location"]
    except (KeyError, IndexError):
        pass
    return mapping


def normalize_ncaam_team_name(name: str) -> str:
    """Map an ESPN full team name to its short location name.

    Lazily fetches the team map from ESPN on first call.
    Falls back to a manual map for Odds API names that don't appear in ESPN's list.
    """
    global _team_map
    if _team_map is None:
        _team_map = _build_team_map()
    return _team_map.get(name, _ODDS_API_FALLBACK.get(name, name))


# ---- box score parsing ------------------------------------------------------


def _parse_box_score_totals(totals: list[str]) -> dict:
    """Parse ESPN's flat totals array into structured stats.

    Indices: 0=MIN(empty), 1=PTS, 2=FG(m-a), 3=3PT(m-a), 4=FT(m-a),
             5=REB, 6=AST, 7=TO, 8=STL, 9=BLK, 10=OREB, 11=DREB, 12=PF

    Returns dict with: pts, fgm, fga, fg3m, fg3a, ftm, fta, orb, drb, to, possessions
    """
    def _split_ma(val: str) -> tuple[int, int]:
        """Split a 'made-attempted' string like '28-58' into (28, 58)."""
        parts = val.split("-")
        return int(parts[0]), int(parts[1])

    pts = int(totals[1])
    fgm, fga = _split_ma(totals[2])
    fg3m, fg3a = _split_ma(totals[3])
    ftm, fta = _split_ma(totals[4])
    orb = int(totals[10])
    drb = int(totals[11])
    to = int(totals[7])

    possessions = fga - orb + to + 0.44 * fta

    return {
        "pts": pts,
        "fgm": fgm,
        "fga": fga,
        "fg3m": fg3m,
        "fg3a": fg3a,
        "ftm": ftm,
        "fta": fta,
        "orb": orb,
        "drb": drb,
        "to": to,
        "possessions": possessions,
    }


# ---- season date range -------------------------------------------------------


def _season_date_range(season: int) -> list[str]:
    """Generate list of YYYY-MM-DD date strings for a NCAAM season.

    Season starts November 1 of `season` year, ends April 10 of `season+1`
    year or today, whichever is earlier.
    """
    start = datetime(season, 11, 1)
    end = min(datetime(season + 1, 4, 10), datetime.now(timezone.utc).replace(tzinfo=None))
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
    for competitor in comp["competitors"]:
        if competitor["homeAway"] == "home":
            home = competitor
        else:
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


def fetch_ncaam_games(
    season: int | None = None,
    dates: list[str] | None = None,
    cache_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch finished NCAAM games and box scores for a season.

    Parameters
    ----------
    season : int or None
        The season start year (e.g. 2025 for 2025-26 season).
        Defaults to current season (Nov start).
    dates : list[str] or None
        Explicit list of YYYY-MM-DD dates to fetch. If None, generates
        the full season date range.
    cache_path : str or None
        Path to the ESPN cache JSON file. If provided, previously fetched
        games and box scores are loaded from disk and only missing/recent
        dates are re-fetched.

    Returns
    -------
    (games_df, box_scores_df)
        games_df columns: date, home_team, away_team, home_goals, away_goals
        box_scores_df columns: game_id, team, date, pts, fgm, fga, fg3m, fg3a,
                               ftm, fta, orb, drb, to, possessions
    """
    if season is None:
        now = datetime.now(timezone.utc)
        season = now.year if now.month >= 10 else now.year - 1

    if dates is None:
        dates = _season_date_range(season)

    # Load cache and restrict fetching to incremental dates
    cache = _load_espn_cache(cache_path)
    fetch_dates = _incremental_dates(cache, dates)

    for date_str in fetch_dates:
        espn_date = date_str.replace("-", "")
        
        final_events = []
        # Fetch groups separately. 50 = NCAA Tournament, 100 = All Division I
        for group in [50, 100]:
            url = f"{NCAAM_ESPN_BASE}/scoreboard?dates={espn_date}&limit=200&groups={group}"
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                continue

            for event in data.get("events", []):
                parsed = _parse_event(event)
                if parsed is not None:
                    final_events.append(parsed)
                    game_id = parsed["event_id"]
                    # Add/update game entry in cache (without touching existing box_scores)
                    existing = cache["games"].get(game_id, {})
                    cache["games"][game_id] = {
                        "date": parsed["date"],
                        "home_team": normalize_ncaam_team_name(parsed["home_name"]),
                        "away_team": normalize_ncaam_team_name(parsed["away_name"]),
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

            time.sleep(_REQUEST_DELAY)
            summary_url = f"{NCAAM_ESPN_BASE}/summary?event={game_id}"
            try:
                s_resp = requests.get(summary_url, timeout=30)
                s_resp.raise_for_status()
                s_data = s_resp.json()

                # Totals are under boxscore.players[], not boxscore.teams[]
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
                            "team": normalize_ncaam_team_name(team_name),
                            **stats,
                        })
                    except (KeyError, IndexError):
                        continue
                cache["games"][game_id]["box_scores"] = box_scores
            except requests.RequestException:
                continue

        time.sleep(_REQUEST_DELAY)

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


def fetch_ncaam_schedule() -> list[dict]:
    """Fetch today's NCAAM games via ESPN API.

    Uses the ESPN scoreboard for today's date in US Eastern Time (UTC-5/UTC-4).
    Stores the game date as the local scoreboard date, not the raw UTC event
    timestamp (which shifts late-night games to the following UTC day).

    Returns
    -------
    list[dict]
        Each dict has keys: home_team, away_team, date.
    """
    et_offset = timedelta(hours=5)  # UTC-5; close enough for schedule purposes
    today_et = (datetime.now(timezone.utc) - et_offset).date()
    game_date_str = today_et.strftime("%Y-%m-%d")
    espn_date = today_et.strftime("%Y%m%d")

    # Fetch default scoreboard (usually includes major tournaments)
    url = f"{NCAAM_ESPN_BASE}/scoreboard?dates={espn_date}&limit=200"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    fixtures = []
    for event in data.get("events", []):
        if "competitions" not in event or not event["competitions"]:
            continue
        comp = event["competitions"][0]
        # Include all games for the UI, regardless of completion status
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

        fixtures.append({
            "home_team": normalize_ncaam_team_name(home["team"]["displayName"]),
            "away_team": normalize_ncaam_team_name(away["team"]["displayName"]),
            # Use the local scoreboard date, not event["date"] (UTC timestamp)
            "date": game_date_str,
            "start_time": comp.get("date", event.get("date")),
            "completed": is_completed,
            "neutral": comp.get("neutralSite", False),
        })

    return fixtures
