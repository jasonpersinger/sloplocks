"""Fetch NCAAM game results, box scores, and schedule from ESPN."""

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from pipeline.config import ESPN_BASE

_REQUEST_DELAY = 0.5

# ---- team-name normalisation ------------------------------------------------

_team_map: dict[str, str] | None = None


def _build_team_map() -> dict[str, str]:
    """Fetch ESPN teams endpoint and build displayName -> location map."""
    url = f"{ESPN_BASE}/teams?limit=400"
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
    """
    global _team_map
    if _team_map is None:
        _team_map = _build_team_map()
    return _team_map.get(name, name)


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


# ---- ESPN scoreboard / summary -----------------------------------------------


def _parse_event(event: dict) -> dict | None:
    """Parse an ESPN scoreboard event into a game dict, or None if not final."""
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

    game_rows = []
    box_rows = []

    for date_str in dates:
        espn_date = date_str.replace("-", "")
        url = f"{ESPN_BASE}/scoreboard?dates={espn_date}&limit=200"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        final_events = []
        for event in data.get("events", []):
            parsed = _parse_event(event)
            if parsed is not None:
                final_events.append(parsed)
                game_rows.append({
                    "game_id": parsed["event_id"],
                    "date": parsed["date"],
                    "home_team": normalize_ncaam_team_name(parsed["home_name"]),
                    "away_team": normalize_ncaam_team_name(parsed["away_name"]),
                    "home_goals": parsed["home_score"],
                    "away_goals": parsed["away_score"],
                })

        # Fetch box scores for each final game
        for parsed in final_events:
            time.sleep(_REQUEST_DELAY)
            summary_url = f"{ESPN_BASE}/summary?event={parsed['event_id']}"
            try:
                s_resp = requests.get(summary_url, timeout=30)
                s_resp.raise_for_status()
                s_data = s_resp.json()

                # Totals are under boxscore.players[], not boxscore.teams[]
                player_groups = s_data.get("boxscore", {}).get("players", [])
                if len(player_groups) < 2:
                    continue

                for player_group in player_groups:
                    try:
                        totals = player_group["statistics"][0]["totals"]
                        stats = _parse_box_score_totals(totals)
                        team_name = player_group["team"]["displayName"]
                        box_rows.append({
                            "game_id": parsed["event_id"],
                            "team": normalize_ncaam_team_name(team_name),
                            "date": parsed["date"],
                            **stats,
                        })
                    except (KeyError, IndexError):
                        continue
            except requests.RequestException:
                continue

        time.sleep(_REQUEST_DELAY)

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
    """Fetch upcoming NCAAM games (today + next 7 days).

    Returns
    -------
    list[dict]
        Each dict has keys: home_team, away_team, date.
    """
    today = datetime.now(timezone.utc).date()
    fixtures = []

    for day_offset in range(8):
        date = today + timedelta(days=day_offset)
        espn_date = date.strftime("%Y%m%d")
        url = f"{ESPN_BASE}/scoreboard?dates={espn_date}&limit=200"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for event in data.get("events", []):
            comp = event["competitions"][0]
            status_type = comp.get("status", {}).get("type", {})
            if status_type.get("completed", False):
                continue

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
                "date": event["date"][:10],
            })

        time.sleep(_REQUEST_DELAY)

    return fixtures
