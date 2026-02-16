"""Fetch EPL match data from football-data.org and odds from The Odds API."""

import requests
import pandas as pd

from pipeline.config import (
    FOOTBALL_DATA_API_KEY,
    FOOTBALL_DATA_BASE,
    EPL_COMPETITION_ID,
    ODDS_API_KEY,
    ODDS_API_BASE,
    ODDS_SPORT,
    ODDS_REGIONS,
    ODDS_MARKETS,
)

# ---- team-name normalisation ------------------------------------------------

_TEAM_NAME_MAP = {
    # football-data.org names
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "AFC Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton",
    "Chelsea FC": "Chelsea",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Ipswich Town FC": "Ipswich",
    "Leicester City FC": "Leicester",
    "Liverpool FC": "Liverpool",
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle",
    "Nottingham Forest FC": "Nottingham Forest",
    "Southampton FC": "Southampton",
    "Tottenham Hotspur FC": "Tottenham",
    "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",
    # The Odds API names
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Brighton and Hove Albion": "Brighton",
    "Tottenham Hotspur": "Tottenham",
    "Wolverhampton Wanderers": "Wolves",
    "Newcastle United": "Newcastle",
    "West Ham United": "West Ham",
    "Leicester City": "Leicester",
    "Ipswich Town": "Ipswich",
}


def normalize_team_name(name: str) -> str:
    """Map a football-data.org team name to its short display name.

    If the name is already short (not in the map), return it unchanged.
    """
    return _TEAM_NAME_MAP.get(name, name)


# ---- football-data.org ------------------------------------------------------


def fetch_epl_matches() -> pd.DataFrame:
    """Fetch finished EPL matches and return a tidy DataFrame.

    Columns: date, home_team, away_team, home_goals, away_goals
    """
    url = f"{FOOTBALL_DATA_BASE}/competitions/{EPL_COMPETITION_ID}/matches"
    headers = {"X-Auth-Token": FOOTBALL_DATA_API_KEY}
    params = {"status": "FINISHED"}

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for match in data.get("matches", []):
        score = match.get("score", {})
        full_time = score.get("fullTime", {})
        home_goals = full_time.get("home")
        away_goals = full_time.get("away")

        # Skip matches without a final score
        if home_goals is None or away_goals is None:
            continue

        rows.append(
            {
                "date": match["utcDate"],
                "home_team": normalize_team_name(match["homeTeam"]["name"]),
                "away_team": normalize_team_name(match["awayTeam"]["name"]),
                "home_goals": int(home_goals),
                "away_goals": int(away_goals),
            }
        )

    return pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_goals", "away_goals"])


def fetch_epl_fixtures() -> list[dict]:
    """Fetch scheduled (upcoming) EPL fixtures.

    Returns a list of dicts with keys: home_team, away_team, date, matchday.
    """
    url = f"{FOOTBALL_DATA_BASE}/competitions/{EPL_COMPETITION_ID}/matches"
    headers = {"X-Auth-Token": FOOTBALL_DATA_API_KEY}
    params = {"status": "SCHEDULED"}

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    fixtures = []
    for match in data.get("matches", []):
        fixtures.append(
            {
                "home_team": normalize_team_name(match["homeTeam"]["name"]),
                "away_team": normalize_team_name(match["awayTeam"]["name"]),
                "date": match["utcDate"],
                "matchday": match.get("matchday"),
            }
        )

    return fixtures


# ---- The Odds API ------------------------------------------------------------


def fetch_odds(sport_key=None) -> list[dict]:
    """Fetch best decimal odds for upcoming matches.

    Parameters
    ----------
    sport_key : str or None
        The Odds API sport key (e.g. "soccer_epl", "basketball_nba").
        Defaults to ``ODDS_SPORT`` from config for backwards compatibility.

    Returns a list of dicts with keys:
        home_team, away_team, commence_time, home_odds, draw_odds, away_odds
    """
    if sport_key is None:
        sport_key = ODDS_SPORT
    url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": ODDS_REGIONS,
        "markets": ODDS_MARKETS,
        "oddsFormat": "decimal",
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    events = resp.json()

    results = []
    for event in events:
        best_home = 0.0
        best_draw = 0.0
        best_away = 0.0

        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                home_price = outcomes.get(event["home_team"], 0.0)
                draw_price = outcomes.get("Draw", 0.0)
                away_price = outcomes.get(event["away_team"], 0.0)

                best_home = max(best_home, home_price)
                best_draw = max(best_draw, draw_price)
                best_away = max(best_away, away_price)

        results.append(
            {
                "home_team": event["home_team"],
                "away_team": event["away_team"],
                "commence_time": event["commence_time"],
                "home_odds": best_home,
                "draw_odds": best_draw,
                "away_odds": best_away,
            }
        )

    return results
