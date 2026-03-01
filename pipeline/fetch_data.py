"""Fetch odds from The Odds API."""

import requests

from pipeline.config import (
    ODDS_API_KEY,
    ODDS_API_BASE,
    ODDS_REGIONS,
    ODDS_MARKETS,
)

# ---- The Odds API ------------------------------------------------------------


def fetch_odds(sport_key: str) -> list[dict]:
    """Fetch best decimal odds for upcoming matches.

    Parameters
    ----------
    sport_key : str
        The Odds API sport key (e.g. "basketball_nba", "basketball_ncaab").

    Returns a list of dicts with keys:
        home_team, away_team, commence_time, home_odds, draw_odds, away_odds
    """
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
