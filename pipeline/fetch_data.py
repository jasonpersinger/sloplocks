"""Fetch odds from The Odds API."""

from collections import Counter

import requests

from pipeline.config import (
    ODDS_API_KEY,
    ODDS_API_BASE,
    ODDS_REGIONS,
    ODDS_MARKETS,
)

# ---- The Odds API ------------------------------------------------------------

def _extract_best_totals_market(bookmakers: list[dict]) -> dict:
    """Return a representative totals market with a consensus line.

    Totals prices vary by line, so we first select the most common points line
    across books, then take the best over/under price available at that exact
    line. If no complete totals market exists, return zeroed placeholders.
    """
    line_counter = Counter()
    totals_by_line: dict[float, list[tuple[float, float]]] = {}

    for bookmaker in bookmakers or []:
        for market in bookmaker.get("markets", []):
            if market.get("key") != "totals":
                continue
            over = next((o for o in market.get("outcomes", []) if o.get("name") == "Over"), None)
            under = next((o for o in market.get("outcomes", []) if o.get("name") == "Under"), None)
            if not over or not under:
                continue
            if over.get("point") != under.get("point"):
                continue
            line = float(over.get("point"))
            line_counter[line] += 1
            totals_by_line.setdefault(line, []).append((float(over.get("price", 0.0)), float(under.get("price", 0.0))))

    if not line_counter:
        return {"total_line": None, "over_odds": 0.0, "under_odds": 0.0}

    consensus_line = sorted(
        line_counter.keys(),
        key=lambda line: (-line_counter[line], abs(line)),
    )[0]
    prices = totals_by_line[consensus_line]
    return {
        "total_line": consensus_line,
        "over_odds": max(price[0] for price in prices),
        "under_odds": max(price[1] for price in prices),
    }


def fetch_odds(sport_key: str, include_totals: bool = False) -> list[dict]:
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
        "markets": "h2h,totals" if include_totals else ODDS_MARKETS,
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

        record = {
            "home_team": event["home_team"],
            "away_team": event["away_team"],
            "commence_time": event["commence_time"],
            "home_odds": best_home,
            "draw_odds": best_draw,
            "away_odds": best_away,
        }
        if include_totals:
            record.update(_extract_best_totals_market(event.get("bookmakers", [])))

        results.append(record)

    return results
