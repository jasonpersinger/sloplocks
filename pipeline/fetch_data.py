"""Fetch odds from The Odds API."""

from collections import Counter
from statistics import median

import requests

from pipeline.config import (
    ODDS_API_KEY,
    ODDS_API_BASE,
    ODDS_REGIONS,
    ODDS_MARKETS,
)
from pipeline.ensemble import no_vig_probabilities

# ---- The Odds API ------------------------------------------------------------


def _book_label(bookmaker: dict) -> str:
    """Return a stable display label for one bookmaker payload."""
    return (
        bookmaker.get("title")
        or bookmaker.get("key")
        or bookmaker.get("name")
        or "unknown_book"
    )

def _median_market_probabilities(book_probs: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-book fair probabilities into a stable benchmark.

    We intentionally use the median of complete-book no-vig probabilities instead
    of re-normalizing the best prices across books. That fallback avoids creating
    synthetic underrounds from prices that could never be bet as one market.
    """
    outcomes = {
        outcome
        for probs in book_probs
        for outcome in probs
    }
    if not outcomes:
        return {}
    return {
        outcome: round(float(median([probs[outcome] for probs in book_probs if outcome in probs])), 4)
        for outcome in sorted(outcomes)
    }


def _extract_best_totals_market(bookmakers: list[dict]) -> dict:
    """Return a representative totals market with a consensus line.

    Totals prices vary by line, so we first select the most common points line
    across books, then take the best over/under price available at that exact
    line. The fair-prob benchmark is derived from complete books at that same
    consensus line; if none exist, callers can fall back to the execution line.
    """
    line_counter = Counter()
    totals_by_line: dict[float, list[tuple[float, float]]] = {}
    fair_probs_by_line: dict[float, list[dict[str, float]]] = {}
    raw_probs_by_line: dict[float, list[dict[str, float]]] = {}
    holds_by_line: dict[float, list[float]] = {}
    books_by_line: dict[float, list[dict]] = {}

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
            over_price = float(over.get("price", 0.0))
            under_price = float(under.get("price", 0.0))
            if over_price <= 1.0 or under_price <= 1.0:
                continue
            line_counter[line] += 1
            totals_by_line.setdefault(line, []).append((over_price, under_price))
            raw_probs, fair_probs, hold = no_vig_probabilities({
                "over": over_price,
                "under": under_price,
            })
            fair_probs_by_line.setdefault(line, []).append(fair_probs)
            raw_probs_by_line.setdefault(line, []).append(raw_probs)
            holds_by_line.setdefault(line, []).append(float(hold))
            books_by_line.setdefault(line, []).append({
                "book": _book_label(bookmaker),
                "last_update": bookmaker.get("last_update"),
                "outcomes": {
                    "over": over_price,
                    "under": under_price,
                },
                "raw_probs": {k: round(float(v), 4) for k, v in raw_probs.items()},
                "fair_probs": {k: round(float(v), 4) for k, v in fair_probs.items()},
                "hold": round(float(hold), 4),
            })

    if not line_counter:
        return {
            "total_line": None,
            "over_odds": 0.0,
            "under_odds": 0.0,
            "totals_benchmark": {},
        }

    consensus_line = sorted(
        line_counter.keys(),
        key=lambda line: (-line_counter[line], abs(line)),
    )[0]
    prices = totals_by_line[consensus_line]
    return {
        "total_line": consensus_line,
        "over_odds": max(price[0] for price in prices),
        "under_odds": max(price[1] for price in prices),
        "totals_benchmark": {
            "line": consensus_line,
            "fair_probs": _median_market_probabilities(fair_probs_by_line.get(consensus_line, [])),
            "raw_probs": _median_market_probabilities(raw_probs_by_line.get(consensus_line, [])),
            "hold": round(float(median(holds_by_line.get(consensus_line, [0.0]))), 4),
            "books_tracked": len(fair_probs_by_line.get(consensus_line, [])),
            "source": "median_complete_book_no_vig",
        },
        "totals_market_snapshot": {
            "line": consensus_line,
            "execution_prices": {
                "over": max(price[0] for price in prices),
                "under": max(price[1] for price in prices),
            },
            "benchmark": {
                "fair_probs": _median_market_probabilities(fair_probs_by_line.get(consensus_line, [])),
                "raw_probs": _median_market_probabilities(raw_probs_by_line.get(consensus_line, [])),
                "hold": round(float(median(holds_by_line.get(consensus_line, [0.0]))), 4),
                "books_tracked": len(fair_probs_by_line.get(consensus_line, [])),
                "source": "median_complete_book_no_vig",
            },
            "books": books_by_line.get(consensus_line, []),
        },
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
        book_fair_probs = []
        book_raw_probs = []
        book_holds = []
        moneyline_books = []

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

                market_odds = {}
                if home_price and home_price > 1.0:
                    market_odds["home"] = float(home_price)
                if away_price and away_price > 1.0:
                    market_odds["away"] = float(away_price)
                if draw_price and draw_price > 1.0:
                    market_odds["draw"] = float(draw_price)

                required_outcomes = {"home", "away"}
                if draw_price and draw_price > 1.0:
                    required_outcomes.add("draw")
                if not required_outcomes.issubset(market_odds):
                    continue

                raw_probs, fair_probs, hold = no_vig_probabilities(market_odds)
                book_raw_probs.append(raw_probs)
                book_fair_probs.append(fair_probs)
                book_holds.append(float(hold))
                moneyline_books.append({
                    "book": _book_label(bookmaker),
                    "last_update": bookmaker.get("last_update"),
                    "outcomes": {k: round(float(v), 4) for k, v in market_odds.items()},
                    "raw_probs": {k: round(float(v), 4) for k, v in raw_probs.items()},
                    "fair_probs": {k: round(float(v), 4) for k, v in fair_probs.items()},
                    "hold": round(float(hold), 4),
                })

        record = {
            "home_team": event["home_team"],
            "away_team": event["away_team"],
            "commence_time": event["commence_time"],
            "home_odds": best_home,
            "draw_odds": best_draw,
            "away_odds": best_away,
            "moneyline_benchmark": {
                "fair_probs": _median_market_probabilities(book_fair_probs),
                "raw_probs": _median_market_probabilities(book_raw_probs),
                "hold": round(float(median(book_holds)), 4) if book_holds else None,
                # Fallback: if no complete books are available, downstream code
                # falls back to the execution-line normalization for continuity.
                "books_tracked": len(book_fair_probs),
                "source": "median_complete_book_no_vig",
            },
            "moneyline_market_snapshot": {
                "execution_prices": {
                    key: value
                    for key, value in {
                        "home": best_home,
                        "draw": best_draw,
                        "away": best_away,
                    }.items()
                    if value and value > 1.0
                },
                "benchmark": {
                    "fair_probs": _median_market_probabilities(book_fair_probs),
                    "raw_probs": _median_market_probabilities(book_raw_probs),
                    "hold": round(float(median(book_holds)), 4) if book_holds else None,
                    "books_tracked": len(book_fair_probs),
                    "source": "median_complete_book_no_vig",
                },
                "books": moneyline_books,
            },
        }
        if include_totals:
            record.update(_extract_best_totals_market(event.get("bookmakers", [])))

        results.append(record)

    return results
