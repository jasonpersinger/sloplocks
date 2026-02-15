"""Fetch expected-goals (xG) data from Understat for the current EPL season."""

import json
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

from pipeline.config import UNDERSTAT_BASE, CURRENT_SEASON

# Understat uses full names; we normalise to the short forms used elsewhere.
_UNDERSTAT_NAME_MAP = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Wolverhampton Wanderers": "Wolves",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nottingham Forest",
    "West Ham United": "West Ham",
    "Leicester City": "Leicester",
    "Ipswich Town": "Ipswich",
    "Tottenham Hotspur": "Tottenham",
    "AFC Bournemouth": "Bournemouth",
    "Brighton and Hove Albion": "Brighton",
    "Crystal Palace": "Crystal Palace",
}


def normalize_understat_name(name: str) -> str:
    """Map an Understat team name to the short canonical name.

    If the name is not in the mapping it is returned unchanged, which handles
    teams whose Understat name already matches (e.g. "Arsenal", "Chelsea").
    """
    return _UNDERSTAT_NAME_MAP.get(name, name)


def parse_understat_data(raw_matches: list[dict]) -> pd.DataFrame:
    """Convert raw Understat match dicts into a tidy DataFrame.

    Parameters
    ----------
    raw_matches : list[dict]
        Each dict is expected to contain at least:
        - ``isResult``  (bool) -- whether the match has been played
        - ``datetime``  (str)  -- ISO-ish date string
        - ``h``         (dict) -- with key ``title`` for the home team name
        - ``a``         (dict) -- with key ``title`` for the away team name
        - ``xG``        (dict) -- with keys ``h`` and ``a`` for xG values

    Returns
    -------
    pd.DataFrame
        Columns: date, home_team, away_team, home_xg, away_xg
    """
    rows: list[dict] = []
    for match in raw_matches:
        if not match.get("isResult", False):
            continue
        rows.append(
            {
                "date": match["datetime"][:10],
                "home_team": normalize_understat_name(match["h"]["title"]),
                "away_team": normalize_understat_name(match["a"]["title"]),
                "home_xg": round(float(match["xG"]["h"]), 2),
                "away_xg": round(float(match["xG"]["a"]), 2),
            }
        )
    return pd.DataFrame(rows)


def fetch_understat_xg() -> pd.DataFrame:
    """Scrape the current EPL season's xG data from Understat.

    Returns
    -------
    pd.DataFrame
        Columns: date, home_team, away_team, home_xg, away_xg
    """
    url = f"{UNDERSTAT_BASE}/league/EPL/{CURRENT_SEASON}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        ),
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    # Understat embeds match data in a script tag like:
    #   var datesData = JSON.parse('...')
    pattern = re.compile(r"var\s+datesData\s*=\s*JSON\.parse\('(.+?)'\)")

    for script in soup.find_all("script"):
        text = script.string or ""
        m = pattern.search(text)
        if m:
            # The embedded string uses escaped unicode (\xHH sequences).
            raw_json = m.group(1).encode("utf-8").decode("unicode_escape")
            data = json.loads(raw_json)
            return parse_understat_data(data)

    raise RuntimeError("Could not find datesData in Understat page")
