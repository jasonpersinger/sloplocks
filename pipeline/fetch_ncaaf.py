"""Fetch NCAAF (FBS college football) results and schedule from ESPN.

College football is not "the NFL with more teams". The three structural
differences this module exists to handle are:

* **Division boundary.** ESPN's college-football scoreboard covers FBS, FCS and
  below. Only FBS is modelled; FBS-vs-FCS games are kept (they are real results
  for the FBS side) but the FCS opponent is collapsed into one synthetic team
  so the rating pool does not fill with hundreds of one-game entries.
* **Conferences and uneven schedules.** Teams play wildly different opponent
  sets, so conference membership and whether a game is in-conference are
  carried through as metadata.
* **Neutral sites.** Week-one kickoff games, conference championships and bowls
  are all neutral, and are common enough that ignoring them would bias the
  home-field term.
"""

import json as _json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

from pipeline.config import NCAAF_CORE_API_BASE, NCAAF_ESPN_BASE

_REQUEST_DELAY = 0.4

# ESPN group id for Division I FBS.
FBS_GROUP = 80

# Every non-FBS opponent is folded into this single rating entity.
FCS_OPPONENT = "FCS Opponent"

_team_map: Optional[dict[str, str]] = None
_team_conferences: Optional[dict[str, Optional[str]]] = None

# Odds API names that do not match an ESPN displayName or location.
_ODDS_API_FALLBACK: dict[str, str] = {
    "Louisiana State": "LSU",
    "Louisiana State Tigers": "LSU",
    "Southern California": "USC",
    "Southern California Trojans": "USC",
    "Texas Christian": "TCU",
    "Texas Christian Horned Frogs": "TCU",
    "Brigham Young": "BYU",
    "Brigham Young Cougars": "BYU",
    "Southern Methodist": "SMU",
    "Southern Methodist Mustangs": "SMU",
    "Central Florida": "UCF",
    "Central Florida Knights": "UCF",
    "Mississippi": "Ole Miss",
    "Mississippi Rebels": "Ole Miss",
    "Connecticut": "UConn",
    "Connecticut Huskies": "UConn",
    "Nevada-Las Vegas": "UNLV",
    "Nevada-Las Vegas Rebels": "UNLV",
    "Miami (FL)": "Miami",
    "Miami Florida": "Miami",
    "Miami Ohio": "Miami (OH)",
    "Massachusetts": "UMass",
    "Massachusetts Minutemen": "UMass",
    "Texas-San Antonio": "UTSA",
    "Texas-El Paso": "UTEP",
    "Alabama-Birmingham": "UAB",
    "Hawai'i": "Hawai'i",
    "Hawaii": "Hawai'i",
}


def _fbs_conference_team_ids(season: int) -> dict[str, str]:
    """Return ``team_id -> conference_id`` for every FBS team in a season.

    The site ``/teams`` endpoint ignores the ``groups`` filter and carries no
    conference field, so FBS membership comes from the core API's group tree
    (group 80 = FBS, whose children are the conferences). Team ids are read out
    of the ``$ref`` URLs rather than dereferenced, which keeps this to roughly
    a dozen requests instead of one per team.
    """
    children_url = (
        f"{NCAAF_CORE_API_BASE}/seasons/{season}/types/2/groups/{FBS_GROUP}/children"
    )
    resp = requests.get(children_url, timeout=30)
    resp.raise_for_status()
    conferences = resp.json().get("items", [])

    team_conference_ids: dict[str, str] = {}
    for conference in conferences:
        ref = conference.get("$ref")
        conference_id = _id_from_ref(ref)
        if not conference_id:
            continue
        teams_url = (
            f"{NCAAF_CORE_API_BASE}/seasons/{season}/types/2"
            f"/groups/{conference_id}/teams?limit=200"
        )
        try:
            teams_resp = requests.get(teams_url, timeout=30)
            teams_resp.raise_for_status()
            items = teams_resp.json().get("items", [])
        except (requests.RequestException, ValueError):
            continue
        for item in items:
            team_id = _id_from_ref(item.get("$ref"))
            if team_id:
                team_conference_ids[team_id] = conference_id
        time.sleep(_REQUEST_DELAY)

    return team_conference_ids


def _id_from_ref(ref: Optional[str]) -> Optional[str]:
    """Extract the trailing numeric id from an ESPN core API ``$ref`` URL."""
    if not ref:
        return None
    path = str(ref).split("?")[0].rstrip("/")
    tail = path.rsplit("/", 1)[-1]
    return tail if tail.isdigit() else None


def _build_team_tables() -> tuple[dict[str, str], dict[str, Optional[str]]]:
    """Build the alias->canonical-name map and the FBS conference lookup.

    ``location`` is used as the canonical short name (e.g. "Ohio State"),
    matching the convention already used for NCAAM, because college nicknames
    are not unique across the division.

    Aliases are built for *all* college teams, FBS or not, so that an FCS name
    normalises consistently before the membership check collapses it.
    """
    url = f"{NCAAF_ESPN_BASE}/teams?limit=900"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    names: dict[str, str] = {}
    canonical_by_id: dict[str, str] = {}
    try:
        teams = data["sports"][0]["leagues"][0]["teams"]
    except (KeyError, IndexError):
        teams = []

    for entry in teams:
        team = entry.get("team") or {}
        canonical = team.get("location") or team.get("displayName")
        if not canonical:
            continue
        for alias in (
            team.get("displayName"),
            team.get("shortDisplayName"),
            team.get("nickname"),
            team.get("location"),
        ):
            # Deliberately not aliasing ``abbreviation``: college abbreviations
            # collide across divisions (multiple schools answer to "SAM").
            if alias:
                names.setdefault(alias, canonical)
        if team.get("id"):
            canonical_by_id[str(team["id"])] = canonical

    season = _current_ncaaf_season()
    team_conference_ids = {}
    for candidate_season in (season, season - 1):
        try:
            team_conference_ids = _fbs_conference_team_ids(candidate_season)
        except (requests.RequestException, ValueError):
            team_conference_ids = {}
        if team_conference_ids:
            break

    if not team_conference_ids:
        # Without this table every team would look non-FBS and the whole slate
        # would be silently discarded. Fail loudly instead.
        raise RuntimeError(
            "Could not load the ESPN FBS conference table; refusing to run "
            "NCAAF with an unknown division membership."
        )

    conferences: dict[str, Optional[str]] = {}
    for team_id, conference_id in team_conference_ids.items():
        canonical = canonical_by_id.get(team_id)
        if canonical:
            conferences[canonical] = conference_id

    return names, conferences


def _ensure_team_tables() -> None:
    global _team_map, _team_conferences
    if _team_map is None or _team_conferences is None:
        _team_map, _team_conferences = _build_team_tables()


def normalize_ncaaf_team_name(name: str) -> str:
    """Map an ESPN or Odds API NCAAF team name to its canonical short name."""
    _ensure_team_tables()
    mapped = _team_map.get(name)
    if mapped:
        return mapped
    fallback = _ODDS_API_FALLBACK.get(name)
    if fallback:
        return _team_map.get(fallback, fallback)
    return name


def is_fbs_team(name: str) -> bool:
    """Return whether a normalised team name belongs to FBS."""
    _ensure_team_tables()
    return name in _team_conferences


def team_conference(name: str) -> Optional[str]:
    """Return the ESPN conference id for a normalised FBS team name."""
    _ensure_team_tables()
    return _team_conferences.get(name)


def _resolve_team(raw_name: str) -> str:
    """Normalise a team name, collapsing non-FBS opponents into one entity."""
    normalized = normalize_ncaaf_team_name(raw_name)
    return normalized if is_fbs_team(normalized) else FCS_OPPONENT


# ---- season handling ---------------------------------------------------------


def _current_ncaaf_season() -> int:
    """Return the start year of the current college football season."""
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 8 else now.year - 1


def _season_date_range(season: int, history_seasons: int = 1) -> list[str]:
    """Generate dates covering one or more seasons, week zero through the title game.

    ``history_seasons`` counts back from ``season`` inclusive. College team
    strength is strongly persistent year over year, so prior seasons are the
    only thing separating 138 teams before conference play begins.
    """
    start = datetime(season - max(0, history_seasons - 1), 8, 20)
    end = min(datetime(season + 1, 1, 25), datetime.now(timezone.utc).replace(tzinfo=None))
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


# ---- ESPN cache helpers -------------------------------------------------------


def _load_espn_cache(cache_path: Optional[str]) -> dict:
    """Load cached ESPN game data."""
    if cache_path is None or not os.path.exists(cache_path):
        return {"games": {}}
    with open(cache_path) as f:
        cache = _json.load(f)
    if not isinstance(cache, dict):
        return {"games": {}}
    cache.setdefault("games", {})
    return cache


def _save_espn_cache(cache_path: Optional[str], cache: dict) -> None:
    """Write the ESPN game cache."""
    if cache_path is None:
        return
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        _json.dump(cache, f)


def _incremental_dates(cache: dict, all_dates: list[str], lookback_days: int = 3) -> list[str]:
    """Return only the dates that still need fetching."""
    games = cache.get("games", {})
    if not games:
        return all_dates
    cached_dates = [v["date"] for v in games.values()]
    max_cached = max(cached_dates)
    min_cached = min(cached_dates)
    cutoff = (datetime.strptime(max_cached, "%Y-%m-%d") - timedelta(days=lookback_days)).date()
    earliest = datetime.strptime(min_cached, "%Y-%m-%d").date()
    # Fetch forward from the cache head, and also backfill anything older than
    # the cache covers, so raising ``history_seasons`` actually pulls the
    # earlier seasons in instead of silently doing nothing.
    return [
        d
        for d in all_dates
        if datetime.strptime(d, "%Y-%m-%d").date() >= cutoff
        or datetime.strptime(d, "%Y-%m-%d").date() < earliest
    ]


# ---- scoreboard parsing -------------------------------------------------------


def _conference_context(home_team: str, away_team: str) -> dict:
    """Describe the conference relationship between two normalised teams."""
    home_conf = team_conference(home_team)
    away_conf = team_conference(away_team)
    return {
        "home_conference": home_conf,
        "away_conference": away_conf,
        "conference_game": bool(home_conf and away_conf and home_conf == away_conf),
        # An FBS team playing the synthetic FCS entity is a guarantee game and
        # is flagged so it can be identified in stored records.
        "fcs_matchup": FCS_OPPONENT in (home_team, away_team),
    }


def _parse_final_event(event: dict) -> Optional[dict]:
    """Parse an ESPN scoreboard event into a finished-game dict.

    Returns ``None`` unless the game is complete. Final scores already include
    college overtime, which cannot end in a tie, so no tie handling is needed
    beyond what the shared pipeline already does.
    """
    competitions = event.get("competitions") or []
    if not competitions:
        return None
    comp = competitions[0]
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

    try:
        home_score = int(home["score"])
        away_score = int(away["score"])
    except (KeyError, TypeError, ValueError):
        return None

    home_team = _resolve_team(home["team"]["displayName"])
    away_team = _resolve_team(away["team"]["displayName"])
    # Two non-FBS teams would both collapse to the synthetic entity; such a
    # game tells us nothing about FBS strength.
    if home_team == FCS_OPPONENT and away_team == FCS_OPPONENT:
        return None

    season = event.get("season") or {}
    week = event.get("week") or {}
    return {
        "date": event["date"][:10],
        "home_team": home_team,
        "away_team": away_team,
        "home_goals": home_score,
        "away_goals": away_score,
        "neutral": bool(comp.get("neutralSite", False)),
        "overtime": bool(comp.get("status", {}).get("period", 0) > 4),
        "season_year": season.get("year"),
        "season_type": season.get("type"),
        "week": week.get("number"),
        **_conference_context(home_team, away_team),
    }


def fetch_ncaaf_games(
    season: Optional[int] = None,
    dates: Optional[list[str]] = None,
    cache_path: Optional[str] = None,
    history_seasons: int = 1,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Fetch finished FBS games via the ESPN scoreboard.

    Returns
    -------
    (games_df, None)
        ``games_df`` columns: game_id, date, home_team, away_team, home_goals,
        away_goals, neutral, overtime, season_year, season_type, week,
        home_conference, away_conference, conference_game, fcs_matchup.
        The second element is ``None``: NCAAF runs no box-score model.
    """
    if season is None:
        season = _current_ncaaf_season()
    if dates is None:
        dates = _season_date_range(season, history_seasons=history_seasons)

    cache = _load_espn_cache(cache_path)
    fetch_dates = _incremental_dates(cache, dates)

    # One request per date. ESPN's scoreboard accepted YYYYMMDD-YYYYMMDD ranges
    # until 2026-09-16 and now answers them with a 400, so this matches the
    # single-date pattern every other fetcher in this repo uses.
    for date_str in fetch_dates:
        url = (
            f"{NCAAF_ESPN_BASE}/scoreboard"
            f"?dates={date_str.replace('-', '')}&limit=900&groups={FBS_GROUP}"
        )
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError):
            continue

        for event in data.get("events", []):
            parsed = _parse_final_event(event)
            if parsed is None:
                continue
            cache["games"][event["id"]] = parsed

        time.sleep(_REQUEST_DELAY)

    _save_espn_cache(cache_path, cache)

    rows = [{"game_id": game_id, **entry} for game_id, entry in cache["games"].items()]
    columns = [
        "game_id", "date", "home_team", "away_team", "home_goals", "away_goals",
        "neutral", "overtime", "season_year", "season_type", "week",
        "home_conference", "away_conference", "conference_game", "fcs_matchup",
    ]
    games_df = pd.DataFrame(rows, columns=columns)
    if not games_df.empty:
        games_df = games_df.sort_values("date").reset_index(drop=True)
    return games_df, None


def fetch_ncaaf_schedule(cache_path: Optional[str] = None) -> list[dict]:
    """Fetch upcoming FBS fixtures from ESPN.

    Walks the window one day at a time, because college football plays Thursday
    through Saturday with midweek MACtion, and bowl season scatters games across
    two weeks. Fixtures where the opponent is not FBS are
    dropped: a rating for the synthetic FCS entity is useful for fitting but
    not something to publish a pick on.

    ``cache_path`` is accepted for signature parity with the other schedule
    fetchers; the schedule is always fetched live.
    """
    et_offset = timedelta(hours=5)
    today_et = (datetime.now(timezone.utc) - et_offset).date()
    window = [today_et + timedelta(days=offset) for offset in range(-1, 3)]

    events = []
    seen_event_ids = set()
    for day in window:
        url = (
            f"{NCAAF_ESPN_BASE}/scoreboard"
            f"?dates={day.strftime('%Y%m%d')}&limit=900&groups={FBS_GROUP}"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        for event in resp.json().get("events", []):
            # A late kickoff can surface on two adjacent scoreboard days.
            event_id = event.get("id")
            if event_id in seen_event_ids:
                continue
            seen_event_ids.add(event_id)
            events.append(event)
        time.sleep(_REQUEST_DELAY)

    fixtures = []
    for event in events:
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        comp = competitions[0]
        status_type = comp.get("status", {}).get("type", {})

        home = away = None
        for competitor in comp.get("competitors", []):
            if competitor.get("homeAway") == "home":
                home = competitor
            elif competitor.get("homeAway") == "away":
                away = competitor
        if home is None or away is None:
            continue

        home_team = _resolve_team(home["team"]["displayName"])
        away_team = _resolve_team(away["team"]["displayName"])
        if FCS_OPPONENT in (home_team, away_team):
            continue

        summary_injuries = []
        try:
            time.sleep(_REQUEST_DELAY)
            summary_resp = requests.get(
                f"{NCAAF_ESPN_BASE}/summary?event={event.get('id')}", timeout=30
            )
            if summary_resp.status_code == 200:
                summary_injuries = summary_resp.json().get("injuries", [])
        except (requests.RequestException, ValueError):
            summary_injuries = []

        start_time = comp.get("date", event.get("date"))
        fixtures.append({
            "home_team": home_team,
            "away_team": away_team,
            "date": _eastern_date(start_time, fallback=today_et),
            "start_time": start_time,
            "completed": status_type.get("completed", False),
            "neutral": bool(comp.get("neutralSite", False)),
            "week": (event.get("week") or {}).get("number"),
            "season_type": (event.get("season") or {}).get("type"),
            "summary_injuries": summary_injuries,
            **_conference_context(home_team, away_team),
        })

    return fixtures


def _eastern_date(start_time: Optional[str], fallback) -> str:
    """Return the US-Eastern calendar date for a UTC kickoff timestamp."""
    if not start_time:
        return fallback.strftime("%Y-%m-%d")
    try:
        parsed = datetime.fromisoformat(str(start_time).replace("Z", "+00:00"))
    except ValueError:
        return fallback.strftime("%Y-%m-%d")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (parsed.astimezone(timezone.utc) - timedelta(hours=5)).strftime("%Y-%m-%d")
