"""Post daily SLOP LOCKS to a Discord webhook.

Reads the current per-sport data files and sends one Discord message with:
  - curated picks from ``slop_locks`` and ``longslop``
  - a fallback radar embed built from the top remaining matchups

Usage:
    python -m pipeline.notify_discord            # reads from data/
    DISCORD_WEBHOOK_URL=https://... python -m pipeline.notify_discord
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

COLOR_SLIME = 0x39FF14
COLOR_RADAR = 0x1F8B4C

SPORT_EMOJIS = {
    "nba": "🏀",
    "ncaam": "🎓",
    "mlb": "⚾",
    "mma": "🥊",
}
PICK_LABELS = {"draw": "DRAW"}
DATA_DIR = Path("data")
SPORT_ORDER = ("nba", "ncaam", "mlb", "mma")
MAX_CURATED_FIELDS = 8
MAX_RADAR_FIELDS = 5

_ET_OFFSET_SPORTS = {"nba", "ncaam"}


def _display_date(date_str: str, sport_key: str) -> str:
    """Return a human-readable date, correcting for UTC->ET shift on NBA/NCAAM."""
    if not date_str:
        return ""
    game_date = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    if sport_key in _ET_OFFSET_SPORTS:
        utc_today = datetime.now(timezone.utc).date()
        if game_date == utc_today + timedelta(days=1):
            game_date = utc_today
    return game_date.strftime("%b %d")


def _fmt_odds(american: int | None) -> str:
    if american is None:
        return "N/A"
    return f"+{american}" if american >= 0 else str(american)


def _fmt_pct(probability: float | None) -> str:
    if probability is None:
        return "N/A"
    return f"{probability * 100:.1f}%"


def _fmt_units(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}u"


def _pick_team(item: dict) -> str:
    """Return the picked side's display name."""
    outcome = item["pick"]
    if outcome == "home":
        return item["home_team"]
    if outcome == "away":
        return item["away_team"]
    return PICK_LABELS.get(outcome, outcome.upper())


def _pick_key(item: dict) -> tuple[str, str, str, str]:
    return (
        item["home_team"],
        item["away_team"],
        str(item.get("date", ""))[:10],
        item["pick"],
    )


def _iter_sport_data() -> list[tuple[str, dict]]:
    rows = []
    for sport_key in SPORT_ORDER:
        pred_path = DATA_DIR / sport_key / "predictions.json"
        if not pred_path.exists():
            continue
        with open(pred_path) as f:
            rows.append((sport_key, json.load(f)))
    return rows


def _match_lookup(matches: list[dict]) -> dict[tuple[str, str, str, str], dict]:
    lookup = {}
    for match in matches or []:
        model_probs = match.get("model_probs") or {}
        outcomes = list(model_probs.keys()) or [match.get("pick")]
        for outcome in outcomes:
            if not outcome:
                continue
            lookup[(
                match["home_team"],
                match["away_team"],
                str(match.get("date", ""))[:10],
                outcome,
            )] = match
    return lookup


def _enrich_item(item: dict, sport_key: str, source: str, match: dict | None = None) -> dict:
    """Fill derived notification fields from the backing match when needed."""
    enriched = dict(item)
    enriched["sport"] = sport_key
    enriched["source"] = source

    match = match or {}
    edge_data = (match.get("edges") or {}).get(enriched.get("pick"), {})

    for field in (
        "model_prob",
        "confidence_score",
        "expected_value",
        "kelly_fraction",
        "fractional_kelly",
        "american_odds",
        "edge",
        "implied_prob",
        "market_implied_prob",
        "decimal_odds",
    ):
        if enriched.get(field) is None and match.get(field) is not None:
            enriched[field] = match.get(field)
        if enriched.get(field) is None and edge_data.get(field) is not None:
            enriched[field] = edge_data.get(field)

    if enriched.get("american_odds") is None:
        best_odds = match.get("best_odds", {})
        if enriched.get("pick") in best_odds:
            enriched["american_odds"] = best_odds[enriched["pick"]]

    if enriched.get("home_pitcher") is None:
        enriched["home_pitcher"] = match.get("home_pitcher")
    if enriched.get("away_pitcher") is None:
        enriched["away_pitcher"] = match.get("away_pitcher")

    return enriched


def _build_curated_candidates() -> list[dict]:
    curated = []
    for sport_key, data in _iter_sport_data():
        lookup = _match_lookup(data.get("matches") or [])

        for lock in data.get("slop_locks") or []:
            curated.append(
                _enrich_item(lock, sport_key, "slop_lock", lookup.get(_pick_key(lock)))
            )

        longslop = data.get("longslop")
        if longslop:
            curated.append(
                _enrich_item(longslop, sport_key, "longslop", lookup.get(_pick_key(longslop)))
            )

    curated.sort(
        key=lambda item: (
            item.get("confidence_score", 0),
            item.get("expected_value", -999),
            item.get("edge", -999),
        ),
        reverse=True,
    )
    return curated


def _build_radar_candidates(excluded_keys: set[tuple[str, str, str, str]]) -> list[dict]:
    radar = []
    for sport_key, data in _iter_sport_data():
        for match in data.get("matches") or []:
            if match.get("completed"):
                continue
            candidate = _enrich_item(match, sport_key, "radar", match)
            key = _pick_key(candidate)
            if key in excluded_keys:
                continue
            radar.append(candidate)

    radar.sort(
        key=lambda item: (
            item.get("confidence_score", 0),
            item.get("model_prob", 0),
            item.get("edge", -999),
        ),
        reverse=True,
    )
    return radar


def _status_label(item: dict) -> str:
    if item.get("source") == "longslop":
        return " 🎯 **LONGSLOP**"

    conf_score = item.get("confidence_score", 0) or 0
    if conf_score >= 85:
        return " 🔥 **MASTER LOCK**"
    if conf_score >= 65:
        return " ✅ **VALIDATED**"
    return " 👀 **RADAR**"


def _pick_subtitle(item: dict) -> str:
    parts = []
    if item.get("source") == "longslop":
        parts.append("Longshot")
    elif item.get("source") == "slop_lock":
        parts.append("Curated")
    else:
        parts.append("Model radar")

    if item.get("expected_value") is not None:
        parts.append(f"EV {_fmt_units(item['expected_value'])}")
    if item.get("fractional_kelly") is not None:
        parts.append(f"Kelly {item['fractional_kelly'] * 100:.1f}%")

    return "  ·  ".join(parts)


def _pitcher_note(item: dict) -> str:
    if item.get("sport") != "mlb":
        return ""
    home_pitcher = item.get("home_pitcher") or "TBD"
    away_pitcher = item.get("away_pitcher") or "TBD"
    return f"\nPitchers: {away_pitcher} vs {home_pitcher}"


def _lock_field(item: dict, sport_key: str) -> dict:
    """Build one Discord embed field for a pick or radar matchup."""
    emoji = SPORT_EMOJIS.get(sport_key, "🎯")
    pick = _pick_team(item)
    date = _display_date(item.get("date", ""), sport_key)
    conf_score = item.get("confidence_score", 0) or 0

    name = f"{emoji}  {item['home_team']} vs {item['away_team']}  ·  {date}{_status_label(item)}"

    parts = [
        f"**{pick}**  ·  {_fmt_odds(item.get('american_odds'))}",
        f"{_fmt_pct(item.get('model_prob'))} model",
    ]
    if item.get("implied_prob") is not None:
        parts.append(f"{_fmt_pct(item.get('implied_prob'))} fair")
    if item.get("edge") is not None:
        parts.append(f"{item['edge'] * 100:+.1f}% edge")

    value = "  ·  ".join(parts)
    value += f"\nConfidence: **{conf_score}/100**  ·  {_pick_subtitle(item)}"

    pitcher_note = _pitcher_note(item)
    if pitcher_note:
        value += pitcher_note
    if item.get("blurb"):
        value += f"\n> {item['blurb']}"

    return {"name": name, "value": value[:1024], "inline": False}


def build_payload() -> dict:
    embeds = []
    et_now = datetime.now(ZoneInfo("America/New_York"))
    header_date = et_now.strftime("%b %d, %Y")

    curated = _build_curated_candidates()
    if curated:
        embeds.append({
            "title": "🔒  TOP PICKS OF THE DAY",
            "color": COLOR_SLIME,
            "fields": [_lock_field(item, item["sport"]) for item in curated[:MAX_CURATED_FIELDS]],
            "footer": {"text": "sloplocks.lol  ·  curated by confidence, EV, and edge"},
        })

    curated_keys = {_pick_key(item) for item in curated}
    radar = _build_radar_candidates(curated_keys)
    if radar:
        embeds.append({
            "title": "📡  MODEL RADAR",
            "color": COLOR_RADAR,
            "fields": [_lock_field(item, item["sport"]) for item in radar[:MAX_RADAR_FIELDS]],
            "footer": {"text": "top remaining matchups by model confidence"},
        })

    summary_parts = []
    if curated:
        count = len(curated)
        summary_parts.append(f"{count} curated pick{'s' if count != 1 else ''}")
    if radar:
        shown = min(len(radar), MAX_RADAR_FIELDS)
        summary_parts.append(f"{shown} radar spot{'s' if shown != 1 else ''}")
    if not summary_parts:
        summary_parts.append("No picks qualified today")

    return {
        "username": "BIG SLIME",
        "content": f"🎯  **SLOP LOCKS  ·  {header_date}**\n" + " · ".join(summary_parts),
        "embeds": embeds[:10],
    }


def main() -> None:
    if not DISCORD_WEBHOOK_URL:
        print("DISCORD_WEBHOOK_URL not set — skipping Discord notification")
        sys.exit(0)

    payload = build_payload()

    if not payload["embeds"]:
        print("No locks to notify about today")
        sys.exit(0)

    resp = requests.post(
        DISCORD_WEBHOOK_URL,
        json=payload,
        timeout=15,
    )

    if resp.status_code == 204:
        print("Discord notification sent successfully")
    else:
        print(f"Discord webhook error: {resp.status_code} — {resp.text}")
        sys.exit(1)


if __name__ == "__main__":
    main()
