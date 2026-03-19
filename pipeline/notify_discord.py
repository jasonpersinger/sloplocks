"""Post daily SLOP LOCKS to a Discord webhook.

Reads the current data files and sends one Discord message with:
  - Top 5 SLOP LOCKS embed (slime green)

Usage:
    python -m pipeline.notify_discord            # reads from data/
    DISCORD_WEBHOOK_URL=https://... python -m pipeline.notify_discord
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

import requests

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

# Discord embed colors (decimal representation of hex)
COLOR_SLIME = 0x39FF14   # #39FF14 — slime green

SPORT_EMOJIS = {"nba": "🏀", "ncaam": "🎓"}
PICK_LABELS  = {"draw": "DRAW"}
DATA_DIR = Path("data")

# NBA/NCAAM evening games (7–10pm ET) are stored as the next UTC day.
# Subtract one day to recover the correct local date for display.
_ET_OFFSET_SPORTS = {"nba", "ncaam"}


def _display_date(date_str: str, sport_key: str) -> str:
    """Return a human-readable date, correcting for UTC→ET shift on NBA/NCAAM."""
    if not date_str:
        return ""
    game_date = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    if sport_key in _ET_OFFSET_SPORTS:
        utc_today = datetime.now(timezone.utc).date()
        # If the stored date is tomorrow UTC, it's actually tonight ET
        if game_date == utc_today + timedelta(days=1):
            game_date = utc_today
    return game_date.strftime("%b %d")


def _fmt_odds(american: int | None) -> str:
    if american is None:
        return "N/A"
    return f"+{american}" if american >= 0 else str(american)


def _fmt_pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def _pick_team(lock: dict) -> str:
    """Return the name of the team being picked (or DRAW)."""
    outcome = lock["pick"]
    if outcome == "home":
        return lock["home_team"]
    if outcome == "away":
        return lock["away_team"]
    return "DRAW"


def _lock_field(lock: dict, sport_key: str) -> dict:
    """Build a single Discord embed field for one lock."""
    emoji = SPORT_EMOJIS.get(sport_key, "🎯")
    pick  = _pick_team(lock)
    date  = _display_date(lock.get("date", ""), sport_key)
    stars = lock.get("confidence_stars", 1)
    star_str = "⭐" * stars

    name  = f"{emoji}  {lock['home_team']} vs {lock['away_team']}  ·  {date}"
    odds  = lock.get("american_odds")
    value = (
        f"**{pick}**  ·  {_fmt_odds(odds)}"
        f"  ·  {_fmt_pct(lock['model_prob'])} conf"
        f"  ·  {lock['edge'] * 100:+.1f}% edge"
        f"\n{star_str} Rating"
    )
    if lock.get("blurb"):
        # Discord quote block for the blurb
        value += f"\n> {lock['blurb']}"

    return {"name": name, "value": value[:1024], "inline": False}


def build_payload() -> dict:
    embeds = []
    et_now = datetime.now(ZoneInfo("America/New_York"))
    header_date = et_now.strftime("%b %d, %Y")

    # --- Collect all candidates across every sport ---
    all_candidates: list[tuple[str, dict]] = []  # (sport_key, match)
    for sport_key in ("nba", "ncaam"):
        pred_path = DATA_DIR / sport_key / "predictions.json"
        if not pred_path.exists():
            continue
        with open(pred_path) as f:
            data = json.load(f)
        
        # Use a dict to deduplicate by (home, away)
        seen_matches = {}
        
        # 1. Add Slop Locks (these already have confidence_stars from run.py)
        for lock in data.get("slop_locks") or []:
            key = (lock["home_team"], lock["away_team"])
            seen_matches[key] = lock
            
        # 2. Add all other matches (all now have confidence_stars from run.py)
        for match in data.get("matches") or []:
            key = (match["home_team"], match["away_team"])
            if key not in seen_matches:
                seen_matches[key] = match
        
        for match in seen_matches.values():
            all_candidates.append((sport_key, match))

    # Sort all candidates by star rating (desc), then model prob (desc)
    all_candidates.sort(key=lambda x: (x[1].get("confidence_stars", 0), x[1].get("model_prob", 0)), reverse=True)
    top10 = all_candidates[:10]

    if top10:
        fields = [_lock_field(match, sport_key) for sport_key, match in top10]
        embeds.append({
            "title": "\u2B50  TOP MATCHUPS",
            "color": COLOR_SLIME,
            "fields": fields,
            "footer": {"text": "sloplocks.lol  ·  ranked by confidence rating"},
        })

    return {
        "username": "BIG SLIME",
        "content": f"🔒  **SLOP LOCKS  ·  {header_date}**",
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
