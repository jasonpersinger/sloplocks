"""Post daily SLOP LOCKS to a Discord webhook.

Reads the current data files and sends one Discord message with:
  - SLOP OF THE DAY embed (gold, highlighted)
  - One embed per sport with locks (slime green)

Usage:
    python -m pipeline.notify_discord            # reads from data/
    DISCORD_WEBHOOK_URL=https://... python -m pipeline.notify_discord
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

# Discord embed colors (decimal representation of hex)
COLOR_SLIME = 0x39FF14   # #39FF14 — slime green
COLOR_GOLD  = 0xFFD60A   # #FFD60A — yellow

SPORT_EMOJIS = {"nba": "🏀", "ncaam": "🎓", "epl": "⚽"}
PICK_LABELS  = {"home": "HOME", "away": "AWAY", "draw": "DRAW"}
DATA_DIR = Path("data")


def _fmt_odds(american: int) -> str:
    return f"+{american}" if american >= 0 else str(american)


def _fmt_pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def _lock_field(lock: dict, sport_key: str) -> dict:
    """Build a single Discord embed field for one lock."""
    emoji = SPORT_EMOJIS.get(sport_key, "🎯")
    pick  = PICK_LABELS.get(lock["pick"], lock["pick"].upper())
    date  = lock.get("date", "")[:10]

    name  = f"{emoji}  {lock['home_team']} vs {lock['away_team']}  ·  {date}"
    value = (
        f"**{pick}**  ·  {_fmt_odds(lock['american_odds'])}"
        f"  ·  {_fmt_pct(lock['model_prob'])} conf"
        f"  ·  {lock['edge'] * 100:+.1f}% edge"
    )
    if lock.get("blurb"):
        # Discord quote block for the blurb
        value += f"\n> {lock['blurb']}"

    return {"name": name, "value": value[:1024], "inline": False}


def _sotd_embed(sotd: dict) -> dict | None:
    """Build the SLOP OF THE DAY embed, or None if no pick."""
    pick = sotd.get("pick")
    if not pick:
        return None

    sport_key  = sotd.get("sport", "")
    sport_name = sotd.get("sport_name", sport_key.upper())
    emoji      = SPORT_EMOJIS.get(sport_key, "⭐")
    pick_label = PICK_LABELS.get(pick["pick"], pick["pick"].upper())

    description = (
        f"{emoji}  **{pick['home_team']} vs {pick['away_team']}**  ·  {pick.get('date','')[:10]}\n"
        f"**{pick_label}**  ·  {_fmt_odds(pick['american_odds'])}"
        f"  ·  {_fmt_pct(pick['model_prob'])} conf"
        f"  ·  {pick['edge'] * 100:+.1f}% edge"
    )
    if pick.get("blurb"):
        description += f"\n> {pick['blurb']}"

    return {
        "title": "⭐  SLOP OF THE DAY",
        "description": description[:4096],
        "color": COLOR_GOLD,
        "footer": {"text": f"Most confident pick across all sports  ·  {sport_name}"},
    }


def build_payload() -> dict:
    today = datetime.now(timezone.utc).strftime("%b %d, %Y")
    embeds = []

    # --- SLOP OF THE DAY ---
    sotd_path = DATA_DIR / "sotd.json"
    if sotd_path.exists():
        with open(sotd_path) as f:
            sotd = json.load(f)
        sotd_embed = _sotd_embed(sotd)
        if sotd_embed:
            embeds.append(sotd_embed)

    # --- Per-sport lock embeds ---
    for sport_key in ("nba", "ncaam", "epl"):
        pred_path = DATA_DIR / sport_key / "predictions.json"
        if not pred_path.exists():
            continue
        with open(pred_path) as f:
            data = json.load(f)

        locks = data.get("slop_locks") or []
        if not locks:
            continue

        sport_name = data.get("sport_name", sport_key.upper())
        emoji      = SPORT_EMOJIS.get(sport_key, "🎯")
        fields     = [_lock_field(lock, sport_key) for lock in locks]

        embeds.append({
            "title": f"{emoji}  {sport_name} SLOP LOCKS",
            "color": COLOR_SLIME,
            "fields": fields,
            "footer": {"text": "sloplocks.lol"},
        })

    return {
        "username": "BIG SLIME",
        "content": f"🔒  **SLOP LOCKS  ·  {today}**",
        "embeds": embeds[:10],  # Discord allows max 10 embeds per message
    }


def main() -> None:
    if not DISCORD_WEBHOOK_URL:
        print("DISCORD_WEBHOOK_URL not set — skipping Discord notification")
        sys.exit(0)

    payload = build_payload()

    if not payload["embeds"]:
        print("No locks or SOTD to notify about today")
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
