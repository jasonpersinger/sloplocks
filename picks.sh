#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if [ -z "$ODDS_API_KEY" ]; then
  read -rp "Enter ODDS_API_KEY: " ODDS_API_KEY
  export ODDS_API_KEY
fi

python3 - <<'EOF'
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline.run import run_sport_pipeline

def fmt_picks(picks, longslop):
    if not picks:
        print("  No locks today.")
    for i, p in enumerate(picks, 1):
        team = p["home_team"] if p["pick"] == "home" else p["away_team"]
        opp  = p["away_team"] if p["pick"] == "home" else p["home_team"]
        print(f"  {i}. {team} vs {opp}  |  {p['american_odds']:+d}  |  model {p['model_prob']:.0%}  edge {p['edge']:+.1%}")
        if p.get("blurb"):
            print(f"     {p['blurb']}")
    if longslop:
        p = longslop
        team = p["home_team"] if p["pick"] == "home" else p["away_team"]
        opp  = p["away_team"] if p["pick"] == "home" else p["home_team"]
        print(f"\n  LONGSLOP: {team} vs {opp}  |  {p['american_odds']:+d}  |  model {p['model_prob']:.0%}")
        if p.get("blurb"):
            print(f"  {p['blurb']}")

for sport in ["nba", "ncaam"]:
    print(f"\n{'='*50}")
    print(f"  {sport.upper()} SLOP LOCKS")
    print(f"{'='*50}")
    try:
        result = run_sport_pipeline(sport)
        fmt_picks(result["slop_locks"], result["longslop"])
    except Exception as e:
        print(f"  Error: {e}")

print()
EOF
