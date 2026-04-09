import json
import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

# Map league keys to ESPN API slugs
LEAGUE_MAP = {
    "epl": "eng.1",
    "ucl": "uefa.champions",
    "uel": "uefa.europa",
    "laliga": "esp.1",
    "bundesliga": "ger.1",
}

_REQUEST_DELAY = 0.5

def _get_base_url(league_key):
    slug = LEAGUE_MAP.get(league_key, "eng.1")
    return f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}"

def fetch_soccer_games(league_key, cache_path=None):
    """Fetch historical games for a specific soccer league."""
    base_url = _get_base_url(league_key)
    
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            cache = json.load(f)
    else:
        cache = {"games": {}}

    today = datetime.now(timezone.utc)
    # Fetch last 30 days to build context
    dates_to_fetch = []
    for i in range(30):
        d = today - timedelta(days=i)
        dates_to_fetch.append(d.strftime("%Y%m%d"))

    for date_str in dates_to_fetch:
        if date_str in cache.get("fetched_dates", []):
            continue
            
        url = f"{base_url}/scoreboard?dates={date_str}"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            for event in data.get("events", []):
                game_id = event["id"]
                comp = event["competitions"][0]
                status = comp["status"]["type"]["name"]
                
                if status == "STATUS_FINAL":
                    home = away = None
                    for team in comp["competitors"]:
                        if team["homeAway"] == "home":
                            home = team
                        else:
                            away = team
                    
                    cache["games"][game_id] = {
                        "date": event["date"][:10],
                        "home_team": normalize_soccer_name(home["team"]["displayName"]),
                        "away_team": normalize_soccer_name(away["team"]["displayName"]),
                        "home_goals": int(home["score"]),
                        "away_goals": int(away["score"]),
                    }
            
            cache.setdefault("fetched_dates", []).append(date_str)
            time.sleep(_REQUEST_DELAY)
        except Exception as e:
            print(f"Error fetching {league_key} date {date_str}: {e}")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(cache, f)

    df = pd.DataFrame(cache["games"].values())
    return df

def fetch_soccer_schedule(league_key):
    """Fetch upcoming fixtures and injury data for a soccer league."""
    base_url = _get_base_url(league_key)
    url = f"{base_url}/scoreboard"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    
    fixtures = []
    for event in data.get("events", []):
        comp = event["competitions"][0]
        status = comp["status"]["type"]["name"]
        
        if status != "STATUS_FINAL":
            home = away = None
            for team in comp["competitors"]:
                if team["homeAway"] == "home":
                    home = team
                else:
                    away = team
            
            # Fetch injuries from summary
            summary_injuries = []
            try:
                summary_url = f"{base_url}/summary?event={event['id']}"
                s_resp = requests.get(summary_url, timeout=10)
                if s_resp.status_code == 200:
                    summary_injuries = s_resp.json().get("injuries", [])
            except:
                pass

            fixtures.append({
                "home_team": normalize_soccer_name(home["team"]["displayName"]),
                "away_team": normalize_soccer_name(away["team"]["displayName"]),
                "date": event["date"][:10],
                "start_time": event["date"],
                "summary_injuries": summary_injuries,
                "completed": False,
            })
            
    return fixtures

def normalize_soccer_name(name):
    """Map ESPN names to The Odds API names."""
    mapping = {
        # EPL
        "Manchester United": "Manchester United",
        "Manchester City": "Manchester City",
        "Tottenham Hotspur": "Tottenham Hotspur",
        "Wolverhampton Wanderers": "Wolverhampton",
        "Brighton & Hove Albion": "Brighton",
        "West Ham United": "West Ham United",
        "Nottingham Forest": "Nottingham Forest",
        "Sheffield United": "Sheffield United",
        "Luton Town": "Luton",
        # UCL / General
        "Bayern Munich": "Bayern Munich",
        "Paris Saint-Germain": "Paris Saint Germain",
        "Real Madrid": "Real Madrid",
        "Atletico Madrid": "Atletico Madrid",
        "Borussia Dortmund": "Borussia Dortmund",
        "Inter Milan": "Inter Milan",
        "AC Milan": "AC Milan",
        "AS Roma": "AS Roma",
        "Napoli": "Inter", # Check mapping
    }
    return mapping.get(name, name)
