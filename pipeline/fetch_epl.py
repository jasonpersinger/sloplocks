import json
import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

# API Base for EPL
EPL_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1"
_REQUEST_DELAY = 0.5

def fetch_epl_games(cache_path=None, seasons=None):
    """
    Fetch historical EPL games. 
    Soccer scores are treated as 'goals' to keep consistent with NHL/MLB schema.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            cache = json.load(f)
    else:
        cache = {"games": {}}

    # For now, we'll fetch the current month's games to populate history
    # In a full implementation, this would loop through seasons.
    today = datetime.now(timezone.utc)
    dates_to_fetch = []
    for i in range(30): # Last 30 days
        d = today - timedelta(days=i)
        dates_to_fetch.append(d.strftime("%Y%m%d"))

    new_count = 0
    for date_str in dates_to_fetch:
        if date_str in cache.get("fetched_dates", []):
            continue
            
        url = f"{EPL_ESPN_BASE}/scoreboard?dates={date_str}"
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
                        "home_team": normalize_epl_name(home["team"]["displayName"]),
                        "away_team": normalize_epl_name(away["team"]["displayName"]),
                        "home_goals": int(home["score"]),
                        "away_goals": int(away["score"]),
                    }
                    new_count += 1
            
            cache.setdefault("fetched_dates", []).append(date_str)
            time.sleep(_REQUEST_DELAY)
        except Exception as e:
            print(f"Error fetching EPL date {date_str}: {e}")

    if cache_path:
        with open(cache_path, 'w') as f:
            json.dump(cache, f)

    df = pd.DataFrame(cache["games"].values())
    return df

def fetch_epl_schedule(cache_path=None):
    """Fetch upcoming EPL fixtures and their injury/context data."""
    url = f"{EPL_ESPN_BASE}/scoreboard"
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
            
            # Fetch summary for injuries
            summary_injuries = []
            try:
                summary_url = f"{EPL_ESPN_BASE}/summary?event={event['id']}"
                s_resp = requests.get(summary_url, timeout=10)
                if s_resp.status_code == 200:
                    summary_injuries = s_resp.json().get("injuries", [])
            except:
                pass

            fixtures.append({
                "home_team": normalize_epl_name(home["team"]["displayName"]),
                "away_team": normalize_epl_name(away["team"]["displayName"]),
                "date": event["date"][:10],
                "start_time": event["date"],
                "summary_injuries": summary_injuries,
                "completed": False,
            })
            
    return fixtures

def normalize_epl_name(name):
    """Normalize common EPL team names to match The Odds API."""
    mapping = {
        "Manchester United": "Manchester United",
        "Manchester City": "Manchester City",
        "Tottenham Hotspur": "Tottenham Hotspur",
        "Wolverhampton Wanderers": "Wolverhampton",
        "Brighton & Hove Albion": "Brighton",
        "West Ham United": "West Ham United",
        "Nottingham Forest": "Nottingham Forest",
        "Sheffield United": "Sheffield United",
        "Luton Town": "Luton",
    }
    return mapping.get(name, name)
