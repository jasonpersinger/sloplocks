import json
from datetime import datetime, timedelta, timezone

def get_game_context(sport_key, fixture, summary_data=None):
    """
    Assembles a raw text blob of qualitative context for a game.
    """
    context_parts = []
    
    # 1. Injuries / Availability
    injury_text = _extract_injury_text(sport_key, fixture, summary_data)
    if injury_text:
        context_parts.append("### Injuries & Availability")
        context_parts.append(injury_text)
    
    # 2. Schedule Context (Rest, Travel, B2B)
    schedule_text = _extract_schedule_text(sport_key, fixture)
    if schedule_text:
        context_parts.append("### Schedule & Fatigue")
        context_parts.append(schedule_text)
        
    # 3. Sport-specific extras (Weather for MLB, etc.)
    extras_text = _extract_sport_extras(sport_key, fixture, summary_data)
    if extras_text:
        context_parts.append("### Additional Factors")
        context_parts.append(extras_text)
        
    return "\n\n".join(context_parts)

def _extract_injury_text(sport_key, fixture, summary_data):
    lines = []
    
    # Prefer summary_data if passed explicitly, otherwise use summary_injuries from fixture
    injuries = []
    if summary_data and "injuries" in summary_data:
        injuries = summary_data.get("injuries", [])
    elif fixture.get("summary_injuries"):
        injuries = fixture.get("summary_injuries", [])
        
    if injuries:
        for injury_block in injuries:
            team_name = (injury_block.get("team") or {}).get("displayName", "Team")
            team_injuries = injury_block.get("injuries", [])
            if team_injuries:
                lines.append(f"{team_name}:")
                for inj in team_injuries:
                    player = (inj.get("athlete") or {}).get("displayName", "Unknown Player")
                    status = inj.get("status", "Unknown Status")
                    desc = inj.get("comment", "")
                    lines.append(f"  - {player} ({status}): {desc}")
    
    return "\n".join(lines)

def _extract_schedule_text(sport_key, fixture):
    # This info is often computed in run_sport_pipeline, 
    # but we can try to derive some basic context here or pass it in.
    lines = []
    
    # Placeholder for travel/rest logic
    # In a real implementation, we'd check previous games dates.
    # For now, we'll use whatever's in the fixture or generic notes.
    if fixture.get("neutral"):
        lines.append("- Neutral site game.")
    
    # Specific sport notes
    if sport_key == "nba":
        # We could add B2B or 3-in-4 notes if we had them in the fixture
        pass
        
    return "\n".join(lines)

def _extract_sport_extras(sport_key, fixture, summary_data):
    lines = []
    if sport_key == "mlb":
        weather = fixture.get("weather")
        if weather:
            lines.append(f"Weather: {weather.get('condition', 'Unknown')}, {weather.get('temp', 'N/A')}°F, Wind {weather.get('wind', 'N/A')}")
        
        home_pitcher = fixture.get("home_pitcher")
        away_pitcher = fixture.get("away_pitcher")
        if home_pitcher:
            lines.append(f"Home Probable Pitcher: {home_pitcher}")
        if away_pitcher:
            lines.append(f"Away Probable Pitcher: {away_pitcher}")
            
    elif sport_key == "nhl":
        home_goalie = fixture.get("home_goalie")
        away_goalie = fixture.get("away_goalie")
        if home_goalie:
            lines.append(f"Home Probable Goalie: {home_goalie} ({fixture.get('home_goalie_status', 'Unknown')})")
        if away_goalie:
            lines.append(f"Away Probable Goalie: {away_goalie} ({fixture.get('away_goalie_status', 'Unknown')})")

    return "\n".join(lines)
