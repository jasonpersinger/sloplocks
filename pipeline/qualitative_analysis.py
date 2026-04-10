from typing import List, Literal, Union, Optional
import os
import json
import logging
import time
import warnings
from datetime import datetime
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)


class IndividualFactor(BaseModel):
    team: str
    description: str
    direction: Literal["positive", "negative"]
    magnitude: float = Field(ge=0.0, le=5.0)
    confidence: float = Field(ge=0.0, le=1.0)


class QualitativeAnalysis(BaseModel):
    sport: str
    home_team: str
    away_team: str
    home_impact: float = Field(ge=-5.0, le=5.0)
    away_impact: float = Field(ge=-5.0, le=5.0)
    individual_factors: List[IndividualFactor]
    net_qualitative_edge: Literal["home", "away", "none"]
    summary: str

# Log file path consistent with other pipeline logs
# We'll put it in data/tracking for permanence
QUALITATIVE_LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "tracking", "qualitative_log.jsonl")

SYSTEM_PROMPT = """You are an expert sports betting analyst specializing in qualitative factors.
Your task is to evaluate non-statistical context (injuries, news, scheduling, weather) and provide impact scores for each team.

### Scoring Rules:
- Output ONLY valid JSON, no markdown or backticks.
- Score conservatively. Most games should score near 0 unless there is a significant event.
- Never hallucinate facts. If context is missing or ambiguous, default to 0.
- Scale: -5 to +5.
    - 0: No impact
    - ±1: Minor (e.g., bench player out, slightly unfavorable travel)
    - ±2: Moderate (e.g., solid starter out, 3rd game in 4 nights)
    - ±3: Significant (e.g., key playmaker out, major travel disadvantage)
    - ±4: Major (e.g., All-Star/Superstar out, extreme weather impact)
    - ±5: Extreme (e.g., MVP candidate + another starter out, team-wide illness)

### Sport Specific Guidance:
- **NBA/NCAAM**: Superstar availability is paramount. Check for "Load Management" or late scratches.
- **MLB**: Evaluate the "Bullpen Tax" or fatigue. If a team's top 3 closers worked 2+ days in a row, they face a moderate (-2) negative impact. Weather (wind blowing out) impacts Totals more than ML.
- **NHL**: Starting Goalie is 50% of the qualitative score. A backup goalie starting against an elite offense is a significant (-3) negative impact.

### Schema:
{
  "sport": "string",
  "home_team": "string",
  "away_team": "string",
  "home_impact": float (-5.0 to 5.0),
  "away_impact": float (-5.0 to 5.0),
  "individual_factors": [
    {
      "team": "string",
      "description": "string",
      "direction": "positive|negative",
      "magnitude": float (0.0 to 5.0),
      "confidence": float (0.0 to 1.0)
    }
  ],
  "net_qualitative_edge": "home|away|none",
  "summary": "one-sentence summary of qualitative impact"
}
"""

def analyze_game_qualitative(game_dict, context_text):
    """
    Calls Gemini API to score qualitative factors using the modern google-genai package.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not found. Returning default qualitative scores.")
        return _get_default_response(game_dict)

    if not context_text or context_text.strip() == "":
        return _get_default_response(game_dict)

    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})
    model_id = "gemini-2.5-flash" 

    user_prompt = f"""Evaluate the qualitative impact for the following game:
Sport: {game_dict.get('sport')}
Home Team: {game_dict.get('home_team')}
Away Team: {game_dict.get('away_team')}
Game Time: {game_dict.get('start_time', game_dict.get('date'))}
Current Line: {game_dict.get('american_odds', 'N/A')}

### Context:
{context_text}
"""

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=QualitativeAnalysis,
            )
        )

        raw_text = response.text
        result = json.loads(raw_text)
        
        # Log raw response and parsed result
        _log_api_call(game_dict, context_text, raw_text, parsed_result=result)
        
        return result

    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        _log_api_call(game_dict, context_text, f"ERROR: {str(e)}")
        return _get_default_response(game_dict)

def _get_default_response(game_dict):
    return {
        "sport": game_dict.get("sport"),
        "home_team": game_dict.get("home_team"),
        "away_team": game_dict.get("away_team"),
        "home_impact": 0.0,
        "away_impact": 0.0,
        "individual_factors": [],
        "net_qualitative_edge": "none",
        "summary": "No significant qualitative factors identified or API error."
    }

def _log_api_call(game_dict, context_sent, raw_response, parsed_result=None):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "game_id": f"{game_dict.get('home_team')}_vs_{game_dict.get('away_team')}_{game_dict.get('date')}",
        "sport": game_dict.get("sport"),
        "home_team": game_dict.get("home_team"),
        "away_team": game_dict.get("away_team"),
        "context_sent": context_sent,
        "raw_response": raw_response,
        "home_impact": parsed_result.get("home_impact") if parsed_result else None,
        "away_impact": parsed_result.get("away_impact") if parsed_result else None,
    }
    
    try:
        os.makedirs(os.path.dirname(QUALITATIVE_LOG_FILE), exist_ok=True)
        with open(QUALITATIVE_LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to write to qualitative log: {e}")
