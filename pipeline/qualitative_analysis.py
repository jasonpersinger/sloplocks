from typing import List, Literal, Optional
import os
import json
import logging
from datetime import datetime
from openai import OpenAI
from pydantic import BaseModel, Field

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


QUALITATIVE_LOG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data", "tracking", "qualitative_log.jsonl",
)

SYSTEM_PROMPT = """You are an expert sports betting analyst specializing in qualitative factors.
Your task is to evaluate non-statistical context (injuries, news, scheduling, weather) and provide impact scores for each team.

### Scoring Rules:
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
- **MLB**: Evaluate bullpen fatigue. If a team's top closers worked 2+ days in a row, that is a moderate (-2) negative. Weather (wind blowing out) impacts Totals more than ML.
- **NHL**: Starting goalie is 50% of the qualitative score. A backup starting against an elite offense is a significant (-3) negative.
"""


def analyze_game_qualitative(game_dict: dict, context_text: str) -> dict:
    """Call OpenAI to score qualitative factors for a game. Returns a dict matching QualitativeAnalysis."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set — returning default qualitative scores.")
        return _default_response(game_dict)

    if not context_text or not context_text.strip():
        return _default_response(game_dict)

    client = OpenAI(api_key=api_key)

    user_prompt = (
        f"Evaluate the qualitative impact for this game:\n"
        f"Sport: {game_dict.get('sport')}\n"
        f"Home Team: {game_dict.get('home_team')}\n"
        f"Away Team: {game_dict.get('away_team')}\n"
        f"Game Time: {game_dict.get('start_time', game_dict.get('date'))}\n"
        f"Current Line: {game_dict.get('american_odds', 'N/A')}\n\n"
        f"### Context:\n{context_text}"
    )

    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format=QualitativeAnalysis,
            temperature=0.1,
        )
        result = response.choices[0].message.parsed
        result_dict = result.model_dump()
        _log_api_call(game_dict, context_text, result_dict)
        return result_dict

    except Exception as e:
        logger.error(f"OpenAI qualitative analysis error: {e}")
        _log_api_call(game_dict, context_text, None, error=str(e))
        return _default_response(game_dict)


def _default_response(game_dict: dict) -> dict:
    return {
        "sport": game_dict.get("sport"),
        "home_team": game_dict.get("home_team"),
        "away_team": game_dict.get("away_team"),
        "home_impact": 0.0,
        "away_impact": 0.0,
        "individual_factors": [],
        "net_qualitative_edge": "none",
        "summary": "No significant qualitative factors identified.",
    }


def _log_api_call(game_dict: dict, context_sent: str, result: Optional[dict], error: str = None):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "game_id": f"{game_dict.get('home_team')}_vs_{game_dict.get('away_team')}_{game_dict.get('date')}",
        "sport": game_dict.get("sport"),
        "home_team": game_dict.get("home_team"),
        "away_team": game_dict.get("away_team"),
        "context_sent": context_sent,
        "home_impact": result.get("home_impact") if result else None,
        "away_impact": result.get("away_impact") if result else None,
        "error": error,
    }
    try:
        os.makedirs(os.path.dirname(QUALITATIVE_LOG_FILE), exist_ok=True)
        with open(QUALITATIVE_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to write qualitative log: {e}")
