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


def _call_structured(system_prompt: str, user_prompt: str, response_model, api_key: str):
    """Shared OpenAI structured-output call. Returns a dict or None on error."""
    client = OpenAI(api_key=api_key)
    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_model,
            temperature=0.1,
        )
        return response.choices[0].message.parsed.model_dump()
    except Exception as e:
        logger.error(f"OpenAI structured call error: {e}")
        return None


def analyze_game_qualitative(game_dict: dict, context_text: str) -> dict:
    """Call OpenAI to score qualitative factors for a game. Returns a dict matching QualitativeAnalysis."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set — returning default qualitative scores.")
        return _default_response(game_dict)

    if not context_text or not context_text.strip():
        return _default_response(game_dict)

    user_prompt = (
        f"Evaluate the qualitative impact for this game:\n"
        f"Sport: {game_dict.get('sport')}\n"
        f"Home Team: {game_dict.get('home_team')}\n"
        f"Away Team: {game_dict.get('away_team')}\n"
        f"Game Time: {game_dict.get('start_time', game_dict.get('date'))}\n"
        f"Current Line: {game_dict.get('american_odds', 'N/A')}\n\n"
        f"### Context:\n{context_text}"
    )

    result_dict = _call_structured(SYSTEM_PROMPT, user_prompt, QualitativeAnalysis, api_key)
    if result_dict is None:
        _log_api_call(game_dict, context_text, None, error="structured call failed")
        return _default_response(game_dict)
    _log_api_call(game_dict, context_text, result_dict)
    return result_dict


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


class TotalsQualitativeFactor(BaseModel):
    description: str
    direction: Literal["over", "under"]
    magnitude: float = Field(ge=0.0, le=5.0)
    confidence: float = Field(ge=0.0, le=1.0)


class TotalsQualitativeAnalysis(BaseModel):
    sport: str
    home_team: str
    away_team: str
    total_line: Optional[float] = None
    total_impact: float = Field(ge=-5.0, le=5.0)  # +over / -under
    individual_factors: List[TotalsQualitativeFactor]
    net_total_edge: Literal["over", "under", "none"]
    summary: str


TOTALS_SYSTEM_PROMPT = """You are an expert sports betting analyst specializing in OVER/UNDER (totals) markets.
Evaluate non-statistical context (weather, bullpen fatigue, lineup/park, pace, injuries to high-usage scorers) and score a single lean toward the OVER or the UNDER.

### Scoring Rules (total_impact, -5 to +5):
- POSITIVE = leans OVER (more scoring). NEGATIVE = leans UNDER (less scoring).
- Score conservatively. Most games should be near 0 unless there is a significant signal.
- Never hallucinate facts. If context is missing or ambiguous, default to 0.
    - 0: No impact
    - ±1: Minor   ±2: Moderate   ±3: Significant   ±4: Major   ±5: Extreme

### Sport Specific Guidance:
- **MLB**: Wind blowing OUT and warm temps push OVER; cold, rain, wind blowing IN push UNDER. A fatigued bullpen (top relievers worked 2+ straight days) pushes OVER. A hitter-friendly park or stacked lineup vs. a weak arm pushes OVER.
- **NBA**: A fast pace matchup pushes OVER. A key high-usage scorer ruled out pushes UNDER. Back-to-back fatigue and elite defenses push UNDER.
"""


def analyze_total_qualitative(total_match: dict, context_text: str) -> dict:
    """Call OpenAI to score Over/Under qualitative lean. Returns a dict matching TotalsQualitativeAnalysis."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set — returning default totals qualitative scores.")
        return _default_totals_response(total_match)

    if not context_text or not context_text.strip():
        return _default_totals_response(total_match)

    user_prompt = (
        f"Evaluate the OVER/UNDER qualitative lean for this game:\n"
        f"Sport: {total_match.get('sport')}\n"
        f"Home Team: {total_match.get('home_team')}\n"
        f"Away Team: {total_match.get('away_team')}\n"
        f"Game Time: {total_match.get('start_time', total_match.get('date'))}\n"
        f"Total Line: {total_match.get('total_line', 'N/A')}\n\n"
        f"### Context:\n{context_text}"
    )

    result_dict = _call_structured(
        TOTALS_SYSTEM_PROMPT, user_prompt, TotalsQualitativeAnalysis, api_key,
    )
    if result_dict is None:
        return _default_totals_response(total_match)
    _log_api_call(total_match, context_text, result_dict)
    return result_dict


def _default_totals_response(total_match: dict) -> dict:
    return {
        "sport": total_match.get("sport"),
        "home_team": total_match.get("home_team"),
        "away_team": total_match.get("away_team"),
        "total_line": total_match.get("total_line"),
        "total_impact": 0.0,
        "individual_factors": [],
        "net_total_edge": "none",
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
