from typing import List, Literal, Optional
import os
import json
import logging
from datetime import datetime
from google import genai
from google.genai import types
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
    # --- explanatory fields -------------------------------------------------
    # These do not feed the probability adjustment; they exist to explain the
    # pick to a reader. ``pick_rationale`` is what the site renders as the
    # analysis blurb.
    pick_rationale: str = Field(
        description=(
            "2-4 sentences explaining WHY this side is the pick. Reference the "
            "model's edge over the market price, what the component models "
            "agree or disagree on, and any qualitative factor that matters."
        )
    )
    model_vs_market: str = Field(
        description=(
            "One sentence on how the model's number compares to the book's "
            "implied probability, and what that disagreement is worth."
        )
    )
    key_risk: str = Field(
        description="One sentence naming the most likely way this pick loses."
    )
    confidence_label: Literal["strong", "moderate", "thin", "avoid"]


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
- **NFL**: One week between games, so rest matters only at the extremes: a short week (Thursday game) is a moderate (-2) negative and a bye week is a moderate (+2) positive. Quarterback availability dwarfs every other position; a starting QB ruled out is major (-4). Weather matters outdoors in December and January: sustained wind above 15mph or heavy precipitation is a significant factor. Offensive line injuries are underrated; two or more starters out is a significant (-3) negative.
- **NCAAF**: Talent gaps between programs are enormous and already priced in, so do not re-penalise a weak team for being weak. Focus on situational spots: a ranked team travelling to a hostile night game, a letdown after a rivalry win, or a lookahead to a marquee opponent. Quarterback availability is again paramount. Neutral-site and conference-championship games remove home-field entirely.

### Explaining the pick:
Beyond scoring the factors, you must explain the selection to a reader.
- `pick_rationale`: 2-4 sentences on WHY this side. Lead with the model-versus-market disagreement, then name the component models that drive it, then any qualitative factor. Be concrete and specific. Never invent injuries, statistics, or news that is not in the provided context.
- `model_vs_market`: one sentence quantifying the gap between the model probability and the book's implied probability.
- `key_risk`: one sentence naming the single most likely way this pick loses. Always fill this in; every pick has a risk.
- `confidence_label`: "strong" only when the edge is large AND the component models agree; "thin" when the edge is small or the models disagree; "avoid" when the qualitative context actively contradicts the model.
Write plainly. No hype, no guarantees, no betting advice framing.
"""


GEMINI_MODEL_ID = "gemini-2.5-flash"


def _call_structured(system_prompt: str, user_prompt: str, response_model, api_key: str):
    """Shared Gemini structured-output call. Returns a dict or None on error.

    Mirrors the client setup already used by ``draft_qualitative_analysis`` so
    both AI surfaces in this repo speak to the same provider and model.
    """
    try:
        client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})
        response = client.models.generate_content(
            model=GEMINI_MODEL_ID,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=response_model,
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"Gemini structured call error: {e}")
        return None


def _format_model_context(game_dict: dict) -> str:
    """Render the model's own numbers so the AI can explain them.

    Without this the analyst only sees team names and a price, which is why
    earlier summaries could only ever recite injury notes.
    """
    lines = ["### Model output:"]
    pick = game_dict.get("pick")
    if pick:
        picked_team = game_dict.get(f"{pick}_team", pick)
        lines.append(f"- Model pick: {picked_team} ({pick})")
    for label, key in (
        ("Model probability", "model_prob"),
        ("Market implied probability", "implied_prob"),
        ("Edge", "edge"),
        ("Expected value", "expected_value"),
    ):
        value = game_dict.get(key)
        if isinstance(value, (int, float)):
            lines.append(f"- {label}: {value:.1%}" if abs(value) <= 1 else f"- {label}: {value}")
    confidence = game_dict.get("confidence_score")
    if confidence is not None:
        lines.append(f"- Confidence score: {confidence}/100")

    individual = game_dict.get("individual_models") or {}
    if individual and pick:
        lines.append("- Component models (probability assigned to the pick):")
        for name, probs in individual.items():
            if isinstance(probs, dict) and pick in probs:
                lines.append(f"    - {name}: {probs[pick]:.1%}")

    ratings = game_dict.get("elo_ratings") or {}
    if ratings:
        lines.append(
            f"- Elo: {game_dict.get('home_team')} {ratings.get('home'):.0f} "
            f"vs {game_dict.get('away_team')} {ratings.get('away'):.0f}"
        )
    if game_dict.get("neutral"):
        lines.append("- Neutral site: no home-field advantage applied.")
    return "\n".join(lines)


def _format_totals_model_context(total_match: dict) -> str:
    """Render the totals projection so the AI can explain the gap to the line."""
    lines = ["### Model output:"]
    expected = total_match.get("expected_total")
    line = total_match.get("total_line")
    if isinstance(expected, (int, float)):
        lines.append(f"- Projected total: {expected:.1f}")
        if isinstance(line, (int, float)):
            lines.append(f"- Posted line: {line:.1f} (model is {expected - line:+.1f} vs the line)")
    for label, key in (("Over probability", "over_prob"), ("Under probability", "under_prob")):
        value = total_match.get(key)
        if isinstance(value, (int, float)):
            lines.append(f"- {label}: {value:.1%}")
    sigma = total_match.get("stddev")
    if isinstance(sigma, (int, float)):
        lines.append(f"- Projection spread (1 sigma): {sigma:.1f} points")
    return "\n".join(lines)


def analyze_game_qualitative(game_dict: dict, context_text: str) -> dict:
    """Call Gemini to score qualitative factors and explain the pick. Returns a dict matching QualitativeAnalysis."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set — returning default qualitative scores.")
        return _default_response(game_dict)

    # Situational context can legitimately be empty — college football has no
    # meaningful public injury feed. The model's own numbers are still worth
    # explaining, so only bail when there is nothing at all to talk about.
    has_context = bool(context_text and context_text.strip())
    has_model_context = game_dict.get("pick") is not None
    if not has_context and not has_model_context:
        return _default_response(game_dict)
    if not has_context:
        context_text = (
            "No injury or situational reporting is available for this game. "
            "Explain the pick from the model output alone and do not speculate "
            "about injuries, weather, or news."
        )

    user_prompt = (
        f"Evaluate the qualitative impact for this game and explain the model's pick.\n"
        f"Sport: {game_dict.get('sport')}\n"
        f"Home Team: {game_dict.get('home_team')}\n"
        f"Away Team: {game_dict.get('away_team')}\n"
        f"Game Time: {game_dict.get('start_time', game_dict.get('date'))}\n"
        f"Current Line: {game_dict.get('american_odds', 'N/A')}\n\n"
        f"{_format_model_context(game_dict)}\n"
        f"### Situational context:\n{context_text}"
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
        "pick_rationale": "",
        "model_vs_market": "",
        "key_risk": "",
        "confidence_label": "moderate",
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
    pick_rationale: str = Field(
        description=(
            "2-4 sentences explaining WHY the projected total sits where it "
            "does relative to the posted line, in terms of how these two teams "
            "have been scoring and conceding."
        )
    )
    model_vs_market: str = Field(
        description="One sentence comparing the model's projected total to the posted line."
    )
    key_risk: str = Field(
        description="One sentence naming the most likely way this over/under loses."
    )
    confidence_label: Literal["strong", "moderate", "thin", "avoid"]


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
- **NFL**: Wind above 15mph is the single biggest UNDER signal; heavy rain or snow also pushes UNDER. Domes and warm early-season games push OVER. A backup quarterback pushes UNDER. Two struggling defenses or a shootout script pushes OVER.
- **NCAAF**: Tempo is the dominant factor and varies far more than in the NFL — a fast-snapping spread offense against a weak defense pushes OVER hard, while a run-heavy clock-control team pushes UNDER. Large talent mismatches often push UNDER late as the favourite empties the bench and runs clock.

### Explaining the pick:
- `pick_rationale`: 2-4 sentences on why the projected total sits where it does versus the posted line, grounded in how these teams have been scoring and conceding.
- `model_vs_market`: one sentence comparing the projection to the posted line.
- `key_risk`: one sentence on the most likely way this over/under loses.
- `confidence_label`: "strong" only for a large, well-supported gap; "thin" for a small one; "avoid" when context contradicts the projection.
Never invent weather, injuries, or statistics that are not in the provided context.
"""


def analyze_total_qualitative(total_match: dict, context_text: str) -> dict:
    """Call Gemini to score the Over/Under lean and explain it. Returns a dict matching TotalsQualitativeAnalysis."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set — returning default totals qualitative scores.")
        return _default_totals_response(total_match)

    has_context = bool(context_text and context_text.strip())
    has_model_context = total_match.get("expected_total") is not None
    if not has_context and not has_model_context:
        return _default_totals_response(total_match)
    if not has_context:
        context_text = (
            "No weather or situational reporting is available for this game. "
            "Explain the projection from the model output alone and do not "
            "speculate about weather, injuries, or news."
        )

    user_prompt = (
        f"Evaluate the OVER/UNDER lean for this game and explain the projection.\n"
        f"Sport: {total_match.get('sport')}\n"
        f"Home Team: {total_match.get('home_team')}\n"
        f"Away Team: {total_match.get('away_team')}\n"
        f"Game Time: {total_match.get('start_time', total_match.get('date'))}\n"
        f"Total Line: {total_match.get('total_line', 'N/A')}\n\n"
        f"{_format_totals_model_context(total_match)}\n"
        f"### Situational context:\n{context_text}"
    )

    result_dict = _call_structured(
        TOTALS_SYSTEM_PROMPT, user_prompt, TotalsQualitativeAnalysis, api_key,
    )
    if result_dict is None:
        _log_api_call(total_match, context_text, None, error="structured call failed")
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
        "pick_rationale": "",
        "model_vs_market": "",
        "key_risk": "",
        "confidence_label": "moderate",
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
        "total_impact": result.get("total_impact") if result else None,
        "net_total_edge": result.get("net_total_edge") if result else None,
        "error": error,
    }
    try:
        os.makedirs(os.path.dirname(QUALITATIVE_LOG_FILE), exist_ok=True)
        with open(QUALITATIVE_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to write qualitative log: {e}")
