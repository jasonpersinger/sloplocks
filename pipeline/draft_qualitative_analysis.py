"""Gemini-powered qualitative commentary for NFL draft prop picks.

This module mirrors the repo's existing game-level qualitative analysis flow,
but targets pre-draft prop cards instead of game fixtures.

Usage:
    python -m pipeline.draft_qualitative_analysis --input-file draft_props.json

Input JSON shape:
{
  "event": "2026 NFL Draft Round 1",
  "market_context": "DraftKings lines as of 2026-04-23 11:00 ET ...",
  "consensus": "Summaries from mocks / intel ...",
  "rumors": "Trade chatter / player-slide buzz ...",
  "picks": [
    {"prop": "Ty Simpson Round 1 YES", "logic": "..."},
    {"prop": "Omar Cooper Jr. Under 23.5", "logic": "..."}
  ]
}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from typing import Literal, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

LOG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "tracking",
    "draft_qualitative_log.jsonl",
)

SYSTEM_PROMPT = """You are the lead analyst for an NFL draft betting model.
Evaluate draft prop picks using only the supplied context.

Rules:
- Output only valid JSON matching the schema.
- Do not invent team needs, medicals, trade rumors, or odds that are not in the payload.
- Distinguish strong signal from rumor noise.
- Treat consensus mock support, market movement, and plausible trade paths as positives.
- Treat medical uncertainty, positional devaluation, and rumor-only support as risks.
- Be concise and specific.

For each pick:
- verdict:
  - buy: the context supports the pick
  - pass: mixed or insufficient edge
  - fade: the context works against the pick
- confidence must be 0.0 to 10.0
- logic should explain why the pick works
- slop_factor should identify the key underpriced rumor, trade dynamic, or market nuance
- risk should identify the cleanest way the pick loses
"""


class DraftPickInput(BaseModel):
    prop: str
    logic: str = ""


class DraftPickCommentary(BaseModel):
    prop: str
    verdict: Literal["buy", "pass", "fade"]
    confidence: float = Field(ge=0.0, le=10.0)
    logic: str
    slop_factor: str
    risk: str


class DraftCardCommentary(BaseModel):
    event: str
    summary: str
    picks: list[DraftPickCommentary]


def analyze_draft_picks(payload: dict) -> dict:
    """Return Gemini commentary for a draft prop card."""
    api_key = os.environ.get("GEMINI_API_KEY")
    picks = payload.get("picks") or []
    if not api_key:
        logger.warning("GEMINI_API_KEY not found. Returning default draft commentary.")
        return _default_response(payload, reason="Missing GEMINI_API_KEY.")
    if not picks:
        return _default_response(payload, reason="No picks were provided.")

    client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})
    model_id = "gemini-2.5-flash"
    prompt = _build_prompt(payload)

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=DraftCardCommentary,
            ),
        )
        result = json.loads(response.text)
        _log_api_call(payload, prompt, response.text, parsed_result=result)
        return result
    except Exception as exc:
        logger.error("Error calling Gemini draft commentary API: %s", exc)
        _log_api_call(payload, prompt, f"ERROR: {exc}")
        return _default_response(payload, reason=f"Gemini error: {exc}")


def _build_prompt(payload: dict) -> str:
    event = payload.get("event") or "NFL Draft Round 1"
    market_context = payload.get("market_context") or "No market context provided."
    consensus = payload.get("consensus") or "No mock consensus provided."
    rumors = payload.get("rumors") or "No rumor context provided."
    picks = payload.get("picks") or []

    serialized_picks = []
    for pick in picks:
        try:
            normalized = DraftPickInput.model_validate(pick)
            serialized_picks.append(normalized.model_dump())
        except Exception:
            serialized_picks.append(
                {
                    "prop": str(pick.get("prop") or "").strip(),
                    "logic": str(pick.get("logic") or "").strip(),
                }
            )

    return (
        f"Event: {event}\n\n"
        f"Market Context:\n{market_context}\n\n"
        f"Consensus:\n{consensus}\n\n"
        f"Rumors / Slop:\n{rumors}\n\n"
        f"Candidate Picks:\n{json.dumps(serialized_picks, indent=2)}\n"
    )


def _default_response(payload: dict, reason: str) -> dict:
    picks = payload.get("picks") or []
    return {
        "event": payload.get("event") or "NFL Draft Round 1",
        "summary": reason,
        "picks": [
            {
                "prop": str(pick.get("prop") or ""),
                "verdict": "pass",
                "confidence": 0.0,
                "logic": reason,
                "slop_factor": "No Gemini commentary available.",
                "risk": "No model output available.",
            }
            for pick in picks
        ],
    }


def _log_api_call(payload: dict, prompt: str, raw_response: str, parsed_result: Optional[dict] = None) -> None:
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": payload.get("event"),
        "picks": payload.get("picks"),
        "prompt": prompt,
        "raw_response": raw_response,
        "summary": (parsed_result or {}).get("summary"),
    }
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a") as handle:
            handle.write(json.dumps(log_entry) + "\n")
    except Exception as exc:
        logger.error("Failed to write draft qualitative log: %s", exc)


def _load_payload(path: Optional[str]) -> dict:
    if path:
        with open(path) as handle:
            return json.load(handle)
    return json.load(os.sys.stdin)


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze NFL draft prop picks with Gemini.")
    parser.add_argument("--input-file", help="Path to a JSON payload. Defaults to stdin.")
    args = parser.parse_args(argv)

    payload = _load_payload(args.input_file)
    result = analyze_draft_picks(payload)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
