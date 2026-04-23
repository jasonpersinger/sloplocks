"""Publish the NFL Draft special tab payload for the static frontend."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pipeline.draft_qualitative_analysis import analyze_draft_picks


ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = ROOT / "data" / "nfl-draft-source.json"
OUTPUT_PATH = ROOT / "data" / "nfl-draft.json"


def _load_source() -> dict:
    with SOURCE_PATH.open() as handle:
        return json.load(handle)


def _build_commentary_payload(source: dict) -> dict:
    return {
        "event": source.get("event"),
        "market_context": source.get("market_context"),
        "consensus": source.get("consensus"),
        "rumors": source.get("rumors"),
        "picks": [
            {
                "prop": pick.get("prop"),
                "logic": pick.get("logic"),
            }
            for pick in (source.get("picks") or [])
        ],
    }


def _merge_output(source: dict, commentary: dict) -> dict:
    commentary_map = {
        item.get("prop"): item
        for item in (commentary.get("picks") or [])
        if item.get("prop")
    }
    merged_picks = []
    buy_count = 0

    for pick in source.get("picks") or []:
        gemini = commentary_map.get(pick.get("prop"), {})
        verdict = gemini.get("verdict")
        if verdict == "buy":
            buy_count += 1
        merged_picks.append(
            {
                **pick,
                "gemini": {
                    "verdict": verdict or "pass",
                    "confidence": gemini.get("confidence", 0.0),
                    "logic": gemini.get("logic") or "",
                    "slop_factor": gemini.get("slop_factor") or "",
                    "risk": gemini.get("risk") or pick.get("risk") or "",
                },
            }
        )

    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "title": source.get("title", "NFL DRAFT"),
        "event": source.get("event", "NFL Draft Round 1"),
        "summary": source.get("summary", ""),
        "market_context": source.get("market_context", ""),
        "consensus": source.get("consensus", ""),
        "rumors": source.get("rumors", ""),
        "sources": source.get("sources") or [],
        "gemini": {
            "enabled": commentary.get("summary") != "Missing GEMINI_API_KEY.",
            "summary": commentary.get("summary", ""),
        },
        "stats": {
            "pick_count": len(merged_picks),
            "buy_count": buy_count,
            "average_confidence": round(
                sum(float(p.get("confidence") or 0.0) for p in merged_picks) / len(merged_picks),
                1,
            )
            if merged_picks
            else 0.0,
        },
        "picks": merged_picks,
    }
    return summary


def build_draft_tab() -> dict:
    source = _load_source()
    commentary = analyze_draft_picks(_build_commentary_payload(source))
    output = _merge_output(source, commentary)
    with OUTPUT_PATH.open("w") as handle:
        json.dump(output, handle, indent=2)
    return output


def _main() -> int:
    build_draft_tab()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
