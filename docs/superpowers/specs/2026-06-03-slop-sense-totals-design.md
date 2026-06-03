# SLOP_SENSE for Over/Under Picks — Design

**Date:** 2026-06-03
**Status:** Approved design, pending spec review

## Problem

The site generates AI qualitative explanations (branded "GEMINI_SENSE", internally
"qualitative analysis") for moneyline picks, but **not** for over/under (totals)
picks. MLB totals picks therefore show no AI analysis box.

Root cause: `analyze_game_qualitative()` is called only inside the moneyline
`matches` loop (`run.py:3314`, `refresh_picks.py:264`). The totals loops never call
it. Additionally, the existing `QualitativeAnalysis` schema is moneyline-shaped
(`home_impact`/`away_impact`, `net_qualitative_edge: home|away|none`) and cannot
express an over/under lean.

## Decisions (from brainstorming)

- **Behavior:** Display the AI summary **and** numerically nudge the over/under
  probability (not display-only).
- **Rename:** `GEMINI_SENSE` → `SLOP_SENSE` in the frontend (the engine switched
  from Gemini to the OpenAI API; the brand should reflect the product, not the
  vendor).
- **Architecture:** Approach A — a dedicated totals analyzer + schema, separate
  from the moneyline path.
- **Sport scope:** MLB **and** NBA totals (every sport with totals + qualitative
  enabled).
- **Nudge cap:** MLB `±0.5` runs. NBA is in points, so the cap is **per-sport**:
  NBA defaults to `±2.5` points (proportional to its existing `2.2`-point
  availability cap).

## How the nudge works

Totals probability is derived from `expected_total` via
`over_prob = 1 - norm.cdf(line, loc=expected_total, scale=sigma)`. The qualitative
nudge **shifts `expected_total`** by a bounded number of runs/points, then the
existing CDF math recomputes `over_prob`. This mirrors exactly how the
weather / lineup / bullpen / availability adjustments already work
(`run.py:3414-3446`, `refresh_picks.py:322-351`), and keeps the sigma-based
probability derivation intact. A positive `total_impact` (lean Over) pushes
`expected_total` up; a negative one (lean Under) pushes it down.

## Components

### 1. `pipeline/qualitative_analysis.py`

Add a totals-specific Pydantic model and analyzer alongside the existing
moneyline ones.

```python
class TotalsQualitativeFactor(BaseModel):
    description: str
    direction: Literal["over", "under"]
    magnitude: float = Field(ge=0.0, le=5.0)
    confidence: float = Field(ge=0.0, le=1.0)

class TotalsQualitativeAnalysis(BaseModel):
    sport: str
    home_team: str
    away_team: str
    total_line: Optional[float]
    total_impact: float = Field(ge=-5.0, le=5.0)   # +over / -under
    individual_factors: List[TotalsQualitativeFactor]
    net_total_edge: Literal["over", "under", "none"]
    summary: str
```

`analyze_total_qualitative(total_match: dict, context_text: str) -> dict`:
- Same OpenAI plumbing as `analyze_game_qualitative` (model `gpt-4o-mini`,
  `temperature=0.1`, structured `response_format`).
- Totals-focused **system prompt**: scoring expresses an Over(+) / Under(-) lean
  driven by pace/run-environment, weather (wind out = Over, cold/rain = Under),
  bullpen fatigue (tired pen = Over), lineup/park, and (NBA) pace/availability of
  high-usage scorers. Same conservative scoring discipline (default 0, never
  hallucinate).
- On missing `OPENAI_API_KEY` or empty context, return a neutral default
  (`total_impact=0.0`, `net_total_edge="none"`, summary "No significant
  qualitative factors identified.").
- Reuse the existing `_log_api_call` logging to `qualitative_log.jsonl`.

To avoid duplicating the OpenAI call across the two analyzers, factor the shared
request/parse/error-handling into a small private helper
`_call_structured(system_prompt, user_prompt, response_model)`. The moneyline
function is refactored to call it; behavior unchanged.

### 2. `pipeline/run.py`

Add two helpers next to the existing qualitative/total helpers:

```python
def _apply_total_qualitative_adjustment(
    expected_total: float,
    qualitative_data: dict,
    weight: float,
    max_points_delta: float,
) -> float:
    impact = float(qualitative_data.get("total_impact", 0.0))
    delta = impact * QUALITATIVE_DEFAULT_WEIGHT * weight  # scale per impact point
    delta = max(-max_points_delta, min(max_points_delta, delta))
    return expected_total + delta   # caller clamps to sport bounds
```

```python
def _format_total_qualitative_summary(total_probs, qualitative_data) -> str:
    # Mirror _format_qualitative_summary but phrased in Over/Under terms.
    # Returns "" / "No qualitative impact." when net_total_edge == "none".
```

Note: `QUALITATIVE_DEFAULT_WEIGHT` (0.005) was tuned for probability-space
moneyline deltas. For totals it scales a *points* delta, so the per-point step is
small and the `max_points_delta` clamp is the real guard. Final `expected_total`
is clamped to the sport's existing bounds (MLB `4.5–16.0`, NBA `180–270`) by the
caller, consistent with the other total adjustments.

### 3. Wiring — both totals paths

In **`run.py`** totals loop (after the weather/lineup/bullpen/availability
adjustments at ~3446, before `over_prob` at ~3448) and in **`refresh_picks.py`**
totals loop (after line 351, before `over_prob` at line 357):

```python
if ENABLE_QUALITATIVE and sport.get("enable_qualitative", False):
    context_text = get_game_context(sport_key, total_match)
    qual = analyze_total_qualitative(total_match, context_text)
    total_match["qualitative_analysis"] = qual
    expected_total = _apply_total_qualitative_adjustment(
        expected_total, qual,
        weight=sport.get("qualitative_weight", 0.4),
        max_points_delta=sport.get("qualitative_total_adjustment_max_points", 0.5),
    )
    expected_total = max(<lo>, min(<hi>, expected_total))  # sport bounds
    total_match["qualitative_summary"] = _format_total_qualitative_summary(
        total_probs_preview, qual,
    )
```

`qualitative_summary` must survive into `totals_locks`. `_compute_totals_locks`
selects/copies from `totals_matches`; verify it preserves unknown keys (it copies
the record dicts) — if it whitelists fields, add `qualitative_summary` and
`qualitative_analysis` to the carried set.

### 4. `pipeline/config.py`

Add `qualitative_total_adjustment_max_points` to the MLB and NBA sport configs:
- MLB: `0.5`
- NBA: `2.5`

(Per-sport because MLB totals are runs and NBA totals are points.)

### 5. `index.html`

Rename the label `GEMINI_SENSE` → `SLOP_SENSE` (line 1011 and any other
occurrence). No other frontend change: `renderPick` already handles
`market_type === 'total'` and renders `qualitative_summary` generically (lines
949, 1004-1011), so totals cards light up automatically once the data carries a
summary.

## Data flow

```
totals loop
  → base expected_total
  → weather / lineup / bullpen / availability adjustments  (existing)
  → analyze_total_qualitative()  ── OpenAI ──▶ TotalsQualitativeAnalysis
  → _apply_total_qualitative_adjustment()  shifts expected_total (clamped)
  → over_prob = 1 - norm.cdf(line, expected_total, sigma)   (existing)
  → total_match.qualitative_summary  ──▶ totals_locks ──▶ dashboard.json
  → frontend renderPick → SLOP_SENSE box
```

## Error handling

- No API key / empty context → neutral default, `expected_total` unchanged, no
  SLOP_SENSE box (summary is "No qualitative impact."). Pipeline never fails on a
  missing key — matches existing moneyline behavior.
- OpenAI error → logged via `_log_api_call`, neutral default returned.
- Adjustment is bounded by `max_points_delta` and the sport total clamp, so a bad
  AI score cannot blow up a projection.

## Testing

- **Unit:** `_apply_total_qualitative_adjustment` — positive impact raises total,
  negative lowers, clamp respected at the cap, neutral (0) is a no-op, sport
  bounds enforced.
- **Unit:** `_format_total_qualitative_summary` — "none" edge → empty/neutral;
  over/under edge → phrased correctly.
- **Unit (mocked OpenAI):** `analyze_total_qualitative` returns schema-valid dict;
  missing key → default; exception → default + logged.
- **Integration:** run a totals refresh for one MLB and one NBA fixture with a
  stubbed analyzer; assert `qualitative_summary` is attached and `expected_total`
  moved by the expected (clamped) amount.
- **Regression:** existing moneyline qualitative tests still pass after the shared
  `_call_structured` refactor.

## Out of scope

- No change to moneyline qualitative behavior (only an internal refactor to share
  the OpenAI call).
- No new data sources; reuse existing `get_game_context`.
- No frontend redesign beyond the label rename.
