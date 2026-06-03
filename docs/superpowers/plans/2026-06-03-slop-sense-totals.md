# SLOP_SENSE for Over/Under Picks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate AI qualitative analysis (rebranded SLOP_SENSE) for MLB and NBA over/under picks, both as a displayed summary and as a bounded nudge to the projected total.

**Architecture:** A dedicated totals analyzer (`analyze_total_qualitative`) returns an Over/Under-directional score. A new helper shifts `expected_total` by a per-sport bounded amount before the existing `norm.cdf` derives `over_prob`, mirroring the existing weather/bullpen total adjustments. Both totals code paths (`run.py` main pipeline and `refresh_picks.py` refresh) call it. The frontend already renders totals cards generically, so only the label changes.

**Tech Stack:** Python 3.11, OpenAI SDK (`gpt-4o-mini`, structured outputs via Pydantic), pytest, scipy `norm`, vanilla JS frontend.

---

## File Structure

- `pipeline/config.py` — add `qualitative_total_adjustment_max_points` to MLB (0.5) and NBA (2.5) configs.
- `pipeline/qualitative_analysis.py` — add `TotalsQualitativeAnalysis` model, `analyze_total_qualitative()`, totals system prompt; refactor shared OpenAI call into `_call_structured()`.
- `pipeline/run.py` — add `_apply_total_qualitative_adjustment()` and `_format_total_qualitative_summary()`; wire into the totals loop; carry summary through `_compute_totals_locks`.
- `pipeline/refresh_picks.py` — wire the analyzer + nudge into the refresh totals loop.
- `index.html` — rename `GEMINI_SENSE` → `SLOP_SENSE`.
- `tests/test_qualitative_totals.py` — new unit tests for the analyzer, the nudge, and the summary formatter.

---

## Task 1: Per-sport totals nudge cap in config

**Files:**
- Modify: `pipeline/config.py` (MLB config near line 446, NBA config near line 188)

- [ ] **Step 1: Add the cap to the MLB sport config**

In `pipeline/config.py`, inside the MLB `SPORTS["mlb"]` dict, next to the existing
`"bullpen_total_adjustment_max_delta"` key, add:

```python
        "qualitative_total_adjustment_max_points": 0.5,
```

- [ ] **Step 2: Add the cap to the NBA sport config**

In the `SPORTS["nba"]` dict, next to `"availability_total_adjustment_max_points"`, add:

```python
        "qualitative_total_adjustment_max_points": 2.5,
```

- [ ] **Step 3: Verify config loads**

Run: `python -c "from pipeline.config import SPORTS; print(SPORTS['mlb']['qualitative_total_adjustment_max_points'], SPORTS['nba']['qualitative_total_adjustment_max_points'])"`
Expected: `0.5 2.5`

- [ ] **Step 4: Commit**

```bash
git add pipeline/config.py
git commit -m "config: add per-sport qualitative totals nudge cap"
```

---

## Task 2: Totals analyzer schema + function

**Files:**
- Modify: `pipeline/qualitative_analysis.py`
- Test: `tests/test_qualitative_totals.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_qualitative_totals.py`:

```python
from unittest.mock import MagicMock
import pipeline.qualitative_analysis as qa


def test_analyze_total_qualitative_returns_default_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    game = {"sport": "mlb", "home_team": "Reds", "away_team": "Cubs", "date": "2026-06-03"}
    result = qa.analyze_total_qualitative(game, "wind blowing out 15mph")
    assert result["total_impact"] == 0.0
    assert result["net_total_edge"] == "none"


def test_analyze_total_qualitative_returns_default_without_context(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    game = {"sport": "mlb", "home_team": "Reds", "away_team": "Cubs", "date": "2026-06-03"}
    result = qa.analyze_total_qualitative(game, "")
    assert result["net_total_edge"] == "none"


def test_analyze_total_qualitative_parses_model_output(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    parsed = qa.TotalsQualitativeAnalysis(
        sport="mlb", home_team="Reds", away_team="Cubs", total_line=9.5,
        total_impact=2.0, individual_factors=[],
        net_total_edge="over", summary="Wind out to RF favors the over.",
    )
    fake_completion = MagicMock()
    fake_completion.choices = [MagicMock(message=MagicMock(parsed=parsed))]
    fake_client = MagicMock()
    fake_client.beta.chat.completions.parse.return_value = fake_completion
    monkeypatch.setattr(qa, "OpenAI", lambda api_key: fake_client)

    game = {"sport": "mlb", "home_team": "Reds", "away_team": "Cubs",
            "date": "2026-06-03", "total_line": 9.5}
    result = qa.analyze_total_qualitative(game, "wind blowing out 15mph")
    assert result["net_total_edge"] == "over"
    assert result["total_impact"] == 2.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_qualitative_totals.py -v`
Expected: FAIL with `AttributeError: module 'pipeline.qualitative_analysis' has no attribute 'TotalsQualitativeAnalysis'`

- [ ] **Step 3: Add the schema and analyzer**

In `pipeline/qualitative_analysis.py`, after the existing `QualitativeAnalysis` class, add:

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
```

Add `_call_structured` near the top of the module (after `SYSTEM_PROMPT`):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_qualitative_totals.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Refactor the moneyline function to share `_call_structured` (keep behavior)**

Replace the body of `analyze_game_qualitative` between building `user_prompt` and the return with:

```python
    result_dict = _call_structured(SYSTEM_PROMPT, user_prompt, QualitativeAnalysis, api_key)
    if result_dict is None:
        _log_api_call(game_dict, context_text, None, error="structured call failed")
        return _default_response(game_dict)
    _log_api_call(game_dict, context_text, result_dict)
    return result_dict
```

Remove the now-unused inline `client = OpenAI(...)` / `try/except` block in that function.

- [ ] **Step 6: Run full qualitative + refresh test suite**

Run: `pytest tests/test_qualitative_totals.py tests/test_refresh_picks.py -v`
Expected: PASS (no regressions)

- [ ] **Step 7: Commit**

```bash
git add pipeline/qualitative_analysis.py tests/test_qualitative_totals.py
git commit -m "feat: add over/under qualitative analyzer (SLOP_SENSE totals)"
```

---

## Task 3: Totals nudge + summary helpers in run.py

**Files:**
- Modify: `pipeline/run.py` (add helpers near `_apply_qualitative_adjustment` ~line 1408 and `_format_qualitative_summary` ~line 2785)
- Test: `tests/test_qualitative_totals.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_qualitative_totals.py`:

```python
from pipeline.run import (
    _apply_total_qualitative_adjustment,
    _format_total_qualitative_summary,
)


def test_total_nudge_positive_raises_total():
    out = _apply_total_qualitative_adjustment(
        9.0, {"total_impact": 3.0}, weight=0.4, max_points_delta=0.5)
    assert out > 9.0


def test_total_nudge_negative_lowers_total():
    out = _apply_total_qualitative_adjustment(
        9.0, {"total_impact": -3.0}, weight=0.4, max_points_delta=0.5)
    assert out < 9.0


def test_total_nudge_respects_cap():
    out = _apply_total_qualitative_adjustment(
        9.0, {"total_impact": 5.0}, weight=999.0, max_points_delta=0.5)
    assert out == 9.5  # clamped to +max_points_delta


def test_total_nudge_zero_impact_is_noop():
    out = _apply_total_qualitative_adjustment(
        9.0, {"total_impact": 0.0}, weight=0.4, max_points_delta=0.5)
    assert out == 9.0


def test_total_summary_none_edge_is_neutral():
    summary = _format_total_qualitative_summary(
        {"over": 0.5, "under": 0.5},
        {"net_total_edge": "none", "total_impact": 0.0, "individual_factors": []})
    assert summary == "No qualitative impact."


def test_total_summary_over_edge_mentions_over():
    summary = _format_total_qualitative_summary(
        {"over": 0.6, "under": 0.4},
        {"net_total_edge": "over", "total_impact": 2.0,
         "individual_factors": [{"description": "wind out to RF"}]})
    assert "Over" in summary
    assert "wind out to RF" in summary
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_qualitative_totals.py -k total_nudge -v`
Expected: FAIL with `ImportError: cannot import name '_apply_total_qualitative_adjustment'`

- [ ] **Step 3: Add the helpers**

In `pipeline/run.py`, after `_apply_qualitative_adjustment` (after line 1408), add:

```python
def _apply_total_qualitative_adjustment(
    expected_total: float,
    qualitative_data: dict,
    weight: float,
    max_points_delta: float,
) -> float:
    """Shift a projected total by a bounded qualitative Over/Under lean.

    Positive total_impact leans Over (raise total); negative leans Under.
    The caller clamps the result to the sport's total bounds.
    """
    impact = float(qualitative_data.get("total_impact", 0.0))
    delta = impact * QUALITATIVE_DEFAULT_WEIGHT * weight
    delta = max(-max_points_delta, min(max_points_delta, delta))
    return expected_total + delta
```

After `_format_qualitative_summary` (after line 2785), add:

```python
def _format_total_qualitative_summary(total_probs, qualitative_data) -> str:
    """Human-readable Over/Under qualitative summary, mirroring the moneyline format."""
    if not qualitative_data or qualitative_data.get("net_total_edge", "none") == "none":
        return "No qualitative impact."

    edge = qualitative_data.get("net_total_edge", "none")
    impact = qualitative_data.get("total_impact", 0.0)

    pre_pick = max(total_probs.keys(), key=lambda k: total_probs[k]) if total_probs else None
    agreement = "no effect"
    if pre_pick is not None:
        agreement = "agreed" if edge == pre_pick else "disagreed"

    direction = "Over" if edge == "over" else "Under"
    factors = [f["description"] for f in qualitative_data.get("individual_factors", [])]
    factors_str = "; ".join(factors[:2])
    return f"Qualitative (Total {impact:+.1f}): leans {direction}, {agreement}. {factors_str}".strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_qualitative_totals.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/run.py tests/test_qualitative_totals.py
git commit -m "feat: add totals qualitative nudge and summary helpers"
```

---

## Task 4: Wire analyzer into run.py totals loop

**Files:**
- Modify: `pipeline/run.py` (totals block ~3407-3489; `_compute_totals_locks` candidate dict ~2265)

- [ ] **Step 1: Insert the qualitative call before the sigma/over_prob computation**

In `pipeline/run.py`, after the MLB/NBA `expected_total` adjustment branches close
(immediately before `sigma = max(1.5, ...)` at line ~3447), add:

```python
            total_for_ai = {
                "sport": sport_key,
                "home_team": home,
                "away_team": away,
                "date": fix["date"],
                "start_time": fix.get("start_time"),
                "total_line": float(match_odds["total_line"]),
            }
            total_qualitative = None
            total_qualitative_summary = ""
            if (
                ENABLE_QUALITATIVE
                and sport.get("enable_qualitative", False)
                and analyze_total_qualitative is not None
            ):
                total_context = get_game_context(sport_key, fix)
                total_qualitative = analyze_total_qualitative(total_for_ai, total_context)
                total_projection["expected_total"] = _apply_total_qualitative_adjustment(
                    total_projection["expected_total"],
                    total_qualitative,
                    weight=sport.get("qualitative_weight", 0.4),
                    max_points_delta=sport.get("qualitative_total_adjustment_max_points", 0.5),
                )
                lo, hi = (4.5, 16.0) if sport_key == "mlb" else (180.0, 270.0)
                total_projection["expected_total"] = max(lo, min(hi, total_projection["expected_total"]))
```

- [ ] **Step 2: Compute the summary after over_prob and attach fields to the record**

After `total_model_probs = {"over": over_prob, "under": 1.0 - over_prob}` (line ~3456), add:

```python
            if total_qualitative is not None:
                total_qualitative_summary = _format_total_qualitative_summary(
                    total_model_probs, total_qualitative)
```

Then in the appended `totals_prediction_records.append({...})` dict (line ~3464-3489),
add these two keys before the closing brace:

```python
                "qualitative_analysis": total_qualitative,
                "qualitative_summary": total_qualitative_summary,
```

- [ ] **Step 3: Import the analyzer (guarded, matching the moneyline import)**

Find where `analyze_game_qualitative` is imported in `run.py` and add `analyze_total_qualitative`
to the same import. If it uses a guarded `try/except ImportError` that sets
`analyze_game_qualitative = None`, mirror it:

```python
try:
    from pipeline.qualitative_analysis import analyze_game_qualitative, analyze_total_qualitative
except ImportError:
    analyze_game_qualitative = None
    analyze_total_qualitative = None
```

Run: `grep -n "analyze_game_qualitative" pipeline/run.py | head -3` to locate the import.

- [ ] **Step 4: Carry the summary through `_compute_totals_locks`**

In `_compute_totals_locks` (the candidate dict at lines ~2246-2266), add before the
closing `})`:

```python
                    "qualitative_analysis": rec.get("qualitative_analysis"),
                    "qualitative_summary": rec.get("qualitative_summary"),
```

- [ ] **Step 5: Verify run.py imports and parses**

Run: `python -c "import pipeline.run"`
Expected: no error

- [ ] **Step 6: Run the run.py test suite**

Run: `pytest tests/test_run.py tests/test_qualitative_totals.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add pipeline/run.py
git commit -m "feat: generate SLOP_SENSE totals analysis in main pipeline"
```

---

## Task 5: Wire analyzer into refresh_picks.py totals loop

**Files:**
- Modify: `pipeline/refresh_picks.py` (totals loop 310-369; imports near line 31)

- [ ] **Step 1: Import the totals analyzer**

In `pipeline/refresh_picks.py`, update the import at line 31:

```python
from pipeline.qualitative_analysis import analyze_game_qualitative, analyze_total_qualitative
```

And add the helpers to the existing `from pipeline.run import (...)` block (which already
imports `_apply_qualitative_adjustment`, `_format_qualitative_summary`):

```python
    _apply_total_qualitative_adjustment,
    _format_total_qualitative_summary,
```

- [ ] **Step 2: Insert the qualitative nudge before over_prob**

In the totals loop, after the MLB/NBA total adjustments and before
`sigma = max(1.5, ...)` (line 352), add:

```python
        total_qualitative = None
        if ENABLE_QUALITATIVE and sport.get("enable_qualitative", False):
            total_context = get_game_context(sport_key, total_match)
            total_for_ai = {
                "sport": sport_key,
                "home_team": total_match["home_team"],
                "away_team": total_match["away_team"],
                "date": total_match["date"],
                "start_time": total_match.get("start_time"),
                "total_line": float(match_odds["total_line"]),
            }
            total_qualitative = analyze_total_qualitative(total_for_ai, total_context)
            expected_total = _apply_total_qualitative_adjustment(
                expected_total,
                total_qualitative,
                weight=sport.get("qualitative_weight", 0.4),
                max_points_delta=sport.get("qualitative_total_adjustment_max_points", 0.5),
            )
            lo, hi = (4.5, 16.0) if sport_key == "mlb" else (180.0, 270.0)
            expected_total = max(lo, min(hi, expected_total))
```

- [ ] **Step 3: Attach summary to the totals record after over_prob**

After `total_probs = {"over": over_prob, "under": 1.0 - over_prob}` (line 359), add:

```python
        if total_qualitative is not None:
            total_match["qualitative_analysis"] = total_qualitative
            total_match["qualitative_summary"] = _format_total_qualitative_summary(
                total_probs, total_qualitative)
```

- [ ] **Step 4: Verify refresh_picks imports and parses**

Run: `python -c "import pipeline.refresh_picks"`
Expected: no error

- [ ] **Step 5: Run refresh test suite**

Run: `pytest tests/test_refresh_picks.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pipeline/refresh_picks.py
git commit -m "feat: generate SLOP_SENSE totals analysis on refresh"
```

---

## Task 6: Frontend rename GEMINI_SENSE → SLOP_SENSE

**Files:**
- Modify: `index.html`

- [ ] **Step 1: Find all occurrences**

Run: `grep -n "GEMINI_SENSE\|GEMINI SENSE\|Gemini" index.html`
Expected: at least line 1011.

- [ ] **Step 2: Replace the label**

In `index.html` line ~1011, change `GEMINI_SENSE` to `SLOP_SENSE`:

```javascript
      senseEl.innerHTML = '<span style="font-weight:bold; color:#ff72f9; font-size:9px; letter-spacing:1px;">SLOP_SENSE</span><br>' + pick.qualitative_summary;
```

Replace any other `GEMINI_SENSE`/`Gemini` user-facing occurrences found in Step 1.

- [ ] **Step 2b (optional hardening): avoid innerHTML for the AI summary**

`pick.qualitative_summary` is AI-generated (its prompt includes scraped game
context), so injecting it via `innerHTML` is an XSS seam. If hardening, build the
node safely instead:

```javascript
      var label = document.createElement('span');
      label.style.cssText = 'font-weight:bold; color:#ff72f9; font-size:9px; letter-spacing:1px;';
      label.textContent = 'SLOP_SENSE';
      senseEl.textContent = '';
      senseEl.appendChild(label);
      senseEl.appendChild(document.createElement('br'));
      senseEl.appendChild(document.createTextNode(pick.qualitative_summary));
```

This is a behavior-preserving change to a line already being edited; skip it only
if matching the existing moneyline rendering style is preferred.

- [ ] **Step 3: Verify no stale label remains**

Run: `grep -c "GEMINI_SENSE" index.html`
Expected: `0`

- [ ] **Step 4: Bump the service worker cache**

In `sw.js`, increment the `CACHE_NAME` version string (per project deploy convention).

Run: `grep -n "CACHE_NAME" sw.js`

- [ ] **Step 5: Commit**

```bash
git add index.html sw.js
git commit -m "feat: rename GEMINI_SENSE to SLOP_SENSE in frontend"
```

---

## Task 7: Full verification

- [ ] **Step 1: Run the whole test suite**

Run: `pytest tests/ -v`
Expected: all PASS

- [ ] **Step 2: Smoke-test a totals run with a stubbed analyzer (optional, no API key)**

Confirm that with `OPENAI_API_KEY` unset, totals picks still build and carry a
neutral `qualitative_summary` of `"No qualitative impact."` (no SLOP_SENSE box),
and the pipeline does not error.

Run: `ENABLE_QUALITATIVE=false python -c "import pipeline.run; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Final commit if any cleanup**

```bash
git add -A && git commit -m "test: verify SLOP_SENSE totals end to end" || echo "nothing to commit"
```

---

## Notes for the implementer

- **`QUALITATIVE_DEFAULT_WEIGHT` (0.005)** was tuned for probability-space moneyline deltas. For totals it scales a *points* delta, so the per-point step is tiny and the `max_points_delta` clamp is the real guard. `test_total_nudge_respects_cap` pins this behavior.
- **Two code paths**: `run.py` builds the initial totals projection; `refresh_picks.py` re-derives it on the live refresh. Both must apply the nudge or the displayed total and SLOP_SENSE box will disagree between build and refresh.
- **Field-whitelisting**: `_compute_totals_locks` copies an explicit field list — the new keys MUST be added there (Task 4 Step 4) or the summary never reaches `dashboard.json`.
- **No moneyline behavior change**: Task 2 Step 5 only refactors the shared OpenAI call; `test_refresh_picks.py` is the regression guard.
