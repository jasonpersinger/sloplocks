# Slop Lock Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Change Slop Lock selection from highest-edge picks (any odds) to most-confident picks within -150 to +195 American odds.

**Architecture:** Two-file change. Add odds-window constants to `config.py`. In `run.py`, swap the `_compute_slop_locks` filter (edge > 0 → odds window) and sort key (edge → model_prob), and update the blurb prompt tone from "value pick" to "confident pick." Fix stale test assertions in `test_run.py` that assumed edge > 0 and narrow odds.

**Tech Stack:** Python, pytest

---

### Task 1: Config constants + `_compute_slop_locks` logic

**Files:**
- Modify: `pipeline/config.py`
- Modify: `pipeline/run.py` (lines ~233–261, `_compute_slop_locks`)
- Test: `tests/test_run.py`

---

**Step 1: Write failing tests**

Read `tests/test_run.py` first. Find the `TestDaysSinceLastGame` class (at the bottom). Add a new `TestComputeSlopLocks` class directly after it:

```python
class TestComputeSlopLocks:
    """Tests for the _compute_slop_locks helper."""

    def _make_record(self, home, away, outcome, model_prob, american_odds, edge=0.0):
        """Build a minimal prediction record for testing."""
        implied_prob = 1 / (1 + abs(american_odds) / 100) if american_odds < 0 else 100 / (american_odds + 100)
        return {
            "home_team": home,
            "away_team": away,
            "date": "2026-03-01",
            "matchday": None,
            "edges": {
                outcome: {
                    "model_prob": model_prob,
                    "implied_prob": implied_prob,
                    "edge": edge,
                    "decimal_odds": 0.0,
                    "american_odds": american_odds,
                    "is_value": edge >= 0.05,
                }
            },
            "best_odds": {outcome: american_odds},
            "model_probs": {outcome: model_prob},
            "individual_models": {},
        }

    def test_filters_to_odds_window(self):
        """Only picks with -150 <= american_odds <= 195 qualify."""
        from pipeline.run import _compute_slop_locks
        records = [
            self._make_record("A", "B", "home", 0.70, -200),   # too short, excluded
            self._make_record("C", "D", "home", 0.65, -140),   # in window
            self._make_record("E", "F", "away", 0.55, 190),    # in window
            self._make_record("G", "H", "away", 0.45, 250),    # too long, excluded
        ]
        locks = _compute_slop_locks(records, ["home", "away"])
        odds = [l["american_odds"] for l in locks]
        assert all(-150 <= o <= 195 for o in odds)
        assert len(locks) == 2

    def test_ranked_by_model_probability(self):
        """Locks are sorted by model_prob descending."""
        from pipeline.run import _compute_slop_locks
        records = [
            self._make_record("A", "B", "home", 0.55, 100),
            self._make_record("C", "D", "home", 0.75, -130),
            self._make_record("E", "F", "away", 0.65, 110),
        ]
        locks = _compute_slop_locks(records, ["home", "away"])
        probs = [l["model_prob"] for l in locks]
        assert probs == sorted(probs, reverse=True)

    def test_no_market_agreement_required(self):
        """A pick qualifies even when model_prob < implied_prob (negative edge)."""
        from pipeline.run import _compute_slop_locks
        # -130 implied = ~56.5%; model says 52% — negative edge, should still qualify
        records = [
            self._make_record("A", "B", "home", 0.52, -130, edge=-0.045),
        ]
        locks = _compute_slop_locks(records, ["home", "away"])
        assert len(locks) == 1

    def test_returns_at_most_five(self):
        """At most 5 locks returned."""
        from pipeline.run import _compute_slop_locks
        records = [
            self._make_record(f"T{i}", f"T{i+1}", "home", 0.60, -100)
            for i in range(10)
        ]
        locks = _compute_slop_locks(records, ["home", "away"])
        assert len(locks) <= 5
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_run.py::TestComputeSlopLocks -v
```

Expected: All 4 tests FAIL — `test_filters_to_odds_window` and `test_no_market_agreement_required` will fail because the current filter requires `edge > 0`, not an odds window; `test_ranked_by_model_probability` fails because current sort is by edge.

**Step 3: Add constants to `pipeline/config.py`**

Read `pipeline/config.py`. Add after `VALUE_EDGE_THRESHOLD`:

```python
SLOP_LOCK_MIN_ODDS = -150   # American odds lower bound for Slop Locks
SLOP_LOCK_MAX_ODDS = 195    # American odds upper bound for Slop Locks
```

**Step 4: Rewrite `_compute_slop_locks` in `pipeline/run.py`**

First, add `SLOP_LOCK_MIN_ODDS, SLOP_LOCK_MAX_ODDS` to the config import at the top of `run.py`. Find the existing import:

```python
from pipeline.config import (
    ANTHROPIC_API_KEY,
    CONGESTION_THRESHOLD_DAYS,
    DATA_DIR,
    NBA_B2B_PENALTY,
    SPORTS,
)
```

Add the two new constants to it.

Then replace the entire `_compute_slop_locks` function (lines ~233–261) with:

```python
def _compute_slop_locks(prediction_records, outcomes):
    """Extract SLOP LOCKS: top 5 most confident picks at -150 to +195 odds."""
    lock_candidates = []
    for rec in prediction_records:
        edges = rec.get("edges", {})
        best_odds = rec.get("best_odds", {})
        for outcome in outcomes:
            e = edges.get(outcome)
            if not e:
                continue
            american = best_odds.get(outcome, e.get("american_odds"))
            if american is None:
                continue
            if not (SLOP_LOCK_MIN_ODDS <= american <= SLOP_LOCK_MAX_ODDS):
                continue
            lock_candidates.append({
                "home_team": rec["home_team"],
                "away_team": rec["away_team"],
                "date": rec["date"],
                "matchday": rec.get("matchday"),
                "pick": outcome,
                "model_prob": round(e["model_prob"], 4),
                "implied_prob": round(e["implied_prob"], 4),
                "edge": round(e["edge"], 4),
                "american_odds": american,
                "decimal_odds": e["decimal_odds"],
                "individual_models": rec.get("individual_models", {}),
            })

    lock_candidates.sort(key=lambda x: x["model_prob"], reverse=True)
    return lock_candidates[:5]
```

**Step 5: Run tests**

```bash
pytest tests/test_run.py::TestComputeSlopLocks -v
```

Expected: All 4 pass.

**Step 6: Fix stale assertions in existing tests**

The `TestRunEPLPipeline::test_produces_valid_epl_predictions` test (in `tests/test_run.py`) has two stale assertions that no longer match the new design:

```python
assert lock["edge"] > 0               # wrong: no edge requirement
assert -200 <= lock["american_odds"] <= 200  # wrong: bounds are -150/+195, and also wrong value
```

Replace those two lines with:

```python
assert -150 <= lock["american_odds"] <= 195
```

**Step 7: Run full test suite**

```bash
pytest tests/ -v -q 2>&1 | tail -10
```

Expected: The previously-failing EPL test (`assert 280 <= 200`) is now fixed. All tests pass except for any that require live network access.

**Step 8: Commit**

```bash
git add pipeline/config.py pipeline/run.py tests/test_run.py
git commit -m "feat: Slop Locks now pick confident plays at -150 to +195 odds"
```

---

### Task 2: Blurb prompt + pipeline run + push

**Files:**
- Modify: `pipeline/run.py` (lines ~190–202, the `pick_type == "lock"` blurb prompt)

---

**Step 1: Update the blurb prompt**

Find the `if pick_type == "lock":` block in `_generate_blurbs` (~line 190). Change the prompt string from:

```python
f"Write exactly 1-2 sentences explaining why this is a value pick. Be direct, "
f"confident, concise. No hedging. Reference specific model data or edge.\n\n"
```

To:

```python
f"Write exactly 1-2 sentences explaining why the model is confident in this pick. Be direct, "
f"confident, concise. No hedging. Reference the model probability and why this outcome is likely.\n\n"
```

**Step 2: Run all tests**

```bash
pytest tests/ -v -q 2>&1 | tail -5
```

Expected: Same results as Task 1 — no regressions.

**Step 3: Run the full pipeline**

```bash
python3 -m pipeline.run
```

Check `data/nba/predictions.json` and `data/epl/predictions.json`:
- `slop_locks` entries (if any) should have `american_odds` between -150 and 195
- Locks should be sorted by `model_prob` descending (highest confidence first)

**Step 4: Commit and push**

```bash
git add pipeline/run.py data/
git diff --staged --quiet || git commit -m "feat: update Slop Lock blurb tone to confident picks"
git push origin master
```
