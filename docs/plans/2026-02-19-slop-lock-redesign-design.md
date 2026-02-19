# Slop Lock Redesign Design

**Date:** 2026-02-19
**Goal:** Change Slop Lock selection from value-edge picks to confident picks within a reasonable odds window (-150 to +195), ranked by model probability.

## Problem

The current Slop Lock algorithm picks the top 5 outcomes by *edge* (model_prob - implied_prob), with no odds filter. This maximises value but surfaces longshot underdogs as locks — including legitimately bad teams that happen to have model surprise value. The Slop Lock category is meant to be the "most likely to hit" bucket, not the highest-value bucket. Longshots belong in LONGSLOP.

## Design

### Selection logic change (`pipeline/run.py` — `_compute_slop_locks`)

| | Before | After |
|---|---|---|
| Filter | `edge > 0` (any odds) | `-150 ≤ american_odds ≤ +195` |
| Rank by | Edge descending | Model probability descending |
| Market agreement required | Yes (implicit via edge > 0) | No |
| Count | Top 5 | Top 5 |

The new filter converts to decimal: roughly 1.51 to 2.95 (covering solid favourites to moderate underdogs). No minimum edge required — the model can be less bullish than the books and the pick still qualifies.

### Config constants (`pipeline/config.py`)

```python
SLOP_LOCK_MIN_ODDS = -150  # American odds lower bound
SLOP_LOCK_MAX_ODDS = 195   # American odds upper bound
```

### Blurb prompt update (`pipeline/run.py` — `_generate_blurbs`)

Change the `pick_type == "lock"` prompt from:
> "explaining why this is a value pick… Reference specific model data or edge."

To:
> "explaining why the model is confident in this pick… Reference specific model probability data."

This reflects the new framing: Slop Locks are confident plays, not value edges.

### LONGSLOP

No changes. Longslop stays: single best pick at +500 or better where model_prob >= implied_prob.

## Files Touched

| File | Change |
|------|--------|
| `pipeline/config.py` | Add `SLOP_LOCK_MIN_ODDS = -150`, `SLOP_LOCK_MAX_ODDS = 195` |
| `pipeline/run.py` | Update `_compute_slop_locks` filter + sort; update blurb prompt |
| `tests/test_run.py` | Update/add `_compute_slop_locks` tests |

## Non-Goals

- Changing LONGSLOP selection
- Changing the frontend display layout (cards, styling, tabs)
- Changing how edge is computed or displayed (edge still shown in output, just not used for ranking)
