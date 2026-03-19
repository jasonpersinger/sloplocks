# Design Spec: SLIMEGRINDER Bankroll Builder

## Status: Approved
## Date: 2026-03-19

## Goal
Introduce "SLIMEGRINDER," a conservative betting strategy focused on a high hit rate (target 66%) to steadily build bankroll.

## 1. Slimegrinder Logic (The Algorithm)
1. **Odds Window:** Only consider picks with American odds between **-250 and +165**.
2. **Positive Edge:** Only consider picks where the Model Probability > Implied Market Probability (Edge > 0).
3. **Primary Rank:** Sort candidates by **Model Probability (descending)**. We want the likeliest winners.
4. **Selection:** Take the **Top 3** qualifying picks across all sports (or per sport, displayed as a cross-sport trio).

## 2. Technical Implementation
### Pipeline (`run.py` & `refresh_picks.py`)
- Add a function `_compute_slimegrinder(prediction_records, outcomes)` similar to `_compute_slop_locks`.
- Update the final `predictions.json` to include a `slimegrinder` array of objects.

### Data Structure
The `slimegrinder` objects will match the `slop_locks` structure:
- `home_team`, `away_team`, `pick`, `model_prob`, `edge`, `american_odds`, `confidence_stars`.

### Frontend (`index.html`)
- Add a **SLIMEGRINDER** section at the very top of the feed (above Slop Locks).
- Visual style: A more industrial, "gritty" look (e.g., darker borders or a different badge color like Cyan/Blue) to distinguish it from the "hot" Slop Locks.
- Show the 3 picks as a cohesive group.

## 3. Visual Layout (Minimalist)
- **Label:** `SLIMEGRINDER TRIO` (Small mono text).
- **Cards:** Slightly more compact than Slop Locks.
- **Goal Text:** Small footer note: "Targeting 66% hit rate • Conservative bankroll growth."

## 4. Success Criteria
- The model identifies 3 likeliest winners within the odds window.
- Slimegrinder picks are clearly distinct from Slop Locks on the single-page feed.
- NCAAM tournament games are included in the Slimegrinder candidates.
