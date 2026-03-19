# Design Spec: NCAAM Tournament Fix & Comprehensive Matchup Ratings

## Status: Approved
## Date: 2026-03-19

## Goals
1. **Fix NCAAM Tournament Ingestion:** Ensure that NCAA Tournament (March Madness) games are correctly fetched and appear in the pipeline.
2. **Comprehensive Matchup List:** Instead of just "Slop Locks," provide a full list of all predicted matchups.
3. **Confidence & Edge Ratings:** Each matchup should display its model probability, edge against market odds, and a 1-5 star "Confidence Rating."

## 1. NCAAM Tournament Ingestion Fix
### The Issue
NCAAM tournament games often require a specific `groups` parameter in the ESPN API (`groups=50` for the tournament, vs `groups=100` for general conference play). The current fetcher uses the default.

### The Solution
- Update `pipeline/fetch_ncaam.py` to include `groups=50` and `groups=100` in the scoreboard requests.
- Update `fetch_ncaam_schedule` to also fetch both groups.
- In `pipeline/run.py`, broaden the date filtering for NCAAM to include games starting later in the evening or early next morning (UTC-wise) to ensure no tournament games are missed due to time zone shifts.

## 2. Confidence & Edge Ratings
### Formula: Simple Weighted Points
The "Confidence Rating" (1-5 stars) will be calculated based on a combination of **Edge** and **Model Probability**.

Points are awarded as follows:
- **Edge Points:**
  - Edge >= 10%: +2 points
  - Edge >= 5%: +1 point
- **Probability Points:**
  - Prob >= 75%: +2 points
  - Prob >= 60%: +1 point
- **Base Point:** Every matchup starts with +1 point (1 star).
- **Total:** Sum of points, capped at 5.

### Data Structure Update
The `predictions.json` file's `matches` array will now include:
- `confidence_stars`: 1-5 integer.
- `model_prob`: Float (probability of the most likely outcome).
- `edge`: Float (edge of the most likely outcome).
- `pick`: The suggested outcome ("home" or "away").

## 3. Comprehensive Output
- **Pipeline:** `run_sport_pipeline` will output *all* valid predictions into the `matches` field of `predictions.json`.
- **Frontend (`index.html`):** Add a section to display "All Today's Matchups" with their star ratings, sorted by Confidence Rating (highest first).
- **Discord (`pipeline/notify_discord.py`):** Update the notification to include the top 10 matchups by star rating if they aren't already in Slop Locks.

## 4. Implementation Details
- **`pipeline/fetch_ncaam.py`:** Update API calls.
- **`pipeline/ensemble.py`:** Add helper to compute star ratings.
- **`pipeline/run.py`:** Use the new star rating helper and ensure all matches are correctly recorded.
- **`index.html`:** Update UI to show the full list.
- **`pipeline/notify_discord.py`:** Refine the summary message.

## 5. Success Criteria
- NCAA Tournament games appear in the data and predictions.
- Every game has a 1-5 star rating.
- The web UI shows all games, not just the top 5 "Slop Locks."
