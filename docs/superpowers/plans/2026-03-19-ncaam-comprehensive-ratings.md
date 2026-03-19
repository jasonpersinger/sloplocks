# NCAAM Tournament Fix & Comprehensive Matchup Ratings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix NCAAM tournament ingestion by adding the correct ESPN group IDs and expand the pipeline to output all predicted matchups with a 1-5 star confidence rating.

**Architecture:** 
- Update NCAAM fetchers to query both standard (100) and tournament (50) group IDs.
- Implement a star-rating formula in `ensemble.py` based on model probability and market edge.
- Modify `run.py` to calculate these ratings for all games and include them in the primary JSON output.
- Update the frontend and Discord notifications to surface these comprehensive ratings.

**Tech Stack:** Python (pandas, requests), HTML/JS (Vanilla), Discord Webhooks.

---

### Task 1: Fix NCAAM Tournament Ingestion

**Files:**
- Modify: `pipeline/fetch_ncaam.py`
- Test: `tests/test_fetch_ncaam.py` (if exists, or manual verification if API keys are needed)

- [ ] **Step 1: Update `fetch_ncaam_games` to include groups.**
  ESPN tournament games are often in group 50.
  Modify the loop in `fetch_ncaam_games` to fetch both `groups=50` and `groups=100`.

- [ ] **Step 2: Update `fetch_ncaam_schedule` to include groups.**
  Modify the URL in `fetch_ncaam_schedule` to include `&groups=50,100`.

- [ ] **Step 3: Commit.**
```bash
git add pipeline/fetch_ncaam.py
git commit -m "fix(ncaam): include tournament group IDs in ESPN API calls"
```

### Task 2: Implement Confidence Star Rating Logic

**Files:**
- Modify: `pipeline/ensemble.py`
- Test: `tests/test_ensemble_stars.py` (Create new)

- [ ] **Step 1: Add `compute_confidence_stars` to `ensemble.py`.**
```python
def compute_confidence_stars(model_prob: float, edge: float) -> int:
    """Calculate 1-5 star rating based on weighted points.
    Edge >= 10%: +2, Edge >= 5%: +1
    Prob >= 75%: +2, Prob >= 60%: +1
    Base: 1 star. Max: 5.
    """
    stars = 1
    if edge >= 0.10: stars += 2
    elif edge >= 0.05: stars += 1
    
    if model_prob >= 0.75: stars += 2
    elif model_prob >= 0.60: stars += 1
    
    return min(5, stars)
```

- [ ] **Step 2: Write tests for the rating logic.**
- [ ] **Step 3: Commit.**
```bash
git add pipeline/ensemble.py tests/test_ensemble_stars.py
git commit -m "feat(ensemble): add star rating calculation logic"
```

### Task 3: Update Pipeline to Include Star Ratings and All Matchups

**Files:**
- Modify: `pipeline/run.py`

- [ ] **Step 1: Update `prediction_records` loop in `run_sport_pipeline`.**
  Import `compute_confidence_stars`. For each record, find the best outcome and calculate its stars.
  Include `confidence_stars`, `pick`, `model_prob`, and `edge` directly in the record.

- [ ] **Step 2: Broaden NCAAM date filter.**
  In `run_sport_pipeline`, for `fixtures` filtering, allow games from `today` OR `tomorrow` (UTC) if they match the ET "today" window.

- [ ] **Step 3: Commit.**
```bash
git add pipeline/run.py
git commit -m "feat(pipeline): calculate star ratings for all matchups and broaden NCAAM window"
```

### Task 4: Update Frontend to Display Comprehensive Matchups

**Files:**
- Modify: `index.html`

- [ ] **Step 1: Add "All Matchups" section.**
  Modify the JS in `index.html` to render all `matches` from `predictions.json`, sorted by `confidence_stars` (desc).
  Display stars (e.g., ⭐⭐⭐⭐⭐) next to each matchup.

- [ ] **Step 2: Commit.**
```bash
git add index.html
git commit -m "feat(ui): display comprehensive matchup list with star ratings"
```

### Task 5: Update Discord Notifications

**Files:**
- Modify: `pipeline/notify_discord.py`

- [ ] **Step 1: Update `build_payload` to include top-rated matchups.**
  If Slop Locks are few, or just as a bonus, list the next 5 best games by star rating.
  Include star icons in the Discord message.

- [ ] **Step 2: Commit.**
```bash
git add pipeline/notify_discord.py
git commit -m "feat(notify): include star ratings in Discord notifications"
```

### Task 6: Final Validation

- [ ] **Step 1: Run the pipeline locally for NCAAM.**
  `python -m pipeline.run`
- [ ] **Step 2: Verify `data/ncaam/predictions.json` contains comprehensive matches with stars.**
- [ ] **Step 3: Verify `index.html` renders correctly.**
- [ ] **Step 4: Commit.**
```bash
git commit --allow-empty -m "chore: final validation of NCAAM tournament and ratings fix"
```
