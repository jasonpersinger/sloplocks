# SLIMEGRINDER Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement "Slimegrinder," a conservative betting strategy targeting a 66% hit rate using picks with odds between -250 and +165 and a positive edge.

**Architecture:**
- Add Slimegrinder logic to the prediction pipeline.
- Include a `slimegrinder` array in `predictions.json`.
- Add a high-visibility Slimegrinder section to the minimalist frontend.

**Tech Stack:** Python, HTML/JS.

---

### Task 1: Update Configuration

**Files:**
- Modify: `pipeline/config.py`

- [ ] **Step 1: Add Slimegrinder constants.**
```python
SLIMEGRINDER_MIN_ODDS = -250
SLIMEGRINDER_MAX_ODDS = 165
```

- [ ] **Step 2: Commit.**
```bash
git add pipeline/config.py
git commit -m "feat(config): add Slimegrinder odds window constants"
```

### Task 2: Implement Slimegrinder Logic in Pipeline

**Files:**
- Modify: `pipeline/run.py`
- Modify: `pipeline/refresh_picks.py`

- [ ] **Step 1: Add `_compute_slimegrinder` to `pipeline/run.py`.**
  Logic: Filter records for odds -250 to +165, Edge > 0, sort by Model Prob desc, take Top 3.
- [ ] **Step 2: Update `run_sport_pipeline` to include `slimegrinder` in output.**
- [ ] **Step 3: Update `pipeline/refresh_picks.py` with `_compute_slimegrinder` logic.**
- [ ] **Step 4: Commit.**
```bash
git add pipeline/run.py pipeline/refresh_picks.py
git commit -m "feat(pipeline): implement Slimegrinder selection logic"
```

### Task 3: Update Frontend UI

**Files:**
- Modify: `index.html`

- [ ] **Step 1: Add Slimegrinder CSS.**
  - Distinct styling (e.g., cyan/blue theme) to differentiate from green Slop Locks.
- [ ] **Step 2: Add Slimegrinder container to HTML.**
  - Position it above the Slop Locks section.
- [ ] **Step 3: Update JS rendering.**
  - Implement `renderSlimegrinder(data.slimegrinder)`.
- [ ] **Step 4: Commit.**
```bash
git add index.html
git commit -m "feat(ui): add Slimegrinder section to feed"
```

### Task 4: Final Validation

- [ ] **Step 1: Run pipeline locally.**
  `python -m pipeline.run`
- [ ] **Step 2: Verify Slimegrinder picks appear in JSON and on local index.html.**
- [ ] **Step 3: Commit and push.**
```bash
git add data/
git commit -m "chore: rollout Slimegrinder data"
git push origin master
```
