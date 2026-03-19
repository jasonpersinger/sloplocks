# Slime Tape (Direction 1) Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the SLOP LOCKS frontend into a high-density, terminal-style "data tape."

**Architecture:**
- Single-file `index.html` with vanilla JS.
- CSS-driven terminal aesthetic (monospace, hairline borders, no border-radius).
- Refactored rendering logic to switch from "Cards" to "Rows."

**Tech Stack:** HTML5, CSS3, Vanilla JS.

---

### Task 1: Terminal Style & Layout Foundation

**Files:**
- Modify: `index.html`

- [ ] **Step 1: Replace `<style>` block with new terminal CSS.**
  - Set background to `#000`.
  - Set global font to `JetBrains Mono`.
  - Define utility classes for hairline borders and dashed lines.
  - Set `.container` max-width to `450px`.
- [ ] **Step 2: Re-scaffold the `<body>` for the tape structure.**
  - Create `#ticker-top`, `#sport-nav`, `#tape-feed`, and `#logs-footer`.
- [ ] **Step 3: Commit.**
```bash
git add index.html
git commit -m "feat(ui): implement terminal style foundation and layout"
```

### Task 2: Ticker & Navigation

**Files:**
- Modify: `index.html`

- [ ] **Step 1: Implement the Top Ticker.**
  - Add JS logic to update the ticker content: `[ TOP EDGE: ... ] [ TOTAL GAMES: ... ] [ BOT: OPTIMAL ]`.
- [ ] **Step 2: Implement Bracket Navigation.**
  - Create bracket-style toggles: `[ NBA ]` (active) vs `( NCAAM )` (inactive).
  - Ensure large hit-areas for mobile touch targets.
- [ ] **Step 3: Commit.**
```bash
git add index.html
git commit -m "feat(ui): add data ticker and bracket navigation"
```

### Task 4: Match Row Rendering

**Files:**
- Modify: `index.html`

- [ ] **Step 1: Refactor `createMatchRow` (replacing `createMatchCard`).**
  - Layout:
    - Row 1: Time/Status + Dashed Line.
    - Row 2: Home Team Name + Win% + Edge%.
    - Row 3: Away Team Name + Win% + Edge%.
    - Row 4: Pick Badge + Odds + Confidence Stars.
  - Use solid green backgrounds for `[ LOCK ]` rows.
- [ ] **Step 2: Update `render` function.**
  - Use the new `createMatchRow`.
  - Apply Edge-descending sort.
- [ ] **Step 3: Commit.**
```bash
git add index.html
git commit -m "feat(ui): implement match row rendering for the tape"
```

### Task 5: System Logs & Footer

**Files:**
- Modify: `index.html`

- [ ] **Step 1: Implement Collapsible "System Logs."**
  - Move ROI and Win Rate stats here.
  - Move Methodology text here.
  - Retain "Entertainment Only" and Ko-fi links.
- [ ] **Step 2: Add "View Full Slate" terminal button.**
- [ ] **Step 3: Commit.**
```bash
git add index.html
git commit -m "feat(ui): add collapsible system logs footer"
```

### Task 6: Validation & Deployment

- [ ] **Step 1: Run pipeline to ensure data is in sync.**
  `python -m pipeline.run`
- [ ] **Step 2: Open `index.html` locally and verify the "Tape" feel.**
- [ ] **Step 3: Verify mobile responsiveness (no horizontal scroll).**
- [ ] **Step 4: Push to master.**
```bash
git push origin master
```
