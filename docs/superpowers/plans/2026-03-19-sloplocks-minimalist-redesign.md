# Single-Page Minimalist Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the entire frontend (`index.html`) to be a single-page minimalist feed with a "Daily Summary" focus, removing tabs and longslops, and ensuring NCAAM data feeds correctly.

**Architecture:**
- Single-column vertical feed.
- "Slop Locks" hero section at the top.
- "View Full Slate" reveal button for the remaining matches.
- Collapsible information footer for stats/methodology.
- Refined "Refined Slime" aesthetic (Neon on Black).

**Tech Stack:** HTML5, CSS3 (Vanilla), Vanilla JS (ES5 for compatibility).

---

### Task 1: Clean Up HTML Scaffold

**Files:**
- Modify: `index.html`

- [ ] **Step 1: Remove `<nav>` and all `tab-panel` wrappers.**
  Consolidate everything into a single `<main>` structure with clear semantic sections (`#section-locks`, `#section-slate`, `#section-info`).
- [ ] **Step 2: Update Header.**
  Scale down the SVG and wordmark. Ensure the sport toggle is clean.
- [ ] **Step 3: Commit.**
```bash
git add index.html
git commit -m "feat(ui): simplify HTML structure for single-page layout"
```

### Task 2: Minimalist CSS Overhaul

**Files:**
- Modify: `index.html`

- [ ] **Step 1: Replace all styles.**
  - Background: `#000`.
  - Typography: `Oswald` for headings, `JetBrains Mono` for data.
  - Border: `#1A1A1A`.
  - Remove all tab-related CSS.
  - Style the new minimalist match cards (Detail A).
- [ ] **Step 2: Add reveal animations.**
  - Smooth slide-down for the full slate.
- [ ] **Step 3: Commit.**
```bash
git add index.html
git commit -m "feat(ui): minimalist CSS overhaul"
```

### Task 3: Refactor JS Rendering Logic

**Files:**
- Modify: `index.html`

- [ ] **Step 1: Update `createMatchCard` to "Detail A".**
  - Teams (Home vs Away)
  - Pick Badge (e.g., `HOUSTON WIN`)
  - Market Info (`+120 · 8.2% EDGE`)
  - 1-5 Star confidence rating.
  - *Remove probability bars and model breakdown.*
- [ ] **Step 2: Update `loadSportData`.**
  - Trigger rendering of both Locks and Slate sections simultaneously.
  - Reset `slateVisible` state on sport change.
- [ ] **Step 3: Implement "View Full Slate" toggle.**
- [ ] **Step 4: Commit.**
```bash
git add index.html
git commit -m "feat(ui): refactor JS for single-page feed"
```

### Task 4: Move Stats & Methodology to Footer

**Files:**
- Modify: `index.html`

- [ ] **Step 1: Create collapsible Info section.**
  - "Season Stats" and "Methodology" sections moved here.
- [ ] **Step 2: Commit.**
```bash
git add index.html
git commit -m "feat(ui): move secondary info to footer"
```

### Task 5: Final Validation & Deployment

- [ ] **Step 1: Run full pipeline locally to generate fresh data.**
  `python -m pipeline.run`
- [ ] **Step 2: Verify both NBA and NCAAM show data in the new UI.**
- [ ] **Step 3: Commit and push.**
```bash
git add .
git commit -m "chore: final minimalist redesign rollout"
git push origin master
```
