# Design Spec: Single-Page Minimalist Redesign

## Status: Approved
## Date: 2026-03-19

## Goals
1. **Consolidate to Single Page:** Remove the tab-based navigation in favor of a single vertical feed.
2. **Minimalist Aesthetic:** Clean up the UI, remove clutter (probability bars, breakdown toggles), and focus on the core pick data.
3. **Refined Slime Theme:** Keep the neon green/pink on black aesthetic but make it look more modern and premium.
4. **Remove Longslops:** Entirely remove the Longslop feature and UI.

## 1. Visual Design (Refined Slime)
- **Background:** Solid deep black (`#000000`).
- **Accent 1 (Slime):** Neon Green (`#39FF14`) for primary picks, positive edge, and buttons.
- **Accent 2 (Contrast):** Neon Pink (`#FF2D95`) for away highlights or secondary emphasis.
- **Typography:** 
    - Headers: `Oswald` (bold, tracked out).
    - Data/Body: `JetBrains Mono`.
- **Borders:** Thin, dark borders (`#1A1A1A`) with subtle neon glows only on "Locked" items.

## 2. Layout Structure
- **Header:**
    - Small, pixel-art logo + "SLOP LOCKS" wordmark.
    - Sport Toggle (NBA / NCAAM) as sleek pills.
    - "Last Updated" timestamp in small mono text.
- **Section A: Slop Locks (Top 5):**
    - Vertical list of high-contrast cards.
    - Each card shows: Teams (Home vs Away), Pick Badge (e.g., "TEAM WIN"), Odds, Edge %, and Confidence Stars.
- **Action:** 
    - Large, low-profile button: `[ + ] VIEW FULL SLATE`.
- **Section B: The Slate (Comprehensive List):**
    - Initially hidden.
    - When revealed, shows all other games from `matches` array in a more compact, list-like format.
- **Footer:**
    - Collapsible "Information" section containing:
        - Season Stats (ROI, Win Rate).
        - Methodology / Ensemble Weights.
        - Ko-fi link / Socials.

## 3. Data & Interactivity
- **Single Page State:** The app maintains the current sport (`currentSport`) and whether the full slate is visible (`slateVisible`).
- **Simplified Rendering:**
    - `createMatchCard` is rewritten to remove the probability bar and model breakdown.
    - `renderLocks` handles the top 5.
    - `renderSlate` (replacing `renderMatches`) handles the rest.
- **Logic:**
    - If `slateVisible` is false, only show Top 5.
    - If `slateVisible` is true, append the remaining matches sorted by Edge.

## 4. Implementation Plan
- **HTML:** Remove `<nav>` and all `tab-panel` divs. Re-scaffold the single `<main>` container.
- **CSS:** Rewrite styles for minimalism. Focus on grid/flex for the feed. Remove all tab-related CSS.
- **JS:** 
    - Remove tab switching logic.
    - Update `loadSportData` to trigger the new rendering pipeline.
    - Update `createMatchCard` to the Approved "A" detail level (Teams + Pick + Odds + Edge + Stars).
    - Add event listener for the "View Full Slate" button.

## 5. Success Criteria
- No more tabs; all info accessible on one page.
- "Locks" feel premium and prominent.
- Full slate is accessible but doesn't clutter the initial view.
- Page looks significantly cleaner and easier to read.
