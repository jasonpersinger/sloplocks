# Design Spec: The Slime Tape (Direction 1)

## Status: Approved
## Date: 2026-03-19

## Goal
A total frontend rewrite of SLOP LOCKS, transforming it from a "dashboard" into a high-density, terminal-style "data tape." The focus is on raw technical utility and a minimalist "hacker" aesthetic.

## 1. Visual Identity (The "Tape" Aesthetic)
- **Palette:** 
    - Background: Pure Black (`#000000`).
    - Primary: Slime Green (`#39FF14`).
    - Secondary: Dim Grey (`#888888`) and Hacker Grey (`#1A1A1A`).
    - Highlights: Solid Green blocks for high-value picks.
- **Mobile Responsiveness:** 
    - Team names will use `overflow: hidden; text-overflow: ellipsis;` to prevent horizontal scrolling on narrow viewports. 
    - Text-based buttons (e.g., `[ NBA ]`) will have a minimum touch target height/width of 44px (using transparent padding) to ensure usability on mobile.
- **Compliance & Data:** 
    - The collapsible footer will retain all data source credits (ESPN, The Odds API) and the "FOR ENTERTAINMENT PURPOSES ONLY" disclaimer.
- **Typography:** 100% `JetBrains Mono`.
- **Styling:** 
    - Square corners only (no border-radius).
    - Hairline borders (`1px solid`).
    - Dashed separators (`1px dashed`).
    - No shadows, gradients, or complex animations.

## 2. Layout Structure
- **Global Ticker (Top):**
    - A narrow, full-width scrolling text line (marquee style or static compact row).
    - Data: `[ TOP EDGE: +12.4% ] [ TOTAL GAMES: 24 ] [ LAST UPDATE: 18:45 ]`.
- **Navigation:**
    - Minimalist text toggles: `[ NBA ] ( NCAAM )`.
    - Selected state indicated by brackets and green text.
- **The Tape (Feed):**
    - A narrow vertical column (max-width 450px).
    - Matches are rendered as structured text blocks.
    - **Header:** Time/Status + Dashed line.
    - **Body:** Head-to-head stats (Prob % and Edge %).
    - **Footer:** Brackets for key data: `[ PICK: HOME ] [ ODDS: +120 ] [ CONF: ★★★ ]`.
- **Footer Info:**
    - Collapsible mono section: `[ + ] SYSTEM LOGS / PERFORMANCE`.

## 3. Data & Rendering Logic
- **Single File:** Pure `index.html` with inline CSS and Vanilla JS.
- **State:** Manage `currentSport` and `feedFilter`.
- **Head-to-Head Stats:** 
    - Probability and Edge for both teams.
    - Positive edge highlighted in Slime Green.
- **Tags:**
    - `[ LOCK ]`: Solid green background highlight.
    - `[ GRINDER ]`: Distinct border highlight.
    - `[ FINAL ]`: Greyed out text.
    - `[ NEUTRAL ]`: Small text tag.

## 4. Implementation Details
- **CSS Reset:** Global `border-radius: 0`, `font-family: 'JetBrains Mono', monospace`.
- **Component: `MatchRow`:** Replacing the current card system. 
- **Sorting:** Default by Edge (descending) for upcoming games.
- **Instant Refresh:** When sport toggle is clicked, the "tape" clears and refills immediately.

## 5. Success Criteria
- The site looks and feels like a professional data feed.
- Information density is significantly higher than the previous design.
- Zero "fluff" (no icons, no bars, no rounded corners).
- Mobile experience feels like a native data terminal.
