# Frontend Redesign: Picks-Focused UI

**Date:** 2026-03-20
**Status:** Approved

## Overview

Redesign `index.html` to focus exclusively on 5 daily picks per view. Remove all secondary content (ticker, slimegrinder, longslop, full slate, verbose logs). The new site is a clean pick card feed with cross-sport and per-sport views.

## What's Removed

- Top ticker bar (top edge %, total games, bot status)
- Slimegrinder Trio section
- Longslop section
- "View Full Slate" / collapsible slate button and full game list
- Verbose System Logs / Performance footer (expanded stats, season stats, methodology blurb, data sources links)

## Page Structure

Single-file `index.html`, no backend changes required.

```
[ SLOP LOCKS ]  [ Mar 20, 2026 ]
[ ALL ] [ NBA ] [ NCAAM ] [ MLB ] [ MMA ]
─────────────────────────────────────────
  Pick Card 1
  Pick Card 2
  Pick Card 3
  Pick Card 4
  Pick Card 5
─────────────────────────────────────────
  14-8 · 58% · ROI +12.4%
```

## Navigation

A pill row below the header: **ALL · NBA · NCAAM · MLB · MMA**. ALL is the default on load. Tapping a pill switches the feed without a page reload. Active pill is slime green; others are dim.

## Pick Selection Logic

### ALL tab (cross-sport)
1. Fetch all 4 sport `predictions.json` files in parallel (`data/nba/predictions.json`, `data/ncaam/predictions.json`, `data/mlb/predictions.json`, `data/mma/predictions.json`)
2. Pool all `slop_locks` arrays from each sport response
3. Sort combined pool by `model_prob` descending
4. Take top 5

No pipeline changes needed — `slop_locks` already enforces positive edge (`edge >= 0`) and is computed daily by the pipeline.

### Per-sport tabs
Use that sport's `slop_locks` array directly. Already top 5 by `model_prob` with positive edge, computed by the pipeline.

## Pick Card Design

Each card shows (in order, top to bottom):

1. **Sport badge + date** — small, dim. e.g. `NBA · MAR 20`
2. **Pick team name** — large (18px), bold, slime green, uppercase. Right-aligned: **odds** (white, 22px bold). e.g. `CELTICS` / `-140`
3. **Opponent line** — dim small text. e.g. `vs KNICKS`
4. **Stats row** — three inline stats:
   - `Model: 67.4%` (slime green)
   - `Market: 58.3%` (dim)
   - `Edge: +9.1%` (slime green)
5. **Edge bar** — thin 3px bar. Background: `#1a1a1a`. Fill: slime green. Width proportional to model_prob (i.e. `model_prob * 100%`). Label row above: `MARKET IMPLIED` left, `MODEL EDGE` right.
6. **AI blurb** — 10px dim text, separated by a dashed top border. The `blurb` field from the pick.

Cards have a `#0d0d0d` background with a `1px solid #1a1a1a` border. No highlight color (the current "lock" rows are slime-green-filled — this is removed).

## Stats Footer

A single always-visible line at the bottom of the page:

```
14-8 · 58.3% · ROI +12.4%
```

- **ALL view**: aggregate wins/losses/ROI across all 4 sports by summing `pick_stats.all` from each loaded sport JSON (wins, losses, evaluated count, sum of (stake * odds * won) for ROI).
- **Per-sport view**: use `pick_stats.all` from that sport's JSON.
- If no evaluated picks yet, show `-- · -- · --`.
- Style: centered, `font-size: 10px`, dim (`#555`), `padding: 12px`.

## Visual Style

Unchanged from current:
- Background: `#000000`
- Accent: `#39FF14` (slime green)
- Text: `#EAEAEA`
- Dim: `#888888` / `#555555`
- Font: JetBrains Mono
- Max container width: 450px, centered

## Data Sources

All data comes from existing pipeline-generated JSON files:
- `data/{sport}/predictions.json` — contains `slop_locks`, `pick_stats`, `generated_at`
- `data/manifest.json` — not required for this redesign (all 4 sports fetched unconditionally)

No changes to pipeline, config, or any Python files.

## Error Handling

- If a sport fetch fails, skip it silently (don't block the other sports from rendering)
- If `slop_locks` is empty for all sports in ALL view, show: `NO PICKS TODAY — CHECK BACK AFTER 6AM UTC`
- If a specific sport has no picks, show the same message in that sport's view

## Out of Scope

- Any changes to the pipeline or JSON output format
- Historical pick browser
- Push notifications
- Dark/light mode toggle
