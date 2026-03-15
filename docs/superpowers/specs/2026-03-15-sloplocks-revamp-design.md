# SLOPLOCKS Revamp — Design Spec
**Date:** 2026-03-15  
**Status:** Approved

---

## Overview

Strip sloplocks down to its core purpose: generate 5 high-quality betting picks per day and post them to Discord. The website is gone. Backtest/ROI tracking is gone. The Discord card is the product.

The models (Elo, AdjustedEfficiency, FourFactors) are retained as-is — they work. The rebuild targets the output layer: pick curation, line shopping, and Discord formatting.

---

## Architecture

### Current flow (broken)
```
run.py → per-sport predictions.json (with slop_locks pre-filtered) → notify_discord.py reads files
```
Pick selection logic is scattered across `run.py` with a leaky fallback (`SLOP_LOCK_FALLBACK_MIN_ODDS = -350`) that produces garbage picks.

### New flow
```
run.py
  └─ per sport: fetch data → fit models → write raw candidates to predictions.json
       ↓
pipeline/curator.py  ← NEW
  └─ reads all sports' raw candidates, applies guardrails, ranks, returns ≤5 picks
       ↓
pipeline/notify_discord.py  ← REWRITTEN
  └─ formats the card, posts to Discord webhook
```

`curator.py` is the single source of truth for "what is a good pick." No fallback that breaks the rules. If 3 picks qualify today, 3 are posted. `predictions.json` files remain for debugging but Discord no longer depends on the pre-filtered `slop_locks` key.

---

## Pick Curation Guardrails (`curator.py`)

### Hard filters — all must pass
| Filter | Value | Rationale |
|--------|-------|-----------|
| Minimum edge | **8%** | Eliminates noise (current system lets 1.2% edge picks through) |
| Odds lower bound | **-130** American | No heavy favorites; -270 picks are low-value |
| Odds upper bound | **+350** American | Caps extreme longshots |
| No fallback | — | If no picks qualify, none are posted |

### Ranking
Picks that pass all filters are ranked by `edge × model_prob` — rewards picks that are both genuinely mispriced AND high-confidence.

### Card size
5 picks max. Posts 2–3 if that's all that qualifies. Never pads with picks that failed the filters.

---

## Line Shopping

The Odds API returns odds from multiple bookmakers in a single request (no extra API cost). For each candidate pick, the curator identifies the best available American odds across the user's books:

**Books:** BetMGM, Caesars, DraftKings, FanDuel, Bet365, Bally Bet

**Odds API keys:** `betmgm`, `williamhill_us`, `draftkings`, `fanduel`, `bet365`, `bally_bet`

The best line and the book offering it are both stored per pick and displayed in the Discord card.

Edge calculation uses the **average implied probability across all 6 books** (removes soft-book inflation), while the displayed odds show the best available line.

---

## Sports Coverage

### Retained
- **NBA** — Elo + AdjustedEfficiency + FourFactors (3-model ensemble)
- **NCAAM** — same stack

### New: NHL
- **Models:** Elo + AdjustedEfficiency (2 models, equal weights — no FourFactors, basketball-specific)
- **Data source:** ESPN NHL API (`site.api.espn.com/apis/site/v2/sports/hockey/nhl`) — free, no key
- **New file:** `pipeline/fetch_nhl.py` — mirrors `fetch_ncaam.py` pattern
- **Config entry:**
```python
"nhl": {
    "name": "NHL",
    "odds_sport": "icehockey_nhl",
    "outcomes": ["home", "away"],
    "models": ["elo", "efficiency"],
    "elo_k_factor": 6,
    "elo_home_advantage": 15,
    "efficiency_home_bonus": 1.5,
    "data_dir": os.path.join(DATA_DIR, "nhl"),
}
```

---

## Discord Format

One embed per day. Clean, scannable. No blurbs, no sport headers, no noise.

```
🔒 SLOP LOCKS · Mar 15

LAKERS · +135 (DK) · 54% conf · +9.1% edge
BRUINS · +140 (FD) · 61% conf · +11.2% edge
ST. JOHN'S · +128 (MGM) · 61% conf · +17.5% edge

3 locks today
```

**Fields per pick:**
- Team name (picked side only, all caps)
- Best odds in American format with book abbreviation in parens
- Model confidence as percentage
- Edge as signed percentage

**Book abbreviations:** DK, FD, MGM, CZR, B365, BALLY

**Footer:** pick count line — helps identify thin days at a glance.

---

## Refresh Workflow (retained)

`pipeline/refresh_picks.py` and `.github/workflows/refresh-picks.yml` are kept as-is. This manual trigger re-fetches odds and recomputes the curator's output in ~20 seconds without retraining models. Useful because odds move throughout the day.

---

## What Gets Removed

| Component | Action |
|-----------|--------|
| `index.html`, `manifest.json`, `sw.js` | Delete — website is down |
| `icons/` | Delete |
| `netlify.toml` | Delete |
| `data/sotd.json` | Delete |
| Pick history evaluation logic in `run.py` | Remove — outcome tracking abandoned |
| `SLOP_LOCK_FALLBACK_MIN_ODDS` in `config.py` | Remove |
| `longslop` key in predictions.json | Remove |
| `slop_locks` pre-filtering in `run.py` | Move to `curator.py` |
| `notify_discord.py` | Rewrite |

`backtest.py`, `pick_history.json`, and `history.json` files are **retained** — the backtest utilities (`compute_model_weights`, `get_rolling_accuracy`) are still used to weight the ensemble, even though ROI tracking is abandoned.

---

## API Budget

Free tier: 500 requests/month on The Odds API.  
Daily pipeline: ~3 sports × 1 odds request each = **3 requests/day = ~93/month**.  
Refresh runs (manual): ~3 requests per trigger.  
Budget is comfortable even with daily refreshes.

---

## Files Created/Modified

| File | Action |
|------|--------|
| `pipeline/curator.py` | **Create** — unified pick selection |
| `pipeline/fetch_nhl.py` | **Create** — ESPN NHL data fetcher |
| `pipeline/notify_discord.py` | **Rewrite** — simplified Discord card |
| `pipeline/config.py` | **Modify** — add NHL, remove fallback odds, add book list |
| `pipeline/run.py` | **Modify** — add NHL sport branch, remove pick evaluation loop, remove slop_locks pre-filtering |
| `pipeline/refresh_picks.py` | **Modify** — update to use curator |
| `index.html`, `sw.js`, `manifest.json`, `netlify.toml` | **Delete** |
