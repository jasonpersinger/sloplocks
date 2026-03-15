# SLOPLOCKS Revamp — Design Spec
**Date:** 2026-03-15  
**Status:** Approved

---

## Overview

Strip sloplocks down to its core purpose: generate 5 high-quality betting picks per day and post them to Discord. The website is gone. Backtest/ROI tracking is gone. The Discord card is the product.

The models (Elo, AdjustedEfficiency, FourFactors) are retained as-is — they work. The rebuild targets the output layer: pick curation, line shopping, and Discord formatting.

---

## Architecture

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

`curator.py` is the single source of truth for "what is a good pick." `predictions.json` files (one per sport) contain all candidates with model probs and edges — their schema changes from having a pre-filtered `slop_locks` key to containing only `matches` (raw candidates). Discord no longer reads from `slop_locks`; the curator does all selection.

---

## Pick Curation Guardrails (`curator.py`)

### Hard filters — all must pass
| Filter | Value | Rationale |
|--------|-------|-----------|
| Minimum edge | **8%** | Eliminates noise (current system lets 1.2% edge picks through) |
| Odds lower bound | **-130** American | No heavy favorites |
| Odds upper bound | **+350** American | Caps extreme longshots |
| No fallback | — | If no picks qualify, card is posted with a "no locks today" message |

### Ranking
Picks that pass all filters are ranked by **edge alone** (descending). Edge is the primary signal — it already incorporates model confidence implicitly (a high-confidence model that disagrees with the line produces a large edge). Multiplying by model_prob would bias toward favorites.

### Cross-sport conflict resolution
Selection is purely global by edge. If 4 NBA picks and 3 NHL picks all qualify, the top 5 by edge are taken regardless of sport. No per-sport cap — let the market inefficiency decide.

### Card size
5 picks max. Posts fewer if quality isn't there. If zero picks qualify, posts a single "no locks today" message.

---

## Line Shopping

### Books
BetMGM (`betmgm`), Caesars (`williamhill_us`), DraftKings (`draftkings`), FanDuel (`fanduel`), Bet365 (`bet365`), Bally Bet (`ballysports`)

### Edge calculation (de-vigged)
For each book that returns a line:
1. Compute raw implied probabilities for both outcomes from the decimal odds
2. De-vig via multiplicative normalization: divide each raw implied prob by their sum — this strips the book's margin and gives fair probabilities
3. Average the fair home and away probabilities across all books that returned a line

Edge = `model_prob − avg_fair_implied_prob`

### Minimum book coverage
If fewer than 2 of the 6 books return a line for a game, skip that game — the market is too thin to trust the implied probability.

### Displayed odds
Best available American odds across the books that returned lines, with the book abbreviation that offers it.

**Abbreviations:** DK, FD, MGM, CZR, B365, BALLY

---

## API Budget

Free tier: 500 requests/month on The Odds API.

| Usage | Requests |
|-------|----------|
| Daily pipeline (3 sports × 1 request each) | 3/day → ~93/month |
| Manual refresh runs (3 requests per trigger) | ~30/month (10 runs) |
| **Total** | **~123/month** — well within 500 |

**On budget exhaustion:** The Odds API returns a 429 with remaining request count in headers. If the daily run hits the limit, odds fetching is skipped for that sport and no picks are generated from it that day. This is logged but does not fail the pipeline.

---

## Sports Coverage

### Retained
- **NBA** — Elo + AdjustedEfficiency + FourFactors (3-model ensemble)
- **NCAAM** — same stack; The Odds API key `basketball_ncaab` is confirmed available

### New: NHL
- **Models:** Elo + AdjustedEfficiency (2 models, equal weights — FourFactors is basketball-specific)
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

## Scheduling

The existing GitHub Actions cron runs at **6am UTC (1am ET)**. This is early enough to capture previous-day results for model updates but the odds are typically set for evening games. This timing is acceptable — lines are available and stable by early morning.

The **manual refresh workflow** (`refresh-picks.yml`) is available for a same-day odds update closer to tip-off if desired. It re-runs the curator and posts a fresh Discord card without retraining models. The refresh workflow is **updated** (not retained as-is) to invoke the new curator → notify flow.

---

## Discord Format

One embed per day. Clean and scannable. Sport emoji distinguishes teams across sports on mixed-sport days.

```
🔒 SLOP LOCKS · Mar 15

🏀 LAKERS · +135 (DK) · 54% conf · +9.1% edge
🏒 BRUINS · +140 (FD) · 61% conf · +11.2% edge
🎓 ST. JOHN'S · +128 (MGM) · 61% conf · +17.5% edge

3 locks today
```

**Fields per pick:** sport emoji, team name (all caps), best American odds with book in parens, model confidence as percentage, edge as signed percentage.

**No locks message:**
```
🔒 SLOP LOCKS · Mar 15
No locks today — nothing cleared the bar.
```

---

## What Gets Removed

| Component | Action |
|-----------|--------|
| `index.html`, `manifest.json` (root), `sw.js` | Delete |
| `icons/` directory | Delete |
| `netlify.toml` | Delete |
| `data/sotd.json` | Delete |
| `SLOP_LOCK_FALLBACK_MIN_ODDS` in `config.py` | Remove |
| `slop_locks` and `longslop` keys in `predictions.json` | Remove (schema simplification) |
| Slop lock pre-filtering logic in `run.py` | Move to `curator.py` |
| Pick history evaluation logic in `run.py` | Remove |

`backtest.py`, `pick_history.json`, and `history.json` are **retained** — backtest utilities (`compute_model_weights`, `get_rolling_accuracy`) are still used to weight the ensemble.

---

## Files Created/Modified

| File | Action |
|------|--------|
| `pipeline/curator.py` | **Create** — unified pick selection with guardrails |
| `pipeline/fetch_nhl.py` | **Create** — ESPN NHL data fetcher |
| `pipeline/notify_discord.py` | **Rewrite** — simplified Discord card |
| `pipeline/config.py` | **Modify** — add NHL config, add BOOKMAKERS list, remove SLOP_LOCK_FALLBACK_MIN_ODDS |
| `pipeline/run.py` | **Modify** — add NHL sport branch, remove pick evaluation loop, remove slop_lock pre-filtering, update predictions.json schema |
| `pipeline/refresh_picks.py` | **Modify** — update to invoke curator → notify flow |
| `data/nba/predictions.json`, `data/ncaam/predictions.json`, `data/nhl/predictions.json` | **Schema change** — remove `slop_locks`/`longslop` keys, keep `matches` |
| `index.html`, `sw.js`, `manifest.json`, `netlify.toml` | **Delete** |
