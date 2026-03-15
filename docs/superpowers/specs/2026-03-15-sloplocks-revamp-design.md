# SLOPLOCKS Revamp — Design Spec
**Date:** 2026-03-15  
**Status:** Approved

---

## Overview

Strip sloplocks down to its core purpose: generate 5 high-quality betting picks per day and post them to Discord. The website is gone. Backtest/ROI tracking is gone. The Discord card is the product.

The models (Elo, AdjustedEfficiency, FourFactors) are retained as-is. The rebuild targets the output layer: pick curation, line shopping, and Discord formatting.

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

`predictions.json` files (one per sport) contain raw candidate matches with model probs and edges. Their schema is simplified: the `slop_locks` and `longslop` keys are removed; only `matches` is retained. The curator reads these and does all selection.

---

## Pick Curation Guardrails (`curator.py`)

### Hard filters — all must pass
| Filter | Value | Rationale |
|--------|-------|-----------|
| Minimum edge | **8%** | Eliminates noise |
| Odds lower bound | **-130** American | No heavy favorites |
| Odds upper bound | **+350** American | Caps extreme longshots |

The odds filter applies to the **best available American odds** (the displayed line from line shopping), not to de-vigged probabilities.

No fallback — if no picks qualify, the card posts "no locks today."

### Ranking
Picks that pass all filters are ranked by **edge alone** (descending). Edge already incorporates model confidence implicitly.

### Cross-sport selection
Selection is purely global by edge rank. No per-sport cap.

### Card size
5 picks max. Posts fewer if quality isn't there.

---

## Line Shopping

### Books
BetMGM (`betmgm`), Caesars (`williamhill_us`), DraftKings (`draftkings`), FanDuel (`fanduel`), Bet365 (`bet365`), Bally Bet (`ballysports`)

### Market
Only **h2h (moneyline)** is fetched — this matches the existing pipeline (`ODDS_MARKETS = "h2h"`). One Odds API request per sport per run.

### Edge calculation (de-vigged)
For each book that returns a line:
1. Compute raw implied probabilities for both outcomes from decimal odds
2. De-vig via multiplicative normalization: divide each raw implied prob by their sum
3. Average the de-vigged fair probabilities across all books with available lines

`edge = model_prob − avg_fair_implied_prob`

**Minimum 2 books required.** If fewer than 2 books return a line for a game, the game is skipped. With exactly 2 books, outlier rejection is not possible — this is acknowledged and acceptable.

### Displayed odds
Best available American odds across available books + the book offering it.

**Book abbreviations in display:** DK, FD, MGM, CZR, B365, BALLY

---

## Discord Format

```
🔒 SLOP LOCKS · Mar 15

🏀 LAKERS · +135 (DK) · 54% model · +9.1% edge
🏒 BRUINS · +140 (FD) · 61% model · +11.2% edge
🎓 ST. JOHN'S · +128 (MGM) · 61% model · +17.5% edge

3 locks today
```

**Field definitions:**
- Sport emoji: 🏀 NBA, 🏒 NHL, 🎓 NCAAM
- Team name: the picked side, all caps
- Best American odds with book abbreviation
- `model` %: the ensemble's predicted probability for the picked outcome (`model_prob`)
- `edge` %: signed difference between model probability and de-vigged implied probability

**No-locks card:**
```
🔒 SLOP LOCKS · Mar 15
No locks today — nothing cleared the bar.
```

---

## API Budget

Only h2h moneylines are fetched — **1 Odds API request per sport per run**.

| Usage | Requests |
|-------|----------|
| Daily pipeline (3 sports) | 3/day → ~93/month |
| Manual refresh runs (est. 10/month × 3) | ~30/month |
| **Total estimate** | **~123/month** (well within 500 free) |

**On 429 / budget exhaustion:** The Odds API returns a 429 with remaining-requests in headers. If the limit is hit, odds fetching is skipped for that sport, no picks are generated from it that day, and the error is logged. The pipeline continues for other sports.

---

## Sports Coverage

### Retained (unchanged fetch modules)
- **NBA** — Elo + AdjustedEfficiency + FourFactors. `fetch_nba.py` is unchanged.
- **NCAAM** — same model stack. `fetch_ncaam.py` is unchanged. Odds API key: `basketball_ncaab` (confirmed available).

### New: NHL
- **Models:** Elo + AdjustedEfficiency (equal weights — FourFactors is basketball-specific)
- **Data source:** ESPN NHL API — free, no key required
  - Base URL: `https://site.api.espn.com/apis/site/v2/sports/hockey/nhl`
  - Key endpoints: `/scoreboard` (results + schedule), `/teams/{id}/statistics` (team stats for efficiency model)
- **New file:** `pipeline/fetch_nhl.py` — mirrors `fetch_ncaam.py` structure; returns `games` DataFrame (date, home_team, away_team, home_score, away_score) and `schedule` list
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

The existing GitHub Actions cron runs at **6am UTC (1am ET)**. Odds are stable and available at this time for evening games. Lines are not re-fetched during the day automatically.

The **manual refresh workflow** (`refresh-picks.yml`) is available for a same-day odds update. It is updated to invoke the new curator → notify flow (re-runs curator with fresh odds, posts a new Discord card).

---

## What Gets Removed

| Component | Action |
|-----------|--------|
| `index.html`, `manifest.json` (root), `sw.js` | Delete |
| `icons/` directory | Delete |
| `netlify.toml` | Delete |
| `data/sotd.json` | Delete |
| `SLOP_LOCK_FALLBACK_MIN_ODDS` in `config.py` | Remove |
| `slop_locks` and `longslop` keys in `predictions.json` | Remove |
| Slop lock pre-filtering logic in `run.py` | Move to `curator.py` |
| Pick history evaluation logic in `run.py` | Remove |

`backtest.py`, `pick_history.json`, and `history.json` are **retained** — `compute_model_weights` and `get_rolling_accuracy` are still used to weight the ensemble.

---

## Files Created / Modified / Deleted

| File | Action |
|------|--------|
| `pipeline/curator.py` | **Create** |
| `pipeline/fetch_nhl.py` | **Create** |
| `pipeline/notify_discord.py` | **Rewrite** |
| `pipeline/config.py` | **Modify** — add NHL, add BOOKMAKERS list, remove SLOP_LOCK_FALLBACK_MIN_ODDS |
| `pipeline/run.py` | **Modify** — add NHL branch, remove pick eval loop, remove slop_lock pre-filtering, update predictions.json schema (drop slop_locks/longslop keys) |
| `pipeline/refresh_picks.py` | **Modify** — invoke curator → notify flow |
| `pipeline/fetch_nba.py` | **Unchanged** |
| `pipeline/fetch_ncaam.py` | **Unchanged** |
| `pipeline/backtest.py` | **Unchanged** |
| `pipeline/models.py` | **Unchanged** |
| `pipeline/ensemble.py` | **Unchanged** |
| `index.html`, `sw.js`, `manifest.json` (root), `netlify.toml` | **Delete** |
